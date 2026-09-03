"""Export reviewed balloons as a YOLO-style training dataset.

Nothing here retrains a model -- this module only produces the dataset
artifacts (images, YOLO ``.txt`` labels, crops, class list, manifest) that a
separate, user-run training script (see README) would consume later.

Export rules:

* Every balloon becomes one row in the manifest, regardless of status, so
  the full history (including rejections) is preserved for analysis.
* Only ``accepted``/``edited`` auto proposals and manual additions that have
  a bounding box become positive YOLO labels. Rejected proposals are never
  exported as positive labels, per the "teach" workflow contract.
* A page is only rendered/exported if it has at least one balloon at all,
  to avoid flooding the dataset folder with blank pages.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PIL import Image

from balloon_app.config import AUTO_BALLOON_DPI, CHARACTERISTIC_CLASSES, DATASETS_IMAGES_DIR
from balloon_app.data_model import Balloon, BalloonSource, Project, ReviewStatus
from balloon_app.pdf_engine import PdfDocument, PdfLoadError, rect_pdf_to_pixel
from balloon_app.pdf_export import resolve_source_path

logger = logging.getLogger("balloon_app.training_export")

CLASS_TO_ID: dict[str, int] = {name: i for i, name in enumerate(CHARACTERISTIC_CLASSES)}


@dataclass
class TeachStats:
    """Summary statistics shown in the Teach / Training Data dialog."""

    total_balloons: int = 0
    auto_proposals: int = 0
    accepted_auto: int = 0
    edited_auto: int = 0
    rejected_auto: int = 0
    manual_additions: int = 0
    labeled_pages: int = 0
    model_versions: list[str] = field(default_factory=list)


def compute_teach_stats(project: Project) -> TeachStats:
    """Compute teach/training statistics purely from in-memory balloon data (no I/O)."""
    auto = [b for b in project.balloons if b.source == BalloonSource.AUTO.value]
    manual = [b for b in project.balloons if b.source == BalloonSource.MANUAL.value]
    accepted_auto = [b for b in auto if b.status == ReviewStatus.ACCEPTED.value]
    edited_auto = [b for b in auto if b.status == ReviewStatus.EDITED.value]
    rejected_auto = [b for b in auto if b.status == ReviewStatus.REJECTED.value]

    labeled_pages = {
        (b.drawing_id, b.page_number)
        for b in project.balloons
        if b.has_bbox()
        and (b.status in (ReviewStatus.ACCEPTED.value, ReviewStatus.EDITED.value) or b.source == BalloonSource.MANUAL.value)
    }
    versions = sorted({b.model_version for b in auto if b.model_version})

    return TeachStats(
        total_balloons=len(project.balloons),
        auto_proposals=len(auto),
        accepted_auto=len(accepted_auto),
        edited_auto=len(edited_auto),
        rejected_auto=len(rejected_auto),
        manual_additions=len(manual),
        labeled_pages=len(labeled_pages),
        model_versions=versions,
    )


@dataclass
class TrainingExportResult:
    output_root: Path
    image_count: int = 0
    label_count: int = 0
    positive_label_count: int = 0
    crop_count: int = 0
    manifest_jsonl_path: Optional[Path] = None
    manifest_csv_path: Optional[Path] = None
    classes_path: Optional[Path] = None
    skipped_drawings: list[str] = field(default_factory=list)


def _is_positive_label(balloon: Balloon) -> bool:
    if balloon.status == ReviewStatus.REJECTED.value:
        return False
    if balloon.status in (ReviewStatus.ACCEPTED.value, ReviewStatus.EDITED.value):
        return True
    return balloon.source == BalloonSource.MANUAL.value


def _write_classes_files(manifests_dir: Path) -> Path:
    txt_path = manifests_dir / "classes.txt"
    yaml_path = manifests_dir / "classes.yaml"

    txt_path.write_text("\n".join(CHARACTERISTIC_CLASSES) + "\n", encoding="utf-8")

    lines = ["# Class mapping for the BalloonApp rules/OCR + future YOLO pipeline.", "names:"]
    for i, name in enumerate(CHARACTERISTIC_CLASSES):
        lines.append(f"  {i}: {name}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def _manifest_row(
    balloon: Balloon,
    image_rel_path: str,
    crop_rel_path: Optional[str],
) -> dict:
    predicted = balloon.original_prediction or {}
    return {
        "balloon_id": balloon.id,
        "drawing_id": balloon.drawing_id,
        "image_path": image_rel_path,
        "crop_path": crop_rel_path,
        "page_number": balloon.page_number + 1,
        "bbox": list(balloon.bbox()) if balloon.has_bbox() else None,
        "predicted_type": predicted.get("char_type", balloon.char_type if balloon.source == "auto" else None),
        "final_type": balloon.char_type,
        "raw_text": balloon.raw_text,
        "final_nominal": balloon.nominal,
        "final_tol_plus": balloon.tol_plus,
        "final_tol_minus": balloon.tol_minus,
        "final_lower_limit": balloon.lower_limit,
        "final_upper_limit": balloon.upper_limit,
        "final_gdt_symbol": balloon.gdt_symbol,
        "final_gdt_tolerance": balloon.gdt_tolerance,
        "final_datums": balloon.datums,
        "status": balloon.status,
        "source": balloon.source,
        "confidence": balloon.confidence,
        "model_version": balloon.model_version,
        "exported_as_positive_label": _is_positive_label(balloon) and balloon.has_bbox(),
    }


def export_training_dataset(
    project: Project,
    output_root: Path | str = DATASETS_IMAGES_DIR.parent,
    dpi: float = AUTO_BALLOON_DPI,
) -> TrainingExportResult:
    """Export images, YOLO labels, crops, class map, and manifest for ``project``."""
    output_root = Path(output_root)
    images_dir = output_root / "images"
    labels_dir = output_root / "labels"
    crops_dir = output_root / "crops"
    manifests_dir = output_root / "manifests"
    for d in (images_dir, labels_dir, crops_dir, manifests_dir):
        d.mkdir(parents=True, exist_ok=True)

    result = TrainingExportResult(output_root=output_root)
    result.classes_path = _write_classes_files(manifests_dir)

    manifest_rows: list[dict] = []

    for drawing in project.drawings:
        page_balloons_map: dict[int, list[Balloon]] = {}
        for b in project.balloons_for(drawing.id):
            page_balloons_map.setdefault(b.page_number, []).append(b)
        if not page_balloons_map:
            continue

        source_path = resolve_source_path(drawing)
        if source_path is None:
            logger.warning("Skipping drawing '%s': source PDF not found", drawing.file_name)
            result.skipped_drawings.append(drawing.file_name or drawing.id)
            continue

        try:
            pdf_doc = PdfDocument(source_path)
            pdf_doc.open()
        except PdfLoadError as exc:
            logger.warning("Skipping drawing '%s': %s", drawing.file_name, exc)
            result.skipped_drawings.append(drawing.file_name or drawing.id)
            continue

        try:
            for page_number, page_balloons in sorted(page_balloons_map.items()):
                try:
                    rgb_bytes, width, height = pdf_doc.render_page_rgb(page_number, dpi)
                except Exception:
                    logger.exception(
                        "Failed to render page %d of drawing '%s' for training export",
                        page_number, drawing.file_name,
                    )
                    continue

                image = Image.frombytes("RGB", (width, height), rgb_bytes)
                stem = f"{drawing.id}_p{page_number + 1}"
                image_path = images_dir / f"{stem}.png"
                image.save(image_path)
                result.image_count += 1

                label_lines: list[str] = []
                for balloon in page_balloons:
                    crop_rel_path = None
                    if balloon.has_bbox():
                        px0, py0, px1, py1 = rect_pdf_to_pixel(balloon.bbox(), dpi)  # type: ignore[arg-type]
                        px0, px1 = sorted((max(0, min(width, px0)), max(0, min(width, px1))))
                        py0, py1 = sorted((max(0, min(height, py0)), max(0, min(height, py1))))
                        if px1 - px0 >= 2 and py1 - py0 >= 2:
                            crop = image.crop((int(px0), int(py0), int(px1), int(py1)))
                            crop_name = f"{stem}_b{balloon.number}_{balloon.id[:8]}.png"
                            crop.save(crops_dir / crop_name)
                            crop_rel_path = str((crops_dir / crop_name).relative_to(output_root)).replace("\\", "/")
                            result.crop_count += 1

                            if _is_positive_label(balloon):
                                class_id = CLASS_TO_ID.get(balloon.char_type, CLASS_TO_ID.get("other", len(CHARACTERISTIC_CLASSES) - 1))
                                x_center = ((px0 + px1) / 2.0) / width
                                y_center = ((py0 + py1) / 2.0) / height
                                box_w = (px1 - px0) / width
                                box_h = (py1 - py0) / height
                                label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

                    manifest_rows.append(
                        _manifest_row(
                            balloon,
                            image_rel_path=str(image_path.relative_to(output_root)).replace("\\", "/"),
                            crop_rel_path=crop_rel_path,
                        )
                    )

                label_path = labels_dir / f"{stem}.txt"
                label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
                result.label_count += 1
                result.positive_label_count += len(label_lines)
        finally:
            pdf_doc.close()

    result.manifest_jsonl_path = manifests_dir / "manifest.jsonl"
    _atomic_write_text(
        result.manifest_jsonl_path, "\n".join(json.dumps(row) for row in manifest_rows) + ("\n" if manifest_rows else "")
    )

    result.manifest_csv_path = manifests_dir / "manifest.csv"
    _write_manifest_csv(result.manifest_csv_path, manifest_rows)

    logger.info(
        "Training export complete: %d images, %d label files (%d positive boxes), %d crops -> %s",
        result.image_count, result.label_count, result.positive_label_count, result.crop_count, output_root,
    )
    return result


def _write_manifest_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "balloon_id", "drawing_id", "image_path", "crop_path", "page_number", "bbox",
        "predicted_type", "final_type", "raw_text", "final_nominal", "final_tol_plus",
        "final_tol_minus", "final_lower_limit", "final_upper_limit", "final_gdt_symbol",
        "final_gdt_tolerance", "final_datums", "status", "source", "confidence",
        "model_version", "exported_as_positive_label",
    ]
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".csv")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                out_row = dict(row)
                if out_row.get("bbox") is not None:
                    out_row["bbox"] = json.dumps(out_row["bbox"])
                writer.writerow(out_row)
        os.replace(tmp_name, path)
    except Exception:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise


def _atomic_write_text(path: Path, text: str) -> None:
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".jsonl")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_name, path)
    except Exception:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise

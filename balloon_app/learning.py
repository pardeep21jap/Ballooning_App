"""Small, local, review-driven memory for recurring drawing callouts."""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

from balloon_app.config import DATASETS_MANIFESTS_DIR
from balloon_app.data_model import Balloon, ReviewStatus

MEMORY_PATH = DATASETS_MANIFESTS_DIR / "learning_memory.json"
_FIELDS = ("char_type", "nominal", "tol_plus", "tol_minus", "lower_limit", "upper_limit",
           "gdt_symbol", "gdt_tolerance", "material_condition", "datums", "surface_finish",
           "thread_callout", "note", "inspection_method", "critical")


def normalize_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().upper())


def load_memory(path: Path = MEMORY_PATH) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(data: dict, path: Path = MEMORY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=str(path.parent), prefix=".learning_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
        os.replace(name, path)
    except Exception:
        if os.path.exists(name):
            os.remove(name)
        raise


def learn_from_balloon(balloon: Balloon, path: Path = MEMORY_PATH) -> None:
    """Persist explicit review feedback. Two consistent rejections suppress an exact callout."""
    key = normalize_key(balloon.raw_text)
    if not key:
        return
    data = load_memory(path)
    entry = data.setdefault(key, {"accepted": 0, "rejected": 0, "correction": None})
    if balloon.status == ReviewStatus.REJECTED.value:
        entry["rejected"] = int(entry.get("rejected", 0)) + 1
    elif balloon.status in (ReviewStatus.ACCEPTED.value, ReviewStatus.EDITED.value):
        entry["accepted"] = int(entry.get("accepted", 0)) + 1
        if balloon.status == ReviewStatus.EDITED.value:
            entry["correction"] = {field: getattr(balloon, field) for field in _FIELDS}
    _save(data, path)


def apply_learned_feedback(balloons: list[Balloon], path: Path = MEMORY_PATH) -> tuple[list[Balloon], int, int]:
    data = load_memory(path)
    kept: list[Balloon] = []
    corrected = suppressed = 0
    for balloon in balloons:
        entry = data.get(normalize_key(balloon.raw_text), {})
        if int(entry.get("rejected", 0)) >= 2 and not int(entry.get("accepted", 0)):
            suppressed += 1
            continue
        correction = entry.get("correction")
        if isinstance(correction, dict):
            for field in _FIELDS:
                if field in correction:
                    setattr(balloon, field, correction[field])
            balloon.confidence = max(balloon.confidence, 0.95)
            balloon.note = (balloon.note + " | " if balloon.note else "") + "Applied from local learning memory"
            balloon.snapshot_prediction()
            corrected += 1
        kept.append(balloon)
    return kept, corrected, suppressed


def memory_stats(path: Path = MEMORY_PATH) -> tuple[int, int]:
    data = load_memory(path)
    corrections = sum(bool(v.get("correction")) for v in data.values())
    suppressed = sum(int(v.get("rejected", 0)) >= 2 and not int(v.get("accepted", 0)) for v in data.values())
    return corrections, suppressed

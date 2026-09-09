# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

BalloonIQ is an offline, local-first Windows desktop app (PyQt6) for
"ballooning" mechanical-engineering PDF drawings: it proposes numbered
balloons for dimensions, tolerances, GD&T frames, threads, and surface
finish callouts on a PDF, lets an engineer review/correct them, then
exports a ballooned PDF and an Excel inspection sheet. See `README.md` for
the full user-facing feature/workflow description (menus, shortcuts,
project-data layout, YOLO training workflow) — it's kept up to date and
worth reading before making UI or workflow changes.

## Commands

Working directory for all commands below: `BalloonApp/` (this file's
directory).

```powershell
# One-time setup
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt   # core deps only
pip install pytest                # needed for the test suite (commented out of requirements.txt)

# Run the app
.\.venv\Scripts\pythonw.exe main.py   # no console window
.\.venv\Scripts\python.exe main.py    # console window, shows startup errors/logs

# Tests
.\.venv\Scripts\python.exe -m pytest balloon_app/tests -v
.\.venv\Scripts\python.exe -m pytest balloon_app/tests/test_ocr_parser.py -v          # one file
.\.venv\Scripts\python.exe -m pytest balloon_app/tests/test_ocr_parser.py::TestThread::test_metric_thread_class_without_pitch -v  # one test
.\.venv\Scripts\python.exe -m pytest balloon_app/tests -k "thread" -v                 # by keyword

# Packaging (Windows EXE via PyInstaller)
pip install pyinstaller
build_windows.bat   # -> dist\BalloonIQ\BalloonIQ.exe (does not bundle projects/, datasets/, models/)
```

There is no linter/formatter configured in this repo (no ruff/flake8/black
config) and no `pyproject.toml`/`pytest.ini` — pytest runs with defaults.

## Architecture

### Auto-ballooning pipeline (the core, non-obvious part of this codebase)

`auto_balloon.auto_balloon_page` runs a multi-stage pipeline over one PDF
page; each stage is a separate module and getting the order/interaction
right matters more than any single function:

1. **Native text extraction** (`pdf_engine.PdfDocument.extract_text_blocks`):
   one `TextBlock` per PDF text *span* (not per line/block), in PDF point
   coordinates.
2. **Span merging** (`auto_balloon._merge_nearby_text_blocks`,
   `_merge_stacked_tolerance_fragments`): CAD PDF exporters often split one
   logical dimension ("50.00", "±0.05", a quantity prefix, a stacked
   +/-tolerance on its own line) across multiple spans/fonts/baselines.
   These merges are deliberately conservative — see
   `_merge_gap_is_safe`'s docstring for why a row of independent ordinate/
   chain dimensions must *not* be merged into one (it looks geometrically
   identical to a value split from its own tolerance/quantity prefix).
3. **Vector-drawn symbol hints** (`pdf_engine.find_vector_diameter_symbol`,
   `find_vector_gdt_frame`, `find_vector_flatness_symbol`,
   `find_vector_depth_symbol`, `find_vector_circle_around_text`): some CAD
   PDF exporters draw a GD&T/dimensioning symbol (Ø, a feature-control
   frame, the depth glyph, a circled item-reference number) as traced
   vector line art instead of a font character, so it never appears in the
   extracted text at all — not even as a mangled substitute. Each detector
   is a best-effort geometric heuristic (segment clustering + size/aspect/
   structural checks) run on the *raw, pre-merge* blocks (merging can
   distort a bbox's "immediately to the left" geometry) and passed down as
   a `*_hint` boolean. A hint only affects the lowest-priority fallback
   classification in `ocr_parser.parse_characteristics` — it never
   overrides a real symbol/pattern already present in the text.
4. **Pre-filter** (`auto_balloon._looks_like_characteristic`): a cheap
   textual gate (decimal number present, or a known symbol/keyword hint)
   that runs *before* a candidate ever reaches the parser. This has its own
   set of hint regexes mirroring what the parser recognizes (e.g. a
   thread's class-only form "M4-6H" with no pitch, or an ISO 228 pipe
   thread "G1/2""), because something the parser could handle fine is
   otherwise discarded here first and never even attempted.
5. **Classification** (`ocr_parser.parse_characteristics`): a regex-based
   rule engine, tried in a specific priority order (thread → surface
   finish/GD&T/general tolerance → shape+secondary-value split → quantity-
   prefixed patterns → mangled-symbol fallbacks → generic symbol/keyword
   detection → bare numeric fallback → note). It returns a *list*, not a
   single result: one line of drawing text often bundles two independently
   inspected requirements (e.g. a tapped hole's thread class *and* its
   depth), and each becomes its own balloon.
6. OCR (Tesseract, via `RulesOcrDetector`) is only used as a fallback when a
   page has no native/selectable text at all (scanned drawings); an
   optional `YoloDetector` can be swapped in from Settings if a trained
   `.pt` model is supplied, otherwise the app silently keeps using the
   rules/OCR path.

When fixing a "wrong/missing balloon" bug, the first question is *which*
stage dropped or misclassified the text — inspect `find_vector_*` hints
before assuming it's an `ocr_parser` regex issue. A recurring real-world
gotcha: CAD-exported dimension text can substitute a typographic dash (en
dash) for a plain hyphen, or draw a symbol as pure vector art with zero
text characters — the fix is almost always to widen the relevant character
class/hint rather than special-case the drawing.

### Data model & persistence

- `data_model.py` — plain dataclasses (`Project` → `Drawing` → `Balloon`),
  with `to_dict`/`from_dict` used by both SQLite and JSON.
- `database.py` — one project = one portable SQLite `.bpdb` file
  (`ProjectDatabase`), plus JSON export/import as a plain-text backup
  format. Original PDFs are **never copied**; the project stores the
  source PDF's file path only.
- `config.AppSettings` — separate from project data: Tesseract path, DPI,
  confidence threshold, recent projects, and per-font "learned symbol"
  mappings, stored via Qt `QSettings` (Windows registry), not in the
  `.bpdb` file.
- Every `Balloon` snapshots its `original_prediction` right after an
  auto-detector creates it (`Balloon.snapshot_prediction`); comparing
  current fields against that snapshot is how accepted/edited/rejected
  status and training-export "predicted vs. final" stats are derived.

### Feedback/learning loop

Two independent, both-local mechanisms consume the same review data:

- `learning.py` — a small per-normalized-raw-text memory
  (`datasets/manifests/learning_memory.json`): two rejections of the exact
  same callout text suppress it next time; an edited/corrected balloon's
  field values are replayed onto future matches of that same text.
- `training_export.py` — exports a YOLO-style labeled dataset
  (images/labels/crops/manifests) from reviewed balloons, for eventually
  training a real detector to replace the regex/OCR pipeline.

### UI layer

`app.py` (`MainWindow`) owns menus/toolbar/wiring and all `QUndoCommand`
subclasses (undo/redo goes through Qt's command stack, not ad hoc state
mutation). `pdf_view.py` is the `QGraphicsView`/`Scene`-based PDF viewer
with the balloon overlay kept in the same PDF coordinate space
`pdf_engine.py` uses. `dialogs.py` holds all secondary dialogs (balloon
edit, settings, teach/training summary, etc.).

### Tests

`balloon_app/tests/` mirrors the module split 1:1. Tests for the
PDF-vector-geometry detectors and the full auto-balloon pipeline build
tiny synthetic PDFs on the fly with PyMuPDF (`fitz.open()`,
`page.insert_text(...)`, `page.draw_line(...)`) rather than fixture files —
follow that pattern when adding a regression test for a parsing/detection
bug rather than sourcing a real drawing.

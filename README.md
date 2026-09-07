# BalloonIQ

BalloonIQ is an **offline, local-first Windows desktop application** for
ballooning mechanical-engineering PDF drawings and generating a generic
Excel inspection sheet. It is built for manufacturing, quality, machining,
injection-molding, and inspection engineers who need to turn a PDF drawing
into a numbered set of inspection characteristics quickly.

You open a PDF drawing, the app proposes numbered balloons for likely
dimensions, tolerances, GD&T frames, threads, and surface-finish callouts,
you review/correct those proposals, and then export:

1. A **ballooned PDF** (vector overlay on a copy of the original drawing).
2. A **generic Excel inspection sheet** (`.xlsx`) with PASS/FAIL formulas.

Every correction you make while reviewing is preserved as local feedback
data, which can later be exported as a YOLO-style object-detection dataset
to train a real vision model -- see [Future YOLO Training
Workflow](#future-yolo-training-workflow) below.

All project data, drawings references, exports, and settings stay on your
computer. No cloud services, accounts, or paid APIs are required.

---

## Key current limitations

Please read this before relying on the tool in production:

- **Automatic detection is assistive, not a replacement for engineering
  review.** The rule-based parser and OCR fallback are meant to speed up
  ballooning, not to be authoritative. Always review every proposed balloon
  before exporting.
- **OCR accuracy varies** with PDF/scan quality, font, resolution, and
  notation style. Low-quality scans will produce more misses and more text
  that needs manual correction.
- **Complex GD&T feature control frames** (especially composite frames,
  unusual symbol fonts, or heavily stylized datums) may not be fully parsed
  and often need manual correction of the symbol, tolerance, material
  condition, and datum fields.
- **GD&T symbol vision** supports all 14 chart symbols: straightness,
  flatness, circularity, cylindricity, angularity, perpendicularity,
  parallelism, position, concentricity, symmetry, line profile, surface
  profile, circular runout, and total runout. It examines upright, enclosed
  feature-control frames in rendered PDFs, including vector symbols and
  scanned images. Weak or ambiguous matches are left unnamed. Scanned
  tolerance/datum text requires Tesseract; symbol recognition itself does
  not. Heavily skewed, broken, tiny, or unusual symbols may need correction.
- The bundled auto-ballooning pipeline is a **transparent, regex-based rule
  engine with geometric symbol matching** (`rules_ocr_v2_gdt`), not a trained machine-learning model. A YOLO
  detector interface exists and is fully wired in, but no pretrained model
  ships with the app -- you would need to train and supply your own `.pt`
  file (see below).
- Long-running operations (auto-ballooning, exporting) run on a background
  thread so the UI stays responsive, but they cannot be forcibly cancelled
  mid-run in this version -- wait for them to finish.

---

## Windows 11 prerequisites

- Windows 11 (Windows 10 should also work, but is not the primary target).
- [Python 3.11 or newer](https://www.python.org/downloads/windows/) (64-bit
  recommended). During installation, check **"Add python.exe to PATH."**
- No administrator rights are required to run or build the app.

## Installing Python

1. Download the Windows installer from python.org (3.11+).
2. Run it, check **"Add python.exe to PATH"**, then **Install Now**.
3. Verify in a new terminal:

   ```
   python --version
   ```

   If `python` is not found but you installed Python, try `py -3 --version`
   instead (the Python Launcher for Windows).

## Setting up a virtual environment

From the `BalloonApp` project folder:

```
python -m venv .venv
.venv\Scripts\activate
```

Your prompt should now start with `(.venv)`.

## Installing dependencies

With the virtual environment activated:

```
pip install -r requirements.txt
```

This installs the **core** dependencies only (PyQt6, PyMuPDF, Pillow,
openpyxl, numpy, pytesseract, opencv-python). Optional extras
(`ultralytics` for YOLO, `pandas`, `pytest`, `pyinstaller`) are commented
out in `requirements.txt` -- uncomment the ones you want, or install them
individually, e.g.:

```
pip install pytest
pip install pyinstaller
```

The application **starts and runs its core workflow even if none of the
optional packages are installed.**

## Installing Tesseract on Windows (optional)

Tesseract OCR is only needed to auto-balloon pages that have **no
selectable/native text** -- i.e. scanned or rasterized drawings. Drawings
with real vector text (most CAD-exported PDFs) do not need OCR at all.

1. Download a Windows installer from the [UB-Mannheim Tesseract
   build](https://github.com/UB-Mannheim/tesseract/wiki) (a common,
   actively maintained Windows build).
2. Run the installer. Note the install path, typically:
   `C:\Program Files\Tesseract-OCR\tesseract.exe`
3. Either:
   - Add that folder to your system `PATH`, **or**
   - Open BalloonIQ's **Tools -> Settings** dialog and paste the full path
     to `tesseract.exe` into the "Tesseract Path" field, **or**
   - Set the `TESSERACT_PATH` environment variable to the full executable
     path before launching the app, e.g. in PowerShell:

     ```
     $env:TESSERACT_PATH = "C:\Program Files\Tesseract-OCR\tesseract.exe"
     ```

If Tesseract is not installed and not configured, BalloonIQ will not
crash -- it shows a clear status message when a page has no native text and
OCR is unavailable, and lets you continue ballooning that page manually.

## Running the app

After completing the installation steps above, open **PowerShell** (or the
PowerShell terminal in your IDE) and run:

```powershell
cd D:\Websites\Ballooning_App\BalloonApp
.\.venv\Scripts\pythonw.exe main.py
```

This starts the latest source version without a console window. No virtual
environment activation is needed. If you installed the project elsewhere,
replace the folder path with your own.

To see startup errors and logs in the terminal, use:

```powershell
.\.venv\Scripts\python.exe main.py
```

Alternatively, from the `BalloonApp` folder with the virtual environment
already activated:

```powershell
python main.py
```

After code updates, save your work and close the running app, then launch
it again with these commands. An existing `.exe` in `dist` will only include
the latest changes after it is rebuilt.

## How project data is stored

- **Projects**: each project is a single portable SQLite file with the
  `.bpdb` extension, stored by default under `projects/<project_name>/`.
  You can also **Save Project As** anywhere, or use **File > (future)
  JSON export/import** logic already implemented in `database.py`
  (`ProjectDatabase.export_json` / `.import_json`) for a plain-text,
  portable backup format.
- **Original PDFs are never copied.** The project stores the source PDF's
  file path. If the file has moved, use **File -> Relink Current
  Drawing...** to point the project at its new location.
- **Datasets** (for future YOLO training) are written under `datasets/`:
  `datasets/images`, `datasets/labels`, `datasets/crops`,
  `datasets/manifests`.
- **Models**: place a trained YOLO `.pt` file under `models/` and select it
  in **Tools -> Settings** to enable the (optional) ML detector.
- **Logs**: a rotating log file is written to `logs/balloon_app.log` for
  troubleshooting.
- **App settings** (Tesseract path, DPI, confidence threshold, recent
  projects, etc.) are stored using Qt's `QSettings` (Windows registry
  under `HKEY_CURRENT_USER`), not in a project file.

## How to use manual ballooning

1. **File -> New Project...** (or **Open Project...**) to start.
2. **File -> Add/Open PDF Drawing...** (`Ctrl+O`) to attach a PDF.
3. Toggle **Edit -> Add Balloon Mode** (`Ctrl+B`), then click anywhere on
   the drawing to place a balloon. A dialog opens immediately so you can
   fill in the characteristic type, nominal, tolerance, GD&T fields, etc.
4. Drag an existing balloon to reposition it; double-click a balloon to
   edit it; select a balloon and press **Delete** to remove it.
5. Use **Edit -> Set Leader Point for Selected Balloon** to draw a leader
   line from the balloon to a specific point on the drawing.
6. Use **Edit -> Duplicate Balloon** (`Ctrl+D`) to clone a balloon (handy
   for a row of identical/similar features).
7. Use **Edit -> Renumber Balloons...** to renumber by current page, by the
   whole drawing (page order, then top-to-bottom/left-to-right), or to just
   compact existing numbers while preserving their current order.

## How to use auto-ballooning

1. Open a PDF drawing as above.
2. **Tools -> Auto-Balloon Current Page** scans just the visible page;
   **Tools -> Auto-Balloon Entire Drawing** scans every page.
3. The pipeline:
   1. Renders the page at a configurable DPI (default 300 for
      auto-ballooning).
   2. Extracts **native PDF text** first (fast, precise, no OCR needed for
      most CAD-exported drawings).
   3. Falls back to **Tesseract OCR** only if a page has no native/selected
      text (e.g. a scanned drawing) -- and only if Tesseract is available.
   4. Classifies each candidate chunk of text with a **regex-based rule
      engine** (`balloon_app/ocr_parser.py`) into a characteristic type
      (linear dimension, diameter, radius, angle, thread, GD&T frame,
      surface finish, general tolerance, or note) and extracts nominal,
      tolerance, limits, GD&T symbol/tolerance/datums, surface finish, or
      thread callout as applicable.
   5. Places a balloon near each detected callout, nudged to avoid
      overlapping already-placed balloons.
4. All auto-generated balloons are created with status **pending** (shown
   in orange) so nothing gets exported without your review.

## How to review and teach it

The right-hand **review panel** lists every characteristic for the active
drawing:

- **Filter** by All / Pending / Accepted / Edited / Rejected / Manual /
  Auto.
- **Search** by balloon number, raw text, or nominal.
- **Sort** by Page, Number, Confidence, or Status.
- Select a row (or click a balloon on the drawing) and use **Accept**,
  **Edit**, or **Reject**.
- **Accept All Above Threshold...** accepts every pending proposal at or
  above a confidence value you choose.
- **Reject Selected/Pending** rejects your current selection, or (if
  nothing is selected) every remaining pending proposal.
- **Add Manual** creates a brand-new characteristic directly from the
  review panel (placed at the center of the current page; move it
  afterward as needed).

Every edit is tracked: the balloon's **original auto-prediction is kept**
alongside your corrected values, and its status becomes `edited`
automatically if you change a pending/accepted auto-proposal's data. This
is exactly the feedback BalloonIQ uses to build a training dataset.

Open **Tools -> Teach / Training Data...** to see a summary (total
balloons, auto proposals, accepted/edited/rejected counts, manual
additions, labeled pages available, and the detector/model version used)
and to run **Export Training Dataset**.

## How to export Excel and ballooned PDF

- **File -> Export Excel Inspection Sheet...** (`Ctrl+E`): choose whether
  to include pending proposals, then pick a save location. The workbook has
  two sheets:
  - **Inspection Data**: one row per characteristic, with a formula-driven
    **Result** column that shows `PASS`/`FAIL` once you type a value into
    **Actual** (and stays blank otherwise, or if no numeric limits apply).
  - **Project Info**: project metadata and a clickable link back to the
    original source PDF.
  - By default, only **accepted**, **edited**, and **manually added**
    characteristics are exported; rejected proposals are never exported.
- **File -> Export Ballooned PDF...**: choose whether to include pending
  proposals (drawn in orange) and/or rejected ones (muted red), then pick a
  save location. The export draws directly onto a copy of the original PDF
  (vector-preserving) whenever possible; if that fails for a particular
  file, it automatically falls back to a high-resolution raster copy with
  the same overlay, and tells you it did so.

## Keyboard shortcuts

| Shortcut | Action |
|---|---|
| `Ctrl+O` | Add/Open PDF drawing |
| `Ctrl+S` | Save Project |
| `Ctrl+Shift+S` | Save Project As |
| `Ctrl+E` | Export Excel Inspection Sheet |
| `Ctrl+B` | Toggle Add Balloon mode |
| `Ctrl+D` | Duplicate selected balloon |
| `Delete` | Delete selected balloon(s) |
| `Ctrl+Z` / `Ctrl+Y` | Undo / Redo |
| `Ctrl+=` / `Ctrl+-` | Zoom In / Zoom Out |
| `Ctrl+0` | Fit Page |
| `PageUp` / `PageDown` | Previous / Next page |
| `Ctrl+Wheel` | Zoom at cursor |

## Running the tests

```
pip install pytest
pytest balloon_app/tests -v
```

Tests cover: tolerance/GD&T/thread/surface-finish parsing
(`test_ocr_parser.py`), Excel export headers/formulas/filtering
(`test_excel_export.py`), SQLite + JSON persistence
(`test_database.py`), the core data model (`test_data_model.py`), and
coordinate-conversion helpers (`test_pdf_engine.py`).

## Packaging into a Windows EXE (PyInstaller)

```
pip install pyinstaller
build_windows.bat
```

`build_windows.bat` will use your existing `.venv` if present (creating one
otherwise is not required, but recommended), install PyInstaller if
missing, and produce a one-folder build under `dist\BalloonIQ\` containing
`BalloonIQ.exe` plus all dependencies. It bundles the `balloon_app`
package and the `resources` folder; it does **not** bundle your
`projects/`, `datasets/`, or `models/` folders (those stay external, next
to the built executable, so your data isn't locked inside the build).

## Future YOLO training workflow

BalloonIQ **does not train models itself**. It only exports a dataset from
your reviewed balloons via **Tools -> Teach / Training Data -> Export
Training Dataset** (or programmatically via
`balloon_app.training_export.export_training_dataset`). The export
produces, under `datasets/`:

- `images/<drawing_id>_p<page>.png` -- full rendered page images (one per
  page that has at least one balloon).
- `labels/<drawing_id>_p<page>.txt` -- YOLO-format labels
  (`class_id x_center y_center width height`, normalized 0-1) for
  **accepted/edited auto proposals and manual additions only**. Rejected
  proposals are never written as positive labels.
- `crops/*.png` -- a cropped image for every balloon that has a bounding
  box (including rejected ones, for later analysis/hard-negative mining).
- `manifests/classes.txt` and `manifests/classes.yaml` -- the fixed class
  list (`balloon_app/config.py:CHARACTERISTIC_CLASSES`), in class-id order.
- `manifests/manifest.jsonl` and `manifests/manifest.csv` -- one row per
  balloon (any status), with predicted vs. final type, raw text, final
  nominal/tolerance/GD&T, status, confidence, model version, and whether it
  was exported as a positive label.

To actually train a YOLO model later (outside of BalloonIQ), a typical
workflow looks like:

```
pip install ultralytics

# 1. Split datasets/images + datasets/labels into train/val folders
#    yourself (or with a small helper script) based on manifest.csv.

# 2. Write a data.yaml, e.g.:
#    path: ./datasets
#    train: images/train
#    val: images/val
#    names:
#      0: linear_dimension
#      1: diameter
#      ...  (see datasets/manifests/classes.yaml)

# 3. Train:
yolo detect train data=data.yaml model=yolov8n.pt epochs=100 imgsz=1280
```

Then, in BalloonIQ's **Tools -> Settings**, point "YOLO Model (.pt) Path"
at your trained weights (or drop the file into `models/`) and check "Use
YOLO model when available." If `ultralytics` is not installed or the model
file is missing/unreadable, the app automatically and silently falls back
to the rules/OCR detector -- it will never fail to start because of a
missing or broken model.

## Project structure

```
BalloonApp/
├── main.py                  Entry point (python main.py)
├── requirements.txt
├── build_windows.bat
├── balloon_app/
│   ├── config.py             Paths, settings, constants, logging setup
│   ├── app.py                MainWindow: menus, toolbar, wiring, undo/redo
│   ├── data_model.py         Project / Drawing / Balloon dataclasses
│   ├── database.py           SQLite persistence + JSON export/import
│   ├── pdf_engine.py         PyMuPDF wrapper + coordinate conversion
│   ├── pdf_view.py           QGraphicsView PDF viewer + balloon overlay
│   ├── pdf_export.py         Ballooned PDF export (vector + raster fallback)
│   ├── excel_export.py       Inspection workbook export (openpyxl)
│   ├── auto_balloon.py       Detector interface + auto-ballooning pipeline
│   ├── ocr_parser.py         Regex-based dimension/GD&T/thread parser
│   ├── training_export.py    YOLO dataset export + teach-dialog stats
│   ├── dialogs.py            All secondary dialogs
│   └── tests/                pytest test suite
├── projects/                 Your saved projects (.bpdb files), gitignored
├── datasets/                 Exported training data, gitignored
└── models/                   Drop a trained .pt file here, gitignored
```

## Troubleshooting

- **The app won't start / ImportError for PyQt6 etc.**: make sure your
  virtual environment is activated and `pip install -r requirements.txt`
  completed without errors.
- **Auto-balloon finds nothing on a scanned drawing**: install Tesseract
  (see above) and confirm **Tools -> Settings** shows the correct path, or
  that `tesseract` is on your system `PATH`.
- **A drawing shows "Drawing Not Found"**: the source PDF was moved or
  renamed. Use **File -> Relink Current Drawing...** to point at its new
  location.
- Check `logs/balloon_app.log` for detailed error messages.

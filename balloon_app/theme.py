"""Consistent desktop themes for the application and its dialogs.

The light theme ("Modernist") is the app's own visual identity: deep-navy
text and chrome on an off-white ground, a single sky-blue accent, and a
fully square, hairline-bordered look (no rounded corners, no gradients).
Every color used below is one flat value or an alpha-blend of it -- no
per-widget one-offs -- so the whole app reads as one system.
"""

from pathlib import Path

from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication

# ---------------------------------------------------------------------------
# "Modernist" light theme palette. Keep in sync with the RGBA tuples in
# config.py's status_color() -- those drive the balloon badges/status chips
# painted directly by app.py, so the two must read as one family of color.
# ---------------------------------------------------------------------------
_INK = "#002049"           # primary text / chrome
_INK_SOFT = "#4d6483"      # secondary text (ink at ~65% over the page bg)
_ACCENT = "#0098f8"        # primary accent
_ACCENT_HOVER = "#007cd0"
_ACCENT_PRESSED = "#005e9d"
_ACCENT_TINT = "#dceefe"   # accent at low opacity, for hover/selection fills
_PAGE_BG = "#f3f2f2"
_SURFACE = "#eae7e7"       # panels, table headers, toolbars
_SURFACE_ALT = "#f8f4f4"   # alternating table rows
_BORDER = "rgba(0, 32, 73, 0.22)"
_BORDER_STRONG = "rgba(0, 32, 73, 0.4)"
_LINE = "#d7d3d3"
_DISABLED = "#9b9797"

_RESOURCE_DIR = Path(__file__).parent / "resources"
_SPIN_UP_LIGHT = (_RESOURCE_DIR / "spin-up-light.svg").as_posix()
_SPIN_DOWN_LIGHT = (_RESOURCE_DIR / "spin-down-light.svg").as_posix()
_SPIN_UP_DARK = (_RESOURCE_DIR / "spin-up-dark.svg").as_posix()
_SPIN_DOWN_DARK = (_RESOURCE_DIR / "spin-down-dark.svg").as_posix()


def apply_theme(app: QApplication, theme: str) -> None:
    if theme == "dark":
        apply_dark_theme(app)
    else:
        apply_light_theme(app)


def apply_light_theme(app: QApplication) -> None:
    """The app's default look: flat, square-cornered, navy-on-off-white
    with a single sky-blue accent."""
    app.setStyle("Fusion")
    palette = QPalette()
    colors = {
        QPalette.ColorRole.Window: _PAGE_BG,
        QPalette.ColorRole.WindowText: _INK,
        QPalette.ColorRole.Base: "#ffffff",
        QPalette.ColorRole.AlternateBase: _SURFACE_ALT,
        QPalette.ColorRole.Text: _INK,
        QPalette.ColorRole.Button: "#ffffff",
        QPalette.ColorRole.ButtonText: _INK,
        QPalette.ColorRole.Highlight: _ACCENT,
        QPalette.ColorRole.HighlightedText: "#ffffff",
        QPalette.ColorRole.ToolTipBase: _INK,
        QPalette.ColorRole.ToolTipText: "#ffffff",
        QPalette.ColorRole.PlaceholderText: _DISABLED,
        QPalette.ColorRole.Link: _ACCENT_HOVER,
        QPalette.ColorRole.Light: "#ffffff",
        QPalette.ColorRole.Midlight: _SURFACE,
        QPalette.ColorRole.Mid: _LINE,
        QPalette.ColorRole.Dark: _DISABLED,
        QPalette.ColorRole.Shadow: "#00000040",
    }
    for role, color in colors.items():
        palette.setColor(role, QColor(color))
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor(_DISABLED))
    app.setPalette(palette)
    app.setStyleSheet(f"""
        QLabel#brandHeader {{ background: {_INK}; color: #f3f2f2; font-weight: 600; padding: 0 12px; }}
        QLabel#panelTitle {{ font-size: 15px; font-weight: 700; }}
        QLabel#mutedLabel {{ color: {_INK_SOFT}; font-size: 11px; }}
        QWidget#reviewActions {{ border-top: 2px solid {_BORDER_STRONG}; }}
        QProgressBar#reviewProgress {{ border: none; background: {_SURFACE}; }}
        QProgressBar#reviewProgress::chunk {{ background: {_INK}; }}
        QTableWidget#reviewTable {{ border: none; }}
        QTableWidget#reviewTable::item {{ border-bottom: 1px solid {_SURFACE}; }}
        QTableWidget#reviewTable QComboBox {{ border: 1px solid transparent; background: transparent; padding: 2px 6px; }}
        QTableWidget#reviewTable QComboBox:hover, QTableWidget#reviewTable QComboBox:focus {{ border-color: {_DISABLED}; background: #ffffff; }}
        QMainWindow, QDialog {{ background: {_PAGE_BG}; }}
        QWidget {{ color: {_INK}; font-family: "Segoe UI"; font-size: 12px; }}

        QMenuBar {{
            background: {_PAGE_BG}; border: none; border-bottom: 2px solid {_BORDER}; padding: 2px 4px;
        }}
        QMenuBar::item {{ background: transparent; padding: 5px 10px; }}
        QMenuBar::item:selected {{ background: {_ACCENT_TINT}; }}
        QMenu {{ background: #ffffff; border: 1px solid {_BORDER_STRONG}; padding: 3px 0; }}
        QMenu::item {{ padding: 6px 24px 6px 14px; }}
        QMenu::item:selected {{ background: {_ACCENT_TINT}; color: {_INK}; }}
        QMenu::separator {{ height: 1px; background: {_LINE}; margin: 4px 8px; }}

        QToolBar {{
            background: {_PAGE_BG}; border: none; border-bottom: 2px solid {_BORDER};
            spacing: 3px; padding: 1px 8px;
        }}
        QToolBar QToolButton {{
            background: transparent; border: 1px solid transparent; border-radius: 0px;
            padding: 1px 6px; font-size: 12px;
        }}
        QToolBar QToolButton:hover {{ background: {_SURFACE}; }}
        QToolBar QToolButton:pressed {{ background: {_LINE}; }}
        QToolBar QToolButton:checked {{ background: {_ACCENT}; color: #ffffff; }}
        QToolBar::separator {{ background: {_LINE}; width: 1px; margin: 2px 5px; }}
        QToolBar QSpinBox, QToolBar QComboBox {{ padding-top: 0px; padding-bottom: 0px; min-height: 9px; }}
        QToolBar QPushButton {{ padding: 1px 9px; font-weight: 400; }}
        QToolBar#mainToolbar QSpinBox, QToolBar#mainToolbar QComboBox,
        QToolBar#mainToolbar QPushButton {{ max-height: 22px; }}

        QStatusBar {{ background: {_SURFACE}; border-top: 1px solid {_BORDER}; color: {_INK_SOFT}; }}
        QStatusBar::item {{ border: none; }}

        QSplitter::handle {{ background: {_BORDER}; }}
        QSplitter::handle:horizontal {{ width: 2px; }}
        QSplitter::handle:vertical {{ height: 2px; }}

        QLineEdit, QSpinBox, QDoubleSpinBox {{
            background: #ffffff; border: 1px solid {_LINE}; border-radius: 0px;
            padding: 5px 8px; selection-background-color: {_ACCENT}; selection-color: #ffffff;
        }}
        QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {{ border-color: {_DISABLED}; }}
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border: 1px solid {_ACCENT}; }}
        QLineEdit:disabled {{ background: {_SURFACE}; color: {_DISABLED}; }}

        /* Styling the border/padding above opts a spin box out of native
           complex-control drawing, so its up/down buttons need explicit
           subcontrol rules -- without these they reserve their usual space
           but paint nothing at all. */
        QSpinBox::up-button, QDoubleSpinBox::up-button {{
            subcontrol-origin: border; subcontrol-position: top right;
            width: 16px; height: 11px; border-left: 1px solid {_LINE}; border-bottom: 1px solid {_LINE};
        }}
        QSpinBox::down-button, QDoubleSpinBox::down-button {{
            subcontrol-origin: border; subcontrol-position: bottom right;
            width: 16px; height: 11px; border-left: 1px solid {_LINE};
        }}
        QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
        QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{ background: {_SURFACE}; }}
        QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed,
        QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {{ background: {_LINE}; }}
        QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
            image: url("{_SPIN_UP_LIGHT}"); width: 7px; height: 4px;
        }}
        QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
            image: url("{_SPIN_DOWN_LIGHT}"); width: 7px; height: 4px;
        }}

        QComboBox {{
            background: #ffffff; border: 1px solid {_LINE}; border-radius: 0px;
            padding: 4px 6px; padding-right: 22px; min-height: 18px;
            selection-background-color: {_ACCENT}; selection-color: #ffffff;
        }}
        QComboBox:hover {{ border-color: {_DISABLED}; }}
        QComboBox:focus {{ border: 1px solid {_ACCENT}; }}
        QComboBox:disabled {{ background: {_SURFACE}; color: {_DISABLED}; }}
        QComboBox::drop-down {{
            subcontrol-origin: padding; subcontrol-position: top right;
            width: 20px; border: none; border-left: 1px solid {_LINE};
        }}
        QComboBox::down-arrow {{
            width: 0; height: 0;
            border-left: 4px solid transparent; border-right: 4px solid transparent;
            border-top: 5px solid {_INK_SOFT};
        }}
        QComboBox::down-arrow:disabled {{ border-top-color: {_DISABLED}; }}
        QComboBox QAbstractItemView {{
            background: #ffffff; border: 1px solid {_BORDER_STRONG};
            selection-background-color: {_ACCENT_TINT}; selection-color: {_INK}; outline: none;
        }}

        QPushButton {{
            background: #ffffff; color: {_INK}; border: 1px solid {_DISABLED};
            border-radius: 0px; padding: 6px 16px; font-weight: 600;
        }}
        QPushButton:hover {{ background: {_SURFACE}; }}
        QPushButton:pressed {{ background: {_LINE}; }}
        QPushButton:disabled {{ color: {_DISABLED}; border-color: {_LINE}; background: {_SURFACE_ALT}; }}
        QPushButton:default {{ border: 1px solid {_ACCENT}; }}
        QPushButton#primaryButton {{ background: {_ACCENT}; color: #ffffff; border: none; }}
        QPushButton#primaryButton:hover {{ background: {_ACCENT_HOVER}; }}
        QPushButton#primaryButton:pressed {{ background: {_INK}; }}
        QPushButton#dangerButton {{ border-color: #ae1800; color: #ae1800; }}
        QPushButton#dangerButton:hover {{ background: #fff2ef; }}

        QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 0px; padding: 4px; }}
        QToolButton:hover {{ background: {_SURFACE}; }}
        QToolButton:pressed {{ background: {_LINE}; }}

        QLabel {{ background: transparent; }}
        QGroupBox {{
            border: 1px solid {_LINE}; border-radius: 0px; margin-top: 14px; padding-top: 6px;
            font-weight: 700;
        }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; color: {_INK}; }}

        QCheckBox::indicator, QRadioButton::indicator {{
            width: 15px; height: 15px; border: 1px solid {_DISABLED}; background: #ffffff;
        }}
        QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border-color: {_ACCENT}; }}
        QCheckBox::indicator:checked {{ background: {_ACCENT}; border-color: {_ACCENT}; }}
        QRadioButton::indicator {{ border-radius: 8px; }}
        QRadioButton::indicator:checked {{
            background: qradialgradient(cx:0.5, cy:0.5, radius:0.5,
                stop:0 #ffffff, stop:0.45 #ffffff, stop:0.5 {_ACCENT}, stop:1 {_ACCENT});
            border-color: {_ACCENT};
        }}

        QTabWidget::pane {{ border: 1px solid {_LINE}; top: -1px; }}
        QTabBar::tab {{
            background: {_SURFACE}; color: {_INK_SOFT}; border: 1px solid {_LINE}; border-bottom: none;
            padding: 7px 16px; margin-right: 2px;
        }}
        QTabBar::tab:selected {{ background: #ffffff; color: {_INK}; font-weight: 700; }}
        QTabBar::tab:hover:!selected {{ background: {_ACCENT_TINT}; }}

        QTableWidget, QTableView {{
            background: #ffffff; border: 1px solid {_LINE}; gridline-color: {_SURFACE};
            alternate-background-color: {_SURFACE_ALT};
            selection-background-color: {_ACCENT_TINT}; selection-color: {_INK};
        }}
        QTableWidget::item, QTableView::item {{ padding: 5px 6px; border: none; }}
        QTableWidget::item:selected, QTableView::item:selected {{ background: {_ACCENT_TINT}; color: {_INK}; }}
        QHeaderView::section {{
            background: {_SURFACE}; color: {_INK_SOFT}; border: none;
            border-right: 1px solid {_LINE}; border-bottom: 2px solid {_BORDER};
            padding: 6px 8px; font-weight: 700; font-size: 11px;
        }}
        QHeaderView::section:horizontal:last {{ border-right: none; }}
        QTableCornerButton::section {{ background: {_SURFACE}; border: none; border-bottom: 2px solid {_BORDER}; }}

        QScrollBar:vertical {{ background: {_SURFACE}; width: 14px; margin: 0; border: none; }}
        QScrollBar::handle:vertical {{ background: {_DISABLED}; min-height: 28px; border: 3px solid {_SURFACE}; }}
        QScrollBar::handle:vertical:hover {{ background: {_INK_SOFT}; }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; border: none; background: none; }}
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}
        QScrollBar:horizontal {{ background: {_SURFACE}; height: 14px; margin: 0; border: none; }}
        QScrollBar::handle:horizontal {{ background: {_DISABLED}; min-width: 28px; border: 3px solid {_SURFACE}; }}
        QScrollBar::handle:horizontal:hover {{ background: {_INK_SOFT}; }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; border: none; background: none; }}
        QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{ background: none; }}

        QProgressBar {{
            background: {_SURFACE}; border: 1px solid {_LINE}; border-radius: 0px;
            text-align: center; color: {_INK}; height: 14px;
        }}
        QProgressBar::chunk {{ background: {_ACCENT}; }}

        QToolTip {{
            color: #ffffff; background-color: {_INK}; border: 1px solid {_INK};
            padding: 4px 7px;
        }}
    """)


def apply_dark_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    palette = QPalette()
    colors = {
        QPalette.ColorRole.Window: "#25282d",
        QPalette.ColorRole.WindowText: "#edf0f4",
        QPalette.ColorRole.Base: "#1d2025",
        QPalette.ColorRole.AlternateBase: "#292d33",
        QPalette.ColorRole.Text: "#edf0f4",
        QPalette.ColorRole.Button: "#343941",
        QPalette.ColorRole.ButtonText: "#edf0f4",
        QPalette.ColorRole.Highlight: "#276ba5",
        QPalette.ColorRole.HighlightedText: "#ffffff",
        QPalette.ColorRole.ToolTipBase: "#343941",
        QPalette.ColorRole.ToolTipText: "#ffffff",
        QPalette.ColorRole.PlaceholderText: "#a4acb8",
        QPalette.ColorRole.Link: "#78baff",
        QPalette.ColorRole.Light: "#515966",
        QPalette.ColorRole.Midlight: "#414750",
        QPalette.ColorRole.Mid: "#363c45",
        QPalette.ColorRole.Dark: "#15181c",
        QPalette.ColorRole.Shadow: "#101215",
    }
    for role, color in colors.items():
        palette.setColor(role, QColor(color))
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor("#818a96"))
    app.setPalette(palette)
    app.setStyleSheet("""
        QLabel#brandHeader { background: #002049; color: #f3f2f2; font-weight: 600; padding: 0 12px; }
        QLabel#panelTitle { font-size: 15px; font-weight: 700; }
        QLabel#mutedLabel { color: #a4acb8; font-size: 11px; }
        QWidget#reviewActions { border-top: 2px solid #454c57; }
        QProgressBar#reviewProgress { border: none; background: #343941; }
        QProgressBar#reviewProgress::chunk { background: #0098f8; }
        QPushButton#primaryButton { background: #007cd0; color: white; padding: 6px 16px; border: none; }
        QTableView { gridline-color: #3b414b; }
        QHeaderView::section {
            background-color: #343941; color: #edf0f4;
            border: 1px solid #454c57; padding: 4px;
        }
        QToolTip { color: #ffffff; background-color: #343941; border: 1px solid #667080; }
        QToolBar#mainToolbar QSpinBox, QToolBar#mainToolbar QComboBox,
        QToolBar#mainToolbar QPushButton { max-height: 22px; }
        QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
            image: url("__SPIN_UP_DARK__"); width: 7px; height: 4px;
        }
        QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
            image: url("__SPIN_DOWN_DARK__"); width: 7px; height: 4px;
        }
    """.replace("__SPIN_UP_DARK__", _SPIN_UP_DARK).replace("__SPIN_DOWN_DARK__", _SPIN_DOWN_DARK))

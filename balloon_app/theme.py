"""Consistent dark desktop theme for the application and its dialogs."""

from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication


def apply_theme(app: QApplication, theme: str) -> None:
    if theme == "dark":
        apply_dark_theme(app)
    else:
        app.setStyleSheet("")
        app.setStyle("Fusion")
        app.setPalette(app.style().standardPalette())


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
        QTableView { gridline-color: #3b414b; }
        QHeaderView::section {
            background-color: #343941; color: #edf0f4;
            border: 1px solid #454c57; padding: 4px;
        }
        QToolTip { color: #ffffff; background-color: #343941; border: 1px solid #667080; }
    """)

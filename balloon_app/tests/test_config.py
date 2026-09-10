"""Tests for AppSettings' in-memory learned-symbols bookkeeping, and for
resource_path()'s PyInstaller-aware lookup.

save()/load() round-trip through the real (registry-backed) QSettings, so
these tests monkeypatch save() to a no-op rather than touching that shared,
machine-wide state from an automated test run.

Bug: the app icon and welcome logo were looked up as
``Path(__file__).parent / "resources" / ...`` in app.py/dialogs.py/
pdf_view.py/theme.py. That resolves correctly when running from source,
but in a PyInstaller one-folder build, pure-Python modules are stored
inside the bundled archive rather than extracted as loose files next to a
real "resources" folder -- so ``__file__``-relative lookups silently
pointed at a path that doesn't exist. QIcon/QPixmap don't raise on a
missing file, they just render nothing, so the built app ran fine with no
error, just no logo. resource_path() centralizes the lookup and, when
frozen, resolves under ``sys._MEIPASS`` instead (where --add-data actually
unpacks resources).
"""

from __future__ import annotations

import sys

from balloon_app.config import AppSettings, resource_path


def test_learn_symbol_adds_a_mapping(monkeypatch):
    settings = AppSettings()
    monkeypatch.setattr(settings, "save", lambda: None)

    settings.learn_symbol("n", "diameter")

    assert settings.learned_symbols == {"n": "diameter"}


def test_learn_symbol_normalizes_case_and_whitespace(monkeypatch):
    settings = AppSettings()
    monkeypatch.setattr(settings, "save", lambda: None)

    settings.learn_symbol(" W ", "countersink")

    assert settings.learned_symbols == {"w": "countersink"}


def test_learn_symbol_overwrites_an_existing_mapping(monkeypatch):
    settings = AppSettings(learned_symbols={"n": "diameter"})
    monkeypatch.setattr(settings, "save", lambda: None)

    settings.learn_symbol("n", "square")

    assert settings.learned_symbols == {"n": "square"}


def test_forget_symbol_removes_a_mapping(monkeypatch):
    settings = AppSettings(learned_symbols={"n": "diameter", "w": "countersink"})
    monkeypatch.setattr(settings, "save", lambda: None)

    settings.forget_symbol("n")

    assert settings.learned_symbols == {"w": "countersink"}


def test_forget_symbol_on_unknown_marker_is_a_no_op(monkeypatch):
    settings = AppSettings(learned_symbols={"w": "countersink"})
    monkeypatch.setattr(settings, "save", lambda: None)

    settings.forget_symbol("does-not-exist")

    assert settings.learned_symbols == {"w": "countersink"}


def test_resource_path_finds_the_real_icon_when_running_from_source():
    path = resource_path("balloonapp.ico")

    assert path.is_file()
    assert path.parent.name == "resources"


def test_resource_path_resolves_under_meipass_when_frozen(tmp_path, monkeypatch):
    # Simulate a PyInstaller one-folder build's layout: --add-data
    # "balloon_app\resources;balloon_app\resources" unpacks resources under
    # sys._MEIPASS/balloon_app/resources, not next to any extracted .py file.
    bundled = tmp_path / "balloon_app" / "resources"
    bundled.mkdir(parents=True)
    (bundled / "balloonapp.ico").write_bytes(b"")

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)

    path = resource_path("balloonapp.ico")

    assert path == bundled / "balloonapp.ico"
    assert path.is_file()

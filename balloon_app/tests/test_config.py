"""Tests for AppSettings' in-memory learned-symbols bookkeeping.

save()/load() round-trip through the real (registry-backed) QSettings, so
these tests monkeypatch save() to a no-op rather than touching that shared,
machine-wide state from an automated test run.
"""

from __future__ import annotations

from balloon_app.config import AppSettings


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

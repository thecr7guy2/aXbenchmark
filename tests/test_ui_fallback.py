from rich.console import Console

from axbench.ui.fallback import detect_terminal_capabilities, resolve_terminal_theme


def test_detect_terminal_capabilities_respects_ascii_and_mono(monkeypatch):
    console = Console(record=True, force_terminal=True, color_system="standard")
    monkeypatch.setenv("AXBENCH_ASCII", "1")
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)

    capabilities = detect_terminal_capabilities(console, no_ui=False)

    assert capabilities.interactive is True
    assert capabilities.unicode_ok is False
    assert capabilities.color_ok is False


def test_resolve_terminal_theme_switches_to_ascii_safe_glyphs():
    theme = resolve_terminal_theme(
        type("Caps", (), {"unicode_ok": False, "color_ok": False})()
    )

    assert theme.glyphs["running"] == ">"
    assert theme.title_text == "AXBench Console"
    assert theme.color_ok is False

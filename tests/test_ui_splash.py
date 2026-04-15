from rich.console import Console

from axbench.ui.splash import render_splash, should_show_splash


def test_render_splash_contains_key_copy():
    console = Console(record=True, width=100)
    console.print(render_splash())
    output = console.export_text()

    assert "AXBench" in output
    assert "LLM Benchmark Orchestrator" in output
    assert "Press Enter to begin" in output


def test_should_show_splash_respects_no_ui(monkeypatch):
    monkeypatch.delenv("AXBENCH_FORCE_SPLASH", raising=False)
    monkeypatch.setenv("CI", "1")
    assert should_show_splash(no_ui=False) is False
    assert should_show_splash(no_ui=True) is False


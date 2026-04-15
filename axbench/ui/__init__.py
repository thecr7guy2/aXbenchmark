"""UI helpers for interactive AXBench terminal experiences."""

from axbench.ui.fallback import TerminalCapabilities, detect_terminal_capabilities, resolve_terminal_theme
from axbench.ui.splash import render_splash, should_show_splash, show_splash
from axbench.ui.state import RunEvent, RunUIState, TaskLifecycle

__all__ = [
    "RunEvent",
    "RunUIState",
    "TerminalCapabilities",
    "TaskLifecycle",
    "detect_terminal_capabilities",
    "render_splash",
    "resolve_terminal_theme",
    "should_show_splash",
    "show_splash",
]

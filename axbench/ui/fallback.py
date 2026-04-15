from __future__ import annotations

import os
import sys
from dataclasses import dataclass

from rich.console import Console

from axbench.ui.theme import UITheme, resolve_theme


@dataclass(frozen=True)
class TerminalCapabilities:
    interactive: bool
    unicode_ok: bool
    color_ok: bool


def detect_terminal_capabilities(console: Console, no_ui: bool) -> TerminalCapabilities:
    force_ascii = os.environ.get("AXBENCH_ASCII") == "1"
    force_mono = os.environ.get("AXBENCH_MONO") == "1" or os.environ.get("NO_COLOR") is not None
    term = os.environ.get("TERM", "").lower()
    interactive = (
        not no_ui
        and not os.environ.get("CI")
        and term != "dumb"
        and sys.stdin.isatty()
        and sys.stdout.isatty()
    )
    encoding = (console.encoding or "").lower()
    unicode_ok = not force_ascii and encoding not in {"ascii", "ansi_x3.4-1968"}
    color_ok = not force_mono and console.color_system is not None and term != "dumb"
    return TerminalCapabilities(
        interactive=interactive,
        unicode_ok=unicode_ok,
        color_ok=color_ok,
    )


def resolve_terminal_theme(capabilities: TerminalCapabilities) -> UITheme:
    return resolve_theme(
        unicode_ok=capabilities.unicode_ok,
        color_ok=capabilities.color_ok,
    )

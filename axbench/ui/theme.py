from __future__ import annotations

from dataclasses import dataclass, field


ASCII_GLYPHS = {
    "queued": ".",
    "running": ">",
    "completed": "*",
    "failed": "x",
    "skipped": "-",
}

UNICODE_GLYPHS = {
    "queued": "○",
    "running": "◉",
    "completed": "●",
    "failed": "✕",
    "skipped": "◌",
}


@dataclass(frozen=True)
class UITheme:
    app_name: str = "AXBench"
    subtitle: str = "LLM Benchmark Orchestrator"
    accent: str = "cyan"
    accent_alt: str = "magenta"
    success: str = "green"
    warning: str = "yellow"
    danger: str = "red"
    muted: str = "bright_black"
    border: str = "blue"
    title_text: str = "AXBench Command Center"
    unicode_ok: bool = True
    color_ok: bool = True
    glyphs: dict[str, str] = field(default_factory=lambda: dict(UNICODE_GLYPHS))
    badges: dict[str, str] = field(
        default_factory=lambda: {
            "queued": "dim",
            "running": "cyan",
            "completed": "green",
            "failed": "red",
            "skipped": "yellow",
        }
    )

    def status_style(self, status: str) -> str:
        return self.badges.get(status, self.muted)


DEFAULT_THEME = UITheme()


def resolve_theme(unicode_ok: bool = True, color_ok: bool = True) -> UITheme:
    if color_ok:
        accent = "cyan"
        accent_alt = "magenta"
        success = "green"
        warning = "yellow"
        danger = "red"
        muted = "bright_black"
        border = "blue"
    else:
        accent = "white"
        accent_alt = "white"
        success = "white"
        warning = "white"
        danger = "white"
        muted = "dim"
        border = "white"

    title_text = "AXBench Command Center" if unicode_ok else "AXBench Console"
    glyphs = dict(UNICODE_GLYPHS if unicode_ok else ASCII_GLYPHS)

    badges = {
        "queued": muted,
        "running": accent_alt,
        "completed": success,
        "failed": danger,
        "skipped": warning,
    }

    return UITheme(
        accent=accent,
        accent_alt=accent_alt,
        success=success,
        warning=warning,
        danger=danger,
        muted=muted,
        border=border,
        title_text=title_text,
        unicode_ok=unicode_ok,
        color_ok=color_ok,
        glyphs=glyphs,
        badges=badges,
    )

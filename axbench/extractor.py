import re


def extract_code(output: str, language: str) -> str:
    """Extract code from a model response. Handles markdown fences."""
    if output is None:
        return ""

    # Try language-specific fence first
    pattern_lang = rf"```{re.escape(language)}\n(.*?)```"
    matches = re.findall(pattern_lang, output, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()

    # Try generic fence
    pattern_generic = r"```(?:\w*\n)?(.*?)```"
    matches = re.findall(pattern_generic, output, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()

    # No fences — return as-is
    return output.strip()

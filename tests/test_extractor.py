from axbench.extractor import extract_code


def test_extracts_python_fenced_block():
    output = "Here is the solution:\n```python\ndef foo():\n    return 1\n```\nLet me explain..."
    assert extract_code(output, "python") == "def foo():\n    return 1"


def test_extracts_cpp_fenced_block():
    output = "```cpp\nint main() { return 0; }\n```"
    assert extract_code(output, "cpp") == "int main() { return 0; }"


def test_extracts_generic_fenced_block_when_no_language_match():
    output = "```\ndef foo(): pass\n```"
    assert extract_code(output, "python") == "def foo(): pass"


def test_returns_longest_block_when_multiple():
    output = "```python\nx = 1\n```\nOr:\n```python\ndef foo():\n    return 42\n```"
    result = extract_code(output, "python")
    assert "def foo" in result


def test_returns_full_output_when_no_fences():
    output = "def foo():\n    return 1"
    assert extract_code(output, "python") == "def foo():\n    return 1"


def test_returns_empty_string_for_none_output():
    assert extract_code(None, "python") == ""

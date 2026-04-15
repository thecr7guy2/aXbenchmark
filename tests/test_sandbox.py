import pytest
from axbench.sandbox import Sandbox, SandboxResult


def test_python_correct_output():
    s = Sandbox()
    result = s.run_python(
        code="def add(a, b):\n    return a + b",
        test_expression="add(2, 3)",
        expected=5,
        timeout=10,
    )
    assert result.passed is True
    assert result.error is None


def test_python_wrong_output():
    s = Sandbox()
    result = s.run_python(
        code="def add(a, b):\n    return a - b",
        test_expression="add(2, 3)",
        expected=5,
        timeout=10,
    )
    assert result.passed is False


def test_python_syntax_error():
    s = Sandbox()
    result = s.run_python(
        code="def add(a b:\n    return a",
        test_expression="add(1, 2)",
        expected=3,
        timeout=10,
    )
    assert result.passed is False
    assert result.error is not None


def test_python_timeout():
    s = Sandbox()
    result = s.run_python(
        code="def hang():\n    while True: pass",
        test_expression="hang()",
        expected=None,
        timeout=1,
    )
    assert result.passed is False
    assert "timeout" in result.error.lower()


def test_python_multiline_expression():
    s = Sandbox()
    result = s.run_python(
        code="def add(a, b):\n    return a + b",
        test_expression=(
            "x = add(1, 2)\n"
            "y = add(3, 4)\n"
            "x + y"
        ),
        expected=10,
        timeout=10,
    )
    assert result.passed is True
    assert result.error is None


def test_python_multiline_try_except_expression():
    s = Sandbox()
    result = s.run_python(
        code="def fail():\n    raise TypeError('bad')",
        test_expression=(
            "try:\n"
            "    fail()\n"
            "    result = 'no_error'\n"
            "except TypeError:\n"
            "    result = 'type_error'\n"
            "result"
        ),
        expected="type_error",
        timeout=10,
    )
    assert result.passed is True


def test_bash_correct_output():
    s = Sandbox()
    result = s.run_bash(
        script="echo hello",
        expected_stdout="hello\n",
        timeout=5,
    )
    assert result.passed is True


def test_bash_wrong_output():
    s = Sandbox()
    result = s.run_bash(
        script="echo world",
        expected_stdout="hello\n",
        timeout=5,
    )
    assert result.passed is False


def test_bash_can_use_setup_and_expected_exit_code():
    s = Sandbox()
    result = s.run_bash(
        script='echo "HEALTHY"',
        setup_script="export TEST_FLAG=1",
        expected_stdout="HEALTHY\n",
        expected_exit_code=0,
        timeout=5,
    )
    assert result.passed is True


def test_bash_fails_on_stderr_when_not_allowed():
    s = Sandbox()
    result = s.run_bash(
        script='echo "oops" 1>&2',
        expected_stdout="",
        timeout=5,
    )
    assert result.passed is False
    assert result.error is not None


def test_cpp_passes():
    s = Sandbox()
    harness = """
#include <iostream>
#include <cassert>

int add(int a, int b) { return a + b; }

int main() {
    assert(add(2, 3) == 5);
    std::cout << "PASS" << std::endl;
    return 0;
}
"""
    result = s.run_cpp(harness, timeout=15)
    assert result.passed is True


def test_cpp_compile_error():
    s = Sandbox()
    harness = "this is not valid c++"
    result = s.run_cpp(harness, timeout=15)
    assert result.passed is False
    assert result.error is not None


def test_cpp_assertion_failure():
    s = Sandbox()
    harness = """
#include <iostream>
#include <cassert>
int main() {
    assert(1 == 2);
    std::cout << "PASS" << std::endl;
    return 0;
}
"""
    result = s.run_cpp(harness, timeout=15)
    assert result.passed is False

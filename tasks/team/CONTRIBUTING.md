# How to contribute benchmark tasks

Drop a YAML file in your folder: `riccardo/`, `tom/`, or `serge_mykyta/`.
Use the following format depending on whether the task is new code generation or a bug fix.

## Code generation task

```yaml
id: <language>_<short_name>          # e.g. bash_log_rotation, cpp_connection_pool
evaluator: code_gen
language: python | cpp | bash | sql
difficulty: easy | medium | hard
source: team/<your_folder>           # e.g. team/riccardo
tags: [docker, logging, ...]

prompt: |
  Describe exactly what you'd type to the AI.

# Python / bash: use test_cases
test_cases:
  - input: "function_call(args)"
    expected: expected_value

# C++: use test_harness with {{GENERATED_CODE}} placeholder
test_harness: |
  #include <iostream>
  ...
  {{GENERATED_CODE}}
  int main() { ... std::cout << "PASS"; }

timeout_seconds: 10
```

## Bug fix task

```yaml
id: <language>_bug_<short_name>
evaluator: bug_fix
language: python | cpp
difficulty: easy | medium | hard
source: team/<your_folder>
tags: [bugs, ...]

prompt: |
  The following code has a bug. Fix it. Return only the corrected code.

  ```<language>
  <buggy code here>
  ```

test_cases:
  - input: "function_call(args)"
    expected: expected_value

timeout_seconds: 10
```

If the test case is hard to define, add a `# TODO:` comment and send it to Maniraj to complete.

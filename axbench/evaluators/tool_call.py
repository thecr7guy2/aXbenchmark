"""
Pillar 5 — Tool Calling Evaluator (Future)

Not implemented in v0.1. The data model for tool calling tasks is defined
in the design doc (`plans/axbench-design.md`, Section 5.1).

To implement: evaluate whether the model:
1. Calls the correct tool (correct_tool weight: 0.4)
2. Passes correct arguments (correct_arguments weight: 0.4)
3. Avoids unnecessary tool calls (no_unnecessary_calls weight: 0.2)

The `LLMClient.generate_with_tools()` method is already implemented and
ready to use.
"""

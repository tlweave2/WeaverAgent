"""The ReAct and Plan-Execute engines."""

import pytest

from agentforge.llm import MockProvider
from agentforge.reasoning import PlanExecuteEngine, ReActEngine, StepType
from agentforge.tools import ToolRegistry, default_registry
from agentforge.types import LLMResponse, StopReason


@pytest.fixture
def tools():
    return default_registry()


# -- ReAct -------------------------------------------------------------------


def test_react_answers_without_tools_when_it_can(tools):
    provider = MockProvider(["Paris."])
    trace = ReActEngine(provider, tools).run("What is the capital of France?")

    assert trace.answer == "Paris."
    assert trace.succeeded
    assert trace.tool_calls == []
    assert [s.type for s in trace.steps] == [StepType.THOUGHT, StepType.ANSWER]


def test_react_loops_through_thought_action_observation(tools):
    provider = MockProvider(
        [
            MockProvider.tool_call("calculator", {"expression": "17*23"}, text="Let me compute."),
            "17 * 23 = 391.",
        ]
    )
    trace = ReActEngine(provider, tools).run("What is 17 * 23?")

    assert trace.answer == "17 * 23 = 391."
    assert trace.succeeded
    assert [s.type for s in trace.steps] == [
        StepType.THOUGHT,
        StepType.ACTION,
        StepType.OBSERVATION,
        StepType.THOUGHT,
        StepType.ANSWER,
    ]
    observation = trace.steps_of(StepType.OBSERVATION)[0]
    assert observation.tool_results[0].content == "391"


def test_react_chains_multiple_tool_rounds(tools):
    provider = MockProvider(
        [
            MockProvider.tool_call("calculator", {"expression": "2+2"}),
            MockProvider.tool_call("calculator", {"expression": "4*10"}),
            "The result is 40.",
        ]
    )
    trace = ReActEngine(provider, tools).run("Add 2 and 2, then multiply by 10.")

    assert len(trace.tool_calls) == 2
    assert trace.answer == "The result is 40."


def test_react_sends_multiple_parallel_calls_in_one_round(tools):
    response = LLMResponse(
        content="Both at once.",
        stop_reason=StopReason.TOOL_USE,
        tool_calls=[
            MockProvider.tool_call("calculator", {"expression": "1+1"}).tool_calls[0],
            MockProvider.tool_call("current_time", {}).tool_calls[0],
        ],
    )
    trace = ReActEngine(MockProvider([response, "Done."]), tools).run("Do two things.")

    observation = trace.steps_of(StepType.OBSERVATION)[0]
    assert len(observation.tool_results) == 2
    assert len(trace.steps_of(StepType.ACTION)) == 1


def test_react_observes_tool_errors_and_recovers(tools):
    provider = MockProvider(
        [
            MockProvider.tool_call("calculator", {"expression": "not math"}),
            MockProvider.tool_call("calculator", {"expression": "1+1"}),
            "Recovered: the answer is 2.",
        ]
    )
    trace = ReActEngine(provider, tools).run("Compute something.")

    first = trace.steps_of(StepType.OBSERVATION)[0]
    assert first.metadata["errors"] == 1
    assert trace.succeeded
    assert "Recovered" in trace.answer


def test_react_forces_an_answer_at_the_step_limit(tools):
    """Hitting the limit still yields the best available answer, flagged as incomplete."""
    provider = MockProvider(
        [MockProvider.tool_call("calculator", {"expression": "1+1"})],
        repeat_last=True,
    )
    engine = ReActEngine(provider, tools, max_steps=3)
    trace = engine.run("Loop forever.")

    assert not trace.succeeded
    assert any(s.type is StepType.ERROR for s in trace.steps)
    answer_step = trace.steps_of(StepType.ANSWER)[-1]
    assert answer_step.metadata.get("truncated") is True
    # The forcing call withholds tools so the model cannot keep looping.
    assert provider.calls[-1]["tools"] == []


def test_react_stops_on_refusal(tools):
    refusal = LLMResponse(content="[refused: cyber] no", stop_reason=StopReason.REFUSAL)
    trace = ReActEngine(MockProvider([refusal]), tools).run("Do something disallowed.")

    assert not trace.succeeded
    assert trace.steps_of(StepType.ERROR)[0].metadata["refusal"] is True
    assert "refused" in trace.answer


def test_react_passes_tool_schemas_and_system_prompt(tools):
    provider = MockProvider(["done"])
    ReActEngine(provider, tools).run("task")

    call = provider.calls[0]
    assert set(call["tools"]) == set(tools.names())
    assert "Available tools:" in call["system"]
    assert "calculator" in call["system"]


def test_react_omits_tools_when_the_registry_is_empty():
    provider = MockProvider(["done"])
    ReActEngine(provider, ToolRegistry()).run("task")
    assert provider.calls[0]["tools"] == []


def test_custom_system_prompt_replaces_the_default(tools):
    provider = MockProvider(["done"])
    ReActEngine(provider, tools, system_prompt="Only speak in haiku.").run("task")
    assert provider.calls[0]["system"] == "Only speak in haiku."


def test_react_conditions_on_history(tools):
    from agentforge.types import Message

    provider = MockProvider(["done"])
    history = [Message.user("earlier"), Message.assistant("noted")]
    ReActEngine(provider, tools).run("now", history=history)

    contents = [m.content for m in provider.calls[0]["messages"]]
    assert contents == ["earlier", "noted", "now"]


def test_trace_accumulates_usage(tools):
    from agentforge.types import Usage

    provider = MockProvider(
        [
            LLMResponse(
                content="",
                stop_reason=StopReason.TOOL_USE,
                tool_calls=MockProvider.tool_call("calculator", {"expression": "1+1"}).tool_calls,
                usage=Usage(input_tokens=10, output_tokens=5),
            ),
            LLMResponse(content="2", stop_reason=StopReason.END_TURN, usage=Usage(20, 7)),
        ]
    )
    trace = ReActEngine(provider, tools).run("1+1?")
    assert trace.usage.input_tokens == 30
    assert trace.usage.output_tokens == 12


def test_trace_render_includes_the_transcript(tools):
    provider = MockProvider([MockProvider.tool_call("calculator", {"expression": "6*7"}), "42."])
    rendered = ReActEngine(provider, tools).run("6*7?").render()

    assert "Task: 6*7?" in rendered
    assert "Engine: react" in rendered
    assert "calculator" in rendered
    assert "42." in rendered


# -- Plan-Execute ------------------------------------------------------------


def test_plan_execute_plans_then_executes_each_step(tools):
    provider = MockProvider(
        [
            '{"steps": ["Compute 17 * 23", "Report the time"]}',  # planner
            MockProvider.tool_call("calculator", {"expression": "17*23"}),  # step 1
            "391",
            MockProvider.tool_call("current_time", {}),  # step 2
            "the current time",
            "17 * 23 is 391, reported at the current time.",  # synthesis
        ]
    )
    trace = PlanExecuteEngine(provider, tools).run("Multiply 17 by 23 and tell me the time.")

    plan_step = trace.steps_of(StepType.PLAN)[0]
    assert "Compute 17 * 23" in plan_step.content
    assert plan_step.metadata["step_count"] == 2
    assert len(trace.tool_calls) == 2
    assert trace.succeeded
    assert "391" in trace.answer
    # Sub-run entries are tagged with the plan step they belong to.
    assert {s.metadata.get("plan_step") for s in trace.steps if "plan_step" in s.metadata} == {0, 1}


def test_planner_does_not_get_tools(tools):
    provider = MockProvider(['{"steps": ["only step"]}', "result", "final"])
    PlanExecuteEngine(provider, tools).run("task")
    assert provider.calls[0]["tools"] == []


def test_plan_execute_respects_max_steps(tools):
    provider = MockProvider(
        ['{"steps": ["a", "b", "c", "d", "e"]}', "r1", "r2", "final"],
        repeat_last=True,
    )
    trace = PlanExecuteEngine(provider, tools, max_steps=2).run("task")
    assert trace.steps_of(StepType.PLAN)[0].metadata["step_count"] == 2


def test_plan_execute_handles_an_empty_plan(tools):
    trace = PlanExecuteEngine(MockProvider(['{"steps": []}']), tools).run("task")
    assert not trace.succeeded
    assert "no usable steps" in trace.steps_of(StepType.ERROR)[0].content


def test_plan_execute_replans_after_a_failed_step(tools):
    provider = MockProvider(
        [
            '{"steps": ["step one", "step two"]}',
            LLMResponse(content="", stop_reason=StopReason.END_TURN),  # step 1 answers emptily
            '{"steps": ["revised step"]}',  # replan
            "revised result",
            "final answer",
        ]
    )
    trace = PlanExecuteEngine(provider, tools, replan_on_failure=True).run("task")

    plans = trace.steps_of(StepType.PLAN)
    assert len(plans) == 2
    assert "revised step" in plans[1].content
    assert plans[1].metadata["replanned_after"] == 0


def test_plan_execute_can_skip_replanning(tools):
    provider = MockProvider(
        [
            '{"steps": ["step one", "step two"]}',
            LLMResponse(content="", stop_reason=StopReason.END_TURN),
            "second result",
            "final",
        ]
    )
    trace = PlanExecuteEngine(provider, tools, replan_on_failure=False).run("task")
    assert len(trace.steps_of(StepType.PLAN)) == 1


def test_plan_execute_stops_on_planner_refusal(tools):
    refusal = LLMResponse(content="[refused: bio] no", stop_reason=StopReason.REFUSAL)
    trace = PlanExecuteEngine(MockProvider([refusal]), tools).run("task")

    assert not trace.succeeded
    assert trace.steps_of(StepType.ERROR)[0].metadata["refusal"] is True


def test_executor_prompt_carries_earlier_step_results(tools):
    provider = MockProvider(
        ['{"steps": ["first", "second"]}', "first result", "second result", "final"]
    )
    PlanExecuteEngine(provider, tools).run("task")

    # calls: 0 planner, 1 step one, 2 step two, 3 synthesis
    second_step_system = provider.calls[2]["system"]
    assert "first result" in second_step_system
    assert "[ok] first" in second_step_system


# -- plan parsing ------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"steps": ["a", "b"]}', ["a", "b"]),
        ('{"plan": ["a"]}', ["a"]),
        ('["a", "b"]', ["a", "b"]),
        ('```json\n{"steps": ["a"]}\n```', ["a"]),
        ('Here is the plan:\n{"steps": ["a", "b"]}\nHope that helps.', ["a", "b"]),
        ("1. first step\n2. second step", ["first step", "second step"]),
        ("- alpha\n- beta", ["alpha", "beta"]),
        ("* only", ["only"]),
        ("", []),
        ("no structure at all here", []),
    ],
)
def test_plan_parsing_tolerates_planner_formatting_drift(text, expected):
    """Planners drift in and out of strict JSON; a formatting slip must not abort a run."""
    from agentforge.reasoning.plan_execute import _parse_plan

    assert _parse_plan(text) == expected

import asyncio

from rllm_model_gateway.models import TraceRecord

from rllm.agents.agent import Episode, Trajectory
from rllm.engine.agentflow_engine import AgentFlowEngine, enrich_episode_with_traces
from rllm.eval.types import EvalOutput
from rllm.types import Step, Task
from rllm.workflows.workflow import TerminationReason


class _Agent:
    def __init__(self):
        self.config = None

    async def arun(self, task, config):
        self.config = config
        return Episode(
            id=task.id,
            termination_reason=TerminationReason.ERROR,
            trajectories=[Trajectory(name="solver")],
        )


class _Evaluator:
    def evaluate(self, task, episode):
        return EvalOutput(reward=0.0, is_correct=False)


class _Gateway:
    def __init__(self):
        self.created = None
        self.deleted = None

    async def acreate_session(self, session_id, is_validation=False):
        self.created = (session_id, is_validation)

    def get_session_url(self, session_id):
        return f"http://gateway/{session_id}"

    async def aget_traces(self, session_id):
        return []

    async def aget_traces_no_flush(self, session_id):
        return []

    async def adelete_session(self, session_id):
        self.deleted = session_id


def test_run_single_passes_validation_flag_and_preserves_termination_reason():
    agent = _Agent()
    gateway = _Gateway()
    engine = AgentFlowEngine(
        agent_flow=agent,
        evaluator=_Evaluator(),
        gateway=gateway,
        model="test-model",
        n_parallel_tasks=1,
    )

    try:
        task = Task(id="task", instruction="q", metadata={"question": "q"})
        episode = asyncio.run(
            engine._run_single(
                task,
                {"question": "q"},
                "task:0",
                is_validation=True,
            )
        )
    finally:
        engine.shutdown()

    assert agent.config.is_validation is True
    assert agent.config.session_uid == "task:0"
    assert episode.termination_reason == TerminationReason.ERROR


def _messages(content: str) -> list[dict]:
    return [{"role": "user", "content": content}]


def _agent_step(content: str, *, action: str | None = None) -> Step:
    return Step(
        chat_completions=_messages(content) + [{"role": "assistant", "content": f"answer {content}"}],
        model_response=f"answer {content}",
        action=action or content,
    )


def _trace(content: str, *, completion_ids: list[int] | None = None, prompt_ids: list[int] | None = None) -> TraceRecord:
    return TraceRecord(
        trace_id=f"trace-{content}",
        session_id="uid",
        model="model",
        messages=_messages(content),
        prompt_token_ids=prompt_ids if prompt_ids is not None else [1, 2],
        response_message={"role": "assistant", "content": f"answer {content}"},
        completion_token_ids=completion_ids if completion_ids is not None else [3, 4],
        logprobs=[0.0] * len(completion_ids if completion_ids is not None else [3, 4]),
    )


def test_enrich_preserves_equal_count_failed_agent_step_without_training_payload():
    episode = Episode(
        trajectories=[
            Trajectory(
                name="seg0",
                steps=[
                    _agent_step("ok-0"),
                    _agent_step("failed", action="llm_failed"),
                    _agent_step("ok-1"),
                ],
                reward=0.5,
            )
        ]
    )
    traces = [
        _trace("ok-0", completion_ids=[10]),
        _trace("failed", completion_ids=[]),
        _trace("ok-1", completion_ids=[11]),
    ]

    enriched = enrich_episode_with_traces(episode, traces, "uid", {"question": "q"}, strict=True)

    steps = enriched.trajectories[0].steps
    assert len(steps) == 3
    assert steps[0].model_output.completion_ids == [10]
    assert steps[1].action == "llm_failed"
    assert steps[1].model_output is None
    assert steps[1].metadata["rllm_enrich_skip_reason"] == "missing_token_ids"
    assert steps[2].model_output.completion_ids == [11]
    assert enriched.metrics["steps_skipped_missing_token_ids"] == 1


def test_enrich_drops_scattered_phantom_trace_before_next_agent_step():
    episode = Episode(
        trajectories=[
            Trajectory(name="seg0", steps=[_agent_step("ok-0")], reward=0.0),
            Trajectory(name="seg1", steps=[_agent_step("ok-1")], reward=1.0),
        ]
    )
    traces = [
        _trace("ok-0", completion_ids=[10]),
        _trace("phantom", completion_ids=[]),
        _trace("ok-1", completion_ids=[11]),
    ]

    enriched = enrich_episode_with_traces(episode, traces, "uid", {"question": "q"}, strict=True)

    assert [step.model_response for traj in enriched.trajectories for step in traj.steps] == [
        "answer ok-0",
        "answer ok-1",
    ]
    assert enriched.metrics["traces_dropped_phantom"] == 1
    assert enriched.metrics["steps_skipped_missing_token_ids"] == 0

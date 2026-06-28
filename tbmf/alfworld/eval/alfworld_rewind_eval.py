"""Rewind-choice ALFWorld evaluator."""

from __future__ import annotations

import rllm
from rllm.eval.types import EvalOutput, Signal
from rllm.types import Episode


@rllm.evaluator
def alfworld_rewind_evaluator(task: dict, episode: Episode) -> EvalOutput:
    won = bool(episode.artifacts.get("won", False))
    reward = 1.0 if won else 0.0
    return EvalOutput(
        reward=reward,
        is_correct=won,
        signals=[
            Signal(name="accuracy", value=reward),
            Signal(name="turns", value=float(episode.artifacts.get("turns", 0))),
            Signal(name="total_llm_calls", value=float(episode.artifacts.get("total_llm_calls", 0))),
            Signal(name="rewinds", value=float(episode.artifacts.get("rewinds", 0))),
            Signal(name="forced_rewinds", value=float(episode.artifacts.get("forced_rewinds", 0))),
            Signal(name="model_rewinds", value=float(episode.artifacts.get("model_rewinds", 0))),
            Signal(name="context_folds", value=float(episode.artifacts.get("context_folds", 0))),
        ],
        metadata={"task_type": episode.artifacts.get("task_type", "unknown")},
    )

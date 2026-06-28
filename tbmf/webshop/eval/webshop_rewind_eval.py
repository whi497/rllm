"""Rewind-choice WebShop evaluator."""

from __future__ import annotations

import rllm
from rllm.eval.types import EvalOutput, Signal
from rllm.types import Episode


@rllm.evaluator
def webshop_rewind_evaluator(task: dict, episode: Episode) -> EvalOutput:
    won = bool(episode.artifacts.get("won", False))
    task_score = float(episode.artifacts.get("task_score", 0.0))
    reward = 1.0 if won else 0.0
    return EvalOutput(
        reward=reward,
        is_correct=won,
        signals=[
            Signal(name="accuracy", value=reward),
            Signal(name="task_score", value=task_score),
            Signal(name="turns", value=float(episode.artifacts.get("turns", 0))),
            Signal(name="env_steps", value=float(episode.artifacts.get("env_steps", 0))),
            Signal(name="total_llm_calls", value=float(episode.artifacts.get("total_llm_calls", 0))),
            Signal(name="rewinds", value=float(episode.artifacts.get("rewinds", 0))),
            Signal(name="forced_rewinds", value=float(episode.artifacts.get("forced_rewinds", 0))),
            Signal(name="model_rewinds", value=float(episode.artifacts.get("model_rewinds", 0))),
            Signal(name="context_folds", value=float(episode.artifacts.get("context_folds", 0))),
        ],
        metadata={
            "won": won,
            "task_score": task_score,
            "session_id": episode.artifacts.get("session_id"),
        },
    )

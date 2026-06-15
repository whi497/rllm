"""MiniSweeper rewind evaluator."""

from __future__ import annotations

import rllm
from rllm.eval.types import EvalOutput, Signal
from rllm.types import Episode


@rllm.evaluator
def minisweeper_rewind_evaluator(task: dict, episode: Episode) -> EvalOutput:
    won = bool(episode.artifacts.get("won", False))
    reward = 1.0 if won else 0.0
    return EvalOutput(
        reward=reward,
        is_correct=won,
        signals=[
            Signal(name="accuracy", value=reward),
            Signal(name="turns", value=float(episode.artifacts.get("total_play_turns", 0))),
            Signal(name="env_steps", value=float(episode.artifacts.get("total_env_steps", 0))),
            Signal(name="rewinds", value=float(episode.artifacts.get("rewinds", 0))),
            Signal(name="segments", value=float(episode.artifacts.get("segments", 0))),
            Signal(name="n_mines", value=float(episode.artifacts.get("n_mines", 0))),
        ],
    )

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
            # Rewind-method structural probes.
            Signal(name="forced_rewinds", value=float(episode.artifacts.get("forced_rewinds", 0))),
            Signal(name="model_rewinds", value=float(episode.artifacts.get("model_rewinds", 0))),
            Signal(name="forced_context_folds", value=float(episode.artifacts.get("forced_context_folds", 0))),
            Signal(name="active_path_len", value=float(episode.artifacts.get("active_path_len", 0))),
            # Per-episode mean trajectory reward, split by reflection vs play segment.
            Signal(name="reflect_reward_avg", value=float(episode.artifacts.get("reflect_reward_avg", 0.0))),
            Signal(name="seg_reward_avg", value=float(episode.artifacts.get("seg_reward_avg", 0.0))),
        ],
    )

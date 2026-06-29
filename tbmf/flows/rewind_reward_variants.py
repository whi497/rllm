"""Reward and grouping helpers for replay-based rewind flows."""

from __future__ import annotations

import re
from typing import Literal

from rllm.types import Episode, Trajectory

PLAY_ROLE = "play"
REFLECT_ROLE = "reflect"


def segment_traj_name(prefix: str, traj_grouping: str, segment_idx: int) -> str:
    if traj_grouping == "per_stage":
        return f"{prefix}_seg{segment_idx}"
    if traj_grouping != "merged":
        raise ValueError(
            f"Unknown traj_grouping={traj_grouping!r}. Valid values are 'merged' and 'per_stage'."
        )
    return f"{prefix}_seg"


def reflect_traj_name(prefix: str, traj_grouping: str, segment_idx: int) -> str:
    if traj_grouping == "per_stage":
        return f"{prefix}_reflect{segment_idx}"
    if traj_grouping != "merged":
        raise ValueError(
            f"Unknown traj_grouping={traj_grouping!r}. Valid values are 'merged' and 'per_stage'."
        )
    return f"{prefix}_reflect"


def _segment_idx(name: str, default: int) -> int:
    match = re.search(r"seg(\d+)$", name)
    return int(match.group(1)) if match else default


def _reflect_idx(name: str, default: int) -> int:
    match = re.search(r"reflect(\d+)$", name)
    return int(match.group(1)) if match else default


def _trajectory_score(traj: Trajectory, score_key: str) -> float:
    if traj.metadata and score_key in traj.metadata:
        return float(traj.metadata.get(score_key) or 0.0)
    for step in reversed(traj.steps):
        metadata = step.metadata or {}
        if score_key in metadata:
            return float(metadata.get(score_key) or 0.0)
        if step.reward is not None:
            return float(step.reward)
    return 0.0


def annotate_rewind_episode(
    episode: Episode,
    *,
    prefix: str,
    traj_grouping: str,
    score_key: str,
    variant: Literal[
        "undiscounted_final",
        "reflect_reward_diff",
        "accumulated_reflect_diff",
        "segment_novelty_gate",
    ],
    reflect_reward_source: str = "cum_reward",
) -> Episode:
    """Rewrite a rewind-choice episode to match a reward variant.

    The ALFWorld/WebShop rewind-choice flows already implement the environment
    and reflection semantics. This helper applies the same naming and reward
    policies used by the MiniSweeper/Sokoban rewind reward family.

    ``reflect_reward_source`` selects how reflection rewards are computed:
      - ``"cum_reward"`` (default): forward diff of the per-segment score
        (``next_score - current_score``).
      - ``"milestone"``: forward diff of the expert-milestone count stored in
        each play segment's ``metadata["milestone_at_end"]`` (mirrors LaMer
        alfworld's separated reflect reward).
      - ``"rubric_progress"``: forward diff of the dense WebShop rubric potential
        stored in each play segment's ``metadata["rubric_at_end"]`` (the
        type/attribute/option/price decomposition scored against the currently
        viewed product). Play-segment rewards are unaffected by this switch.
    """
    # Maps the progress source to the play-segment metadata key it differences.
    _PROGRESS_METADATA_KEY = {
        "milestone": "milestone_at_end",
        "rubric_progress": "rubric_at_end",
    }
    if reflect_reward_source not in ("cum_reward", "milestone", "rubric_progress"):
        raise ValueError(
            f"Unknown reflect_reward_source={reflect_reward_source!r}; valid values are "
            "'cum_reward', 'milestone', and 'rubric_progress'."
        )
    progress_key = _PROGRESS_METADATA_KEY.get(reflect_reward_source)

    won = bool(episode.artifacts.get("won", episode.artifacts.get("success", False)))
    final_reward = 1.0 if won else 0.0
    play_entries: list[tuple[int, int, float]] = []
    reflect_entries: list[tuple[int, int]] = []
    progress_by_segment: dict[int, float] = {}

    for order, traj in enumerate(episode.trajectories):
        if traj.name.startswith(f"{prefix}_seg"):
            segment_idx = _segment_idx(traj.name, order)
            produced_new_state = any(_step_produced_new_state(step) for step in traj.steps)
            score = _trajectory_score(traj, score_key)
            metadata = dict(traj.metadata or {})
            if progress_key is not None:
                progress_by_segment[segment_idx] = float(metadata.get(progress_key, 0.0))
            metadata.update(
                {
                    "role": PLAY_ROLE,
                    "segment_idx": segment_idx,
                    "cum_reward_at_end": score,
                    "produced_new_state": produced_new_state,
                }
            )
            traj.metadata = metadata
            traj.name = segment_traj_name(prefix, traj_grouping, segment_idx)
            play_entries.append((segment_idx, order, score))
        elif traj.name.startswith(f"{prefix}_reflect"):
            segment_idx = _reflect_idx(traj.name, order)
            metadata = dict(traj.metadata or {})
            metadata.update({"role": REFLECT_ROLE, "segment_idx": segment_idx})
            traj.metadata = metadata
            traj.name = reflect_traj_name(prefix, traj_grouping, segment_idx)
            reflect_entries.append((segment_idx, order))

    play_entries.sort()
    reflect_entries.sort()

    if variant == "undiscounted_final":
        for _, idx, _ in play_entries:
            episode.trajectories[idx].reward = final_reward
        for _, idx in reflect_entries:
            episode.trajectories[idx].reward = final_reward
        return episode

    scores_by_segment = {segment_idx: score for segment_idx, _, score in play_entries}
    for _, idx, score in play_entries:
        if variant == "accumulated_reflect_diff":
            # Play segments get the path-cumulative env reward AS-IS (no
            # discounting, no terminal collapse): each segment's own score.
            episode.trajectories[idx].reward = score
        elif variant == "segment_novelty_gate":
            produced_new_state = bool((episode.trajectories[idx].metadata or {}).get("produced_new_state", True))
            episode.trajectories[idx].reward = final_reward if produced_new_state else 0.0
        else:
            episode.trajectories[idx].reward = final_reward

    for segment_idx, idx in reflect_entries:
        if progress_key is not None:
            # Forward diff of the chosen progress signal (expert milestones or the
            # dense rubric potential); positive when the next attempt advanced
            # further than where this reflection was triggered.
            current = progress_by_segment.get(segment_idx, 0.0)
            nxt = progress_by_segment.get(segment_idx + 1, current)
            episode.trajectories[idx].reward = nxt - current
        else:
            current_score = scores_by_segment.get(segment_idx, 0.0)
            next_score = scores_by_segment.get(segment_idx + 1, current_score)
            episode.trajectories[idx].reward = next_score - current_score

    return episode


def _step_produced_new_state(step) -> bool:
    metadata = step.metadata or {}
    if metadata.get("outcome") in {"advanced", "won"}:
        return True
    if metadata.get("won") is True:
        return True
    if metadata.get("action_is_valid") is True and step.action not in {
        "invalid",
        "invalid_rewind",
        "truncated",
    }:
        return True
    return False

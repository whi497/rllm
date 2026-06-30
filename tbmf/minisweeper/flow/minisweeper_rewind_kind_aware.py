"""Kind-aware MiniSweeper rewind ablation flows.

This module deliberately reuses the rollout mechanics from
``minesweeper_rewind_reflect_reward_diff`` and layers the A1--A6 ablation
changes on top:

A1 kind_force_discount  : terminal success is discounted by forced rewinds.
A2 kind_force_penalty   : A1 + explicit forced-rewind event penalty.
A3 kind_model_rewind    : A2 + model-rewind decision trajectory credit.
A4 kind_potential       : A3 + segment potential / efficiency reward.
A5 kind_opportunity     : A4 + rewind opportunity card in the action prompt.
A6 kind_structured      : A5 + structured branch memory prompt + repeat penalty.

A0 remains the existing ``accumulated_reflect_diff`` flow so that experiments can
compare against the current best baseline without changing its implementation.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
import math
import re
from typing import Any, Callable

import rllm
from rllm.types import AgentConfig, Episode, Task, Trajectory

try:
    from . import minisweeper_rewind_reflect_reward_diff as base
except (ImportError, ValueError):
    import minisweeper_rewind_reflect_reward_diff as base


# ---------------------------------------------------------------------------
# Ablation levels
# ---------------------------------------------------------------------------

LEVEL_FORCE_DISCOUNT = 1  # A1
LEVEL_FORCE_PENALTY = 2   # A2
LEVEL_MODEL_REWIND = 3    # A3
LEVEL_POTENTIAL = 4       # A4
LEVEL_OPPORTUNITY = 5     # A5
LEVEL_STRUCTURED = 6      # A6

_LEVEL_NAMES = {
    LEVEL_FORCE_DISCOUNT: "kind_force_discount",
    LEVEL_FORCE_PENALTY: "kind_force_penalty",
    LEVEL_MODEL_REWIND: "kind_model_rewind",
    LEVEL_POTENTIAL: "kind_potential",
    LEVEL_OPPORTUNITY: "kind_opportunity",
    LEVEL_STRUCTURED: "kind_structured",
}

# Conservative defaults. They can be overridden per run via
# +rllm.task_metadata_overrides.<name>=<value>.
DEFAULT_FORCE_DISCOUNT = 0.75
DEFAULT_FORCE_MINE_DISCOUNT = 0.50
DEFAULT_FORCE_PENALTY = 0.25
DEFAULT_INVALID_PENALTY = 0.10
DEFAULT_STALL_PENALTY = 0.10
DEFAULT_DEAD_PENALTY = 0.30
DEFAULT_PROGRESS_WEIGHT = 0.40
DEFAULT_EFFICIENCY_WEIGHT = 0.15
DEFAULT_MODEL_NEXT_WEIGHT = 0.50
DEFAULT_MODEL_EFF_WEIGHT = 0.20
DEFAULT_MODEL_SAVED_WEIGHT = 0.20
DEFAULT_MODEL_COST = 0.03
DEFAULT_BAD_MODEL_REWIND_PENALTY = 0.30
DEFAULT_REFLECT_DIFF_WEIGHT = 1.00
DEFAULT_REFLECT_MODEL_ALPHA = 1.00
DEFAULT_REFLECT_FORCE_ALPHA = 0.50
DEFAULT_REPEAT_PENALTY = 0.20

_EPS = 1e-8
_COORD_RE = re.compile(r"\((\d+)\s*,\s*(\d+)\)")
_REWIND_ACTION_RE = re.compile(r"^rewind\s+to\s+C?_?(\d+)$", re.IGNORECASE)

_ORIGINAL_BUILD_OBSERVATION_PROMPT = base._build_observation_prompt
_ORIGINAL_REFLECT_PROMPT = base.REWIND_REFLECT_PROMPT
_ORIGINAL_REFLECT_PROMPT_WITH_CKPT_CHOICE = base.REWIND_REFLECT_PROMPT_WITH_CKPT_CHOICE
_ORIGINAL_ASSIGN_REWARDS = base._assign_cross_segment_rewards


STRUCTURED_MEMORY_ADDENDUM = """

# Structured branch memory requirement
Inside <remark>, use the following compact XML-like fields. Keep every field
short and concrete; do not paste the full trajectory.

<remark>
  <rewind_from>C_{rewind_from_step}</rewind_from>
  <rewind_to>C_j</rewind_to>
  <failed_branch_summary>What this branch tried and why it failed or stalled.</failed_branch_summary>
  <avoid>Specific coordinates/actions that should not be repeated.</avoid>
  <learned_constraints>Concrete numbered-clue constraints, likely mines, and likely safe cells.</learned_constraints>
  <next_plan>A different next attempt from the selected checkpoint.</next_plan>
</remark>
"""


def _as_float(meta: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(meta.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _clip(value: float, lo: float = -0.5, hi: float = 0.5) -> float:
    return max(lo, min(hi, float(value)))


def _is_forced_rewind(log: dict[str, Any]) -> bool:
    return str(log.get("kind", "")).lower() == "forced"


def _is_model_rewind(log: dict[str, Any]) -> bool:
    return str(log.get("kind", "")).lower() == "model"


def _is_mine_or_dead_reason(reason: str) -> bool:
    lowered = (reason or "").lower()
    return "mine" in lowered or "dead" in lowered or "done=true" in lowered


def _force_severity(log: dict[str, Any]) -> float:
    reason = str(log.get("reason", ""))
    lowered = reason.lower()
    severity = 1.0
    if _is_mine_or_dead_reason(reason):
        severity += 1.0
    if "attempt budget" in lowered or "budget exhausted" in lowered:
        severity += 0.5
    if "truncated" in lowered or "invalid" in lowered or "failed" in lowered:
        severity += 0.5
    return severity


def _step_action(step: Any) -> str:
    return str(getattr(step, "action", "") or "")


def _count_invalid_steps(steps: list[Any]) -> int:
    count = 0
    for step in steps:
        action = _step_action(step).lower()
        if action in {"invalid", "invalid_rewind", "truncated"} or action.startswith("llm_"):
            count += 1
    return count


def _count_stall_steps(steps: list[Any]) -> int:
    # The base flow stores no-change details in BranchEvent rather than Step.  We
    # use a conservative step-level proxy here so the reward path remains robust
    # without changing the base rollout implementation.
    count = 0
    for step in steps:
        action = _step_action(step).lower()
        if "no visible change" in action or "already" in action or "ineffective" in action:
            count += 1
    return count


def _role(traj: Trajectory) -> str:
    return str((getattr(traj, "metadata", None) or {}).get("role", ""))


def _segment_idx(traj: Trajectory, fallback: int) -> int:
    meta = getattr(traj, "metadata", None) or {}
    try:
        return int(meta.get("segment_idx", meta.get("segment", fallback)))
    except (TypeError, ValueError):
        return fallback


def _cum_reward(traj: Trajectory) -> float:
    meta = getattr(traj, "metadata", None) or {}
    try:
        return float(meta.get("cum_reward_at_end", meta.get("cum_reward_at_trigger", 0.0)) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _copy_traj_with_steps(traj: Trajectory, steps: list[Any]) -> Trajectory:
    return Trajectory(
        name=traj.name,
        steps=steps,
        reward=getattr(traj, "reward", None),
        metadata=copy.deepcopy(getattr(traj, "metadata", None) or {}),
    )


def _traj_name(prefix: str, traj_grouping: str, segment_idx: int) -> str:
    if traj_grouping == "per_stage":
        return f"{prefix}_{segment_idx}"
    return f"minisweeper_{prefix}"


def _append_rewind_decision_trajectories(
    trajectories: list[Trajectory],
    traj_grouping: str,
) -> list[Trajectory]:
    """Move explicit model-rewind steps into separate decision trajectories.

    The underlying base flow records the model rewind as the final step of the
    current play segment.  For A3+ we train it with its own reward by extracting
    that step into a role="rewind_decision" trajectory.  Keeping a copy inside
    the play trajectory would double-count the same log-prob, so it is removed
    from the play segment.  Non-rewind play steps stay untouched.
    """
    rewritten: list[Trajectory] = []
    decision_trajs: list[Trajectory] = []

    for fallback_idx, traj in enumerate(trajectories):
        if _role(traj) != base.PLAY_ROLE:
            rewritten.append(traj)
            continue

        segment_idx = _segment_idx(traj, fallback_idx)
        kept_steps: list[Any] = []
        for step in list(traj.steps or []):
            action = _step_action(step).strip()
            match = _REWIND_ACTION_RE.match(action)
            if match:
                target = int(match.group(1))
                decision_trajs.append(
                    Trajectory(
                        name=_traj_name("model_rewind", traj_grouping, segment_idx),
                        steps=[step],
                        reward=None,
                        metadata={
                            "role": "rewind_decision",
                            "segment_idx": segment_idx,
                            "rewind_to": target,
                            "source_traj": traj.name,
                            "cum_reward_at_decision": _cum_reward(traj),
                        },
                    )
                )
            else:
                kept_steps.append(step)

        if kept_steps:
            rewritten.append(_copy_traj_with_steps(traj, kept_steps))
        else:
            # Empty play segment: keep no trainable play trajectory.  Its rewind
            # decision trajectory above still carries the policy-gradient signal.
            continue

    rewritten.extend(decision_trajs)
    return rewritten


def _build_log_maps(artifacts: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    rewind_log = list(artifacts.get("rewind_log", []) or [])
    by_segment: dict[int, dict[str, Any]] = {}
    for log in rewind_log:
        try:
            by_segment[int(log.get("segment", -1))] = log
        except (TypeError, ValueError):
            continue
    return by_segment, rewind_log


def _extract_avoid_coords(memory: str) -> set[tuple[int, int]]:
    if not memory:
        return set()
    lowered = memory.lower()
    # Prefer coordinates mentioned around explicit avoid fields; fall back to all
    # coordinates in the memory if the model did not follow the schema exactly.
    avoid_chunks = re.findall(r"<avoid>(.*?)</avoid>", lowered, flags=re.DOTALL)
    text = "\n".join(avoid_chunks) if avoid_chunks else lowered
    return {(int(r), int(c)) for r, c in _COORD_RE.findall(text)}


def _trajectory_reveals_coords(traj: Trajectory, coords: set[tuple[int, int]]) -> bool:
    if not coords:
        return False
    for step in traj.steps or []:
        match = _COORD_RE.search(_step_action(step))
        if not match:
            continue
        coord = (int(match.group(1)), int(match.group(2)))
        if coord in coords:
            return True
    return False


def _repeat_bad_action_segments(artifacts: dict[str, Any], play_by_segment: dict[int, int], trajectories: list[Trajectory]) -> set[int]:
    repeated: set[int] = set()
    for record in artifacts.get("branch_memories", []) or []:
        try:
            segment = int(record.get("segment", -1))
        except (TypeError, ValueError):
            continue
        avoid_coords = _extract_avoid_coords(str(record.get("memory", "")))
        if not avoid_coords:
            continue
        next_segments = sorted(s for s in play_by_segment if s > segment)
        if not next_segments:
            continue
        next_segment = next_segments[0]
        next_traj = trajectories[play_by_segment[next_segment]]
        if _trajectory_reveals_coords(next_traj, avoid_coords):
            repeated.add(next_segment)
    return repeated


def _assign_kind_aware_rewards(
    trajectories: list[Trajectory],
    won: bool,
    artifacts: dict[str, Any],
    meta: dict[str, Any],
    level: int,
) -> dict[str, Any]:
    force_discount = _as_float(meta, "force_discount", DEFAULT_FORCE_DISCOUNT)
    force_mine_discount = _as_float(meta, "force_mine_discount", DEFAULT_FORCE_MINE_DISCOUNT)
    force_penalty_w = _as_float(meta, "force_penalty", DEFAULT_FORCE_PENALTY)
    invalid_penalty_w = _as_float(meta, "invalid_penalty", DEFAULT_INVALID_PENALTY)
    stall_penalty_w = _as_float(meta, "stall_penalty", DEFAULT_STALL_PENALTY)
    dead_penalty_w = _as_float(meta, "dead_penalty", DEFAULT_DEAD_PENALTY)
    progress_w = _as_float(meta, "progress_weight", DEFAULT_PROGRESS_WEIGHT)
    efficiency_w = _as_float(meta, "efficiency_weight", DEFAULT_EFFICIENCY_WEIGHT)
    model_next_w = _as_float(meta, "model_next_weight", DEFAULT_MODEL_NEXT_WEIGHT)
    model_eff_w = _as_float(meta, "model_eff_weight", DEFAULT_MODEL_EFF_WEIGHT)
    model_saved_w = _as_float(meta, "model_saved_weight", DEFAULT_MODEL_SAVED_WEIGHT)
    model_cost = _as_float(meta, "model_rewind_cost", DEFAULT_MODEL_COST)
    bad_model_penalty = _as_float(meta, "bad_model_rewind_penalty", DEFAULT_BAD_MODEL_REWIND_PENALTY)
    reflect_diff_w = _as_float(meta, "reflect_diff_weight", DEFAULT_REFLECT_DIFF_WEIGHT)
    reflect_model_alpha = _as_float(meta, "reflect_model_alpha", DEFAULT_REFLECT_MODEL_ALPHA)
    reflect_force_alpha = _as_float(meta, "reflect_force_alpha", DEFAULT_REFLECT_FORCE_ALPHA)
    repeat_penalty_w = _as_float(meta, "repeat_penalty", DEFAULT_REPEAT_PENALTY)

    log_by_segment, rewind_log = _build_log_maps(artifacts)
    forced_logs = [log for log in rewind_log if _is_forced_rewind(log)]
    forced_mine_logs = [log for log in forced_logs if _is_mine_or_dead_reason(str(log.get("reason", "")))]

    terminal_reward = 1.0 if won else 0.0
    if level >= LEVEL_FORCE_DISCOUNT:
        terminal_reward *= force_discount ** len(forced_logs)
        terminal_reward *= force_mine_discount ** len(forced_mine_logs)

    play_indices = [i for i, t in enumerate(trajectories) if _role(t) == base.PLAY_ROLE]
    reflect_indices = [i for i, t in enumerate(trajectories) if _role(t) == base.REFLECT_ROLE]
    decision_indices = [i for i, t in enumerate(trajectories) if _role(t) == "rewind_decision"]

    play_indices.sort(key=lambda i: _segment_idx(trajectories[i], i))
    reflect_indices.sort(key=lambda i: _segment_idx(trajectories[i], i))
    decision_indices.sort(key=lambda i: _segment_idx(trajectories[i], i))

    play_by_segment = {_segment_idx(trajectories[i], i): i for i in play_indices}
    play_cum_by_segment = {seg: _cum_reward(trajectories[idx]) for seg, idx in play_by_segment.items()}
    ordered_segments = sorted(play_by_segment)

    repeat_segments = _repeat_bad_action_segments(artifacts, play_by_segment, trajectories) if level >= LEVEL_STRUCTURED else set()

    # Play reward: terminal force discount for A1, explicit force penalty for A2,
    # and potential / efficiency shaping for A4+.
    segment_stats: dict[int, dict[str, float]] = {}
    prev_cum = 0.0
    for order, segment in enumerate(ordered_segments):
        traj_idx = play_by_segment[segment]
        traj = trajectories[traj_idx]
        cum = play_cum_by_segment[segment]
        turns = max(1, len(traj.steps or []))
        delta = cum - prev_cum
        eff = delta / turns
        invalid_rate = _count_invalid_steps(list(traj.steps or [])) / turns
        stall_rate = _count_stall_steps(list(traj.steps or [])) / turns
        log = log_by_segment.get(segment, {})
        forced = _is_forced_rewind(log)
        dead_or_mine = _is_mine_or_dead_reason(str(log.get("reason", "")))

        reward = terminal_reward
        if level >= LEVEL_POTENTIAL:
            reward += progress_w * _clip(delta)
            reward += efficiency_w * _clip(eff)
            reward -= invalid_penalty_w * invalid_rate
            reward -= stall_penalty_w * stall_rate
            if dead_or_mine:
                reward -= dead_penalty_w
        if level >= LEVEL_FORCE_PENALTY and forced:
            reward -= force_penalty_w * _force_severity(log)
        if level >= LEVEL_STRUCTURED and segment in repeat_segments:
            reward -= repeat_penalty_w

        traj.reward = float(reward)
        segment_stats[segment] = {
            "cum": cum,
            "delta": delta,
            "eff": eff,
            "turns": float(turns),
            "forced": 1.0 if forced else 0.0,
        }
        prev_cum = cum

    # Reflection reward: retain the existing forward diff signal, with optional
    # A6 kind scaling and repeat penalty.
    for reflect_order, traj_idx in enumerate(reflect_indices):
        traj = trajectories[traj_idx]
        segment = _segment_idx(traj, reflect_order)
        this_cum = play_cum_by_segment.get(segment, 0.0)
        later_segments = [s for s in ordered_segments if s > segment]
        next_segment = later_segments[0] if later_segments else None
        next_cum = play_cum_by_segment.get(next_segment, this_cum) if next_segment is not None else this_cum
        diff = next_cum - this_cum
        log = log_by_segment.get(segment, {})
        alpha = 1.0
        if level >= LEVEL_STRUCTURED:
            alpha = reflect_model_alpha if _is_model_rewind(log) else reflect_force_alpha
        reward = reflect_diff_w * alpha * _clip(diff)
        if level >= LEVEL_STRUCTURED and next_segment in repeat_segments:
            reward -= repeat_penalty_w
        traj.reward = float(reward)

    # Model rewind decision reward: reward the terminate-and-rollback decision
    # only when the next segment improves relative to the abandoned branch.
    for traj_idx in decision_indices:
        traj = trajectories[traj_idx]
        segment = _segment_idx(traj, 0)
        log = log_by_segment.get(segment, {})
        current = segment_stats.get(segment, {"cum": play_cum_by_segment.get(segment, 0.0), "eff": 0.0})
        later_segments = [s for s in ordered_segments if s > segment]
        next_segment = later_segments[0] if later_segments else None
        next_stats = segment_stats.get(next_segment, {"cum": current["cum"], "eff": 0.0}) if next_segment is not None else {"cum": current["cum"], "eff": 0.0}
        next_gain = float(next_stats["cum"]) - float(current["cum"])
        eff_gain = float(next_stats["eff"]) - float(current.get("eff", 0.0))
        try:
            rewind_from = float(log.get("from", 0) or 0)
            rewind_to = float(log.get("to", 0) or 0)
        except (TypeError, ValueError):
            rewind_from = 0.0
            rewind_to = 0.0
        saved = max(0.0, rewind_from - rewind_to) / max(1.0, rewind_from)
        bad = 1.0 if next_gain <= _EPS else 0.0
        reward = (
            model_next_w * _clip(next_gain)
            + model_eff_w * _clip(eff_gain)
            + model_saved_w * saved
            - model_cost
            - bad_model_penalty * bad
        )
        traj.reward = float(reward)
        traj.metadata = dict(getattr(traj, "metadata", None) or {})
        traj.metadata.update(
            {
                "next_gain": next_gain,
                "eff_gain": eff_gain,
                "saved_budget_proxy": saved,
                "bad_model_rewind": bool(bad),
            }
        )

    # Ensure no None rewards leak downstream.
    for traj in trajectories:
        if getattr(traj, "reward", None) is None:
            traj.reward = 0.0

    def _avg_reward(indices: list[int]) -> float:
        vals = [float(trajectories[i].reward or 0.0) for i in indices]
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "kind_level": level,
        "kind_flow": _LEVEL_NAMES[level],
        "terminal_reward_after_force_discount": terminal_reward,
        "force_discount_factor": terminal_reward if won else 0.0,
        "forced_rewind_penalty_events": len(forced_logs),
        "forced_mine_penalty_events": len(forced_mine_logs),
        "repeat_bad_action_segments": sorted(repeat_segments),
        "play_reward_avg_kind": _avg_reward(play_indices),
        "reflect_reward_avg_kind": _avg_reward(reflect_indices),
        "model_rewind_reward_avg_kind": _avg_reward(decision_indices),
        "model_rewind_decision_trajs": len(decision_indices),
    }


def _build_observation_prompt_kind_aware(*args: Any, **kwargs: Any) -> str:
    prompt = _ORIGINAL_BUILD_OBSERVATION_PROMPT(*args, **kwargs)
    level = int(getattr(base, "_TBMF_KIND_AWARE_LEVEL", 0) or 0)
    if level < LEVEL_OPPORTUNITY:
        return prompt

    branch_attempts_used = int(kwargs.get("branch_attempts_used", args[1] if len(args) > 1 else 0))
    game_position = int(kwargs.get("game_position", args[2] if len(args) > 2 else 0))
    segment_max_turns = int(kwargs.get("segment_max_turns", args[5] if len(args) > 5 else 20))
    active_branch_events = kwargs.get("active_branch_events", None)
    if active_branch_events is None and len(args) > 6:
        active_branch_events = args[6]
    active_branch_events = list(active_branch_events or [])

    branch_remaining = max(0, segment_max_turns - branch_attempts_used)
    recent_events = active_branch_events[-6:]
    invalid_like = sum(1 for e in recent_events if str(getattr(e, "kind", "")).lower() in {"invalid", "invalid_rewind", "truncated"})
    no_change_like = sum(1 for e in recent_events if "no visible change" in str(getattr(e, "outcome", "")).lower())
    mine_or_dead = any(_is_mine_or_dead_reason(str(getattr(e, "outcome", ""))) for e in recent_events)
    should_show = branch_remaining <= 3 or invalid_like >= 2 or no_change_like >= 2 or mine_or_dead
    if not should_show or game_position <= 0:
        return prompt

    recent_target = max(0, game_position - 2)
    full_restart = 0
    candidate_lines = [
        f"- C_{recent_target}: recent rollback target; useful if only the last few reveals caused uncertainty.",
    ]
    if full_restart != recent_target:
        candidate_lines.append("- C_0: full restart; useful after a mine hit or repeated invalid/stalled branch.")

    card = "\n\n# Rewind opportunity card\n"
    card += "The current branch shows a possible trap/stagnation signal. Continuing is allowed only if you have a concrete safe reveal plan.\n"
    card += f"Branch budget remaining: {branch_remaining}/{segment_max_turns}.\n"
    card += f"Recent invalid/truncated events: {invalid_like}; recent no-change events: {no_change_like}.\n"
    card += "Candidate rollback targets:\n" + "\n".join(candidate_lines) + "\n"
    card += "If the next reveal is mostly a guess or repeats an avoided coordinate, proactively use <action>rewind to C_j</action>."
    return prompt + card


@contextmanager
def _configured_base(level: int):
    """Configure the imported base rollout for one kind-aware experiment.

    Rollout workers normally run one flow key per process.  We still keep the
    configuration localized and idempotent so tests can call the wrappers in a
    deterministic way.
    """
    previous_level = getattr(base, "_TBMF_KIND_AWARE_LEVEL", 0)
    previous_builder = base._build_observation_prompt
    previous_reflect = base.REWIND_REFLECT_PROMPT
    previous_reflect_choice = base.REWIND_REFLECT_PROMPT_WITH_CKPT_CHOICE
    previous_assign = base._assign_cross_segment_rewards
    try:
        base._TBMF_KIND_AWARE_LEVEL = level
        base._build_observation_prompt = _build_observation_prompt_kind_aware
        if level >= LEVEL_STRUCTURED:
            base.REWIND_REFLECT_PROMPT = _ORIGINAL_REFLECT_PROMPT + STRUCTURED_MEMORY_ADDENDUM
            base.REWIND_REFLECT_PROMPT_WITH_CKPT_CHOICE = _ORIGINAL_REFLECT_PROMPT_WITH_CKPT_CHOICE + STRUCTURED_MEMORY_ADDENDUM
        else:
            base.REWIND_REFLECT_PROMPT = _ORIGINAL_REFLECT_PROMPT
            base.REWIND_REFLECT_PROMPT_WITH_CKPT_CHOICE = _ORIGINAL_REFLECT_PROMPT_WITH_CKPT_CHOICE
        # Rewards are recomputed after post-processing, but no-oping the base
        # assignment avoids transient reward values in logs if the base ever adds
        # debug snapshots before return.
        base._assign_cross_segment_rewards = lambda trajectories, won, traj_gamma: None
        yield
    finally:
        base._TBMF_KIND_AWARE_LEVEL = previous_level
        base._build_observation_prompt = previous_builder
        base.REWIND_REFLECT_PROMPT = previous_reflect
        base.REWIND_REFLECT_PROMPT_WITH_CKPT_CHOICE = previous_reflect_choice
        base._assign_cross_segment_rewards = previous_assign


async def _run_kind_aware(task: Task, config: AgentConfig, level: int) -> Episode:
    meta = dict(task.metadata or {})
    traj_grouping = str(meta.get("traj_grouping", "merged"))
    with _configured_base(level):
        episode = await base.minisweeper_rewind_reflect_reward_diff_flow(task, config)

    trajectories = list(episode.trajectories or [])
    if level >= LEVEL_MODEL_REWIND:
        trajectories = _append_rewind_decision_trajectories(trajectories, traj_grouping=traj_grouping)

    artifacts = dict(episode.artifacts or {})
    reward_debug = _assign_kind_aware_rewards(
        trajectories=trajectories,
        won=bool(episode.is_correct),
        artifacts=artifacts,
        meta=meta,
        level=level,
    )

    episode.trajectories = trajectories
    artifacts.update(reward_debug)
    episode.artifacts = artifacts
    metrics = dict(episode.metrics or {})
    metrics.update({f"kind/{k}": v for k, v in reward_debug.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
    episode.metrics = metrics
    return episode


@rllm.rollout(name="minisweeper_kind_force_discount")
async def minisweeper_rewind_kind_force_discount_flow(task: Task, config: AgentConfig) -> Episode:
    return await _run_kind_aware(task, config, LEVEL_FORCE_DISCOUNT)


@rllm.rollout(name="minisweeper_kind_force_penalty")
async def minisweeper_rewind_kind_force_penalty_flow(task: Task, config: AgentConfig) -> Episode:
    return await _run_kind_aware(task, config, LEVEL_FORCE_PENALTY)


@rllm.rollout(name="minisweeper_kind_model_rewind")
async def minisweeper_rewind_kind_model_rewind_flow(task: Task, config: AgentConfig) -> Episode:
    return await _run_kind_aware(task, config, LEVEL_MODEL_REWIND)


@rllm.rollout(name="minisweeper_kind_potential")
async def minisweeper_rewind_kind_potential_flow(task: Task, config: AgentConfig) -> Episode:
    return await _run_kind_aware(task, config, LEVEL_POTENTIAL)


@rllm.rollout(name="minisweeper_kind_opportunity")
async def minisweeper_rewind_kind_opportunity_flow(task: Task, config: AgentConfig) -> Episode:
    return await _run_kind_aware(task, config, LEVEL_OPPORTUNITY)


@rllm.rollout(name="minisweeper_kind_structured")
async def minisweeper_rewind_kind_structured_flow(task: Task, config: AgentConfig) -> Episode:
    return await _run_kind_aware(task, config, LEVEL_STRUCTURED)

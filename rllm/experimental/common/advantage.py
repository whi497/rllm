"""
Generic advantage computation algorithms and utilities that work on TrajectoryGroups.

Each advantage estimator will return a tuple of (advantages, returns).
"""

import logging
import re
from collections import defaultdict
from collections.abc import Callable

import numpy as np

from rllm.experimental.common.config import AlgorithmConfig, rLLMAdvantageEstimator
from rllm.experimental.common.rl_algo import calculate_grpo_advantages_per_group, calculate_rloo_advantages_per_group
from rllm.types import TrajectoryGroup
from rllm.utils.logging import DuplicateLoggingFilter

logger = logging.getLogger(__name__)
logger.addFilter(DuplicateLoggingFilter())  # prevent duplicate logging messages

_STRUCTURAL_SUFFIX_RE = re.compile(r'_(ep\d+|reflect\d+)$')


def _extract_metric_suffix(group_role: str) -> str | None:
    """Extract structural suffix (ep0, reflect0) from group_role, stripping env prefix and mode suffix (_step)."""
    m = _STRUCTURAL_SUFFIX_RE.search(group_role)
    return m.group(1) if m else None


RLLM_ADV_ESTIMATOR_REGISTRY: dict[str, Callable] = {}


def register_rllm_adv_estimator(name: str | rLLMAdvantageEstimator) -> Callable:
    """Register a rLLM advantage estimator — either built-in or custom.

    Registered estimators must follow the canonical signature:

        def my_estimator(
            rewards: list[np.ndarray],
            algorithm_config: AlgorithmConfig,
            **kwargs,
        ) -> tuple[list[np.ndarray], list[np.ndarray]]

    `rewards` is one entry per `TrajectoryGroup` of the same `group_role`;
    each entry is a 1-D array of scalar trajectory rewards. The output
    `(advantages_by_group, returns_by_group)` must be aligned with
    `rewards` (same outer length and same inner shapes).

    `algorithm_config` is the resolved `AlgorithmConfig`; pull whatever
    config the estimator needs (e.g. `norm_adv_by_std_in_grpo`).

    `**kwargs` carries optional per-call data injected by
    `collect_reward_and_advantage_from_trajectory_groups`. The orchestrator
    currently injects:

    * `traj_groups: list[TrajectoryGroup]` — aligned with `rewards`,
      so estimators can read per-trajectory metadata (response lengths,
      step counts, etc.) from `traj_groups[i].trajectories[j].steps`.

    Args:
        name: Name of the advantage estimator.
    """

    def decorator(func: Callable) -> Callable:
        RLLM_ADV_ESTIMATOR_REGISTRY[name] = func
        return func

    return decorator


def get_rllm_adv_estimator(name: str | rLLMAdvantageEstimator) -> Callable:
    """Get a rLLM advantage estimator by name.

    Args:
        name: Name of the advantage estimator.
    """
    if name not in RLLM_ADV_ESTIMATOR_REGISTRY:
        raise ValueError(f"Unknown advantage estimator {name}. If you have a custom advantage estimator, please register it using `register_rllm_adv_estimator`.")
    return RLLM_ADV_ESTIMATOR_REGISTRY[name]


@register_rllm_adv_estimator(rLLMAdvantageEstimator.GRPO)
def calculate_grpo_advantages(rewards: list[np.ndarray], algorithm_config: AlgorithmConfig, **kwargs) -> tuple[list[np.ndarray], list[np.ndarray]]:
    norm_adv_by_std_in_grpo = algorithm_config.norm_adv_by_std_in_grpo
    advantages_by_group, returns_by_group = zip(
        *[calculate_grpo_advantages_per_group(group_rewards, norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo) for group_rewards in rewards],
        strict=True,
    )

    return advantages_by_group, returns_by_group


@register_rllm_adv_estimator(rLLMAdvantageEstimator.REINFORCE)
def calculate_reinforce_advantages(rewards: list[np.ndarray], algorithm_config: AlgorithmConfig, **kwargs) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """REINFORCE: advantage = reward (no baseline)"""
    return rewards, rewards


@register_rllm_adv_estimator(rLLMAdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE)
def calculate_reinforce_plus_plus_baseline_advantages(rewards: list[np.ndarray], algorithm_config: AlgorithmConfig, epsilon: float = 1e-6, **kwargs) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """REINFORCE++ baseline estimator.

    In line with Verl's REINFORCE++ baseline logic for grouped rollouts:
    1. Use per-group mean baseline when group size > 1, else baseline = 0.
    2. Whiten centered scores using role-level batch statistics.
    """
    if len(rewards) == 0:
        return [], []

    centered_rewards_by_group: list[np.ndarray] = []
    for group_rewards in rewards:
        centered_rewards_by_group.append(group_rewards - np.mean(group_rewards))

    all_centered_rewards = np.concatenate(centered_rewards_by_group)
    batch_std = np.std(all_centered_rewards)

    advantages_by_group = [centered_rewards / (batch_std + epsilon) for centered_rewards in centered_rewards_by_group]

    return advantages_by_group, advantages_by_group


@register_rllm_adv_estimator(rLLMAdvantageEstimator.RLOO)
def calculate_rloo_advantages(rewards: list[np.ndarray], algorithm_config: AlgorithmConfig, **kwargs) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Reinforce Leave-one-out (RLOO): https://arxiv.org/abs/2402.14740"""
    advantages_by_group, returns_by_group = zip(*[calculate_rloo_advantages_per_group(group_rewards) for group_rewards in rewards], strict=True)
    return advantages_by_group, returns_by_group


def _grpo_1d(values: np.ndarray, *, remove_std: bool, epsilon: float = 1e-6) -> np.ndarray:
    """Group-relative (GRPO-style) normalization of a 1-D array of scalars.

    ``remove_std=True``  -> mean-centering only (``mean_norm``).
    ``remove_std=False`` -> mean/std normalization (``mean_std_norm``).
    Groups of size <= 1 get a zero advantage (no relative signal).
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1:
        return np.zeros_like(arr)
    mean = arr.mean()
    if remove_std:
        return arr - mean
    return (arr - mean) / (arr.std() + epsilon)


@register_rllm_adv_estimator(rLLMAdvantageEstimator.GIGPO)
def calculate_gigpo_advantages(
    rewards: list[np.ndarray],
    algorithm_config: AlgorithmConfig,
    traj_groups: list[TrajectoryGroup] | None = None,
    **kwargs,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """GiGPO (Group-in-Group) advantage: ``A = A_macro + w * A_micro``.

    https://arxiv.org/abs/2505.10978 — adapted to the rLLM trajectory-row model.

    Each trajectory row is expected to carry GiGPO metadata stamped by the flow:

    * ``gigpo_macro_key``   — identifier of the segment instance the row belongs
      to (all rows of one play segment share it). The *episode-level* (macro)
      term is GRPO over the one ``gigpo_macro_reward`` per unique macro key, then
      broadcast back to every row of that segment.
    * ``gigpo_macro_reward`` — the segment's macro reward S_i (kept as the macro
      signal, matching the ``accumulated_reflect_diff`` reward flow).
    * ``gigpo_anchor``      — the environment observation (board) BEFORE the
      row's action. The *step-level* (micro) term clusters all rows in the group
      (i.e. across all rollouts/branches of the same task) by identical anchor
      and GRPO-normalizes ``gigpo_step_return`` within each anchor cluster.
    * ``gigpo_step_return`` — the row's discounted return-to-go within its branch.

    Groups whose rows lack ``gigpo_anchor`` (e.g. reflection groups) fall back to
    plain GRPO over the trajectory rewards, so the reflection reward flow is left
    untouched with no per-role routing config required.

    Macro normalization follows ``norm_adv_by_std_in_grpo``; micro normalization
    follows ``gigpo_mode`` (``mean_norm`` -> mean-only, ``mean_std_norm`` ->
    mean/std).
    """
    assert traj_groups is not None, "gigpo estimator requires traj_groups (injected by the orchestrator)"

    w = float(algorithm_config.gigpo_step_advantage_w)
    micro_remove_std = algorithm_config.gigpo_mode == "mean_norm"

    advantages_by_group: list[np.ndarray] = []
    returns_by_group: list[np.ndarray] = []

    for group_rewards, group in zip(rewards, traj_groups, strict=True):
        trajs = group.trajectories
        metas = [t.metadata or {} for t in trajs]
        n = len(trajs)

        # A group is a "gigpo" (play) group if ANY row carries an anchor. Rows
        # without an anchor (invalid/truncated/rewind turns interleaved in a
        # reveal-granularity play group) get macro-only credit. Groups with no
        # anchored row at all (e.g. reflections) fall back to plain GRPO so the
        # reflection reward flow is left untouched.
        if not any("gigpo_anchor" in m for m in metas):
            adv, ret = calculate_grpo_advantages_per_group(
                np.asarray(group_rewards, dtype=np.float64),
                norm_adv_by_std_in_grpo=algorithm_config.norm_adv_by_std_in_grpo,
            )
            advantages_by_group.append(adv)
            returns_by_group.append(ret)
            continue

        macro_keys = [str(m.get("gigpo_macro_key", trajs[i].uid)) for i, m in enumerate(metas)]
        macro_rewards = [float(m.get("gigpo_macro_reward", group_rewards[i])) for i, m in enumerate(metas)]

        # Episode-level (macro): GRPO over one value per unique segment instance,
        # then broadcast back to all rows of that segment. Uses the repo's own
        # per-group GRPO so the macro term is byte-identical to plain GRPO,
        # including its singleton convention (size-1 group -> raw value, mean=0
        # std=1), which also matches the reference GiGPO episode_norm_reward.
        first_seen: dict[str, int] = {}
        uniq_macro_vals: list[float] = []
        for key, val in zip(macro_keys, macro_rewards):
            if key not in first_seen:
                first_seen[key] = len(uniq_macro_vals)
                uniq_macro_vals.append(val)
        macro_adv_uniq, _ = calculate_grpo_advantages_per_group(
            np.asarray(uniq_macro_vals, dtype=np.float64),
            norm_adv_by_std_in_grpo=algorithm_config.norm_adv_by_std_in_grpo,
        )
        macro_adv_by_key = {key: float(macro_adv_uniq[pos]) for key, pos in first_seen.items()}

        # Step-level (micro): cluster ONLY anchored rows by identical anchor board
        # and GRPO-normalize their per-row discounted return within each cluster.
        # Non-anchored rows keep micro=0 (macro-only credit).
        step_returns = [float(m.get("gigpo_step_return", 0.0)) for m in metas]
        anchor_clusters: dict[str, list[int]] = defaultdict(list)
        for i, m in enumerate(metas):
            if "gigpo_anchor" in m:
                anchor_clusters[str(m.get("gigpo_anchor"))].append(i)

        micro_adv = np.zeros(n, dtype=np.float64)
        for idxs in anchor_clusters.values():
            cluster_adv = _grpo_1d(np.asarray([step_returns[i] for i in idxs]), remove_std=micro_remove_std)
            for j, i in enumerate(idxs):
                micro_adv[i] = cluster_adv[j]

        final = np.asarray([macro_adv_by_key[macro_keys[i]] + w * micro_adv[i] for i in range(n)], dtype=np.float64)
        advantages_by_group.append(final)
        returns_by_group.append(final)

    return advantages_by_group, returns_by_group



def _collect_precomputed_advantages(group: TrajectoryGroup, group_role: str) -> list[float]:
    """Collect pre-computed per-token advantages from all steps.

    Called when use_precomputed_advantage is True. Steps with None or length-mismatched
    advantages are defaulted to zero lists. Raises if step.advantage is a scalar float
    (pre-computed advantages must be per-token lists).
    """
    flattened_advantages = []
    steps_missing = 0
    total_steps = 0

    for traj in group.trajectories:
        for step in traj.steps:
            total_steps += 1
            if isinstance(step.advantage, float):
                step.advantage = [step.advantage] * len(step.response_ids)
            elif isinstance(step.advantage, list):
                if len(step.advantage) != len(step.response_ids):
                    logger.warning(f"[group={group_role}] Step has advantage length {len(step.advantage)} but response_ids length {len(step.response_ids)}. Defaulting to zeros.")
                    step.advantage = [0.0] * len(step.response_ids)
                    steps_missing += 1
            else:
                raise ValueError(f"[group={group_role}] step.advantage must be a scalar or a list when use_precomputed_advantage is True, got {type(step.advantage)}")

            flattened_advantages.extend(step.advantage)

    if steps_missing > 0:
        logger.warning(f"[group={group_role}] {steps_missing}/{total_steps} steps missing pre-computed advantages, defaulted to zeros.")

    return flattened_advantages


def collect_reward_and_advantage_from_trajectory_groups(
    groups: list[TrajectoryGroup],
    algorithm_config: AlgorithmConfig,
    collect_advantage: bool = True,
) -> dict:
    """
    Collect reward and advantage from trajectory groups. Return a dictionary of metrics.
    If collect_advantage is False, only collect rewards.

    Args:
        groups: List of TrajectoryGroup objects
        algorithm_config: Algorithm configuration
        collect_advantage: Whether to collect advantage

    Returns:
        Dictionary of metrics
    """
    assert algorithm_config.stepwise_advantage_mode == "broadcast", "Only broadcast mode is supported in experimental unified trainer."

    advantages_by_role = defaultdict(list)
    rewards_by_role = defaultdict(list)
    traj_rewards_by_role = defaultdict(list)
    traj_groups_by_role = defaultdict(list)

    for group in groups:
        group_role = group.group_role
        has_precomputed_advantage = any(step.advantage is not None for traj in group.trajectories for step in traj.steps)

        if has_precomputed_advantage and algorithm_config.use_precomputed_advantage:
            # Precompute mode (e.g. OPD, SFT): always use pre-computed per-token advantages from the workflow.
            if collect_advantage:
                flattened_advantages = _collect_precomputed_advantages(group, group_role)
                advantages_by_role[group_role].extend(flattened_advantages)
        else:
            # RL mode: compute advantages from trajectory rewards.
            if collect_advantage and has_precomputed_advantage:
                logger.warning(f"[group={group_role}] Steps have pre-computed advantages but use_precomputed_advantage is False. Overwriting with {algorithm_config.estimator.value}.")

            assert all(traj.reward is not None for traj in group.trajectories), "Trajectory reward cannot be None in broadcast mode"
            traj_rewards = np.array([traj.reward for traj in group.trajectories])
            rewards_by_role[group_role].extend(traj_rewards)

            if collect_advantage:
                traj_groups_by_role[group_role].append(group)
                traj_rewards_by_role[group_role].append(traj_rewards)

    if collect_advantage:
        for group_role, traj_groups in traj_groups_by_role.items():
            advantage_fn = get_rllm_adv_estimator(algorithm_config.estimator_map.get(group_role, algorithm_config.estimator))
            traj_rewards = traj_rewards_by_role[group_role]
            advantages_by_group, _ = advantage_fn(  # ignore returns here
                rewards=traj_rewards,
                algorithm_config=algorithm_config,
                traj_groups=traj_groups,
            )
            assert len(advantages_by_group) == len(traj_groups), "length mismatch between advantages and trajectory groups"
            for traj_group, advantages_by_traj in zip(traj_groups, advantages_by_group, strict=True):
                assert len(advantages_by_traj) == len(traj_group.trajectories), "length mismatch between trajectory rewards and computed advantages"
                advantages_by_role[group_role].extend(np.asarray(advantages_by_traj).tolist())  # for metrics calculation
                for traj, advantage in zip(traj_group.trajectories, advantages_by_traj, strict=True):
                    for step in traj.steps:
                        step.advantage = float(advantage)

    # Reduce metrics. Metric keys strip the environment prefix (carried by experiment_name),
    # emitting canonical reward/mean plus per-suffix detail (reward/ep0/mean, reward/reflect0/mean).
    final_metrics = {}

    all_rewards = []
    for rewards in rewards_by_role.values():
        all_rewards.extend(rewards)
    if all_rewards:
        final_metrics["reward/mean"] = np.mean(all_rewards)
        final_metrics["reward/std"] = np.std(all_rewards)
        final_metrics["reward/max"] = np.max(all_rewards)
        final_metrics["reward/min"] = np.min(all_rewards)

    for group_role, rewards in rewards_by_role.items():
        suffix = _extract_metric_suffix(group_role)
        if suffix is not None:
            final_metrics[f"reward/{suffix}/mean"] = np.mean(rewards)
            final_metrics[f"reward/{suffix}/std"] = np.std(rewards)
            final_metrics[f"reward/{suffix}/max"] = np.max(rewards)
            final_metrics[f"reward/{suffix}/min"] = np.min(rewards)

    if collect_advantage:
        all_advantages = []
        for advantages in advantages_by_role.values():
            all_advantages.extend(advantages)
        if all_advantages:
            final_metrics["advantage/mean"] = np.mean(all_advantages)
            final_metrics["advantage/std"] = np.std(all_advantages)
            final_metrics["advantage/max"] = np.max(all_advantages)
            final_metrics["advantage/min"] = np.min(all_advantages)
            final_metrics["advantage/fraction_zero"] = np.sum(np.abs(np.array(all_advantages)) < 1e-8) / len(all_advantages)

        for group_role, advantages in advantages_by_role.items():
            suffix = _extract_metric_suffix(group_role)
            if suffix is not None:
                final_metrics[f"advantage/{suffix}/mean"] = np.mean(advantages)
                final_metrics[f"advantage/{suffix}/std"] = np.std(advantages)
                final_metrics[f"advantage/{suffix}/max"] = np.max(advantages)
                final_metrics[f"advantage/{suffix}/min"] = np.min(advantages)
                final_metrics[f"advantage/{suffix}/fraction_zero"] = np.sum(np.abs(np.array(advantages)) < 1e-8) / len(advantages)

    return final_metrics

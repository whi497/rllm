"""Runtime adapter for kind-aware MiniSweeper rewind flows.

The implementation module contains the reward/prompt logic.  This adapter patches
its base-flow configuration to be process-stable rather than episode-scoped so
parallel rollout coroutines running the same ablation level cannot temporarily
restore the shared base module while another episode is still generating.
"""

from __future__ import annotations

from contextlib import contextmanager

try:
    from . import minisweeper_rewind_kind_aware as impl
except (ImportError, ValueError):
    import minisweeper_rewind_kind_aware as impl


@contextmanager
def _process_stable_configured_base(level: int):
    impl.base._TBMF_KIND_AWARE_LEVEL = level
    impl.base._build_observation_prompt = impl._build_observation_prompt_kind_aware
    if level >= impl.LEVEL_STRUCTURED:
        impl.base.REWIND_REFLECT_PROMPT = impl._ORIGINAL_REFLECT_PROMPT + impl.STRUCTURED_MEMORY_ADDENDUM
        impl.base.REWIND_REFLECT_PROMPT_WITH_CKPT_CHOICE = (
            impl._ORIGINAL_REFLECT_PROMPT_WITH_CKPT_CHOICE + impl.STRUCTURED_MEMORY_ADDENDUM
        )
    else:
        impl.base.REWIND_REFLECT_PROMPT = impl._ORIGINAL_REFLECT_PROMPT
        impl.base.REWIND_REFLECT_PROMPT_WITH_CKPT_CHOICE = impl._ORIGINAL_REFLECT_PROMPT_WITH_CKPT_CHOICE
    impl.base._assign_cross_segment_rewards = lambda trajectories, won, traj_gamma: None
    yield


# Patch the implementation module.  _run_kind_aware resolves _configured_base at
# call time, so all exported rollout functions below use the stable version.
impl._configured_base = _process_stable_configured_base

minisweeper_rewind_kind_force_discount_flow = impl.minisweeper_rewind_kind_force_discount_flow
minisweeper_rewind_kind_force_penalty_flow = impl.minisweeper_rewind_kind_force_penalty_flow
minisweeper_rewind_kind_model_rewind_flow = impl.minisweeper_rewind_kind_model_rewind_flow
minisweeper_rewind_kind_potential_flow = impl.minisweeper_rewind_kind_potential_flow
minisweeper_rewind_kind_opportunity_flow = impl.minisweeper_rewind_kind_opportunity_flow
minisweeper_rewind_kind_structured_flow = impl.minisweeper_rewind_kind_structured_flow

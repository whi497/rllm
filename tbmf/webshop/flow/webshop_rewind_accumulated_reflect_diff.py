"""WebShop rewind reward variant: accumulated play + reflection forward diff."""

from __future__ import annotations

import rllm
from rllm.types import AgentConfig, Episode, Task
from tbmf.flows.rewind_reward_variants import annotate_rewind_episode

try:
    from .webshop_rewind_choice_flow import webshop_rewind_choice_flow
except (ImportError, ValueError):
    from webshop_rewind_choice_flow import webshop_rewind_choice_flow


@rllm.rollout(name="webshop_rewind_accumulated_reflect_diff")
async def webshop_rewind_accumulated_reflect_diff_flow(task: Task, config: AgentConfig) -> Episode:
    episode = await webshop_rewind_choice_flow.arun(task, config)
    meta = task.metadata or {}
    traj_grouping = str(meta.get("traj_grouping", "merged"))
    # Default to the dense rubric-progress reflection reward (forward diff of the
    # mid-episode WebShop rubric potential). Set reflect_reward_source="cum_reward"
    # to fall back to the purchase-only task_score diff.
    reflect_reward_source = str(meta.get("reflect_reward_source", "rubric_progress"))
    return annotate_rewind_episode(
        episode,
        prefix="webshop",
        traj_grouping=traj_grouping,
        score_key="task_score",
        variant="accumulated_reflect_diff",
        reflect_reward_source=reflect_reward_source,
    )

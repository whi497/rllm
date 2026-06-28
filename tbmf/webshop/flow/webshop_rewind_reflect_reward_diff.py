"""WebShop rewind reward variant: reflection forward task-score difference."""

from __future__ import annotations

import rllm
from rllm.types import AgentConfig, Episode, Task
from tbmf.flows.rewind_reward_variants import annotate_rewind_episode

try:
    from .webshop_rewind_choice_flow import webshop_rewind_choice_flow
except (ImportError, ValueError):
    from webshop_rewind_choice_flow import webshop_rewind_choice_flow


@rllm.rollout(name="webshop_rewind_reflect_reward_diff")
async def webshop_rewind_reflect_reward_diff_flow(task: Task, config: AgentConfig) -> Episode:
    episode = await webshop_rewind_choice_flow.arun(task, config)
    traj_grouping = str((task.metadata or {}).get("traj_grouping", "merged"))
    return annotate_rewind_episode(
        episode,
        prefix="webshop",
        traj_grouping=traj_grouping,
        score_key="task_score",
        variant="reflect_reward_diff",
    )

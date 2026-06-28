"""ALFWorld rewind reward variant: segment novelty gate."""

from __future__ import annotations

import rllm
from rllm.types import AgentConfig, Episode, Task
from tbmf.flows.rewind_reward_variants import annotate_rewind_episode

try:
    from .alfworld_rewind_choice_flow import alfworld_rewind_choice_flow
except (ImportError, ValueError):
    from alfworld_rewind_choice_flow import alfworld_rewind_choice_flow


@rllm.rollout(name="alfworld_rewind_segment_novelty_gate")
async def alfworld_rewind_segment_novelty_gate_flow(task: Task, config: AgentConfig) -> Episode:
    episode = await alfworld_rewind_choice_flow.arun(task, config)
    traj_grouping = str((task.metadata or {}).get("traj_grouping", "merged"))
    return annotate_rewind_episode(
        episode,
        prefix="alfworld",
        traj_grouping=traj_grouping,
        score_key="won",
        variant="segment_novelty_gate",
    )

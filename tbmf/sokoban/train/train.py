"""Unified Rewind Sokoban training with selectable reward flow.

A single entry point for the sokoban rewind family. The training rollout (and the
``rewind`` validation pass) use whichever rewind flow is selected by the
``rllm.flow`` hydra parameter, instead of having a separate train module per flow.
All variants share the same training/validation harness; they differ only in how
rewards are assigned and grouped inside the flow itself.

Available flows (``+rllm.flow=<key>``):
  - ``refacted``           : reference rewind-choice flow (discounted cross-segment reward).
  - ``undiscounted_final`` : undiscounted terminal 0/1 reward to every play segment
                             and reflection.
  - ``reflect_reward_diff``: play segments get terminal 0/1; reflections get the
                             forward path-cumulative env-reward diff S_{k+1}-S_k.
  - ``accumulated_reflect_diff``: play segments get the path-cumulative env reward
                             AS-IS (S_i, no discounting/terminal collapse);
                             reflections get the forward diff S_{k+1}-S_k.
  - ``segment_novelty_gate``: like reflect_reward_diff plus a segment-level play
                             reward zeroed when the segment produced no new env state.

Trajectory grouping is an independent knob (``+rllm.traj_grouping=<mode>``), read by
the flows from task metadata:
  - ``merged`` (default): all play segments grouped together, all reflections together.
  - ``per_stage``        : segment_{idx} / reflection_{idx} grouped by stage index.
Segments and reflections are always in separate groups regardless of mode.

Action-only history is another independent knob (``+rllm.action_only=<bool>``), also
read by the flows from task metadata:
  - ``false`` (default): reflection-prompt branch history keeps the model reasoning.
  - ``true``           : branch history keeps only each step's final <action> tag,
                         dropping reasoning.

Both knobs are injected into task metadata (e.g. via
``+rllm.task_metadata_overrides.traj_grouping`` /
``+rllm.task_metadata_overrides.action_only`` in the launcher) and have no effect on
flow selection.

Usage::

    python3 -m tbmf.sokoban.train.train rllm/backend=verl \\
        +rllm.flow=reflect_reward_diff +rllm.traj_grouping=per_stage
"""

import hydra
import os
from omegaconf import DictConfig

try:
    from ..eval.sokoban_eval import sokoban_evaluator
    from ..eval.sokoban_rewind_eval import sokoban_rewind_evaluator
    from ..flow.sokoban_flow import sokoban_flow
    from ..flow.sokoban_rewind_choice_flow_refacted import sokoban_rewind_flow
    from ..flow.sokoban_rewind_undiscounted_final import sokoban_rewind_undiscounted_final_flow
    from ..flow.sokoban_rewind_reflect_reward_diff import sokoban_rewind_reflect_reward_diff_flow
    from ..flow.sokoban_rewind_accumulated_reflect_diff import sokoban_rewind_accumulated_reflect_diff_flow
    from ..flow.sokoban_rewind_segment_novelty_gate import sokoban_rewind_segment_novelty_gate_flow
except (ImportError, ValueError):
    from eval.sokoban_eval import sokoban_evaluator
    from eval.sokoban_rewind_eval import sokoban_rewind_evaluator
    from flow.sokoban_flow import sokoban_flow
    from flow.sokoban_rewind_choice_flow_refacted import sokoban_rewind_flow
    from flow.sokoban_rewind_undiscounted_final import sokoban_rewind_undiscounted_final_flow
    from flow.sokoban_rewind_reflect_reward_diff import sokoban_rewind_reflect_reward_diff_flow
    from flow.sokoban_rewind_accumulated_reflect_diff import sokoban_rewind_accumulated_reflect_diff_flow
    from flow.sokoban_rewind_segment_novelty_gate import sokoban_rewind_segment_novelty_gate_flow

try:
    from .multi_pass import MultiPassConfig, MultiPassEvaluator, MultiPassFlow, ValidationPass
    from .task_metadata_overrides import with_task_metadata_overrides
except (ImportError, ValueError):
    from multi_pass import MultiPassConfig, MultiPassEvaluator, MultiPassFlow, ValidationPass
    from task_metadata_overrides import with_task_metadata_overrides

from rllm.data.dataset import DatasetRegistry
from rllm.experimental.unified_trainer import AgentTrainer


# Flow key -> rewind flow function. Select with `+rllm.flow=<key>`.
REWIND_FLOWS = {
    "refacted": sokoban_rewind_flow,
    "undiscounted_final": sokoban_rewind_undiscounted_final_flow,
    "reflect_reward_diff": sokoban_rewind_reflect_reward_diff_flow,
    "accumulated_reflect_diff": sokoban_rewind_accumulated_reflect_diff_flow,
    "segment_novelty_gate": sokoban_rewind_segment_novelty_gate_flow,
}

DEFAULT_FLOW = "refacted"


def _resolve_flow(config: DictConfig):
    flow_key = config.get("rllm", {}).get("flow", DEFAULT_FLOW)
    if flow_key not in REWIND_FLOWS:
        raise ValueError(
            f"Unknown rllm.flow={flow_key!r}. Valid flows: {sorted(REWIND_FLOWS)}."
        )
    return REWIND_FLOWS[flow_key]


def _build_multi_pass(config: DictConfig):
    train_flow = _resolve_flow(config)

    val_cfg = config.get("rllm", {}).get("validation", {}).get("passes", {})
    single_ep_enabled = val_cfg.get("single_episode", {}).get("enabled", True)
    rewind_enabled = val_cfg.get("rewind", {}).get("enabled", True)

    mp_config = MultiPassConfig(
        train_flow=train_flow,
        train_evaluator=sokoban_evaluator,
        val_passes=[
            ValidationPass("single_episode", sokoban_flow, sokoban_evaluator, enabled=single_ep_enabled),
            ValidationPass("rewind", train_flow, sokoban_rewind_evaluator, enabled=rewind_enabled, sample_budget=1),
        ],
    )
    return MultiPassFlow(mp_config), MultiPassEvaluator(mp_config)


@hydra.main(config_path="pkg://rllm.experimental.config", config_name="unified", version_base=None)
def main(config: DictConfig):
    dataset_name = config.get("rllm", {}).get("dataset", {}).get("name") or os.environ.get("SOKOBAN_DATASET_NAME", "sokoban")
    train_dataset = DatasetRegistry.load_dataset(dataset_name, "train")
    val_dataset = DatasetRegistry.load_dataset(dataset_name, "test")

    if train_dataset is None or val_dataset is None:
        raise RuntimeError(
            f"Sokoban dataset '{dataset_name}' not found. "
            "Run: python3 tbmf/sokoban/prepare_sokoban_data.py "
            "--dataset-name sokoban_7x7_3box --dim-room 7x7 --num-boxes 3"
        )

    metadata_overrides = config.get("rllm", {}).get("task_metadata_overrides")
    train_dataset = with_task_metadata_overrides(train_dataset, metadata_overrides)
    val_dataset = with_task_metadata_overrides(val_dataset, metadata_overrides)

    flow, evaluator = _build_multi_pass(config)

    trainer = AgentTrainer(
        backend=config.rllm.get("backend", "verl"),
        agent_flow=flow,
        evaluator=evaluator,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()

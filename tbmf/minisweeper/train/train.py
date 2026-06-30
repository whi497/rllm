"""Unified Rewind MiniSweeper training with selectable reward flow.

A single entry point for the minisweeper rewind family. The training rollout
(and the ``rewind`` validation pass) use whichever rewind flow is selected by
the ``rllm.flow`` hydra parameter, instead of having a separate train module per
flow. All variants share the same training/validation harness; they differ only
in how rewards are assigned and grouped inside the flow itself.

Available flows (``+rllm.flow=<key>``):
  - ``refacted``           : reference rewind-choice flow (discounted cross-segment reward).
  - ``undiscounted_final`` : undiscounted terminal 0/1 reward to every play segment
                             and reflection.
  - ``reflect_reward_diff``: play segments get terminal 0/1; reflections get the
                             forward path-cumulative env-reward diff S_{k+1}-S_k.
  - ``discounted_reflect_diff``: play segments get the DISCOUNTED terminal reward
                             (refacted-style traj_gamma); reflections get the same
                             forward diff S_{k+1}-S_k.
  - ``accumulated_reflect_diff``: play segments get the path-cumulative env reward
                             AS-IS (segment i gets S_i, no discount / no terminal
                             collapse); reflections get the same forward diff
                             S_{k+1}-S_k.  This is the A0 baseline for the
                             kind-aware ablation sweep below.
  - ``gigpo``              : GiGPO (group-in-group) advantage on the play side.
                             Macro = S_i (as in accumulated_reflect_diff); micro =
                             per-anchor-board step return. Set
                             ``algorithm.adv_estimator=gigpo`` and pick the step
                             granularity with
                             ``+rllm.task_metadata_overrides.gigpo_granularity=segment|reveal``.
                             Reflections are untouched (auto plain-GRPO fallback).
  - ``segment_novelty_gate``: like reflect_reward_diff plus a segment-level play
                             reward zeroed when the segment produced no new env state.

Kind-aware active-rewind ablation flows:
  - ``kind_force_discount``: A1, terminal success is discounted by forced rewinds.
  - ``kind_force_penalty`` : A2, A1 + explicit forced-rewind event penalty.
  - ``kind_model_rewind``  : A3, A2 + model-rewind decision trajectory reward.
  - ``kind_potential``     : A4, A3 + segment potential / efficiency shaping.
  - ``kind_opportunity``   : A5, A4 + rewind opportunity card in the action prompt.
  - ``kind_structured``    : A6, A5 + structured branch memory + repeat penalty.

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

    python3 -m tbmf.minisweeper.train.train rllm/backend=verl \
        +rllm.flow=kind_structured +rllm.traj_grouping=per_stage
"""

import hydra
from omegaconf import DictConfig

try:
    from ..eval.minisweeper_eval import minisweeper_evaluator
    from ..eval.minisweeper_rewind_eval import minisweeper_rewind_evaluator
    from ..flow.minisweeper_flow import minisweeper_flow
    from ..flow.minisweeper_rewind_choice_flow_refacted import minisweeper_rewind_choice_flow
    from ..flow.minisweeper_rewind_undiscounted_final import minisweeper_rewind_undiscounted_final_flow
    from ..flow.minisweeper_rewind_reflect_reward_diff import minisweeper_rewind_reflect_reward_diff_flow
    from ..flow.minisweeper_rewind_discounted_reflect_diff import minisweeper_rewind_discounted_reflect_diff_flow
    from ..flow.minisweeper_rewind_accumulated_reflect_diff import minisweeper_rewind_accumulated_reflect_diff_flow
    from ..flow.minisweeper_rewind_gigpo import minisweeper_rewind_gigpo_flow
    from ..flow.minisweeper_rewind_segment_novelty_gate import minisweeper_rewind_segment_novelty_gate_flow
    from ..flow.minisweeper_rewind_kind_aware import (
        minisweeper_rewind_kind_force_discount_flow,
        minisweeper_rewind_kind_force_penalty_flow,
        minisweeper_rewind_kind_model_rewind_flow,
        minisweeper_rewind_kind_potential_flow,
        minisweeper_rewind_kind_opportunity_flow,
        minisweeper_rewind_kind_structured_flow,
    )
except (ImportError, ValueError):
    from eval.minisweeper_eval import minisweeper_evaluator
    from eval.minisweeper_rewind_eval import minisweeper_rewind_evaluator
    from flow.minisweeper_flow import minisweeper_flow
    from flow.minisweeper_rewind_choice_flow_refacted import minisweeper_rewind_choice_flow
    from flow.minisweeper_rewind_undiscounted_final import minisweeper_rewind_undiscounted_final_flow
    from flow.minisweeper_rewind_reflect_reward_diff import minisweeper_rewind_reflect_reward_diff_flow
    from flow.minisweeper_rewind_discounted_reflect_diff import minisweeper_rewind_discounted_reflect_diff_flow
    from flow.minisweeper_rewind_accumulated_reflect_diff import minisweeper_rewind_accumulated_reflect_diff_flow
    from flow.minisweeper_rewind_gigpo import minisweeper_rewind_gigpo_flow
    from flow.minisweeper_rewind_segment_novelty_gate import minisweeper_rewind_segment_novelty_gate_flow
    from flow.minisweeper_rewind_kind_aware import (
        minisweeper_rewind_kind_force_discount_flow,
        minisweeper_rewind_kind_force_penalty_flow,
        minisweeper_rewind_kind_model_rewind_flow,
        minisweeper_rewind_kind_potential_flow,
        minisweeper_rewind_kind_opportunity_flow,
        minisweeper_rewind_kind_structured_flow,
    )

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
    "refacted": minisweeper_rewind_choice_flow,
    "undiscounted_final": minisweeper_rewind_undiscounted_final_flow,
    "reflect_reward_diff": minisweeper_rewind_reflect_reward_diff_flow,
    "discounted_reflect_diff": minisweeper_rewind_discounted_reflect_diff_flow,
    "accumulated_reflect_diff": minisweeper_rewind_accumulated_reflect_diff_flow,
    "gigpo": minisweeper_rewind_gigpo_flow,
    "segment_novelty_gate": minisweeper_rewind_segment_novelty_gate_flow,
    # A1--A6 kind-aware active-rewind ablations.
    "kind_force_discount": minisweeper_rewind_kind_force_discount_flow,
    "kind_force_penalty": minisweeper_rewind_kind_force_penalty_flow,
    "kind_model_rewind": minisweeper_rewind_kind_model_rewind_flow,
    "kind_potential": minisweeper_rewind_kind_potential_flow,
    "kind_opportunity": minisweeper_rewind_kind_opportunity_flow,
    "kind_structured": minisweeper_rewind_kind_structured_flow,
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
        train_evaluator=minisweeper_rewind_evaluator,
        val_passes=[
            ValidationPass("single_episode", minisweeper_flow, minisweeper_evaluator, enabled=single_ep_enabled),
            ValidationPass("rewind", train_flow, minisweeper_rewind_evaluator, enabled=rewind_enabled, sample_budget=1),
        ],
    )
    return MultiPassFlow(mp_config), MultiPassEvaluator(mp_config)


@hydra.main(config_path="pkg://rllm.experimental.config", config_name="unified", version_base=None)
def main(config: DictConfig):
    train_dataset = DatasetRegistry.load_dataset("minisweeper", "train")
    val_dataset = DatasetRegistry.load_dataset("minisweeper", "test")

    if train_dataset is None or val_dataset is None:
        raise RuntimeError("MiniSweeper dataset not found. Run: python3 tbmf/minisweeper/prepare_minisweeper_data.py")

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

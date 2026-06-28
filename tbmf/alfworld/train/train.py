"""Unified Rewind ALFWorld training with selectable reward flow."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

try:
    from ..eval.alfworld_eval import alfworld_evaluator
    from ..eval.alfworld_rewind_eval import alfworld_rewind_evaluator
    from ..flow.alfworld_flow import alfworld_flow
    from ..flow.alfworld_rewind_choice_flow import alfworld_rewind_choice_flow
    from ..flow.alfworld_rewind_reflect_reward_diff import alfworld_rewind_reflect_reward_diff_flow
    from ..flow.alfworld_rewind_accumulated_reflect_diff import alfworld_rewind_accumulated_reflect_diff_flow
    from ..flow.alfworld_rewind_segment_novelty_gate import alfworld_rewind_segment_novelty_gate_flow
    from ..flow.alfworld_rewind_undiscounted_final import alfworld_rewind_undiscounted_final_flow
except (ImportError, ValueError):
    from eval.alfworld_eval import alfworld_evaluator
    from eval.alfworld_rewind_eval import alfworld_rewind_evaluator
    from flow.alfworld_flow import alfworld_flow
    from flow.alfworld_rewind_choice_flow import alfworld_rewind_choice_flow
    from flow.alfworld_rewind_reflect_reward_diff import alfworld_rewind_reflect_reward_diff_flow
    from flow.alfworld_rewind_accumulated_reflect_diff import alfworld_rewind_accumulated_reflect_diff_flow
    from flow.alfworld_rewind_segment_novelty_gate import alfworld_rewind_segment_novelty_gate_flow
    from flow.alfworld_rewind_undiscounted_final import alfworld_rewind_undiscounted_final_flow

try:
    from .multi_pass import MultiPassConfig, MultiPassEvaluator, MultiPassFlow, ValidationPass
    from .task_metadata_overrides import with_task_metadata_overrides
except (ImportError, ValueError):
    from multi_pass import MultiPassConfig, MultiPassEvaluator, MultiPassFlow, ValidationPass
    from task_metadata_overrides import with_task_metadata_overrides

from rllm.data.dataset import DatasetRegistry
from rllm.experimental.unified_trainer import AgentTrainer


REWIND_FLOWS = {
    "refacted": alfworld_rewind_choice_flow,
    "undiscounted_final": alfworld_rewind_undiscounted_final_flow,
    "reflect_reward_diff": alfworld_rewind_reflect_reward_diff_flow,
    "segment_novelty_gate": alfworld_rewind_segment_novelty_gate_flow,
}

DEFAULT_FLOW = "refacted"


def _resolve_flow(config: DictConfig):
    flow_key = config.get("rllm", {}).get("flow", DEFAULT_FLOW)
    if flow_key not in REWIND_FLOWS:
        raise ValueError(f"Unknown rllm.flow={flow_key!r}. Valid flows: {sorted(REWIND_FLOWS)}.")
    return REWIND_FLOWS[flow_key]


def _build_multi_pass(config: DictConfig):
    train_flow = _resolve_flow(config)
    val_cfg = config.get("rllm", {}).get("validation", {}).get("passes", {})
    single_ep_enabled = val_cfg.get("single_episode", {}).get("enabled", True)
    rewind_enabled = val_cfg.get("rewind", {}).get("enabled", True)

    mp_config = MultiPassConfig(
        train_flow=train_flow,
        train_evaluator=alfworld_rewind_evaluator,
        val_passes=[
            ValidationPass("single_episode", alfworld_flow, alfworld_evaluator, enabled=single_ep_enabled),
            ValidationPass("rewind", train_flow, alfworld_rewind_evaluator, enabled=rewind_enabled, sample_budget=1),
        ],
    )
    return MultiPassFlow(mp_config), MultiPassEvaluator(mp_config)


@hydra.main(config_path="pkg://rllm.experimental.config", config_name="unified", version_base=None)
def main(config: DictConfig):
    train_dataset = DatasetRegistry.load_dataset("alfworld", "train")
    val_dataset = DatasetRegistry.load_dataset("alfworld", "test")

    if train_dataset is None or val_dataset is None:
        raise RuntimeError("ALFWorld dataset not found. Run: python3 tbmf/alfworld/prepare_alfworld_data.py")

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

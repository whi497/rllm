"""Unified Rewind WebShop training with selectable reward flow."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

try:
    from ..eval.webshop_eval import webshop_evaluator
    from ..eval.webshop_rewind_eval import webshop_rewind_evaluator
    from ..flow.webshop_flow import webshop_flow
    from ..flow.webshop_rewind_choice_flow import webshop_rewind_choice_flow
    from ..flow.webshop_rewind_reflect_reward_diff import webshop_rewind_reflect_reward_diff_flow
    from ..flow.webshop_rewind_accumulated_reflect_diff import webshop_rewind_accumulated_reflect_diff_flow
    from ..flow.webshop_rewind_segment_novelty_gate import webshop_rewind_segment_novelty_gate_flow
    from ..flow.webshop_rewind_undiscounted_final import webshop_rewind_undiscounted_final_flow
except (ImportError, ValueError):
    from eval.webshop_eval import webshop_evaluator
    from eval.webshop_rewind_eval import webshop_rewind_evaluator
    from flow.webshop_flow import webshop_flow
    from flow.webshop_rewind_choice_flow import webshop_rewind_choice_flow
    from flow.webshop_rewind_reflect_reward_diff import webshop_rewind_reflect_reward_diff_flow
    from flow.webshop_rewind_accumulated_reflect_diff import webshop_rewind_accumulated_reflect_diff_flow
    from flow.webshop_rewind_segment_novelty_gate import webshop_rewind_segment_novelty_gate_flow
    from flow.webshop_rewind_undiscounted_final import webshop_rewind_undiscounted_final_flow

try:
    from .multi_pass import MultiPassConfig, MultiPassEvaluator, MultiPassFlow, ValidationPass
    from .task_metadata_overrides import with_task_metadata_overrides
except (ImportError, ValueError):
    from multi_pass import MultiPassConfig, MultiPassEvaluator, MultiPassFlow, ValidationPass
    from task_metadata_overrides import with_task_metadata_overrides

from rllm.data.dataset import DatasetRegistry
from rllm.experimental.unified_trainer import AgentTrainer


REWIND_FLOWS = {
    "refacted": webshop_rewind_choice_flow,
    "undiscounted_final": webshop_rewind_undiscounted_final_flow,
    "reflect_reward_diff": webshop_rewind_reflect_reward_diff_flow,
    "accumulated_reflect_diff": webshop_rewind_accumulated_reflect_diff_flow,
    "segment_novelty_gate": webshop_rewind_segment_novelty_gate_flow,
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
        train_evaluator=webshop_rewind_evaluator,
        val_passes=[
            ValidationPass("single_episode", webshop_flow, webshop_evaluator, enabled=single_ep_enabled),
            ValidationPass("rewind", train_flow, webshop_rewind_evaluator, enabled=rewind_enabled, sample_budget=1),
        ],
    )
    return MultiPassFlow(mp_config), MultiPassEvaluator(mp_config)


def _shutdown_webshop_pool():
    try:
        try:
            from ..webshop_env import WebShopEnv
        except (ImportError, ValueError):
            from webshop_env import WebShopEnv
        WebShopEnv.shutdown_pool()
    except Exception:
        pass


@hydra.main(config_path="pkg://rllm.experimental.config", config_name="unified", version_base=None)
def main(config: DictConfig):
    train_dataset = DatasetRegistry.load_dataset("webshop", "train")
    val_dataset = DatasetRegistry.load_dataset("webshop", "test")

    if train_dataset is None or val_dataset is None:
        raise RuntimeError("WebShop dataset not found. Run: python3 tbmf/webshop/prepare_webshop_data.py")

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
    try:
        trainer.train()
    finally:
        _shutdown_webshop_pool()


if __name__ == "__main__":
    main()

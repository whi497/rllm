"""Rewind Sokoban training with multi-pass validation.

Train: rewind-capable rollouts (model decides when to checkpoint/rewind).
Val pass 1 (single_episode): standard GRPO flow -> pass@1/pass@4.
Val pass 2 (rewind): rewind flow -> measures exploration efficiency.

Usage::

    python3 -m tbmf.sokoban.train.train_rewind rllm/backend=verl
"""

import hydra
from omegaconf import DictConfig

try:
    from ..eval.sokoban_eval import sokoban_evaluator
    from ..flow.sokoban_flow import sokoban_flow
    from ..flow.sokoban_rewind_choice_flow import sokoban_rewind_flow
except (ImportError, ValueError):
    from eval.sokoban_eval import sokoban_evaluator
    from flow.sokoban_flow import sokoban_flow
    from flow.sokoban_rewind_choice_flow import sokoban_rewind_flow

try:
    from .multi_pass import MultiPassConfig, MultiPassEvaluator, MultiPassFlow, ValidationPass
except (ImportError, ValueError):
    from multi_pass import MultiPassConfig, MultiPassEvaluator, MultiPassFlow, ValidationPass

from rllm.data.dataset import DatasetRegistry
from rllm.experimental.unified_trainer import AgentTrainer


def _build_multi_pass(config: DictConfig):
    val_cfg = config.get("rllm", {}).get("validation", {}).get("passes", {})
    single_ep_enabled = val_cfg.get("single_episode", {}).get("enabled", True)
    rewind_enabled = val_cfg.get("rewind", {}).get("enabled", True)

    mp_config = MultiPassConfig(
        train_flow=sokoban_rewind_flow,
        train_evaluator=sokoban_evaluator,
        val_passes=[
            ValidationPass("single_episode", sokoban_flow, sokoban_evaluator, enabled=single_ep_enabled),
            ValidationPass("rewind", sokoban_rewind_flow, sokoban_evaluator, enabled=rewind_enabled, sample_budget=1),
        ],
    )
    return MultiPassFlow(mp_config), MultiPassEvaluator(mp_config)


@hydra.main(config_path="pkg://rllm.experimental.config", config_name="unified", version_base=None)
def main(config: DictConfig):
    train_dataset = DatasetRegistry.load_dataset("sokoban", "train")
    val_dataset = DatasetRegistry.load_dataset("sokoban", "test")

    if train_dataset is None or val_dataset is None:
        raise RuntimeError("Sokoban dataset not found. Run: python3 tbmf/sokoban/prepare_sokoban_data.py")

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

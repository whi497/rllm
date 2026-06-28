"""Rewind MiniSweeper training using the refacted rewind-choice flow.

Trains/validates the *refacted* rewind-choice flow
(``minisweeper_rewind_choice_flow_refacted``), which fixes the
reflection-stacking bug: a re-rewind to the same checkpoint keeps exactly one
reflection on that prefix and folds the superseded reflection into the next
reflection's branch history.

Train: refacted rewind-capable rollouts (model decides when to rewind).
Val pass 1 (single_episode): standard GRPO flow -> pass@1/pass@4.
Val pass 2 (rewind): refacted rewind-choice flow -> measures exploration efficiency.

Usage::

    python3 -m tbmf.minisweeper.train.train_rewind_refacted rllm/backend=verl
"""

import hydra
from omegaconf import DictConfig

try:
    from ..eval.minisweeper_eval import minisweeper_evaluator
    from ..eval.minisweeper_rewind_eval import minisweeper_rewind_evaluator
    from ..flow.minisweeper_flow import minisweeper_flow
    from ..flow.minisweeper_rewind_choice_flow_refacted import minisweeper_rewind_choice_flow
except (ImportError, ValueError):
    from eval.minisweeper_eval import minisweeper_evaluator
    from eval.minisweeper_rewind_eval import minisweeper_rewind_evaluator
    from flow.minisweeper_flow import minisweeper_flow
    from flow.minisweeper_rewind_choice_flow_refacted import minisweeper_rewind_choice_flow

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
        train_flow=minisweeper_rewind_choice_flow,
        train_evaluator=minisweeper_rewind_evaluator,
        val_passes=[
            ValidationPass("single_episode", minisweeper_flow, minisweeper_evaluator, enabled=single_ep_enabled),
            # Use the rewind-specific evaluator so rewind/turns, rewind/env_steps,
            # rewind/rewinds, rewind/segments read the rewind flow's artifact names
            # (total_play_turns / total_env_steps / rewinds / segments) instead of
            # the single-episode evaluator's turns/env_steps (absent in rewind
            # artifacts and silently reported as 0).
            ValidationPass("rewind", minisweeper_rewind_choice_flow, minisweeper_rewind_evaluator, enabled=rewind_enabled, sample_budget=1),
        ],
    )
    return MultiPassFlow(mp_config), MultiPassEvaluator(mp_config)


@hydra.main(config_path="pkg://rllm.experimental.config", config_name="unified", version_base=None)
def main(config: DictConfig):
    train_dataset = DatasetRegistry.load_dataset("minisweeper", "train")
    val_dataset = DatasetRegistry.load_dataset("minisweeper", "test")

    if train_dataset is None or val_dataset is None:
        raise RuntimeError("MiniSweeper dataset not found. Run: python3 tbmf/minisweeper/prepare_minisweeper_data.py")

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

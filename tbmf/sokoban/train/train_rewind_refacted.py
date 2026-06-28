"""Rewind Sokoban training using the refacted rewind-choice flow.

Identical wiring to ``train_rewind`` but trains/validates the *refacted*
rewind-choice flow (``sokoban_rewind_choice_flow_refacted``), which fixes the
reflection-stacking bug: a re-rewind to the same checkpoint keeps exactly one
reflection on that prefix and folds the superseded reflection into the next
reflection's branch history.

Train: refacted rewind-capable rollouts (model decides when to checkpoint/rewind).
Val pass 1 (single_episode): standard GRPO flow -> pass@1/pass@4.
Val pass 2 (rewind): refacted rewind flow -> measures exploration efficiency.

Usage::

    python3 -m tbmf.sokoban.train.train_rewind_refacted rllm/backend=verl
"""

import hydra
import os
from omegaconf import DictConfig

try:
    from ..eval.sokoban_eval import sokoban_evaluator
    from ..eval.sokoban_rewind_eval import sokoban_rewind_evaluator
    from ..flow.sokoban_flow import sokoban_flow
    from ..flow.sokoban_rewind_choice_flow_refacted import sokoban_rewind_flow
except (ImportError, ValueError):
    from eval.sokoban_eval import sokoban_evaluator
    from eval.sokoban_rewind_eval import sokoban_rewind_evaluator
    from flow.sokoban_flow import sokoban_flow
    from flow.sokoban_rewind_choice_flow_refacted import sokoban_rewind_flow

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
            # Use the rewind-specific evaluator so rewind/turns, rewind/env_steps,
            # rewind/rewinds, rewind/segments read the rewind flow's artifact names
            # (total_play_turns / total_env_steps / rewinds / segments) instead of
            # the single-episode evaluator's turns/env_steps (which are absent in
            # rewind artifacts and were silently reported as 0).
            ValidationPass("rewind", sokoban_rewind_flow, sokoban_rewind_evaluator, enabled=rewind_enabled, sample_budget=1),
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

"""GRPO Sokoban training with multi-pass validation.

Train: single-episode GRPO rollouts.
Val pass 1 (single_episode): standard GRPO flow -> pass@1/pass@4.
Val pass 2 (multi_episode): LaMer 3-episode flow -> success_at1/2/3 (runs once per task).

Usage::

    python3 -m tbmf.sokoban.train.train_grpo rllm/backend=verl
"""

import os

import hydra
from omegaconf import DictConfig, OmegaConf

try:
    from ..eval.sokoban_eval import sokoban_evaluator
    from ..eval.sokoban_lamer_eval import sokoban_lamer_evaluator
    from ..flow.sokoban_flow import sokoban_flow
    from ..flow.sokoban_lamer_flow import sokoban_lamer_flow
except (ImportError, ValueError):
    from eval.sokoban_eval import sokoban_evaluator
    from eval.sokoban_lamer_eval import sokoban_lamer_evaluator
    from flow.sokoban_flow import sokoban_flow
    from flow.sokoban_lamer_flow import sokoban_lamer_flow

try:
    from .multi_pass import MultiPassConfig, MultiPassEvaluator, MultiPassFlow, ValidationPass
except (ImportError, ValueError):
    from multi_pass import MultiPassConfig, MultiPassEvaluator, MultiPassFlow, ValidationPass

from rllm.data.dataset import Dataset, DatasetRegistry
from rllm.experimental.unified_trainer import AgentTrainer


def _with_task_metadata_overrides(dataset: Dataset, overrides: DictConfig | None) -> Dataset:
    if not overrides:
        return dataset

    override_dict = OmegaConf.to_container(overrides, resolve=True)
    if not isinstance(override_dict, dict) or not override_dict:
        return dataset

    rows = []
    for row in dataset.get_data():
        updated = dict(row)
        updated.update(override_dict)
        rows.append(updated)
    return Dataset(rows, name=dataset.name, split=dataset.split)


def _build_multi_pass(config: DictConfig):
    val_cfg = config.get("rllm", {}).get("validation", {}).get("passes", {})
    single_ep_enabled = val_cfg.get("single_episode", {}).get("enabled", True)
    multi_ep_enabled = val_cfg.get("multi_episode", {}).get("enabled", True)

    mp_config = MultiPassConfig(
        train_flow=sokoban_flow,
        train_evaluator=sokoban_evaluator,
        val_passes=[
            ValidationPass("single_episode", sokoban_flow, sokoban_evaluator, enabled=single_ep_enabled),
            # ValidationPass("multi_episode", sokoban_lamer_flow, sokoban_lamer_evaluator, enabled=multi_ep_enabled, sample_budget=1),
        ],
    )
    return MultiPassFlow(mp_config), MultiPassEvaluator(mp_config)


@hydra.main(config_path="pkg://rllm.experimental.config", config_name="unified", version_base=None)
def main(config: DictConfig):
    # Select the dataset the same way as the rewind training entry, so this GRPO
    # baseline can train/validate on the exact same dataset (e.g. sokoban_7x7_3box).
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
    train_dataset = _with_task_metadata_overrides(train_dataset, metadata_overrides)
    val_dataset = _with_task_metadata_overrides(val_dataset, metadata_overrides)

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

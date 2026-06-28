"""Helpers for applying task metadata overrides in MiniSweeper trainers."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from rllm.data.dataset import Dataset, DatasetRegistry


class _DatasetWithVerlPath(Dataset):
    def __init__(self, data: list[dict[str, Any]], *, name: str | None, split: str | None, verl_data_path: str):
        super().__init__(data=data, name=name, split=split)
        self._verl_data_path = verl_data_path

    def get_verl_data_path(self) -> str:
        return self._verl_data_path


def _safe_path_part(value: str | None, fallback: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value or fallback).strip("_") or fallback


def _merge_override(row: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    updated = dict(row)
    extra_info = updated.get("extra_info")
    if isinstance(extra_info, dict):
        merged_extra = dict(extra_info)
        merged_extra.update(overrides)
        updated["extra_info"] = merged_extra
    else:
        updated.update(overrides)
    return updated


def _override_verl_path(dataset: Dataset, overrides: dict[str, Any]) -> Path:
    original_verl_path = dataset.get_verl_data_path()
    if original_verl_path:
        output_dir = Path(original_verl_path).parent / "_metadata_overrides"
    else:
        output_dir = Path(os.environ.get("RLLM_HOME", "~/.rllm")).expanduser() / "datasets" / "_metadata_overrides"
    output_dir.mkdir(parents=True, exist_ok=True)

    digest_input = {
        "dataset": dataset.name,
        "split": dataset.split,
        "len": len(dataset),
        "overrides": overrides,
    }
    digest = hashlib.sha1(json.dumps(digest_input, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]
    name = _safe_path_part(dataset.name, "dataset")
    split = _safe_path_part(dataset.split, "split")
    return output_dir / f"{name}_{split}_{digest}_verl.parquet"


def with_task_metadata_overrides(dataset: Dataset, overrides: DictConfig | None) -> Dataset:
    if not overrides:
        return dataset

    override_dict = OmegaConf.to_container(overrides, resolve=True)
    if not isinstance(override_dict, dict) or not override_dict:
        return dataset

    rows = [_merge_override(dict(row), override_dict) for row in dataset.get_data()]
    verl_path = _override_verl_path(dataset, override_dict)

    import pandas as pd

    if rows and {"prompt", "reward_model", "extra_info"}.issubset(rows[0]):
        verl_rows = rows
    else:
        verl_rows = DatasetRegistry.apply_verl_postprocessing(rows)
    pd.DataFrame(verl_rows).to_parquet(verl_path)

    print(
        "Applied task metadata overrides "
        f"{override_dict} to {dataset.name or 'dataset'}:{dataset.split or 'split'}; "
        f"Verl data path: {verl_path}"
    )
    return _DatasetWithVerlPath(rows, name=dataset.name, split=dataset.split, verl_data_path=str(verl_path))

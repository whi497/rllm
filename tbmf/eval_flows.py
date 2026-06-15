"""Run TBMF flow evaluations and export trajectories.

This script evaluates normal, LaMer, and rewind-choice flows for Sokoban and
MiniSweeper against an OpenAI-compatible endpoint. It writes:
  - summary.json: aggregate metrics for each flow
  - episodes/<run_name>/episode_*.json: full rLLM Episode dumps
  - trajectories.jsonl: one compact record per evaluated task
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from rllm.eval.runner import run_dataset
from rllm.types import Episode, Task

from tbmf.minisweeper.eval.minisweeper_eval import minisweeper_evaluator
from tbmf.minisweeper.eval.minisweeper_lamer_eval import minisweeper_lamer_evaluator
from tbmf.minisweeper.eval.minisweeper_rewind_eval import minisweeper_rewind_evaluator
from tbmf.minisweeper.flow.minisweeper_flow import minisweeper_flow
from tbmf.minisweeper.flow.minisweeper_lamer_flow import minisweeper_lamer_flow
from tbmf.minisweeper.flow.minisweeper_rewind_choice_flow import minisweeper_rewind_choice_flow
from tbmf.minisweeper.prepare_minisweeper_data import prepare_minisweeper_data
from tbmf.sokoban.eval.sokoban_eval import sokoban_evaluator
from tbmf.sokoban.eval.sokoban_lamer_eval import sokoban_lamer_evaluator
from tbmf.sokoban.eval.sokoban_rewind_eval import sokoban_rewind_evaluator
from tbmf.sokoban.flow.sokoban_flow import sokoban_flow
from tbmf.sokoban.flow.sokoban_lamer_flow import sokoban_lamer_flow
from tbmf.sokoban.flow.sokoban_rewind_choice_flow import sokoban_rewind_flow as sokoban_rewind_choice_flow
from tbmf.sokoban.prepare_sokoban_data import prepare_sokoban_data


@dataclass(frozen=True)
class EvalSpec:
    env: str
    variant: str
    flow: Any
    evaluator: Any

    @property
    def run_name(self) -> str:
        return f"{self.env}_{self.variant}"


def _json_default(obj: Any) -> Any:
    import dataclasses

    import numpy as np

    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    return str(obj)


def _task_from_row(row: dict[str, Any], idx: int) -> Task:
    return Task(
        id=str(row.get("uid") or row.get("id") or idx),
        instruction=str(row.get("question") or row.get("instruction") or ""),
        metadata=dict(row),
        dataset_dir=Path("."),
        sub_dir=None,
    )


def _prepare_tasks(args: argparse.Namespace) -> dict[str, list[Task]]:
    _, sokoban_test = prepare_sokoban_data(
        dataset_name="sokoban_eval_flows",
        train_size=0,
        test_size=args.max_examples,
        env_seed=args.sokoban_env_seed,
    )
    _, minesweeper_test = prepare_minisweeper_data(
        train_size=0,
        test_size=args.max_examples,
        env_seed=args.minesweeper_env_seed,
    )
    return {
        "sokoban": [_task_from_row(row, idx) for idx, row in enumerate(sokoban_test.get_data())],
        "minisweeper": [_task_from_row(row, idx) for idx, row in enumerate(minesweeper_test.get_data())],
    }


def _episode_summary(spec: EvalSpec, task: Task, episode: Episode, idx: int) -> dict[str, Any]:
    artifacts = episode.artifacts or {}
    return {
        "idx": idx,
        "task_id": task.id,
        "env": spec.env,
        "variant": spec.variant,
        "is_correct": bool(episode.is_correct),
        "won": bool(artifacts.get("won", episode.is_correct)),
        "reward": episode.trajectories[0].reward if episode.trajectories else None,
        "num_trajectories": len(episode.trajectories),
        "trajectory_names": [traj.name for traj in episode.trajectories],
        "trajectory_rewards": [traj.reward for traj in episode.trajectories],
        "num_steps": sum(len(traj.steps) for traj in episode.trajectories),
        "artifacts": artifacts,
        "metrics": episode.metrics,
    }


def _episode_summary_from_json(spec: EvalSpec, task: Task, raw: dict[str, Any], idx: int) -> dict[str, Any]:
    artifacts = raw.get("artifacts") or raw.get("info", {}).get("artifacts") or {}
    trajectories = raw.get("trajectories") or []
    return {
        "idx": idx,
        "task_id": task.id,
        "env": spec.env,
        "variant": spec.variant,
        "is_correct": bool(raw.get("is_correct", False)),
        "won": bool(artifacts.get("won", raw.get("is_correct", False))),
        "reward": trajectories[0].get("reward") if trajectories else None,
        "num_trajectories": len(trajectories),
        "trajectory_names": [traj.get("name") for traj in trajectories],
        "trajectory_rewards": [traj.get("reward") for traj in trajectories],
        "num_steps": sum(len(traj.get("steps") or []) for traj in trajectories),
        "artifacts": artifacts,
        "metrics": raw.get("metrics") or {},
    }


def _write_episode(path: Path, episode: Episode, idx: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = episode.model_dump(mode="json")
    data["eval_idx"] = idx
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)


async def _run_spec(
    spec: EvalSpec,
    tasks: list[Task],
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, Any]:
    episode_dir = out_dir / "episodes" / spec.run_name
    episode_records: list[dict[str, Any]] = []

    def on_episode_complete(idx: int, episode: Episode) -> None:
        _write_episode(episode_dir / f"episode_{idx:06d}_{tasks[idx].id}.json", episode, idx)

    result, episodes = await run_dataset(
        tasks=tasks,
        agent_flow=spec.flow,
        base_url=args.base_url,
        model=args.model,
        concurrency=args.concurrency,
        agent_name=spec.run_name,
        dataset_name=spec.env,
        on_episode_complete=on_episode_complete,
        evaluator_override=spec.evaluator,
    )

    # run_dataset drops errored episodes from the returned list, so derive compact
    # records from the saved successful episodes and EvalResult items separately.
    saved_paths = sorted(episode_dir.glob("episode_*.json"))
    for path in saved_paths:
        with path.open(encoding="utf-8") as f:
            raw = json.load(f)
        idx = int(raw.get("eval_idx", len(episode_records)))
        episode_records.append(_episode_summary_from_json(spec, tasks[idx], raw, idx))

    aggregate = {
        "env": spec.env,
        "variant": spec.variant,
        "total": result.total,
        "correct": result.correct,
        "errors": result.errors,
        "score": result.score,
        "signal_averages": result.signal_averages,
        "result_items": [dataclasses.asdict(item) for item in result.items],
        "episodes_saved": len(saved_paths),
    }
    return {"aggregate": aggregate, "episodes": episode_records}


async def _main_async(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir).expanduser()
    if args.timestamp:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir = out_dir / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks_by_env = _prepare_tasks(args)
    specs = [
        EvalSpec("sokoban", "normal", sokoban_flow, sokoban_evaluator),
        EvalSpec("sokoban", "lamer", sokoban_lamer_flow, sokoban_lamer_evaluator),
        EvalSpec("sokoban", "rewind_choice", sokoban_rewind_choice_flow, sokoban_rewind_evaluator),
        EvalSpec("minisweeper", "normal", minisweeper_flow, minisweeper_evaluator),
        EvalSpec("minisweeper", "lamer", minisweeper_lamer_flow, minisweeper_lamer_evaluator),
        EvalSpec("minisweeper", "rewind_choice", minisweeper_rewind_choice_flow, minisweeper_rewind_evaluator),
    ]
    if args.only:
        allowed = set(args.only)
        specs = [spec for spec in specs if spec.run_name in allowed]

    all_summary: dict[str, Any] = {
        "base_url": args.base_url,
        "model": args.model,
        "max_examples": args.max_examples,
        "concurrency": args.concurrency,
        "runs": {},
    }
    jsonl_path = out_dir / "trajectories.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as jsonl:
        for spec in specs:
            print(f"Running {spec.run_name} on {len(tasks_by_env[spec.env])} tasks...")
            run_data = await _run_spec(spec, tasks_by_env[spec.env], args, out_dir)
            all_summary["runs"][spec.run_name] = run_data["aggregate"]
            for record in run_data["episodes"]:
                jsonl.write(json.dumps(record, default=_json_default, ensure_ascii=False) + "\n")
            jsonl.flush()

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(all_summary, f, indent=2, default=_json_default)
    print(f"Wrote {out_dir / 'summary.json'}")
    print(f"Wrote {jsonl_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TBMF normal, LaMer, and rewind-choice flows")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-examples", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--output-dir", default="eval_outputs/tbmf_flows")
    parser.add_argument("--timestamp", action="store_true")
    parser.add_argument("--sokoban-env-seed", type=int, default=4608)
    parser.add_argument("--minesweeper-env-seed", type=int, default=0)
    parser.add_argument(
        "--only",
        nargs="*",
        help="Optional run names, e.g. sokoban_normal minisweeper_rewind_choice",
    )
    return parser.parse_args()


def main() -> None:
    asyncio.run(_main_async(parse_args()))


if __name__ == "__main__":
    main()

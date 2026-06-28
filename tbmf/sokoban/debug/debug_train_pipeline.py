#!/usr/bin/env python3
"""Pausable, CPU-only walkthrough of the Sokoban *training* pipeline.

The sibling ``debug_rewind_choice_flow.py`` pauses the ROLLOUT (env <-> LLM).
This script picks up where that leaves off and pauses the TRAINING transforms
that the rollout feeds into:

  Stage 1  rollout            run N sibling episodes of one task (the GRPO group)
  Stage 2  segment rewards    _assign_cross_segment_rewards (the real flow fn)
  Stage 3  token stitching    enrich_episode_with_traces (the real engine fn)
  Stage 4  GRPO advantage     A = (R - mean) / std within the group, + rejection

Stages 2 and 3 call the *real* rLLM functions. Stage 4 re-implements the GRPO
advantage math that normally lives inside verl's ``compute_advantage`` so you
can read it line-by-line on CPU without Ray/GPU. The formula matches verl's
GRPO outcome-advantage: per group, normalize trajectory rewards by mean (and
optionally std), then broadcast the scalar across that trajectory's tokens.

It needs a running OpenAI-compatible model server for the rollout (same as the
flow debugger) and a local HF tokenizer to turn each step's chat_completions
into the prompt/response token ids that the gateway would have captured.

Usage::

    RLLM_HOME=$PWD python3 rllm/tbmf/sokoban/debug/debug_train_pipeline.py \
        --base-url http://localhost:8000/v1 \
        --model Qwen/Qwen3-4B \
        --tokenizer Qwen/Qwen3-4B \
        --group-size 4 --task-index 0 \
        --step-budget 12 --segment-max-turns 4

Set ``--auto`` or ``DEBUG_NO_PAUSE=1`` to stream without pausing.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import textwrap
from typing import Any

# --- Make the package importable whether run as a module or a script ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SOKOBAN_DIR = os.path.dirname(_THIS_DIR)
_FLOW_DIR = os.path.join(_SOKOBAN_DIR, "flow")
_TBMF_DIR = os.path.dirname(_SOKOBAN_DIR)
_RLLM_ROOT = os.path.dirname(_TBMF_DIR)
_WORKSPACE_ROOT = os.path.dirname(_RLLM_ROOT)
if "RLLM_HOME" not in os.environ and os.path.exists(
    os.path.join(_WORKSPACE_ROOT, "datasets", "registry.json")
):
    os.environ["RLLM_HOME"] = _WORKSPACE_ROOT
for p in (_SOKOBAN_DIR, _FLOW_DIR, _RLLM_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from rllm.data.dataset import DatasetRegistry
from rllm.types import AgentConfig, Task

# Real training-side functions we are here to study.
from rllm.engine.agentflow_engine import enrich_episode_with_traces
from rllm_model_gateway.models import TraceRecord

# The flow module owns _assign_cross_segment_rewards and the rollout.
from flow import sokoban_rewind_choice_flow as flow_mod


# ----------------------------------------------------------------------------
# Pretty printing (mirrors the flow debugger's palette)
# ----------------------------------------------------------------------------

_RESET, _BOLD, _DIM = "\033[0m", "\033[1m", "\033[2m"
_CYAN, _GREEN, _YELLOW = "\033[36m", "\033[32m", "\033[33m"
_RED, _MAGENTA, _BLUE = "\033[31m", "\033[35m", "\033[34m"


def _c(text: str, color: str) -> str:
    return text if os.environ.get("NO_COLOR") else f"{color}{text}{_RESET}"


def _banner(title: str, color: str = _CYAN) -> None:
    print()
    print(_c("=" * 88, color))
    print(_c(f" {title}", color + _BOLD))
    print(_c("=" * 88, color))


def _wait(prompt: str = "Press <Enter> for the next stage (q + Enter to quit)... ") -> None:
    if os.environ.get("DEBUG_NO_PAUSE"):
        print(_c("[auto-continue]", _DIM))
        return
    try:
        ans = input(_c("\n>>> " + prompt, _MAGENTA + _BOLD))
    except EOFError:
        print(_c("\n[stdin closed -> continuing]", _DIM))
        return
    if ans.strip().lower() in {"q", "quit", "exit"}:
        raise KeyboardInterrupt


# ----------------------------------------------------------------------------
# Build real TraceRecords from a finished episode using a local tokenizer.
#
# In real training, the gateway captures one TraceRecord per LLM call straight
# from vLLM (prompt_token_ids, completion_token_ids, logprobs). Here there is no
# gateway, so we reconstruct equivalent TraceRecords by tokenizing each step's
# chat_completions exactly the way rllm.engine.rollout.openai_engine does:
#   prompt  = chat_parser.parse(messages_without_last, add_generation_prompt=True)
#   response= the final assistant message content
# Then we hand them to the REAL enrich_episode_with_traces so the positional
# matching + Step backfill you studied runs unmodified.
# ----------------------------------------------------------------------------

def _make_tracer(tokenizer, chat_parser):
    def _trace_for_step(step, session_id: str, idx: int) -> TraceRecord:
        messages = list(step.chat_completions or [])
        if messages and messages[-1].get("role") == "assistant":
            prompt_messages = messages[:-1]
            response_message = dict(messages[-1])
        else:
            prompt_messages = messages
            response_message = {"role": "assistant", "content": step.model_response or ""}

        # Render the prompt prefix the same way the rollout engine does.
        try:
            prompt_text = chat_parser.parse(
                prompt_messages, add_generation_prompt=True, is_first_msg=True
            )
        except Exception:
            prompt_text = "\n".join(m.get("content", "") for m in prompt_messages)
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

        completion_text = response_message.get("content", "") or ""
        completion_ids = tokenizer.encode(completion_text, add_special_tokens=False)

        return TraceRecord(
            trace_id=f"{session_id}:trace{idx}",
            session_id=session_id,
            messages=prompt_messages,
            prompt_token_ids=prompt_ids,
            response_message=response_message,
            completion_token_ids=completion_ids,
            logprobs=[0.0] * len(completion_ids),  # placeholder; real run uses vLLM logprobs
            finish_reason="stop",
        )

    return _trace_for_step


def _episode_traces(episode, tracer, session_id: str) -> list[TraceRecord]:
    """One trace per Step, in chronological (trajectory, step) order.

    This is the exact order enrich_episode_with_traces expects: it walks
    trajectories in order and consumes traces positionally.
    """
    traces: list[TraceRecord] = []
    idx = 0
    for traj in episode.trajectories:
        for step in traj.steps:
            traces.append(tracer(step, session_id, idx))
            idx += 1
    return traces


# ----------------------------------------------------------------------------
# Stage helpers
# ----------------------------------------------------------------------------

def _sample_task(dataset_name: str, split: str, index: int, overrides: dict) -> Task:
    ds = DatasetRegistry.load_dataset(dataset_name, split)
    if ds is None:
        raise RuntimeError(
            f"Dataset '{dataset_name}/{split}' not found. Prepare it first with "
            "prepare_sokoban_data.py."
        )
    rows = ds.get_data()
    if not rows:
        raise RuntimeError(f"'{split}' dataset is empty.")
    row = rows[index % len(rows)]
    task = Task(
        id=str(row.get("uid", f"{split}_{index}")),
        instruction=str(row.get("question", "Sokoban puzzle")),
        metadata=dict(row),
        dataset_dir="",
    )
    for k, v in overrides.items():
        if v is not None:
            task.metadata[k] = v
    return task


def _print_segment_rewards(episode, group_idx: int) -> None:
    _banner(f"STAGE 2  segment rewards (sibling #{group_idx})  won={episode.artifacts.get('won')}", _YELLOW)
    print(_c(
        "  _assign_cross_segment_rewards already ran inside the flow. Only the LAST\n"
        "  play segment scores 1.0 on a win; earlier segments get gamma-discounted\n"
        "  credit; each reflection inherits the reward of the segment it set up.\n",
        _DIM,
    ))
    for traj in episode.trajectories:
        kind = "PLAY    " if traj.name.startswith("sokoban_seg") else "REFLECT "
        print(f"    {kind} {traj.name:<20} steps={len(traj.steps):<3} reward={traj.reward}")


def _print_stitching(enriched, group_idx: int) -> None:
    _banner(f"STAGE 3  token stitching via enrich_episode_with_traces (sibling #{group_idx})", _GREEN)
    print(_c(
        "  Each Step now carries prompt_ids / response_ids / logprobs pulled from the\n"
        "  (reconstructed) traces and backfilled in Step.model_post_init. response_ids\n"
        "  are the ONLY tokens the PPO loss will train on.\n",
        _DIM,
    ))
    for traj in enriched.trajectories:
        print(_c(f"    {traj.name}  (reward={traj.reward})", _GREEN + _BOLD))
        for i, step in enumerate(traj.steps):
            print(
                f"      step{i}: action={str(step.action)[:28]:<28} "
                f"|prompt_ids|={len(step.prompt_ids):<5} "
                f"|response_ids|={len(step.response_ids):<5} "
                f"|logprobs|={len(step.logprobs)}"
            )


def _grpo_advantages(group_rewards: list[float], norm_by_std: bool) -> list[float]:
    """Outcome-level GRPO advantage, matching verl's grpo estimator.

    A_i = (R_i - mean(R)) [/ std(R)]   over the sibling group sharing a task.
    The scalar is later broadcast across every response token of trajectory i.
    """
    if not group_rewards:
        return []
    mean = statistics.fmean(group_rewards)
    if norm_by_std and len(group_rewards) > 1:
        std = statistics.pstdev(group_rewards)
        denom = std if std > 1e-8 else 1.0
        return [(r - mean) / denom for r in group_rewards]
    return [r - mean for r in group_rewards]


def _print_grpo(group_episodes: list[Any], norm_by_std: bool) -> None:
    _banner("STAGE 4  GRPO advantage across the sibling group + rejection sampling", _CYAN)

    # Episode-level reward used for grouping/rejection == 1.0 on win else 0.0,
    # matching sokoban_evaluator and the trainer's is_correct keying.
    episode_rewards = [1.0 if ep.artifacts.get("won") else 0.0 for ep in group_episodes]
    print("  Episode-level rewards (win=1, lose=0):", episode_rewards)

    all_win = all(r > 0 for r in episode_rewards)
    all_lose = all(r == 0 for r in episode_rewards)
    if all_win or all_lose:
        label = "solve_all" if all_win else "solve_none"
        print(_c(
            f"  -> {label}: zero reward variance in this group. Rejection sampling\n"
            f"     (agent_workflow_trainer.py:235-242) DROPS it — no usable gradient.",
            _RED + _BOLD,
        ))
    else:
        print(_c("  -> solve_partial: group is KEPT for training.", _GREEN + _BOLD))

    advs = _grpo_advantages(episode_rewards, norm_by_std=norm_by_std)
    print(_c(f"\n  Episode-level GRPO advantages (norm_by_std={norm_by_std}):", _BOLD))
    for i, (r, a) in enumerate(zip(episode_rewards, advs, strict=True)):
        print(f"    sibling #{i}: R={r}  A={a:+.4f}")

    print(_c(
        "\n  In real training the advantage is computed per-TRAJECTORY (segments +\n"
        "  reflections each carry their shaped reward from Stage 2), then each scalar\n"
        "  is broadcast across that trajectory's response tokens and masked by\n"
        "  response_mask in the PPO loss. Below is the per-trajectory view:",
        _DIM,
    ))
    # Per-trajectory GRPO: group trajectories by name across siblings, since
    # GRPO compares like-for-like positions. (Simplified: we group by traj.name.)
    by_name: dict[str, list[tuple[int, float]]] = {}
    for si, ep in enumerate(group_episodes):
        for traj in ep.trajectories:
            by_name.setdefault(traj.name, []).append((si, float(traj.reward or 0.0)))
    for name, items in sorted(by_name.items()):
        rewards = [r for _, r in items]
        t_advs = _grpo_advantages(rewards, norm_by_std=norm_by_std)
        cells = "  ".join(
            f"#{si}:R={r:+.3f}/A={a:+.3f}"
            for (si, r), a in zip(items, t_advs, strict=True)
        )
        print(f"    {name:<20} {cells}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

async def _run(args) -> None:
    overrides = {
        "step_budget": args.step_budget,
        "segment_max_turns": args.segment_max_turns,
        "max_segments": args.max_segments,
        "actions_per_turn": args.actions_per_turn,
    }
    task = _sample_task(args.dataset_name, args.split, args.task_index, overrides)

    _banner("SAMPLED TASK (shared by the whole GRPO group)", _CYAN)
    print(f"  id              = {task.id}")
    print(f"  seed            = {task.metadata.get('seed')}")
    print(f"  solution_length = {task.metadata.get('solution_length')}")
    print(f"  group_size      = {args.group_size}")
    print(f"  step_budget     = {task.metadata.get('step_budget')}")
    print(f"  base_url        = {args.base_url}")
    print(f"  model           = {args.model}")

    # Tokenizer + chat parser: the faithful stand-in for the gateway's tokenization.
    from transformers import AutoTokenizer
    from rllm.parser.chat_template_parser import ChatTemplateParser

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    chat_parser = ChatTemplateParser.get_parser(tokenizer)
    tracer = _make_tracer(tokenizer, chat_parser)

    group_episodes = []
    for g in range(args.group_size):
        cfg = AgentConfig(
            base_url=args.base_url,
            model=args.model,
            session_uid=f"{task.id}:{g}",
            metadata={},
            is_validation=False,  # training path
            sampling_params={"temperature": args.temperature, "max_tokens": args.max_tokens},
        )

        _banner(f"STAGE 1  rollout sibling #{g} of {args.group_size}", _BLUE)
        episode = await flow_mod.sokoban_rewind_flow.arun(task, cfg)
        print(_c(
            f"  done: won={episode.artifacts.get('won')} "
            f"segments={episode.artifacts.get('segments')} "
            f"trajectories={len(episode.trajectories)} "
            f"rewinds={episode.artifacts.get('rewinds')}",
            _BLUE + _BOLD,
        ))

        _print_segment_rewards(episode, g)
        _wait()
        
        import pdb; pdb.set_trace()
        traces = _episode_traces(episode, tracer, session_id=cfg.session_uid)
        enriched = enrich_episode_with_traces(
            episode, traces, uid=cfg.session_uid, task=task.metadata, strict=False
        )
        _print_stitching(enriched, g)
        # Keep the enriched episode (it carries the token ids + rewards).
        enriched.artifacts = episode.artifacts
        group_episodes.append(enriched)
        _wait()

    _print_grpo(group_episodes, norm_by_std=not args.no_std_norm)
    print()
    print(_c("Pipeline walkthrough complete.", _CYAN + _BOLD))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=os.environ.get("DEBUG_BASE_URL", "http://localhost:8000/v1"))
    ap.add_argument("--model", default=os.environ.get("DEBUG_MODEL", ""))
    ap.add_argument("--tokenizer", default=os.environ.get("DEBUG_TOKENIZER", ""),
                    help="HF tokenizer id/path. Defaults to --model.")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--dataset-name", default=os.environ.get("DEBUG_DATASET_NAME", "sokoban"))
    ap.add_argument("--task-index", type=int, default=0)
    ap.add_argument("--group-size", type=int, default=4, help="GRPO siblings of the same task.")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--step-budget", type=int, default=12)
    ap.add_argument("--segment-max-turns", type=int, default=4)
    ap.add_argument("--max-segments", type=int, default=None)
    ap.add_argument("--actions-per-turn", type=int, default=None)
    ap.add_argument("--no-std-norm", action="store_true",
                    help="Use mean-only advantage (norm_adv_by_std_in_grpo=False).")
    ap.add_argument("--auto", action="store_true", help="Do not pause between stages.")
    args = ap.parse_args()

    if args.auto:
        os.environ["DEBUG_NO_PAUSE"] = "1"
    if not args.model:
        ap.error("--model is required (or set $DEBUG_MODEL).")

    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        print(_c("\nInterrupted.", _RED))
        sys.exit(130)


if __name__ == "__main__":
    main()

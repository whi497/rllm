#!/usr/bin/env python3
"""Interactive, step-by-step debugger for the rewind-choice Sokoban flow.

It runs the real, unmodified ``sokoban_rewind_flow`` from
``flow/sokoban_rewind_choice_flow.py`` on one sampled Sokoban task, while
wrapping LLM calls and environment interactions so you can inspect:

  - action prompts and raw model responses,
  - parsed move sequences or explicit rewind commands,
  - reflection prompts and parsed rewind targets,
  - environment feedback after each move,
  - checkpoint saves and rewind operations,
  - reflection prompts and generated branch memories.

The debugger pauses before each model call. Press <Enter> to continue, or run
with ``--auto`` / ``DEBUG_NO_PAUSE=1`` to stream the whole rollout.

Usage::

    python3 rllm/tbmf/sokoban/debug/debug_rewind_choice_flow.py \
        --base-url http://localhost:8000/v1 \
        --model Qwen/Qwen3-4B \
        --task-index 0

Or set DEBUG_BASE_URL / DEBUG_MODEL instead of the flags.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import textwrap
from typing import Any

# --- Make the package importable whether run as a module or a script ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SOKOBAN_DIR = os.path.dirname(_THIS_DIR)          # .../tbmf/sokoban
_FLOW_DIR = os.path.join(_SOKOBAN_DIR, "flow")     # .../tbmf/sokoban/flow
_TBMF_DIR = os.path.dirname(_SOKOBAN_DIR)          # .../tbmf
_RLLM_ROOT = os.path.dirname(_TBMF_DIR)            # .../rllm
_WORKSPACE_ROOT = os.path.dirname(_RLLM_ROOT)
if "RLLM_HOME" not in os.environ and os.path.exists(os.path.join(_WORKSPACE_ROOT, "datasets", "registry.json")):
    os.environ["RLLM_HOME"] = _WORKSPACE_ROOT
for p in (_SOKOBAN_DIR, _FLOW_DIR, _RLLM_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from rllm.data.dataset import DatasetRegistry
from rllm.types import AgentConfig, Task

# Import the flow module so we can monkeypatch symbols it resolves at runtime.
from flow import sokoban_rewind_choice_flow_refacted as flow_mod


# ----------------------------------------------------------------------------
# Pretty printing
# ----------------------------------------------------------------------------

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_MAGENTA = "\033[35m"
_BLUE = "\033[34m"


def _c(text: str, color: str) -> str:
    if os.environ.get("NO_COLOR"):
        return text
    return f"{color}{text}{_RESET}"


def _rule(char: str = "=", width: int = 88) -> str:
    return char * width


def _banner(title: str, color: str = _CYAN) -> None:
    print()
    print(_c(_rule(), color))
    print(_c(f" {title}", color + _BOLD))
    print(_c(_rule(), color))


def _section(title: str, body: Any, color: str = _BLUE, indent: int = 2) -> None:
    print(_c(f"  -- {title} --", color + _BOLD))
    if body is None:
        body = "(none)"
    body = str(body)
    pad = " " * indent
    for line in body.splitlines() or [""]:
        print(pad + line)
    print()


def _wait(prompt: str = "Press <Enter> for the next model call (q + Enter to quit)... ") -> None:
    if os.environ.get("DEBUG_NO_PAUSE"):
        print(_c("[auto-continue]", _DIM))
        return
    try:
        ans = input(_c("\n>>> " + prompt, _MAGENTA + _BOLD))
    except EOFError:
        print(_c("\n[stdin closed -> continuing without pause]", _DIM))
        return
    if ans.strip().lower() in {"q", "quit", "exit"}:
        print(_c("\nUser requested quit. Aborting debug run.", _RED + _BOLD))
        raise KeyboardInterrupt


# ----------------------------------------------------------------------------
# Wrappers: intercept LLM calls and env interactions issued by the flow.
# ----------------------------------------------------------------------------

def _classify_llm_call(messages: list[dict[str, str]]) -> str:
    if len(messages) == 1 and messages[0].get("role") == "user":
        return "reflection"
    return "action"


def _format_actions(actions: list[int] | None) -> str:
    if not actions:
        return "(none)"
    return ", ".join(flow_mod._action_label(action) for action in actions)


class _DebugCompletions:
    """Wraps client.chat.completions.create."""

    def __init__(self, real_completions, actions_per_turn: int):
        self._real = real_completions
        self._actions_per_turn = actions_per_turn
        self._llm_call_no = 0

    async def create(self, *args, **kwargs):
        self._llm_call_no += 1
        messages = kwargs.get("messages", [])
        call_kind = _classify_llm_call(messages)

        if call_kind == "reflection":
            _banner(f"LLM CALL #{self._llm_call_no}  (REFLECTION / branch memory)", _YELLOW)
        else:
            _banner(f"LLM CALL #{self._llm_call_no}  (ACTION decision)", _CYAN)

        for msg in messages:
            role = msg.get("role", "?")
            if role == "system":
                color = _DIM
            elif call_kind == "reflection":
                color = _YELLOW
            else:
                color = _GREEN
            _section(f"PROMPT [{role}]", msg.get("content", ""), color)

        _wait()

        resp = await self._real.create(*args, **kwargs)
        content = ""
        finish = None
        try:
            content = resp.choices[0].message.content or ""
            finish = resp.choices[0].finish_reason
        except Exception:
            pass

        _section("MODEL RESPONSE (raw)", content, _GREEN)
        print(_c(f"  finish_reason = {finish}", _DIM))

        if call_kind == "reflection":
            remark = flow_mod.parse_remark(content) if content else ""
            _section(
                "PARSED BRANCH MEMORY (<remark>...</remark>)",
                remark or "(no <remark> tag; flow will fall back to raw text / heuristic)",
                _YELLOW,
            )
            final = flow_mod._extract_final_action_text(content)
            _section("FINAL <action> EXTRACTED", repr(final), _MAGENTA)
            target_match = re.fullmatch(
                r"rewind\s+to\s+(?:C\s*[_-]?\s*)?(\d+)\s*$",
                final,
                flags=re.IGNORECASE,
            )
            parsed = f"REFLECTION REWIND to C_{target_match.group(1)}" if target_match else "NO VALID REWIND ACTION"
            _section("PARSED REFLECTION REWIND", parsed, _BLUE if target_match else _RED)
        else:
            command = flow_mod.parse_agent_command(
                content, actions_per_turn=self._actions_per_turn
            )
            final = flow_mod._extract_final_action_text(content)
            _section("FINAL <action> EXTRACTED", repr(final), _MAGENTA)
            if command.kind == "move":
                parsed = f"MOVE sequence: {_format_actions(command.actions)}"
                color = _GREEN
            elif command.kind == "rewind":
                parsed = f"MODEL REWIND to C_{command.rewind_to}"
                color = _BLUE
            else:
                parsed = f"INVALID -> {command.error}"
                color = _RED
            _section("PARSED COMMAND", parsed, color)

        return resp


class _DebugChat:
    def __init__(self, real_chat, actions_per_turn: int):
        self.completions = _DebugCompletions(real_chat.completions, actions_per_turn)


class _DebugAsyncOpenAI:
    """Drop-in replacement for AsyncOpenAI used inside the flow module."""

    _actions_per_turn = 3

    def __init__(self, *args, **kwargs):
        from openai import AsyncOpenAI as _RealAsyncOpenAI

        self._real = _RealAsyncOpenAI(*args, **kwargs)
        self.chat = _DebugChat(self._real.chat, _DebugAsyncOpenAI._actions_per_turn)


def _make_session_wrapper(real_session):
    """Wrap a session so reset/step/checkpoint/rewind are traced."""

    class _DebugSession:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        async def __aenter__(self):
            await self._inner.__aenter__()
            return self

        async def __aexit__(self, *exc):
            return await self._inner.__aexit__(*exc)

        async def reset(self):
            obs, info = await self._inner.reset()
            _banner("ENV RESET (initial Sokoban state = C_0)", _GREEN)
            _section("INITIAL OBSERVATION", obs, _GREEN)
            _section("RESET INFO", info, _DIM)
            return obs, info

        async def step(self, action):
            action_label = flow_mod._action_label(action)
            _banner(f"ENV STEP  action={action_label} ({action})", _GREEN)
            result = await self._inner.step(action)
            if result.won:
                outcome = "WON"
            elif result.done:
                outcome = "DONE but not won"
            else:
                outcome = "continuing"
            _section("ENV FEEDBACK / observation after action", result.observation, _GREEN)
            print(_c(
                f"  done={result.done}  won={result.won}  reward={result.reward}  "
                f"=> {outcome}",
                _GREEN + _BOLD,
            ))
            print(_c(
                f"  budget: total_steps={getattr(self._inner, 'total_steps', '?')}  "
                f"remaining={getattr(self._inner, 'step_budget_remaining', '?')}",
                _DIM,
            ))
            return result

        async def save_checkpoint(self, *args, **kwargs):
            saved = await self._inner.save_checkpoint(*args, **kwargs)
            print(_c(f"  [checkpoint saved -> id={saved!r}]", _DIM))
            return saved

        async def rewind(self, checkpoint_id):
            _banner(f"ENV REWIND -> checkpoint id={checkpoint_id!r}", _BLUE)
            result = await self._inner.rewind(checkpoint_id)
            _section("OBSERVATION AFTER REWIND", result.observation, _BLUE)
            print(_c(
                f"  budget after rewind: total_steps={getattr(self._inner, 'total_steps', '?')}  "
                f"remaining={getattr(self._inner, 'step_budget_remaining', '?')}",
                _DIM,
            ))
            return result

    return _DebugSession(real_session)


def _make_create_env_session(real_create):
    async def _wrapped(*args, **kwargs):
        real_session = await real_create(*args, **kwargs)
        return _make_session_wrapper(real_session)

    return _wrapped


# ----------------------------------------------------------------------------
# Task sampling
# ----------------------------------------------------------------------------

def _sample_task(dataset_name: str, split: str, index: int) -> Task:
    ds = DatasetRegistry.load_dataset(dataset_name, split)
    if ds is None:
        raise RuntimeError(
            f"Sokoban dataset '{dataset_name}/{split}' not found. Prepare it first, e.g.:\n"
            "  RLLM_HOME=$PWD python3 rllm/tbmf/sokoban/prepare_sokoban_data.py "
            "--dataset-name sokoban_7x7_3box --dim-room 7x7 --num-boxes 3"
        )
    rows = ds.get_data()
    if not rows:
        raise RuntimeError(f"Sokoban '{split}' dataset is empty.")
    row = rows[index % len(rows)]

    return Task(
        id=str(row.get("uid", f"{split}_{index}")),
        instruction=str(row.get("question", "Sokoban puzzle")),
        metadata=dict(row),
        dataset_dir="",
    )


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def _print_episode_summary(episode) -> None:
    _banner("EPISODE SUMMARY", _CYAN)
    art = episode.artifacts or {}
    print(_c(f"  won                         = {art.get('won')}", _GREEN if art.get("won") else _RED))
    for key in (
        "segments",
        "total_play_turns",
        "total_reflection_turns",
        "total_llm_calls",
        "total_env_steps",
        "step_budget",
        "rewinds",
        "forced_rewinds",
        "model_rewinds",
        "forced_context_folds",
        "exhausted_reason",
        "final_game_position",
    ):
        print(f"  {key:<28}= {art.get(key)}")

    rewind_log = art.get("rewind_log") or []
    if rewind_log:
        _section("REWIND LOG", "", _BLUE)
        for record in rewind_log:
            performed = "env" if record.get("performed_env_rewind", True) else "context-only"
            print(_c(
                f"    seg {record.get('segment')}: C_{record.get('from')} -> C_{record.get('to')} "
                f"(ckpt={record.get('checkpoint_id_used')}, {performed}, kind={record.get('kind')}) "
                f"reason={record.get('reason')}",
                _BLUE,
            ))
        print()

    mems = art.get("branch_memories") or []
    if mems:
        _section("BRANCH MEMORIES (reflection output, in injection order)", "", _YELLOW)
        for mem in mems:
            print(_c(
                f"    [seg {mem.get('segment')}] {mem.get('kind')} rewind "
                f"C_{mem.get('rewind_from')}->C_{mem.get('rewind_to')}: "
                f"{mem.get('reason')}",
                _YELLOW + _BOLD,
            ))
            print(textwrap.indent(str(mem.get("memory", "")), "      "))
            print()

    print(_c(f"  trajectories ({len(episode.trajectories)}):", _CYAN + _BOLD))
    for traj in episode.trajectories:
        print(f"    - {traj.name}: {len(traj.steps)} step(s), reward={traj.reward}")


async def _run(args) -> None:
    task = _sample_task(args.dataset_name, args.split, args.task_index)

    # Override flow knobs through task metadata. The flow reads these keys.
    overrides = {
        "step_budget": args.step_budget,
        "segment_max_turns": args.segment_max_turns,
        "max_segments": args.max_segments,
        "max_total_turns": args.max_total_turns,
        "actions_per_turn": args.actions_per_turn,
    }
    for key, value in overrides.items():
        if value is not None:
            task.metadata[key] = value

    dim_room = flow_mod._parse_dim_room(task.metadata.get("dim_room", (6, 6)))
    num_boxes = int(task.metadata.get("num_boxes", flow_mod.LAMER_SOKOBAN_CONFIG["num_boxes"]))
    actions_per_turn = int(
        task.metadata.get("actions_per_turn", flow_mod.LAMER_SOKOBAN_CONFIG["actions_per_turn"])
    )

    _banner("SAMPLED TASK", _CYAN)
    print(f"  dataset                       = {args.dataset_name}/{args.split}")
    print(f"  id                            = {task.id}")
    print(f"  instruction                   = {task.instruction}")
    print(f"  dim_room                      = {dim_room}")
    print(f"  num_boxes                     = {num_boxes}")
    print(f"  seed                          = {task.metadata.get('seed')}")
    print(f"  solution_length               = {task.metadata.get('solution_length')}")
    print(f"  actions_per_turn              = {actions_per_turn}")
    print(f"  step_budget                   = {task.metadata.get('step_budget', flow_mod.DEFAULT_STEP_BUDGET)}")
    print(f"  segment_max_turns             = {task.metadata.get('segment_max_turns', task.metadata.get('max_turns', flow_mod.DEFAULT_SEGMENT_MAX_TURNS))}")
    print(f"  base_url                      = {args.base_url}")
    print(f"  model                         = {args.model}")

    cfg = AgentConfig(
        base_url=args.base_url,
        model=args.model,
        session_uid=f"debug-{task.id}",
        metadata={},
        is_validation=True,
        sampling_params={
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        },
    )

    _DebugAsyncOpenAI._actions_per_turn = actions_per_turn
    orig_openai = flow_mod.AsyncOpenAI
    orig_create_env = flow_mod.create_env_session
    flow_mod.AsyncOpenAI = _DebugAsyncOpenAI
    flow_mod.create_env_session = _make_create_env_session(orig_create_env)

    try:
        episode = await flow_mod.sokoban_rewind_flow.arun(task, cfg)
    finally:
        flow_mod.AsyncOpenAI = orig_openai
        flow_mod.create_env_session = orig_create_env

    _print_episode_summary(episode)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--base-url",
        default=os.environ.get("DEBUG_BASE_URL", "http://localhost:8000/v1"),
        help="OpenAI-compatible model server base URL. Default: $DEBUG_BASE_URL or http://localhost:8000/v1",
    )
    ap.add_argument(
        "--model",
        default=os.environ.get("DEBUG_MODEL", ""),
        help="Model name as registered on the server. Default: $DEBUG_MODEL",
    )
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--dataset-name", default=os.environ.get("DEBUG_DATASET_NAME", "sokoban"))
    ap.add_argument("--task-index", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--step-budget", type=int, default=60)
    ap.add_argument("--segment-max-turns", type=int, default=flow_mod.DEFAULT_SEGMENT_MAX_TURNS)
    ap.add_argument("--max-segments", type=int, default=None)
    ap.add_argument("--max-total-turns", type=int, default=None)
    ap.add_argument("--actions-per-turn", type=int, default=None)
    ap.add_argument(
        "--auto",
        action="store_true",
        help="Run without pausing before model calls (equivalent to DEBUG_NO_PAUSE=1).",
    )
    args = ap.parse_args()

    if args.auto:
        os.environ["DEBUG_NO_PAUSE"] = "1"

    if not args.model:
        ap.error("--model is required (or set $DEBUG_MODEL). It must match a model served at --base-url.")

    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        print(_c("\nInterrupted.", _RED))
        sys.exit(130)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Interactive, step-by-step debugger for the rewind MiniSweeper flow.

It runs the REAL, unmodified ``minisweeper_rewind_flow`` (from
``flow/minisweeper_rewind_flow_v1.py``) on a single sampled task, but wraps every
LLM call and every environment interaction so you can watch:

  - the exact prompt sent to the model (system + observation, or reflection prompt),
  - the raw model response,
  - the parsed action (reveal / rewind / invalid),
  - the environment feedback (observation, won/done/outcome),
  - rewind operations (from C_x -> C_y, which checkpoint id was used),
  - reflexion: the reflection prompt and the branch memory it produces.

It pauses and waits for you to press <Enter> before each model call (i.e. before
each step), so you can step through the trajectory one interaction at a time.

Because it monkeypatches the OpenAI client and the env session *inside the flow
module*, the flow logic itself is untouched — what you see is what the flow
actually does at train/eval time.

Usage::

    # point this at a running vLLM/OpenAI-compatible server
    python3 -m tbmf.minisweeper.debug.debug_rewind_flow \
        --base-url http://localhost:8000/v1 \
        --model Qwen/Qwen3-4B \
        --task-index 0

Or set env vars DEBUG_BASE_URL / DEBUG_MODEL instead of the flags.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import textwrap

# --- Make the package importable whether run as a module or a script ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MINISWEEPER_DIR = os.path.dirname(_THIS_DIR)          # .../tbmf/minisweeper
_TBMF_DIR = os.path.dirname(_MINISWEEPER_DIR)          # .../tbmf
_RLLM_ROOT = os.path.dirname(_TBMF_DIR)                # .../rllm
for p in (_MINISWEEPER_DIR, _RLLM_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from rllm.data.dataset import DatasetRegistry
from rllm.types import AgentConfig, Task

# Import the flow MODULE (not just the function) so we can monkeypatch the
# symbols it looks up at runtime: AsyncOpenAI and create_env_session.
from flow import minisweeper_rewind_flow_v1 as flow_mod


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


def _section(title: str, body: str, color: str = _BLUE, indent: int = 2) -> None:
    print(_c(f"  -- {title} --", color + _BOLD))
    if body is None:
        body = "(none)"
    body = str(body)
    pad = " " * indent
    for line in body.splitlines() or [""]:
        print(pad + line)
    print()


def _wait(prompt: str = "Press <Enter> for the next step (q + Enter to quit)... ") -> None:
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
# Wrappers: intercept LLM calls and env interactions issued BY the flow.
# ----------------------------------------------------------------------------

class _DebugCompletions:
    """Wraps client.chat.completions.create.

    The flow uses two kinds of calls:
      1. action calls   -> messages = [system, observation_prompt]
      2. reflection calls-> messages = [reflection_prompt] (single user msg)
    We detect which by message shape and label them accordingly.
    """

    def __init__(self, real_completions, board_size: int):
        self._real = real_completions
        self._board_size = board_size
        self._llm_call_no = 0

    async def create(self, *args, **kwargs):
        self._llm_call_no += 1
        messages = kwargs.get("messages", [])
        is_reflection = len(messages) == 1 and messages[0].get("role") == "user"

        # Gate every model call on Enter so the user can step the trajectory.
        if is_reflection:
            _banner(f"LLM CALL #{self._llm_call_no}  (REFLEXION / branch-memory)", _YELLOW)
        else:
            _banner(f"LLM CALL #{self._llm_call_no}  (ACTION decision)", _CYAN)

        # Show the prompt that the flow built and is about to send.
        for msg in messages:
            role = msg.get("role", "?")
            color = _DIM if role == "system" else (_YELLOW if is_reflection else _GREEN)
            _section(f"PROMPT [{role}]", msg.get("content", ""), color)

        _wait()

        # Forward to the real model.
        resp = await self._real.create(*args, **kwargs)
        content = ""
        finish = None
        try:
            content = resp.choices[0].message.content or ""
            finish = resp.choices[0].finish_reason
        except Exception:  # pragma: no cover - defensive
            pass

        _section("MODEL RESPONSE (raw)", content, _GREEN)
        print(_c(f"  finish_reason = {finish}", _DIM))

        # Show how the flow's own parser will interpret this response.
        if not is_reflection:
            # The flow validates reveals against the current board (rejecting
            # already-opened cells), so feed the same observation in here —
            # otherwise this preview would disagree with the flow's decision.
            user_msgs = [m.get("content", "") for m in messages if m.get("role") == "user"]
            observation = user_msgs[-1] if user_msgs else None
            cmd = flow_mod.parse_agent_command(
                content, board_size=self._board_size, observation=observation
            )
            final = flow_mod._extract_final_action_text(content)
            _section("FINAL <action> EXTRACTED", repr(final), _MAGENTA)
            if cmd.kind == "reveal":
                parsed = f"REVEAL cell (row={cmd.row}, col={cmd.col})"
                color = _GREEN
            elif cmd.kind == "rewind":
                parsed = f"REWIND to C_{cmd.rewind_to}"
                color = _BLUE
            else:
                parsed = f"INVALID -> {cmd.error}"
                color = _RED
            _section("PARSED COMMAND", parsed, color)
        else:
            remark = flow_mod.parse_remark(content) if content else ""
            _section(
                "PARSED BRANCH MEMORY (<remark>...</remark>)",
                remark or "(no <remark> tag; flow will fall back to raw text / heuristic)",
                _YELLOW,
            )

        return resp


class _DebugChat:
    def __init__(self, real_chat, board_size: int):
        self.completions = _DebugCompletions(real_chat.completions, board_size)


class _DebugAsyncOpenAI:
    """Drop-in replacement for AsyncOpenAI used inside the flow module."""

    _board_size = 6  # set by the debug driver before the flow runs

    def __init__(self, *args, **kwargs):
        from openai import AsyncOpenAI as _RealAsyncOpenAI

        self._real = _RealAsyncOpenAI(*args, **kwargs)
        self.chat = _DebugChat(self._real.chat, _DebugAsyncOpenAI._board_size)


def _make_session_wrapper(real_session):
    """Wrap a session so step/rewind/save_checkpoint/reset are traced."""

    class _DebugSession:
        def __init__(self, inner):
            self._inner = inner

        # Pass through attributes the flow reads (step_budget_remaining,
        # total_steps, current_checkpoint, save_checkpoint presence, etc.).
        def __getattr__(self, name):
            return getattr(self._inner, name)

        async def __aenter__(self):
            await self._inner.__aenter__()
            return self

        async def __aexit__(self, *exc):
            return await self._inner.__aexit__(*exc)

        async def reset(self):
            obs, info = await self._inner.reset()
            _banner("ENV RESET (initial board = C_0)", _GREEN)
            _section("INITIAL OBSERVATION", obs, _GREEN)
            _section("RESET INFO", info, _DIM)
            return obs, info

        async def step(self, action):
            _banner(f"ENV STEP  action={action}", _GREEN)
            result = await self._inner.step(action)
            outcome = (
                "MINE HIT (done, lost)" if (result.done and not result.won)
                else ("WON" if result.won else "safe reveal")
            )
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

        async def save_checkpoint(self, *a, **k):
            saved = await self._inner.save_checkpoint(*a, **k)
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

def _sample_task(split: str, index: int) -> Task:
    ds = DatasetRegistry.load_dataset("minisweeper", split)
    if ds is None:
        raise RuntimeError(
            f"MiniSweeper '{split}' dataset not found. Prepare it first:\n"
            "  python3 tbmf/minisweeper/prepare_minisweeper_data.py"
        )
    rows = ds.get_data()
    if not rows:
        raise RuntimeError(f"MiniSweeper '{split}' dataset is empty.")
    row = rows[index % len(rows)]

    return Task(
        id=str(row.get("uid", f"{split}_{index}")),
        instruction=str(row.get("question", "MiniSweeper puzzle")),
        metadata=dict(row),
        dataset_dir="",
    )


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def _print_episode_summary(episode) -> None:
    _banner("EPISODE SUMMARY", _CYAN)
    art = episode.artifacts or {}
    print(_c(f"  won                = {art.get('won')}", _GREEN if art.get("won") else _RED))
    for key in (
        "segments", "total_play_turns", "total_reflection_turns", "total_llm_calls",
        "total_env_steps", "step_budget", "rewinds", "forced_rewinds", "model_rewinds",
        "exhausted_reason", "final_game_position",
    ):
        print(f"  {key:<19}= {art.get(key)}")

    rewind_log = art.get("rewind_log") or []
    if rewind_log:
        _section("REWIND LOG", "", _BLUE)
        for r in rewind_log:
            print(_c(
                f"    seg {r.get('segment')}: C_{r.get('from')} -> C_{r.get('to')} "
                f"(ckpt={r.get('checkpoint_id_used')}) reason={r.get('reason')}",
                _BLUE,
            ))
        print()

    mems = art.get("branch_memories") or []
    if mems:
        _section("BRANCH MEMORIES (reflexion output, in injection order)", "", _YELLOW)
        for m in mems:
            print(_c(f"    [seg {m.get('segment')}] rewind "
                     f"C_{m.get('rewind_from')}->C_{m.get('rewind_to')}: "
                     f"{m.get('reason')}", _YELLOW + _BOLD))
            print(textwrap.indent(str(m.get("memory", "")), "      "))
            print()

    print(_c(f"  trajectories ({len(episode.trajectories)}):", _CYAN + _BOLD))
    for t in episode.trajectories:
        print(f"    - {t.name}: {len(t.steps)} step(s), reward={t.reward}")


async def _run(args) -> None:
    task = _sample_task(args.split, args.task_index)

    # Override flow knobs via task metadata (the flow reads these from there).
    if args.step_budget is not None:
        task.metadata["step_budget"] = int(args.step_budget)
    if args.segment_max_turns is not None:
        task.metadata["segment_max_turns"] = int(args.segment_max_turns)

    board_size = int(task.metadata.get("board_size", 6))
    n_mines = int(task.metadata.get("n_mines", 3))

    _banner("SAMPLED TASK", _CYAN)
    print(f"  id                = {task.id}")
    print(f"  instruction       = {task.instruction}")
    print(f"  board_size        = {board_size}, n_mines = {n_mines}, seed = {task.metadata.get('seed')}")
    print(f"  step_budget       = {task.metadata.get('step_budget', flow_mod.DEFAULT_STEP_BUDGET)}")
    print(f"  segment_max_turns = {task.metadata.get('segment_max_turns', flow_mod.DEFAULT_SEGMENT_MAX_TURNS)}")
    print(f"  base_url          = {args.base_url}")
    print(f"  model             = {args.model}")

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

    # Install the instrumentation INSIDE the flow module namespace.
    _DebugAsyncOpenAI._board_size = board_size
    orig_openai = flow_mod.AsyncOpenAI
    orig_create_env = flow_mod.create_env_session
    flow_mod.AsyncOpenAI = _DebugAsyncOpenAI
    flow_mod.create_env_session = _make_create_env_session(orig_create_env)

    try:
        # ``minisweeper_rewind_flow`` is wrapped by @rllm.rollout into an
        # AgentFlowFn. Calling it directly would invoke its own asyncio.run()
        # and clash with our running loop, so use .arun(), which awaits the
        # underlying coroutine in-place.
        episode = await flow_mod.minisweeper_rewind_flow.arun(task, cfg)
    finally:
        flow_mod.AsyncOpenAI = orig_openai
        flow_mod.create_env_session = orig_create_env

    _print_episode_summary(episode)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=os.environ.get("DEBUG_BASE_URL", "http://localhost:8000/v1"),
                    help="OpenAI-compatible base URL of the model server (vLLM). "
                         "Default: $DEBUG_BASE_URL or http://localhost:8000/v1")
    ap.add_argument("--model", default=os.environ.get("DEBUG_MODEL", ""),
                    help="Model name as registered on the server. Default: $DEBUG_MODEL")
    ap.add_argument("--split", default="test", choices=["train", "test"],
                    help="Dataset split to sample the task from (default: test).")
    ap.add_argument("--task-index", type=int, default=0,
                    help="Which row to sample (wraps around). Default: 0.")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--step-budget", type=int, default=60,
                    help="Total environment reveal-action budget (flow step_budget). Default: 60.")
    ap.add_argument("--segment-max-turns", type=int, default=30,
                    help="Per-segment turn limit before a forced rewind (flow segment_max_turns). Default: 30.")
    ap.add_argument("--auto", action="store_true",
                    help="Run the whole episode without pausing for <Enter> "
                         "(equivalent to setting DEBUG_NO_PAUSE=1).")
    args = ap.parse_args()

    if args.auto:
        os.environ["DEBUG_NO_PAUSE"] = "1"

    if not args.model:
        ap.error("--model is required (or set $DEBUG_MODEL). "
                 "It must match a model served at --base-url.")

    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        print(_c("\nInterrupted.", _RED))
        sys.exit(130)


if __name__ == "__main__":
    main()

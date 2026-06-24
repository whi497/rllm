"""Rewind-capable MiniSweeper agent flow.

The model operates within a fixed step budget and can:
  1. Reveal a cell (row, col) — normal play action
  2. Rewind to any previous step N via `<action>rewind to N</action>`

Every env step is auto-checkpointed. When the model rewinds:
  - A reflection LLM call is triggered on the failed segment
  - The env restores to step N
  - The model continues from step N with the reflection injected into context

Forced rewind happens when:
  - Segment turn limit exceeded
  - LLM call failed / token truncated
  - Mine hit (done but not won)

The flow ends ONLY when:
  - Task is solved (won=True — all non-mine cells revealed)
  - Step budget is exhausted
  - Safety caps for total LLM turns or rewind segments are exhausted

Reward: cross-segment backward discounting (same as LaMer).

KEY DESIGN: The model sees "game position" (len of current path from start),
which always equals the checkpoint_id, NOT session.total_steps (which never
decreases). This ensures "rewind to N" maps directly to checkpoint N.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from env_service import Checkpoint, RewindResult, StepResult, create_env_session, parse_remark
from env_service.minesweeper import MineSweeperEnv
from openai import AsyncOpenAI

import rllm
from rllm.types import AgentConfig, Episode, Step, Task, Trajectory

try:
    from ..prepare_minisweeper_data import LAMER_MINISWEEPER_CONFIG
    from ..minisweeper_prompt import MINESWEEPER_REFLECT_PROMPT
    from .minisweeper_flow import parse_action
except (ImportError, ValueError):
    from prepare_minisweeper_data import LAMER_MINISWEEPER_CONFIG
    from minisweeper_prompt import MINESWEEPER_REFLECT_PROMPT
    from minisweeper_flow import parse_action

logger = logging.getLogger(__name__)

DEFAULT_STEP_BUDGET = 30
DEFAULT_TRAJ_GAMMA = 0.7
DEFAULT_SEGMENT_MAX_TURNS = 5
DEFAULT_MAX_SEGMENTS = 8
DEFAULT_MAX_TOTAL_TURNS = 32
MAX_REFLECTIONS_IN_CONTEXT = 3
MAX_REFLECTION_CHARS = 1200
MAX_REFLECTION_HISTORY_CHARS = 8000

_REWIND_RE = re.compile(
    r"(?:<action>\s*)?rewind\s+to\s+(\d+)(?:\s*</action>)?",
    re.IGNORECASE,
)

# --- Prompts ---

REWIND_SYSTEM_PROMPT = """\
You are an expert agent operating in the Minesweeper game with REWIND capability.
You will be given a {board_size} by {board_size} board, with {n_mines} hidden mines.
The rows and columns are indexed from 1 to {board_size}.

# Cell States
- Unopened cells (?): cells that are yet to be revealed and may contain a mine.
- Blank cells (.): opened and non-mine cells with no neighboring mines.
- Numbered cells (1-8): opened non-mine cells showing how many mines are in the eight neighboring cells.
- Mine cells (*): opened cells that contain a mine (game over).

# Goal
Reveal all non-mine cells without hitting any mine.

# Reveal Rules
- Choose ONE unopened cell (?) to reveal per turn.
- Blank cell (.): auto-cascades to reveal contiguous blank cells and bordering numbered cells.
- Numbered cell (1-8): only that single cell is revealed.
- Mine (*): game ends immediately in a loss.

# Rewind Capability
You have a total step budget of {step_budget} steps. At any point you can:
- `<action>rewind to N</action>` — rewind the game to step N (N can be 0 to go back to start).

When you rewind, the board returns to what it was at step N. Steps already used still count toward your budget. Use rewind when you realize you made a wrong deduction or hit a mine.

# Response Format
- First reason step-by-step about the current board state. Analyze numbered clues to deduce safe cells.
- Then choose ONE unopened cell to reveal: `<action>(row, col)</action>`
- OR rewind: `<action>rewind to N</action>`
"""

REWIND_REFLECT_PROMPT = """\
You are an expert Minesweeper player reflecting on a failed segment.
Board: {board_size} by {board_size}, {n_mines} mines. Rows/columns indexed 1 to {board_size}.

# Cell States
- Unopened cells (?): cells yet to be revealed.
- Blank cells (.): no neighboring mines.
- Numbered cells (1-8): count of neighboring mines.
- Mine cells (*): mine hit = loss.

# Context
The game was at position {rewind_from_step} when you decided to rewind back to position {rewind_to_step}.

# Board at position {rewind_to_step} (where you're rewinding to):
{rewind_to_observation}

# Actions taken from position {rewind_to_step} to position {rewind_from_step}:
{history_after_rewind}

# Your Task
Reflect on what went wrong. Identify the incorrect deduction or risky guess that led to failure. Devise a concise improved plan starting from position {rewind_to_step}.

- First reason step-by-step about what went wrong.
- Then devise a new plan with reference to specific cells and deductions.
- End with your reflection inside <remark> </remark> tags to guide the next attempt.
"""


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return "[truncated]\n" + text[-max_chars:]


def _build_system_prompt(board_size: int, n_mines: int, step_budget: int) -> str:
    return REWIND_SYSTEM_PROMPT.format(
        board_size=board_size, n_mines=n_mines, step_budget=step_budget
    )


def _build_observation_prompt(
    observation: str,
    turn: int,
    game_position: int,
    step_budget: int,
    budget_used: int,
    reflections: list[str] | None = None,
    action_is_valid: bool = True,
) -> str:
    budget_remaining = max(0, step_budget - budget_used)

    if turn == 0 and game_position == 0:
        header = "The initial state of the game is:"
    else:
        header = f"Current observation (position {game_position}, turn {turn}):"

    retry = ""
    if not action_is_valid:
        retry = (
            "\nYour last response did not contain a valid coordinate or rewind command. "
            "Choose one unopened cell (row, col) or rewind to N.\n"
        )

    reflection_context = ""
    if reflections:
        reflection_context = "\n# Reflections from previous attempts:\n"
        recent_reflections = reflections[-MAX_REFLECTIONS_IN_CONTEXT:]
        for i, r in enumerate(recent_reflections):
            reflection_context += f"  Reflection #{i+1}: {_truncate_text(r, MAX_REFLECTION_CHARS)}\n"

    return (
        f"{header}\n{observation}\n"
        f"{retry}"
        f"\nBudget used: {budget_used}/{step_budget} (remaining: {budget_remaining}). "
        f"Current game position: {game_position}.\n"
        f"{reflection_context}\n"
        f"Choose one unopened cell to reveal: <action>(row, col)</action>, "
        f"or <action>rewind to N</action> to go back to position N."
    )


def _parse_rewind(content: str) -> int | None:
    match = _REWIND_RE.search(content)
    if match:
        return int(match.group(1))
    return None


def _build_history_string(
    interaction_history: list[dict[str, str]],
    start: int,
    end: int | None = None,
) -> str:
    entries = interaction_history[start:end]
    if not entries:
        return "(no actions taken)"
    lines = []
    for i, entry in enumerate(entries):
        step_num = start + i + 1
        lines.append(f"Position {step_num}: reveal {entry['action']} → {entry['outcome']}")
    result = "\n".join(lines)
    return _truncate_text(result, MAX_REFLECTION_HISTORY_CHARS)


async def _do_reflection(
    client: AsyncOpenAI,
    model: str,
    sampling: dict,
    board_size: int,
    n_mines: int,
    rewind_to: int,
    current_position: int,
    rewind_to_obs: str,
    interaction_history: list[dict[str, str]],
    task_id: str,
) -> tuple[str, str, str]:
    """Run reflection LLM call. Returns (full_response, remark_text, prompt)."""
    history_after = _build_history_string(interaction_history, rewind_to)

    reflect_prompt = REWIND_REFLECT_PROMPT.format(
        board_size=board_size,
        n_mines=n_mines,
        rewind_from_step=current_position,
        rewind_to_step=rewind_to,
        rewind_to_observation=rewind_to_obs,
        history_after_rewind=history_after,
    )
    reflect_messages = [{"role": "user", "content": reflect_prompt}]

    try:
        reflect_resp = await client.chat.completions.create(
            model=model, messages=reflect_messages, **sampling, timeout=120,
        )
        reflect_content = reflect_resp.choices[0].message.content or ""
    except Exception as e:
        logger.warning("minisweeper_rewind task %s: reflection failed: %s", task_id, e)
        reflect_content = ""

    remark_text = parse_remark(reflect_content) if reflect_content else ""
    remark_text = _truncate_text(remark_text, MAX_REFLECTION_CHARS) if remark_text else ""
    return reflect_content, remark_text, reflect_prompt


@rllm.rollout(name="minisweeper_rewind")
async def minisweeper_rewind_flow(task: Task, config: AgentConfig) -> Episode:
    """Drive MiniSweeper with rewind capability under a fixed step budget.

    KEY: game_position = len(interaction_history) = checkpoint_id at that point.
    This ensures "rewind to N" from the model always maps to checkpoint N.
    session.total_steps tracks budget consumption (never decreases).
    """
    meta = task.metadata or {}

    seed = int(meta.get("seed", LAMER_MINISWEEPER_CONFIG["env_seed"]))
    board_size = int(meta.get("board_size", LAMER_MINISWEEPER_CONFIG["board_size"]))
    n_mines = int(meta.get("n_mines", LAMER_MINISWEEPER_CONFIG["n_mines"]))
    board_type = meta.get("board_type", LAMER_MINISWEEPER_CONFIG["board_type"])
    mode = meta.get("mode", LAMER_MINISWEEPER_CONFIG["mode"])
    puzzle_state = meta.get("puzzle_state")

    step_budget = int(meta.get("step_budget", DEFAULT_STEP_BUDGET))
    segment_max_turns = int(meta.get("segment_max_turns", meta.get("max_turns", DEFAULT_SEGMENT_MAX_TURNS)))
    max_segments = int(meta.get("max_segments", DEFAULT_MAX_SEGMENTS))
    max_total_turns = int(meta.get("max_total_turns", DEFAULT_MAX_TOTAL_TURNS))
    traj_gamma = float(meta.get("traj_gamma", DEFAULT_TRAJ_GAMMA))

    client = AsyncOpenAI(base_url=config.base_url, api_key="EMPTY")
    sampling = {k: v for k, v in config.sampling_params.items() if k != "top_k"}

    t = time.perf_counter()
    session = await create_env_session(
        MineSweeperEnv,
        session_mode="local",
        step_budget=step_budget,
        board_size=board_size,
        n_mines=n_mines,
        board_type=board_type,
        seed=seed,
        puzzle_state=puzzle_state,
    )

    trajectories: list[Trajectory] = []
    env_init_s = 0.0
    env_step_s = 0.0

    async with session:
        observation, _info = await session.reset()
        env_init_s = time.perf_counter() - t
        initial_observation = observation

        # step_observations[i] = board observation AFTER checkpoint i is saved.
        # step_observations[0] = initial board (checkpoint 0 from reset).
        step_observations: list[str] = [observation]
        # interaction_history[i] = action taken at game position i+1
        # len(interaction_history) == current game position == latest checkpoint_id
        interaction_history: list[dict[str, str]] = []

        segment_idx = 0
        segment_steps: list[Step] = []
        segment_turn = 0
        segment_start_position = 0
        reflections: list[str] = []

        system_prompt = _build_system_prompt(board_size, n_mines, step_budget)
        won = False
        total_turns = 0
        exhausted_reason: str | None = None

        while True:
            budget_remaining = session.step_budget_remaining
            if budget_remaining is not None and budget_remaining <= 0:
                exhausted_reason = "step budget exhausted"
                break
            if segment_idx >= max_segments:
                exhausted_reason = "segment limit exhausted"
                break
            if total_turns >= max_total_turns:
                exhausted_reason = "turn limit exhausted"
                break
            if won:
                break

            game_position = len(interaction_history)
            obs_prompt = _build_observation_prompt(
                observation=observation,
                turn=segment_turn,
                game_position=game_position,
                step_budget=step_budget,
                budget_used=session.total_steps,
                reflections=reflections,
            )
            if segment_turn == 0:
                messages: list[dict] = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": obs_prompt},
                ]
            else:
                messages.append({"role": "user", "content": obs_prompt})

            force_rewind = False
            force_rewind_reason = ""
            rewind_target: int | None = None

            while True:
                budget_remaining = session.step_budget_remaining
                if budget_remaining is not None and budget_remaining <= 0:
                    exhausted_reason = "step budget exhausted"
                    break
                if total_turns >= max_total_turns:
                    exhausted_reason = "turn limit exhausted"
                    break
                if won:
                    break

                if segment_turn >= segment_max_turns:
                    force_rewind = True
                    force_rewind_reason = "segment turn limit exceeded"
                    break

                try:
                    resp = await client.chat.completions.create(
                        model=config.model, messages=messages, **sampling, timeout=120,
                    )
                    content = resp.choices[0].message.content or ""
                except Exception as e:
                    logger.warning("minisweeper_rewind task %s turn %d: LLM failed: %s", task.id, total_turns, e)
                    force_rewind = True
                    force_rewind_reason = "LLM call failed"
                    break

                finish_reason = resp.choices[0].finish_reason
                if finish_reason == "length":
                    force_rewind = True
                    force_rewind_reason = "response truncated (token limit)"
                    messages.append({"role": "assistant", "content": content})
                    segment_steps.append(Step(
                        chat_completions=list(messages),
                        observation=messages[-2]["content"],
                        model_response=content,
                        action="(truncated)",
                        thought=content,
                    ))
                    total_turns += 1
                    segment_turn += 1
                    break

                messages.append({"role": "assistant", "content": content})
                total_turns += 1
                segment_turn += 1

                rewind_target = _parse_rewind(content)

                if rewind_target is not None:
                    segment_steps.append(Step(
                        chat_completions=list(messages),
                        observation=messages[-2]["content"],
                        model_response=content,
                        action=f"rewind to {rewind_target}",
                        thought=content,
                    ))
                    force_rewind = True
                    force_rewind_reason = f"model requested rewind to {rewind_target}"
                    break

                # --- NORMAL ACTION: parse coordinate ---
                action = parse_action(content, board_size=board_size)
                action_is_valid = action is not None
                env_action = action if action is not None else (-1, -1)
                action_label = f"({env_action[0]}, {env_action[1]})"

                segment_steps.append(Step(
                    chat_completions=list(messages),
                    observation=messages[-2]["content"],
                    model_response=content,
                    action=action_label if action_is_valid else "invalid",
                    thought=content,
                ))

                # Execute action
                t = time.perf_counter()
                result = await session.step(("L", env_action[0], env_action[1]))
                env_step_s += time.perf_counter() - t

                observation = result.observation
                won = result.won
                done = result.done

                # Save checkpoint immediately after each step
                await session.save_checkpoint()
                step_observations.append(observation)

                outcome = "mine hit!" if (done and not won) else ("won!" if won else "safe")
                interaction_history.append({
                    "observation": observation,
                    "outcome": outcome,
                    "action": action_label,
                })

                if won:
                    break

                if done and not won:
                    force_rewind = True
                    force_rewind_reason = "mine hit (game over)"
                    break

                # Build next turn prompt — self-contained around latest board
                game_position = len(interaction_history)
                obs_prompt = _build_observation_prompt(
                    observation=observation,
                    turn=segment_turn,
                    game_position=game_position,
                    step_budget=step_budget,
                    budget_used=session.total_steps,
                    reflections=reflections,
                    action_is_valid=action_is_valid,
                )
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": obs_prompt},
                ]

            # === END OF INNER LOOP ===

            if won:
                if segment_steps:
                    trajectories.append(Trajectory(
                        name=f"minisweeper_seg{segment_idx}",
                        steps=segment_steps,
                        reward=None,
                    ))
                break

            budget_remaining = session.step_budget_remaining
            limit_reached = budget_remaining is not None and budget_remaining <= 0
            limit_reached = limit_reached or total_turns >= max_total_turns
            limit_reached = limit_reached or (force_rewind and segment_idx + 1 >= max_segments)
            if limit_reached:
                if exhausted_reason is None:
                    if budget_remaining is not None and budget_remaining <= 0:
                        exhausted_reason = "step budget exhausted"
                    elif total_turns >= max_total_turns:
                        exhausted_reason = "turn limit exhausted"
                    else:
                        exhausted_reason = "segment limit exhausted"
                if segment_steps:
                    trajectories.append(Trajectory(
                        name=f"minisweeper_seg{segment_idx}",
                        steps=segment_steps,
                        reward=None,
                    ))
                break

            # === FORCED OR VOLUNTARY REWIND ===
            if force_rewind:
                current_position = len(interaction_history)

                if rewind_target is not None:
                    rewind_to = max(0, min(rewind_target, current_position))
                else:
                    rewind_to = segment_start_position

                if segment_steps:
                    trajectories.append(Trajectory(
                        name=f"minisweeper_seg{segment_idx}",
                        steps=segment_steps,
                        reward=None,
                    ))

                # --- REFLECTION ---
                rewind_to_obs = step_observations[rewind_to] if rewind_to < len(step_observations) else initial_observation

                reflect_content, remark_text, reflect_prompt = await _do_reflection(
                    client=client,
                    model=config.model,
                    sampling=sampling,
                    board_size=board_size,
                    n_mines=n_mines,
                    rewind_to=rewind_to,
                    current_position=current_position,
                    rewind_to_obs=rewind_to_obs,
                    interaction_history=interaction_history,
                    task_id=task.id,
                )
                reflections.append(remark_text)

                reflect_step = Step(
                    chat_completions=[{"role": "user", "content": reflect_prompt},
                                      {"role": "assistant", "content": reflect_content}],
                    observation=reflect_prompt,
                    model_response=reflect_content,
                    action="reflect",
                    thought=reflect_content,
                )
                trajectories.append(Trajectory(
                    name=f"minisweeper_reflect{segment_idx}",
                    steps=[reflect_step],
                    reward=None,
                ))

                # --- PERFORM REWIND ---
                # rewind_to maps directly to checkpoint_id because:
                # checkpoint 0 = initial state (from reset)
                # checkpoint N = state after Nth action (each save_checkpoint adds one)
                try:
                    rw_result = await session.rewind(rewind_to)
                    observation = rw_result.observation
                except ValueError:
                    rw_result = await session.rewind(0)
                    observation = rw_result.observation
                    rewind_to = 0

                # Truncate histories to match rewound state
                step_observations = step_observations[:rewind_to + 1]
                interaction_history = interaction_history[:rewind_to]

                # Start new segment
                segment_idx += 1
                segment_steps = []
                segment_turn = 0
                segment_start_position = rewind_to
                rewind_target = None
                force_rewind = False

    # --- Compute rewards: cross-segment backward discounting ---
    play_indices = [i for i, t in enumerate(trajectories) if t.name.startswith("minisweeper_seg")]
    reflect_indices = [i for i, t in enumerate(trajectories) if t.name.startswith("minisweeper_reflect")]

    n_play = len(play_indices)
    if n_play > 0:
        play_rewards = [0.0] * n_play
        if won:
            play_rewards[-1] = 1.0

        discounted = [0.0] * n_play
        discounted[-1] = play_rewards[-1]
        for i in range(n_play - 2, -1, -1):
            discounted[i] = play_rewards[i] + traj_gamma * discounted[i + 1]

        for i, traj_idx in enumerate(play_indices):
            trajectories[traj_idx].reward = discounted[i]

        for traj_idx in reflect_indices:
            traj_name = trajectories[traj_idx].name
            seg_num = int(traj_name[len("minisweeper_reflect"):])
            next_play_idx = seg_num + 1
            if next_play_idx < n_play:
                trajectories[traj_idx].reward = discounted[next_play_idx]
            else:
                trajectories[traj_idx].reward = 0.0

    return Episode(
        trajectories=trajectories,
        metrics={"time/env_init_s": env_init_s, "time/env_step_s": env_step_s},
        artifacts={
            "won": won,
            "segments": n_play,
            "total_turns": total_turns,
            "total_env_steps": session.total_steps,
            "step_budget": step_budget,
            "max_segments": max_segments,
            "max_total_turns": max_total_turns,
            "exhausted_reason": exhausted_reason,
            "rewinds": segment_idx,
            "reflections": reflections,
            "board_size": board_size,
            "n_mines": n_mines,
            "mode": mode,
        },
        is_correct=won,
    )

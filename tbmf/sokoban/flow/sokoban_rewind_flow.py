"""Rewind-capable Sokoban agent flow.

The model operates within a fixed step budget and can:
  1. Take normal actions (up/down/left/right)
  2. Rewind to any previous step N via `<action>rewind to N</action>`

Every env step is auto-checkpointed. When the model rewinds:
  - A reflection LLM call is triggered on the failed segment (step N → current)
  - The env restores to step N
  - The model continues from step N with the reflection injected into context

Forced rewind happens when:
  - Segment turn limit exceeded (model ran out of turns without solving)
  - LLM call failed / token truncated
  - Env done but not won (dead state, e.g. box stuck in corner)

The flow ends ONLY when:
  - Task is solved (won=True)
  - Step budget is exhausted
  - Safety caps for total LLM turns or rewind segments are exhausted

Reward: cross-segment backward discounting (same as LaMer).
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from env_service import Checkpoint, RewindResult, StepResult, create_env_session, parse_remark
from env_service.sokoban import SokobanEnv
from openai import AsyncOpenAI

import rllm
from rllm.types import AgentConfig, Episode, Step, Task, Trajectory

try:
    from ..prepare_sokoban_data import LAMER_SOKOBAN_CONFIG
    from ..sokoban_prompt import SOKOBAN_PLAY_PROMPT
    from .sokoban_flow import parse_actions, _parse_dim_room, _ACTION_LABELS, _ACTION_COUNT_WORDS
except (ImportError, ValueError):
    from prepare_sokoban_data import LAMER_SOKOBAN_CONFIG
    from sokoban_prompt import SOKOBAN_PLAY_PROMPT
    from sokoban_flow import parse_actions, _parse_dim_room, _ACTION_LABELS, _ACTION_COUNT_WORDS

logger = logging.getLogger(__name__)

DEFAULT_STEP_BUDGET = 100
DEFAULT_TRAJ_GAMMA = 0.7
DEFAULT_SEGMENT_MAX_TURNS = 7
DEFAULT_MAX_SEGMENTS = 6
DEFAULT_MAX_TOTAL_TURNS = 64
MAX_REFLECTIONS_IN_CONTEXT = 3
MAX_REFLECTION_CHARS = 1200
MAX_REFLECTION_HISTORY_CHARS = 12000

_REWIND_RE = re.compile(
    r"(?:<action>\s*)?rewind\s+to\s+(\d+)(?:\s*</action>)?",
    re.IGNORECASE,
)

# --- Prompts ---

REWIND_SYSTEM_PROMPT = """\
You are an expert agent operating in the Sokoban environment with REWIND capability.

# Symbols and Their Meaning
- Walls (`#`): These block movement.
- Floor (`_`): Open spaces where you can walk and move boxes.
- Targets (`O`): The spots where boxes need to go.
- Boxes (`X`): These are what you need to push onto the targets.
- Player (`P`): That's you!
- Box on Target (`√`): A box successfully placed on a target.
- Player on Target (`S`): You standing on a target.

# Goal
Push all boxes (`X`) onto targets (`O`).

# Rules
- Admissible actions: ["up", "down", "left", "right"]
- You can only push one box at a time. You can't pull boxes.
- You can't walk through or push boxes into walls or other boxes.
- Avoid pushing boxes into corners or against walls where they can't be moved.


# Rewind Capability
You have a total step budget of {step_budget} steps. At any point you can:
- `<action>rewind to N</action>` — rewind the game to step N (N can be 0 to current step).

When you rewind, the game state returns to what it was at step N. Steps already used still count toward your budget. Use rewind when you realize you're stuck or made a wrong move.

# Response Format
- First reason step-by-step about the current state.
- Then choose {num_actions_per_turn} actions within <action> </action> tags.
- OR rewind: <action>rewind to N</action>
"""

REWIND_REFLECT_PROMPT = """\
You are an expert agent operating in the Sokoban environment.

# Symbols and Their Meaning
- Walls (`#`): These block movement. You can't move through or push anything into walls.
- Floor (`_`): Open spaces where you can walk and move boxes.
- Targets (`O`): The spots where boxes need to go.
- Boxes (`X`): These are what you need to push onto the targets.
- Player (`P`): That's you! You'll move around the grid to push boxes.
- Box on Target (`√`): A box successfully placed on a target.
- Player on Target (`S`): You standing on a target.

# Your Goal
Your goal is to push all the boxes (`X`) onto the target spots (`O`). Once all boxes are on the targets, you win!. Now You are reflecting on a failed segment of a Sokoban puzzle.

# Context
The game was at step {rewind_from_step} when you decided to rewind back to step {rewind_to_step}.

# State at step {rewind_to_step} (where you're rewinding to):
{rewind_to_observation}

# History Before Rewind (interactions before step {rewind_to_step})
{history_before_rewind}

# History After Rewind (interactions after step {rewind_to_step} to step {rewind_from_step})
{history_after_rewind}


# Your Task
Reflect on what went wrong in the segment from step {rewind_to_step} to step {rewind_from_step}. Identify the mistake or inefficiency, and devise a concise improved plan starting from step {rewind_to_step}.

- Your response should first be step-by-step reasoning about the strategy and path you took to attempt to complete the task. Identify where things went wrong or could be better.
- Then devise a concise, new plan of action that accounts for your mistake with reference to specific actions that you should have taken.
- Finally, end the response with your reflection and improved plan inside <remark> </remark> tags, to guide the next trial.
"""


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return "[truncated]\n" + text[-max_chars:]


def _build_system_prompt(step_budget: int, actions_per_turn: int) -> str:
    word = _ACTION_COUNT_WORDS.get(actions_per_turn, str(actions_per_turn))
    return REWIND_SYSTEM_PROMPT.format(step_budget=step_budget, num_actions_per_turn=word)


def _build_observation_prompt(
    observation: str,
    turn: int,
    game_position: int,
    step_budget: int,
    budget_used: int,
    actions_per_turn: int,
    reflections: list[str] | None = None,
    action_is_valid: bool = True,
) -> str:
    word = _ACTION_COUNT_WORDS.get(actions_per_turn, str(actions_per_turn))
    budget_remaining = max(0, step_budget - budget_used)

    if turn == 0 and game_position == 0:
        header = "The initial state of the game is:"
    else:
        header = f"Current observation (position {game_position}, turn {turn}):"

    retry = ""
    if not action_is_valid:
        retry = (
            "\nYour last response did not contain a valid action or rewind command. "
            "Choose from up/down/left/right or rewind to N.\n"
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
        f"Choose the next {word} actions within <action> </action>, "
        f"or <action>rewind to N</action> to go back to position N."
    )


def _parse_rewind(content: str) -> int | None:
    """Extract rewind target step from model output. Returns None if not a rewind."""
    match = _REWIND_RE.search(content)
    if match:
        return int(match.group(1))
    return None


def _build_history_strings(
    interaction_history: list[dict[str, str]],
    rewind_to: int,
) -> tuple[str, str]:
    """Build history_before_rewind and history_after_rewind strings."""
    history_before_lines = []
    for entry in interaction_history[:rewind_to]:
        history_before_lines.append(
            f"[Observation] {entry['observation']}\n"
            f"[Action] {entry['action']}"
        )
    history_before = "\n\n".join(history_before_lines) if history_before_lines else "(start of game)"
    history_before = _truncate_text(history_before, MAX_REFLECTION_HISTORY_CHARS)

    history_after_lines = []
    for entry in interaction_history[rewind_to:]:
        history_after_lines.append(
            f"[Observation] {entry['observation']}\n"
            f"[Action] {entry['action']}"
        )
    history_after = "\n\n".join(history_after_lines) if history_after_lines else "(no actions taken)"
    history_after = _truncate_text(history_after, MAX_REFLECTION_HISTORY_CHARS)

    return history_before, history_after


async def _do_reflection(
    client: AsyncOpenAI,
    model: str,
    sampling: dict,
    rewind_to: int,
    current_step: int,
    rewind_to_obs: str,
    interaction_history: list[dict[str, str]],
    task_id: str,
) -> tuple[str, str]:
    """Run reflection LLM call. Returns (full_response, remark_text)."""
    history_before, history_after = _build_history_strings(interaction_history, rewind_to)

    reflect_prompt = REWIND_REFLECT_PROMPT.format(
        rewind_from_step=current_step,
        rewind_to_step=rewind_to,
        rewind_to_observation=rewind_to_obs,
        history_before_rewind=history_before,
        history_after_rewind=history_after,
    )
    reflect_messages = [{"role": "user", "content": reflect_prompt}]

    try:
        reflect_resp = await client.chat.completions.create(
            model=model, messages=reflect_messages, **sampling, timeout=120,
        )
        reflect_content = reflect_resp.choices[0].message.content or ""
    except Exception as e:
        logger.warning("sokoban_rewind task %s: reflection failed: %s", task_id, e)
        reflect_content = ""

    remark_text = parse_remark(reflect_content) if reflect_content else ""
    remark_text = _truncate_text(remark_text, MAX_REFLECTION_CHARS) if remark_text else ""
    return reflect_content, remark_text, reflect_prompt


@rllm.rollout(name="sokoban_rewind")
async def sokoban_rewind_flow(task: Task, config: AgentConfig) -> Episode:
    """Drive Sokoban with rewind capability under a fixed step budget.

    The flow terminates on:
      - won=True (task solved)
      - step_budget exhausted
      - max_total_turns or max_segments exhausted

    Forced rewinds happen on:
      - segment turn limit exceeded
      - LLM call failure / token truncation
      - env done but not won (dead state)
    """
    meta = task.metadata or {}

    seed = int(meta.get("seed", LAMER_SOKOBAN_CONFIG["env_seed"]))
    dim_room = _parse_dim_room(meta.get("dim_room", (meta.get("dim_x", 6), meta.get("dim_y", 6))))
    num_boxes = int(meta.get("num_boxes", LAMER_SOKOBAN_CONFIG["num_boxes"]))
    max_env_steps = int(meta.get("max_steps", LAMER_SOKOBAN_CONFIG["max_steps"]))
    search_depth = int(meta.get("search_depth", LAMER_SOKOBAN_CONFIG["search_depth"]))
    min_steps = int(meta.get("min_steps", LAMER_SOKOBAN_CONFIG["min_steps"]))
    max_sol_steps = int(meta.get("max_sol_steps", LAMER_SOKOBAN_CONFIG["max_sol_steps"]))
    actions_per_turn = max(1, int(meta.get("actions_per_turn", LAMER_SOKOBAN_CONFIG["actions_per_turn"])))
    mode = str(meta.get("mode", LAMER_SOKOBAN_CONFIG["mode"]))
    puzzle_state = meta.get("puzzle_state")

    step_budget = int(meta.get("step_budget", DEFAULT_STEP_BUDGET))
    segment_max_turns = int(meta.get("segment_max_turns", meta.get("max_turns", DEFAULT_SEGMENT_MAX_TURNS)))
    # max_segments is the single source of truth for how many play segments (and
    # thus rewinds: rewinds = segments - 1) an episode may run. Default 6.
    max_segments = int(meta.get("max_segments", DEFAULT_MAX_SEGMENTS))
    max_total_turns = int(meta.get("max_total_turns", DEFAULT_MAX_TOTAL_TURNS))
    traj_gamma = float(meta.get("traj_gamma", DEFAULT_TRAJ_GAMMA))

    client = AsyncOpenAI(base_url=config.base_url, api_key="EMPTY")
    sampling = {k: v for k, v in config.sampling_params.items() if k != "top_k"}

    t = time.perf_counter()
    session = await create_env_session(
        SokobanEnv,
        session_mode="local",
        step_budget=step_budget,
        mode=mode,
        dim_room=dim_room,
        num_boxes=num_boxes,
        max_steps=max_env_steps,
        search_depth=search_depth,
        min_steps=min_steps,
        max_sol_steps=max_sol_steps,
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

        step_observations: list[str] = [observation]
        step_action_labels: list[str] = []
        interaction_history: list[dict[str, str]] = []

        segment_idx = 0
        segment_steps: list[Step] = []
        segment_turn = 0
        segment_start_step = 0
        reflections: list[str] = []

        system_prompt = _build_system_prompt(step_budget, actions_per_turn)
        won = False
        total_turns = 0
        exhausted_reason: str | None = None

        # === OUTER LOOP: only exits on won or budget exhausted ===
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

            # Build initial prompt for this segment
            game_position = len(interaction_history)
            obs_prompt = _build_observation_prompt(
                observation=observation,
                turn=segment_turn,
                game_position=game_position,
                step_budget=step_budget,
                budget_used=session.total_steps,
                actions_per_turn=actions_per_turn,
                reflections=reflections,
            )
            if segment_turn == 0:
                messages: list[dict] = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": obs_prompt},
                ]
            else:
                messages.append({"role": "user", "content": obs_prompt})

            # === INNER LOOP: one segment (until rewind/forced-rewind/won/budget) ===
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

                # Check segment turn limit
                if segment_turn >= segment_max_turns:
                    force_rewind = True
                    force_rewind_reason = "segment turn limit exceeded"
                    break

                # LLM call
                try:
                    resp = await client.chat.completions.create(
                        model=config.model, messages=messages, **sampling, timeout=120,
                    )
                    content = resp.choices[0].message.content or ""
                except Exception as e:
                    logger.warning("sokoban_rewind task %s turn %d: LLM failed: %s", task.id, total_turns, e)
                    force_rewind = True
                    force_rewind_reason = "LLM call failed"
                    break

                # Check for token truncation (finish_reason != "stop")
                finish_reason = resp.choices[0].finish_reason
                if finish_reason == "length":
                    force_rewind = True
                    force_rewind_reason = "response truncated (token limit)"
                    # Still record the truncated response
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

                # Check if model wants to rewind
                rewind_target = _parse_rewind(content)

                if rewind_target is not None:
                    # --- MODEL-INITIATED REWIND ---
                    segment_steps.append(Step(
                        chat_completions=list(messages),
                        observation=messages[-2]["content"],
                        model_response=content,
                        action=f"rewind to {rewind_target}",
                        thought=content,
                    ))
                    # Will be handled below as a rewind
                    force_rewind = True
                    force_rewind_reason = f"model requested rewind to {rewind_target}"
                    break

                # --- NORMAL ACTIONS ---
                actions = parse_actions(content, max_actions=actions_per_turn)
                action_is_valid = actions is not None
                if actions is None:
                    actions = [0]
                action_labels = [_ACTION_LABELS[a] for a in actions if a in _ACTION_LABELS]
                last_action = ", ".join(action_labels) if action_labels else "still"

                segment_steps.append(Step(
                    chat_completions=list(messages),
                    observation=messages[-2]["content"],
                    model_response=content,
                    action=last_action,
                    thought=content,
                ))

                # Execute actions
                env_done_not_won = False
                prev_obs = step_observations[-1] if step_observations else observation
                for action in actions:
                    if session.step_budget_remaining is not None and session.step_budget_remaining <= 0:
                        break
                    t = time.perf_counter()
                    result = await session.step(action)
                    env_step_s += time.perf_counter() - t
                    observation = result.observation
                    won = result.won
                    done = result.done

                    action_label = _ACTION_LABELS.get(action, "?")
                    step_observations.append(observation)
                    step_action_labels.append(action_label)
                    interaction_history.append({
                        "observation": prev_obs,
                        "action": action_label,
                    })
                    await session.save_checkpoint()
                    prev_obs = observation

                    if won:
                        break
                    if done and not won:
                        env_done_not_won = True
                        break

                if won:
                    break

                # Env dead state -> force rewind
                if env_done_not_won:
                    force_rewind = True
                    force_rewind_reason = "env done but not won (dead state)"
                    break

                # Build next turn prompt
                game_position = len(interaction_history)
                obs_prompt = _build_observation_prompt(
                    observation=observation,
                    turn=segment_turn,
                    game_position=game_position,
                    step_budget=step_budget,
                    budget_used=session.total_steps,
                    actions_per_turn=actions_per_turn,
                    reflections=reflections,
                    action_is_valid=action_is_valid,
                )
                # Keep each turn self-contained around the latest board state.
                # Carrying every prior assistant response made validation prompts
                # grow until vLLM stalled on model-max-length requests.
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": obs_prompt},
                ]

            # === END OF INNER LOOP ===

            if won:
                # Finalize winning segment
                if segment_steps:
                    trajectories.append(Trajectory(
                        name=f"sokoban_seg{segment_idx}",
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
                # Budget/cap exhausted, finalize last segment without another reflection.
                if segment_steps:
                    trajectories.append(Trajectory(
                        name=f"sokoban_seg{segment_idx}",
                        steps=segment_steps,
                        reward=None,
                    ))
                break

            # === FORCED OR VOLUNTARY REWIND ===
            if force_rewind:
                current_position = len(interaction_history)

                # Determine rewind target — game_position == checkpoint_id
                if rewind_target is not None:
                    rewind_to = max(0, min(rewind_target, current_position))
                else:
                    rewind_to = segment_start_step

                # Finalize current segment
                if segment_steps:
                    trajectories.append(Trajectory(
                        name=f"sokoban_seg{segment_idx}",
                        steps=segment_steps,
                        reward=None,
                    ))

                # --- REFLECTION ---
                rewind_to_obs = step_observations[rewind_to] if rewind_to < len(step_observations) else initial_observation

                reflect_content, remark_text, reflect_prompt = await _do_reflection(
                    client=client,
                    model=config.model,
                    sampling=sampling,
                    rewind_to=rewind_to,
                    current_step=current_position,
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
                    name=f"sokoban_reflect{segment_idx}",
                    steps=[reflect_step],
                    reward=None,
                ))

                # --- PERFORM REWIND ---
                try:
                    rw_result = await session.rewind(rewind_to)
                    observation = rw_result.observation
                except ValueError:
                    rw_result = await session.rewind(0)
                    observation = rw_result.observation
                    rewind_to = 0

                # Truncate histories
                step_observations = step_observations[:rewind_to + 1]
                step_action_labels = step_action_labels[:rewind_to]
                interaction_history = interaction_history[:rewind_to]

                # Start new segment
                segment_idx += 1
                segment_steps = []
                segment_turn = 0
                segment_start_step = rewind_to
                rewind_target = None
                force_rewind = False

    # --- Compute rewards: cross-segment backward discounting ---
    play_indices = [i for i, t in enumerate(trajectories) if t.name.startswith("sokoban_seg")]
    reflect_indices = [i for i, t in enumerate(trajectories) if t.name.startswith("sokoban_reflect")]

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
            seg_num = int(traj_name[len("sokoban_reflect"):])
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
            "dim_room": dim_room,
            "num_boxes": num_boxes,
            "mode": mode,
        },
        is_correct=won,
    )

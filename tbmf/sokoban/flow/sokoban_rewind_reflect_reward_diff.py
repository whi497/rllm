"""Rewind-capable Sokoban agent flow.

Travel Back to Move Forward version.

Core semantics:
  - Environment state can rewind to a previous checkpoint C_k.
  - Knowledge state only moves forward: before each rewind, the failed branch is
    compressed into a lightweight branch memory and injected into later context.

The model operates within a fixed decision-step budget and can:
  1. Take normal Sokoban moves: `<action>up, right, down</action>`
  2. Rewind to a previous checkpoint: `<action>rewind to C_k</action>`
     The parser also accepts `<action>rewind to k</action>` for compatibility.

Forced rewind happens when:
  - Current segment turn limit is exceeded
  - LLM call failed or response was truncated by token length
  - Environment reaches done=True and won=False, e.g. a dead/stuck state

For forced rewind, the normal reflection prompt asks the model to select the
rollback checkpoint and compress the failed branch in one response.

Model rewind happens when:
  - The model explicitly emits a valid rewind action during normal play.

Important invariants:
  - game_position == len(interaction_history)
  - C_0 is the initial Sokoban state.
  - C_k is the state after k executed model action steps on the current path.
  - total_play_turns is the model-facing step budget consumption and never decreases.
  - session.total_steps is primitive Sokoban action consumption and never decreases.
  - Rewind truncates the active checkpoint path, but branch memories persist.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
import textwrap
import time
from typing import Any, Literal

from env_service import create_env_session, parse_remark
from env_service.sokoban import SokobanEnv
from openai import AsyncOpenAI

import rllm
from tbmf.flow_utils import classify_llm_failure
from rllm.types import AgentConfig, Episode, Step, Task, Trajectory

try:
    from ..prepare_sokoban_data import LAMER_SOKOBAN_CONFIG
    from .sokoban_flow import (
        parse_actions,
        _parse_dim_room,
        _ACTION_LABELS,
        _ACTION_COUNT_WORDS,
    )
except (ImportError, ValueError):
    from prepare_sokoban_data import LAMER_SOKOBAN_CONFIG
    from sokoban_flow import (
        parse_actions,
        _parse_dim_room,
        _ACTION_LABELS,
        _ACTION_COUNT_WORDS,
    )

logger = logging.getLogger(__name__)

DEFAULT_STEP_BUDGET = 60
DEFAULT_TRAJ_GAMMA = 0.7
DEFAULT_SEGMENT_MAX_TURNS = 20
DEFAULT_MAX_SEGMENTS = 6
DEFAULT_MAX_TOTAL_TURNS = 64

MAX_BRANCH_MEMORY_CHARS = 4800
MAX_REFLECTIONS_IN_CONTEXT = 3
MAX_REFLECTION_CHARS = 4800
MAX_BRANCH_HISTORY_CHARS = 30000
MAX_ACTIVE_BRANCH_EVENTS = 10
MAX_MODEL_RESPONSE_IN_HISTORY_CHARS = 900

# --- Action parsing ---

_ACTION_TAG_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.IGNORECASE | re.DOTALL)
_REWIND_FULL_RE = re.compile(
    r"rewind\s+to\s+(?:C\s*[_-]?\s*)?(\d+)\s*$",
    re.IGNORECASE,
)
_NOOP_LABELS = {"still", "stay", "wait", "noop", "no-op", "none"}


@dataclass(frozen=True)
class AgentCommand:
    kind: Literal["move", "rewind", "invalid"]
    actions: list[int] | None = None
    rewind_to: int | None = None
    raw: str = ""
    error: str = ""


@dataclass
class BranchEvent:
    """A compact event used for current-branch context and reflection."""

    kind: str
    position_before: int
    position_after: int
    action: str
    outcome: str
    observation_after: str
    model_response: str = ""


def _copy_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Copy chat messages before storing snapshots that may outlive rewinds."""

    return [dict(message) for message in messages]


def _extract_checkpoint_id(obj: Any, fallback: Any = None) -> Any:
    """Best-effort extraction of a checkpoint id from env_service objects."""
    if obj is None:
        return fallback
    if isinstance(obj, int):
        return obj
    if isinstance(obj, dict):
        for key in ("checkpoint_id", "id", "checkpoint", "current_checkpoint_id"):
            if key in obj and obj[key] is not None:
                return obj[key]
        return fallback
    for attr in ("checkpoint_id", "id", "checkpoint", "current_checkpoint_id"):
        value = getattr(obj, attr, None)
        if value is not None:
            return value
    return fallback


def _infer_checkpoint_id_from_session(session: Any, fallback: Any) -> Any:
    for attr in ("current_checkpoint_id", "latest_checkpoint_id", "checkpoint_id"):
        value = getattr(session, attr, None)
        if value is not None:
            return value
    current_checkpoint = getattr(session, "current_checkpoint", None)
    inferred = _extract_checkpoint_id(current_checkpoint, fallback=None)
    if inferred is not None:
        return inferred
    return fallback


async def _save_checkpoint_for_position(session: Any, position: int) -> Any:
    """Save or identify the env checkpoint for visible model-step position C_position."""
    saved = None
    save_checkpoint = getattr(session, "save_checkpoint", None)
    if callable(save_checkpoint):
        saved = await save_checkpoint()
    checkpoint_id = _extract_checkpoint_id(saved, fallback=None)
    if checkpoint_id is None:
        checkpoint_id = _infer_checkpoint_id_from_session(session, fallback=position)
    return checkpoint_id


async def _rewind_session_to_position(
    session: Any,
    position_to_checkpoint_id: list[Any],
    position: int,
) -> tuple[Any, Any]:
    """Rewind env to visible path position C_position."""
    if not (0 <= position < len(position_to_checkpoint_id)):
        raise ValueError(
            f"Cannot rewind to C_{position}: only have positions "
            f"C_0..C_{len(position_to_checkpoint_id) - 1}."
        )

    checkpoint_id = position_to_checkpoint_id[position]
    try:
        result = await session.rewind(checkpoint_id)
        return result, checkpoint_id
    except ValueError:
        if checkpoint_id != position:
            logger.warning(
                "checkpoint id %r failed for visible position C_%d; falling back to raw position id",
                checkpoint_id,
                position,
            )
            result = await session.rewind(position)
            return result, position
        raise


def _message(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


# --- Prompts ---

REWIND_SYSTEM_PROMPT = """\
You are an expert Sokoban agent operating with REWIND capability.
The task is guaranteed to be solvable. You must never conclude that the task is unsolvable, hopeless, or impossible, and you must never give up. Even if some branches appear unpromising or lead to dead ends, you can still learn from them, rewind when necessary, and continue exploring.
Your environment state may move backward through rewind, but your knowledge must keep moving forward. Use each failed attempt to refine your understanding, avoid repeating the same mistakes, and make better decisions in future branches until the task is solved.

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
- A blocked move may consume budget without improving the state, so avoid repeating it.

# Checkpoints and Rewind Capability
You operate under two budgets, and you should manage them differently.

The first is a global action budget: {step_budget} model action steps for the entire episode. Each model action step may contain up to {num_actions_per_turn} primitive Sokoban moves. This is a hard ceiling shared across the whole episode. You may use it for exploration, trial and error, and ultimately solving the task. This budget is accumulated globally and is never refunded by rewinding.

The second is a task-solve budget: {branch_attempt_budget} model action steps for making progress along the current active path. You should aim to solve the task within this allowance. Unlike the global action budget, this budget is tied to the checkpoint path. When you rewind to checkpoint C_j, the task-solve budget is rolled back to the amount already consumed at C_j. In other words, rewinding to C_j refunds the task-solve steps spent after C_j, while everything consumed up to C_j remains spent.

Use the global action budget and the task-solve budget together to balance exploration, learning from failed branches, trial and error, and direct progress toward solving the task.

Checkpoints track your progress along the current path as C_0, C_1, ..., C_k:
- C_0 is the initial state.
- C_i is the state reached after i executed model action steps on this path.

Once you have executed at least one model action step, you may travel back to any earlier
checkpoint:
- `<action>rewind to C_j</action>`, where 0 <= j < the current checkpoint index.
- `<action>rewind to j</action>` is equivalent.

Rewinding restores the environment to checkpoint C_j and rolls your branch attempt budget
back to its value at C_j, but what you learned on the abandoned branch is not lost — it
persists as memory. you can rewind when you find 
* a better plan to start at C_j 
* recognize the current branch is stuck(an action trap, an invalid loop, a dead end, or unproductive exploration) 
* You are in good path to success but find insufficient budget to make progress(the task-solve budget is exhausted) and want to move to current state in less turns and context(more actions each turn)

# Response Format
- First reason step-by-step about the current state and box/target geometry.
- Then choose up to {num_actions_per_turn} moves inside one final action tag, e.g.
  `<action>up, right, right</action>`.
- OR rewind to a previous checkpoint: `<action>rewind to C_j</action>`.
Only the final `<action>...</action>` tag will be executed.
"""
# When you rewind, the environment state returns to checkpoint C_j. However, the
# knowledge gained from the failed branch persists as branch memory. Use rewind
# when you realize the current branch is an action trap, invalid loop, dead end,
# unproductive exploration, or when an earlier plan conflicts with new observations.
# You have a total decision-step budget of {step_budget} model action steps.
# Each model action step may contain up to {num_actions_per_turn} primitive Sokoban moves.
# You also have a current-branch attempt budget of {branch_attempt_budget} model responses
# before the system forces a rewind decision. The total decision-step budget is global
# for the whole episode and does not reset after rewinds. The current-branch attempt
# budget resets after each rewind; use it to avoid spending too many attempts on a
# bad branch before traveling back.
# The current path is indexed by checkpoints C_0, C_1, ..., C_k:
# - C_0 is the initial state.
# - C_i is the state after i executed model action steps on the current path.

# You operate under two budgets, and you should manage them differently.

# The first is a global action budget: {step_budget} model action steps for the entire
# episode, where each step may contain up to {num_actions_per_turn} primitive Sokoban moves.
# This is a hard ceiling shared across the whole episode, you can use this to explore, trail and error and solve tasks.

# The second is a task solve budget: {branch_attempt_budget} model steps to make progress
# on your current branch. You should aim to solve the task within this allowance. Rewinding 
# rolled back the budget used at the checkpoint you return to. But the global action steps is still accumulated.
# So rewinding to C_j refunds the steps you spent after C_j,
# while everything consumed up to C_j stays spent — returning to an earlier checkpoint
# reclaims more of this budget than returning to a recent one. 你应该使用好global action steps budget 和task solve budget 来平衡探索，学习，试错和解决任务。



REWIND_REFLECT_PROMPT = """\
You are an expert Sokoban player independently reflecting on a branch attempt.

# Symbols and Their Meaning
- Walls (`#`): These block movement.
- Floor (`_`): Open spaces where you can walk and move boxes.
- Targets (`O`): The spots where boxes need to go.
- Boxes (`X`): These are what you need to push onto the targets.
- Player (`P`): The agent.
- Box on Target (`√`): A box successfully placed on a target.
- Player on Target (`S`): Agent standing on a target.

# Rules
- Admissible actions: ["up", "down", "left", "right"]
- You can only push one box at a time. You can't pull boxes.
- You can't walk through or push boxes into walls or other boxes.
- Avoid pushing boxes into corners or against walls where they can't be moved.
- A blocked move may consume budget without improving the state, so avoid repeating it.


# Core rollback semantics
The game was at step C_{rewind_from_step} when you decided to rewind back to step C_{rewind_to_step}.

Thus the env state returns to step C_{rewind_to_step}, Your goal is to push the knowledge state forward.
compress the branch attempt into a small branch memory that record what have you done that lead to what result,
what have you learned, what you should avoid and what you should do next.
Do not preserve the full trajectory. Preserve only information useful for future decisions
the core target is to help move closer to the target during the next new attempt or try explore other states.

# Context
{rewind_target_context}

# Rewind trigger
{rewind_reason}

# Relevant checkpoint states
{checkpoint_context}

# State at C_{rewind_from_step} before rewinding
{current_observation}

# Branch history from C_{rewind_to_step} to C_{rewind_from_step}
{history_after_rewind}

# Prior reflection outputs
{reflection_history_context}

# Your task
Reflect on the branch attempt and produce a concise branch memory for the next new attempt direction from the checkpoint you return to.


Focus on:
- Invaild actions that environment not accepted or skipped by the environment or stuck by the walls or boxes;
- failure reason or likely bad push/walk sequence;
- boxes pushed into corners, walls, dead zones, or away from reachable targets;
- blocked/no-op moves or invalid loops that should not be repeated;
- useful spatial facts about player access, box positions, target positions, and required order;
- the task is always solvable, so you should always find a concrete improved plan from the checkpoint you return to.

Include a compact branch memory inside <remark> </remark> tags.
Use this structure, replacing C_j with the selected checkpoint:
<remark>
# History Attempt from C_{rewind_to_step} to C_{rewind_from_step}
State at C_{rewind_to_step}:
{rewind_target_context}
State at C_{rewind_from_step}:
{current_observation}
Actions I have done: ...
What's bad about the actions: ...
How to move forward: ...
Next new Plan from C_j that Must be different from the actions I have done 
and independently and carefully think about and forward towards the goal and to a different state or to a good observed state but in less turns and context(more actions each turn): ...
</remark>
"""
# What I have learned: ...
# What I should avoid: ...

REWIND_REFLECT_PROMPT_WITH_CKPT_CHOICE = """\
You are an expert Sokoban player independently reflecting on a branch attempt.

# Symbols and Their Meaning
- Walls (`#`): These block movement.
- Floor (`_`): Open spaces where you can walk and move boxes.
- Targets (`O`): The spots where boxes need to go.
- Boxes (`X`): These are what you need to push onto the targets.
- Player (`P`): The agent.
- Box on Target (`√`): A box successfully placed on a target.
- Player on Target (`S`): Agent standing on a target.

# Rules
- Admissible actions: ["up", "down", "left", "right"]
- You can only push one box at a time. You can't pull boxes.
- You can't walk through or push boxes into walls or other boxes.
- Avoid pushing boxes into corners or against walls where they can't be moved.
- A blocked move may consume budget without improving the state, so avoid repeating it.


# Core rollback semantics
A forced rewind was triggered at C_{rewind_from_step}. The controller has not
selected the rollback checkpoint yet. You must choose exactly one valid previous
checkpoint C_j to restore, then compress the failed branch into useful memory.

The environment state will return to the checkpoint you choose. The knowledge
state must move forward: record what you did, what result it caused, what you
learned, what to avoid, and what to try next from the selected checkpoint.
Do not preserve the full trajectory. Preserve only information useful for future decisions.

# Rewind trigger
{rewind_reason}

# Valid rollback targets
{valid_targets}

# Candidate checkpoint states
{checkpoint_context}

# State at C_{rewind_from_step} before rewinding
{current_observation}

# Branch history from C_{history_start_step} to C_{rewind_from_step}
{history_after_rewind}

# Prior reflection outputs
{reflection_history_context}

# Your task
Reflect on the branch attempt and produce a concise branch memory for the next new attempt direction from the checkpoint you return to.


Focus on:
- Invaild actions that environment not accepted or skipped by the environment;
- failure reason or likely bad push/walk sequence;
- boxes pushed into corners, walls, dead zones, or away from reachable targets;
- blocked/no-op moves or invalid loops that should not be repeated;
- useful spatial facts about player access, box positions, target positions, and required order;
- the task is always solvable, so you should always find a concrete improved plan from the checkpoint you return to.

Include a compact branch memory inside <remark> </remark> tags.
Use this structure, replacing C_j with the checkpoint you choose:
<remark>
# History Attempt from C_{history_start_step} to C_{rewind_from_step}
Selected checkpoint: C_j
State at C_{rewind_from_step}:
{current_observation}
Actions I have done: ...
What's bad about the actions: ...
How to move forward: ...
Next new Plan from C_j that Must be different from the actions I have done 
and independently and carefully think about and forward towards the goal and to a different state or to a good observed state but in less turns and context(more actions each turn): ...
</remark>

End with exactly one final action tag selecting the checkpoint to restore:
<action>rewind to C_j</action>
"""



def _action_label(action: int) -> str:
    return _ACTION_LABELS.get(action, str(action))


def _format_action_sequence(actions: list[int]) -> str:
    return ", ".join(_action_label(action) for action in actions)


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return "[truncated]\n" + text[-max_chars:]


def _format_checkpoint_range(current_position: int) -> str:
    if current_position <= 0:
        return "(none)"
    if current_position == 1:
        return "C_0"
    return f"C_0..C_{current_position - 1}"


def _build_system_prompt(
    step_budget: int,
    actions_per_turn: int,
    branch_attempt_budget: int,
) -> str:
    word = _ACTION_COUNT_WORDS.get(actions_per_turn, str(actions_per_turn))
    return REWIND_SYSTEM_PROMPT.format(
        step_budget=step_budget,
        num_actions_per_turn=word,
        branch_attempt_budget=branch_attempt_budget,
    )


_REFLECTION_INJECTION_MARKER = "# Reflection on the abandoned branch C_"


def _format_reflection_injection(
    branch_memory: str,
    rewind_from: int,
    rewind_to: int,
) -> str:
    """Wrap a branch memory into the message appended to the rewind target's prefix.

    The reflection rides the conversation prefix of checkpoint C_{rewind_to}: every
    future turn generated at C_{rewind_to} and its new descendants inherits it,
    while abandoned-sibling checkpoints are dropped by rewind truncation.
    """
    memory = _truncate_text(branch_memory, MAX_BRANCH_MEMORY_CHARS)
    return (
        f"{_REFLECTION_INJECTION_MARKER}{rewind_to} -> C_{rewind_from} "
        f"(now resuming from C_{rewind_to}):\n{memory}"
    )


def _is_reflection_injection(message: dict[str, str]) -> bool:
    return (
        message.get("role") == "user"
        and message.get("content", "").startswith(_REFLECTION_INJECTION_MARKER)
    )


def _find_trailing_reflection(messages: list[dict[str, str]]) -> str | None:
    """Return the content of the trailing reflection-injection message, if any.

    A reflection is only ever appended as the last message of a checkpoint prefix
    (see the injection block), so the stale reflection at a re-rewound checkpoint
    is the final message in that prefix.
    """
    if messages and _is_reflection_injection(messages[-1]):
        return messages[-1]["content"]
    return None


def _strip_trailing_reflections(messages: list[dict[str, str]]) -> None:
    """Drop trailing reflection-injection messages in place.

    Keeps the base prefix (system + obs/assistant turns) so a re-rewind to the
    same checkpoint replaces the previous reflection instead of stacking on it.
    """
    while messages and _is_reflection_injection(messages[-1]):
        messages.pop()


def _build_reflection_history_context(reflection_history: list[str] | None) -> str:
    if not reflection_history:
        return "(none)"
    lines = []
    recent_reflections = reflection_history[-MAX_REFLECTIONS_IN_CONTEXT:]
    for i, reflection in enumerate(recent_reflections, start=1):
        lines.append(f"Reflection #{i}: {_truncate_text(reflection, MAX_REFLECTION_CHARS)}")
    return "\n".join(lines)


def _format_branch_event(event: BranchEvent) -> str:
    return (
        f"- C_{event.position_before} -> C_{event.position_after}: "
        f"{event.kind} {event.action} => {event.outcome}"
    )


def _build_observation_prompt(
    observation: str,
    branch_attempts_used: int,
    game_position: int,
    step_budget: int,
    budget_used: int,
    segment_max_turns: int,
    actions_per_turn: int,
    active_branch_events: list[BranchEvent] | None = None,
    action_is_valid: bool = True,
    invalid_reason: str = "",
) -> str:
    word = _ACTION_COUNT_WORDS.get(actions_per_turn, str(actions_per_turn))
    budget_remaining = max(0, step_budget - budget_used)
    branch_remaining = max(0, segment_max_turns - branch_attempts_used)
    segment_turn = branch_attempts_used + 1

    if game_position == 0 and budget_used == 0:
        header = "Initial observation at checkpoint C_0:"
    else:
        header = f"Current observation at checkpoint C_{game_position} (segment turn {segment_turn}):"

    retry = ""
    if not action_is_valid:
        retry = (
            "\nYour previous response did not execute a valid action. "
            f"{invalid_reason.strip()}\n"
        )

    active_context = ""
    if active_branch_events:
        recent_events = active_branch_events[-MAX_ACTIVE_BRANCH_EVENTS:]
        active_context = "\n\n# Recent events in the current branch:\n"
        active_context += "\n".join(_format_branch_event(event) for event in recent_events)

    return (
        f"{header}\n{observation}\n"
        f"{retry}\n"
        f"Global action budget used: {budget_used}/{step_budget} "
        f"(remaining: {budget_remaining}).\n"
        f"Task-solve budget used on this branch: {branch_attempts_used}/{segment_max_turns} "
        f"(remaining: {branch_remaining}).\n"
        f"Current checkpoint: C_{game_position}.\n"
        f"Valid model rewind targets: {_format_checkpoint_range(game_position)}."
        f"{active_context}\n"
        f"Choose up to {word} moves inside one final action tag, e.g. "
        f"<action>up, right</action>, or travel back: <action>rewind to C_j</action>."
    )


def _build_history_after_rewind(
    interaction_history: list[dict[str, str]],
    extra_events: list[BranchEvent] | None,
    start: int,
    end: int,
    action_only: bool = False,
) -> str:
    lines: list[str] = []
    bounded_start = max(0, start)
    bounded_end = max(bounded_start, end)
    for entry in interaction_history[bounded_start:bounded_end]:
        before = entry.get("position_before", "?")
        after = entry.get("position_after", "?")
        lines.extend(
            [
                f"C_{before} -> C_{after}",
                f"Action: {entry.get('action', '')}",
                f"Primitive outcomes: {entry.get('primitive_outcomes', '(none)')}",
                f"Outcome: {entry.get('outcome', '')}",
                "Observation before action:",
                entry.get("observation_before", ""),
                "Observation after action:",
                entry.get("observation_after", entry.get("observation", "")),
                "",
            ]
        )

    if extra_events:
        relevant_events = [
            event
            for event in extra_events
            if bounded_start <= event.position_before <= bounded_end
        ]
        if relevant_events:
            lines.append("Non-environment events in this branch:")
            for event in relevant_events:
                lines.append(_format_branch_event(event))
                if event.model_response:
                    if action_only:
                        # Keep only the model's action, dropping the reasoning, to
                        # shrink reflection-prompt cost and stay under the branch
                        # history cap. Take the final <action> tag verbatim; fall
                        # back to the recorded event.action when no tag is present
                        # (e.g. truncated responses) so we never leak reasoning back
                        # in.
                        action_tags = _ACTION_TAG_RE.findall(event.model_response)
                        action_text = action_tags[-1].strip() if action_tags else event.action
                        if action_text:
                            lines.append(f"  Model action: {action_text}")
                    else:
                        lines.append("  Model response excerpt:")
                        lines.append(textwrap.indent(_truncate_text(event.model_response, MAX_MODEL_RESPONSE_IN_HISTORY_CHARS), "  "))

    history = "\n".join(lines).strip() if lines else "(no branch history)"
    return _truncate_text(history, MAX_BRANCH_HISTORY_CHARS)


def _fallback_branch_memory(
    rewind_reason: str,
    rewind_to: int,
    current_position: int,
    interaction_history: list[dict[str, str]],
    action_only: bool = False,
) -> str:
    recent = _build_history_after_rewind(
        interaction_history=interaction_history,
        extra_events=None,
        start=rewind_to,
        end=current_position,
        action_only=action_only,
    )
    return _truncate_text(
        (
            f"Failure reason: {rewind_reason}\n"
            f"Useful facts learned: branch C_{rewind_to}->C_{current_position} failed.\n"
            f"Avoid repeating: {recent}\n"
            f"Plan from C_{rewind_to}: try a different action sequence from this checkpoint."
        ),
        MAX_BRANCH_MEMORY_CHARS,
    )


def _build_checkpoint_choice_context(
    step_observations: list[str] | None,
    current_position: int,
    fallback_observation: str,
) -> str:
    if current_position <= 0:
        return "No previous checkpoint exists."

    lines: list[str] = []
    for position in range(current_position):
        observation = (
            step_observations[position]
            if step_observations is not None and position < len(step_observations)
            else fallback_observation
        )
        lines.append(f"State at C_{position}:\n{observation}")
    return _truncate_text("\n\n".join(lines), MAX_BRANCH_HISTORY_CHARS)


def _extract_final_action_text(content: str) -> str:
    matches = _ACTION_TAG_RE.findall(content or "")
    if matches:
        return matches[-1].strip()
    lines = [line.strip() for line in (content or "").splitlines() if line.strip()]
    return lines[-1] if lines else ""


def parse_agent_command(content: str, actions_per_turn: int) -> AgentCommand:
    raw = _extract_final_action_text(content)
    if not raw:
        return AgentCommand(kind="invalid", raw=raw, error="missing final <action> tag")

    rewind_match = _REWIND_FULL_RE.fullmatch(raw)
    if rewind_match:
        return AgentCommand(kind="rewind", rewind_to=int(rewind_match.group(1)), raw=raw)

    parts = [part.strip().lower() for part in re.split(r"[,;\n]+", raw) if part.strip()]
    if len(parts) == 1 and parts[0] in _NOOP_LABELS:
        return AgentCommand(kind="invalid", raw=raw, error="no-op actions are not useful")
    if not parts:
        return AgentCommand(kind="invalid", raw=raw, error="empty action list")
    if len(parts) > actions_per_turn:
        return AgentCommand(
            kind="invalid",
            raw=raw,
            error=f"too many primitive moves: got {len(parts)}, max is {actions_per_turn}",
        )

    try:
        actions = parse_actions(f"<action>{', '.join(parts)}</action>", max_actions=actions_per_turn)
    except Exception as e:
        return AgentCommand(kind="invalid", raw=raw, error=f"could not parse actions: {e}")

    if not actions:
        return AgentCommand(kind="invalid", raw=raw, error="no valid Sokoban actions found")
    if len(actions) > actions_per_turn:
        return AgentCommand(
            kind="invalid",
            raw=raw,
            error=f"too many parsed primitive moves: got {len(actions)}, max is {actions_per_turn}",
        )
    return AgentCommand(kind="move", actions=actions, raw=raw)


def _parse_rewind_target_only(content: str, current_position: int) -> tuple[int | None, str]:
    raw = _extract_final_action_text(content)
    match = _REWIND_FULL_RE.fullmatch(raw)
    if not match:
        return None, f"final action {raw!r} is not a rewind command"
    target = int(match.group(1))
    if not (0 <= target < current_position):
        return None, (
            f"rewind target C_{target} is invalid; valid targets are "
            f"{_format_checkpoint_range(current_position)}"
        )
    return target, ""


async def _do_reflection(
    client: AsyncOpenAI,
    model: str,
    sampling: dict[str, Any],
    rewind_to: int | None,
    current_position: int,
    rewind_to_obs: str | None,
    current_observation: str,
    step_observations: list[str] | None,
    interaction_history: list[dict[str, str]],
    segment_events: list[BranchEvent],
    reflection_history: list[str],
    rewind_reason: str,
    task_id: str,
    history_start: int | None = None,
    prior_reflection_at_target: str | None = None,
    action_only: bool = False,
) -> tuple[str, str, str, int | None, str]:
    """Run reflection LLM call and parse any final rewind target."""

    def _fold_prior_reflection(history: str, target: int | None) -> str:
        if not prior_reflection_at_target:
            return history
        label = f"C_{target}" if target is not None else "this checkpoint"
        folded = (
            f"# Prior reflection already applied at {label} (now superseded):\n"
            f"{_truncate_text(prior_reflection_at_target, MAX_BRANCH_MEMORY_CHARS)}"
        )
        return _truncate_text(f"{history}\n\n{folded}", MAX_BRANCH_HISTORY_CHARS)

    if rewind_to is None:
        # Forced/choice path: the destination is not chosen yet, so list the
        # branch history from where this segment began.
        prompt_history_start = (
            max(0, min(history_start, current_position)) if history_start is not None else 0
        )
        history_after = _build_history_after_rewind(
            interaction_history=interaction_history,
            extra_events=segment_events,
            start=prompt_history_start,
            end=current_position,
            action_only=action_only,
        )
        history_after = _fold_prior_reflection(history_after, prompt_history_start)
        valid_targets = _format_checkpoint_range(current_position)
        checkpoint_context = _build_checkpoint_choice_context(
            step_observations=step_observations,
            current_position=current_position,
            fallback_observation=current_observation,
        )
        reflect_prompt = REWIND_REFLECT_PROMPT_WITH_CKPT_CHOICE.format(
            rewind_from_step=current_position,
            valid_targets=valid_targets,
            checkpoint_context=checkpoint_context,
            current_observation=current_observation,
            history_start_step=prompt_history_start,
            history_after_rewind=history_after,
            reflection_history_context=_build_reflection_history_context(reflection_history),
            rewind_reason=rewind_reason,
        )
    else:
        # Model rewind: the destination C_{rewind_to} is known. The prompt's
        # rewind_to_step is literally that destination, and the branch history
        # spans C_{rewind_to} -> C_{current_position}.
        history_after = _build_history_after_rewind(
            interaction_history=interaction_history,
            extra_events=segment_events,
            start=rewind_to,
            end=current_position,
            action_only=action_only,
        )
        history_after = _fold_prior_reflection(history_after, rewind_to)
        rewind_target_context = (
            f"The environment will resume from C_{rewind_to} after this reflection. "
        )
        checkpoint_context = f"State at C_{rewind_to} where execution will resume:\n{rewind_to_obs or ''}"
        reflect_prompt = REWIND_REFLECT_PROMPT.format(
            rewind_from_step=current_position,
            rewind_to_step=rewind_to,
            rewind_target_context=rewind_target_context,
            checkpoint_context=checkpoint_context,
            current_observation=current_observation,
            history_after_rewind=history_after,
            reflection_history_context=_build_reflection_history_context(reflection_history),
            rewind_reason=rewind_reason,
        )
    reflect_messages = [{"role": "user", "content": reflect_prompt}]

    try:
        reflect_resp = await client.chat.completions.create(
            model=model,
            messages=reflect_messages,
            **sampling,
            timeout=120,
        )
        # from pprint import pprint
        # pprint(reflect_messages)
        # pprint(reflect_resp.choices[0].message.content)
        # import pdb; pdb.set_trace()
        reflect_content = reflect_resp.choices[0].message.content or ""
    except Exception as e:
        logger.warning("sokoban_rewind task %s: reflection failed: %s", task_id, e)
        reflect_content = ""

    branch_memory = parse_remark(reflect_content) if reflect_content else ""
    if not branch_memory and reflect_content:
        branch_memory = reflect_content.strip()
    branch_memory = _truncate_text(branch_memory, MAX_BRANCH_MEMORY_CHARS) if branch_memory else ""
    selected_target, parse_error = _parse_rewind_target_only(
        reflect_content,
        current_position=current_position,
    )
    return reflect_content, branch_memory, reflect_prompt, selected_target, parse_error


# --- Reward assignment ---


# Default ("merged") trajectory names: every play segment shares one name and
# every reflection shares another, so GRPO groups all play segments of a task
# together and all reflections together.
PLAY_TRAJ_NAME = "sokoban_seg"
REFLECT_TRAJ_NAME = "sokoban_reflect"

# Trajectory roles are stamped into metadata and drive reward assignment, so the
# reward logic is independent of the (grouping-controlled) trajectory name.
PLAY_ROLE = "play"
REFLECT_ROLE = "reflect"


def _segment_traj_name(traj_grouping: str, segment_idx: int) -> str:
    """Play-segment trajectory name for the active grouping mode.

    ``merged`` (default): all segments share ``sokoban_seg``.
    ``per_stage``: segments are grouped by stage index -> ``segment_{idx}``.
    """
    if traj_grouping == "per_stage":
        return f"segment_{segment_idx}"
    return PLAY_TRAJ_NAME


def _reflect_traj_name(traj_grouping: str, segment_idx: int) -> str:
    """Reflection trajectory name for the active grouping mode.

    ``merged`` (default): all reflections share ``sokoban_reflect``.
    ``per_stage``: reflections are grouped by stage index -> ``reflection_{idx}``.

    Reflections always use a name distinct from segments, so play and reflection
    trajectories are never grouped together regardless of mode.
    """
    if traj_grouping == "per_stage":
        return f"reflection_{segment_idx}"
    return REFLECT_TRAJ_NAME


def _assign_cross_segment_rewards(
    trajectories: list[Trajectory],
    won: bool,
    traj_gamma: float,
) -> None:
    """Variant ``reflect_reward_diff`` reward assignment.

    ``traj_gamma`` is accepted for signature compatibility but ignored (no
    discounting).

    Play segments all get the terminal outcome reward (1.0 if won else 0.0).

    Reflections get the forward cumulative-reward difference. Trajectories are
    appended in execution order (seg0, reflect0, seg1, reflect1, ..., final_seg),
    and every play trajectory carries ``metadata["cum_reward_at_end"] = S_i`` (the
    path-cumulative env reward when that segment ended). ``reflection_k`` is the
    ``k``-th reflection and sits between play segment ``k`` and play segment
    ``k+1``; its reward is ``S_{k+1} - S_k`` -- the extra cumulative reward the
    next attempt earned relative to where this reflection was triggered.

    Play vs reflection is identified by ``metadata["role"]`` (not by name), so
    this is independent of the trajectory grouping mode (merged or per_stage).

    Alignment is taken from the ordered play trajectory list itself (not a
    separate counter), so it is robust to segments that produce no trajectory.
    The last play segment has no following reflection, so there is always one
    more (or equal) play trajectory than reflections; a reflection lacking a
    following play segment falls back to 0.0.
    """
    final_reward = 1.0 if won else 0.0

    play_indices = [i for i, t in enumerate(trajectories) if (t.metadata or {}).get("role") == PLAY_ROLE]
    reflect_indices = [i for i, t in enumerate(trajectories) if (t.metadata or {}).get("role") == REFLECT_ROLE]

    play_cum = [
        float((trajectories[i].metadata or {}).get("cum_reward_at_end", 0.0))
        for i in play_indices
    ]

    for traj_idx in play_indices:
        trajectories[traj_idx].reward = final_reward

    for reflect_order, traj_idx in enumerate(reflect_indices):
        next_seg = reflect_order + 1
        if next_seg < len(play_cum) and reflect_order < len(play_cum):
            diff = play_cum[next_seg] - play_cum[reflect_order]
        else:
            diff = 0.0
        trajectories[traj_idx].reward = float(diff)

    for traj in trajectories:
        if traj.reward is None:
            traj.reward = 0.0


# --- Main rollout ---


@rllm.rollout(name="sokoban_rewind_reflect_reward_diff")
async def sokoban_rewind_reflect_reward_diff_flow(task: Task, config: AgentConfig) -> Episode:
    """Drive Sokoban with explicit rollback and persistent branch memory."""
    meta = task.metadata or {}

    # "merged" (default): all segments grouped together, all reflections together.
    # "per_stage": segment_{idx} / reflection_{idx} grouped by stage index.
    traj_grouping = str(meta.get("traj_grouping", "merged"))
    # When True, reflection-prompt branch history keeps only each step's final
    # <action> tag, dropping the model reasoning (cheaper prompts; reasoning never
    # leaks back into later attempts).
    action_only = bool(meta.get("action_only", False))

    seed = int(meta.get("seed", LAMER_SOKOBAN_CONFIG["env_seed"]))
    dim_room = _parse_dim_room(
        meta.get("dim_room", (meta.get("dim_x", 6), meta.get("dim_y", 6)))
    )
    num_boxes = int(meta.get("num_boxes", LAMER_SOKOBAN_CONFIG["num_boxes"]))
    max_env_steps = int(meta.get("max_steps", LAMER_SOKOBAN_CONFIG["max_steps"]))
    search_depth = int(meta.get("search_depth", LAMER_SOKOBAN_CONFIG["search_depth"]))
    min_steps = int(meta.get("min_steps", LAMER_SOKOBAN_CONFIG["min_steps"]))
    max_sol_steps = int(meta.get("max_sol_steps", LAMER_SOKOBAN_CONFIG["max_sol_steps"]))
    actions_per_turn = max(
        1,
        int(meta.get("actions_per_turn", LAMER_SOKOBAN_CONFIG["actions_per_turn"])),
    )
    mode = str(meta.get("mode", LAMER_SOKOBAN_CONFIG["mode"]))
    puzzle_state = meta.get("puzzle_state")

    step_budget = int(meta.get("step_budget", DEFAULT_STEP_BUDGET))
    segment_max_turns = int(meta.get("segment_max_turns", DEFAULT_SEGMENT_MAX_TURNS))
    # max_segments is the single source of truth for how many play segments (and
    # thus rewinds: rewinds = segments - 1) an episode may run. Default 6.
    max_segments = int(meta.get("max_segments", DEFAULT_MAX_SEGMENTS))
    max_total_turns = int(
        meta.get(
            "max_total_turns",
            max(DEFAULT_MAX_TOTAL_TURNS, step_budget + max_segments),
        )
    )
    traj_gamma = float(meta.get("traj_gamma", DEFAULT_TRAJ_GAMMA))

    client = AsyncOpenAI(base_url=config.base_url, api_key="EMPTY")
    sampling = {k: v for k, v in config.sampling_params.items() if k != "top_k"}

    primitive_step_budget = int(
        meta.get(
            "primitive_step_budget",
            meta.get("env_step_budget", max(step_budget * actions_per_turn, max_env_steps)),
        )
    )

    session = await create_env_session(
        SokobanEnv,
        session_mode="local",
        step_budget=primitive_step_budget,
        mode=mode,
        dim_room=dim_room,
        num_boxes=num_boxes,
        max_steps=max(max_env_steps, primitive_step_budget),
        search_depth=search_depth,
        min_steps=min_steps,
        max_sol_steps=max_sol_steps,
        seed=seed,
        puzzle_state=puzzle_state,
    )

    trajectories: list[Trajectory] = []
    env_init_s = 0.0
    env_step_s = 0.0
    llm_action_s = 0.0
    llm_reflection_s = 0.0

    async with session:
        t0 = time.perf_counter()
        observation, reset_info = await session.reset()
        env_init_s = time.perf_counter() - t0
        initial_observation = observation

        initial_checkpoint_id = _extract_checkpoint_id(
            reset_info,
            fallback=_infer_checkpoint_id_from_session(session, fallback=0),
        )

        # step_observations[i] = Sokoban observation at visible checkpoint C_i.
        step_observations: list[str] = [observation]

        # position_to_checkpoint_id[i] = env_service checkpoint id for visible C_i.
        # This avoids assuming env checkpoint ids always equal visible positions.
        position_to_checkpoint_id: list[Any] = [initial_checkpoint_id]

        # Path-cumulative env reward (fine-grained). cum_reward follows the active
        # path; cum_reward_checkpoints[pos] is its value at checkpoint C_pos so a
        # rewind to C_j can restore it. The value at the moment a play segment ends
        # is stamped into that segment trajectory's metadata (cum_reward_at_end) and
        # used as S_i by the reflection forward-diff reward.
        cum_reward = 0.0
        cum_reward_checkpoints: list[float] = [0.0]

        # interaction_history[i] = executed model action step from C_i to C_{i+1}.
        interaction_history: list[dict[str, str]] = []

        reflection_history: list[str] = []
        branch_memory_records: list[dict[str, Any]] = []
        rewind_log: list[dict[str, Any]] = []

        system_prompt = _build_system_prompt(step_budget, actions_per_turn, segment_max_turns)

        won = False
        exhausted_reason: str | None = None
        segment_idx = 0
        total_play_turns = 0
        total_reflection_turns = 0
        total_llm_calls = 0
        branch_attempts_used = 0
        forced_rewinds = 0
        model_rewinds = 0
        forced_context_folds = 0
        active_messages: list[dict[str, str]] = [_message("system", system_prompt)]
        global_messages: list[dict[str, str]] = [_message("system", system_prompt)]
        # message_checkpoints[i] is the active conversation prefix to use when
        # generating from visible checkpoint C_i. Rewind truncates only this
        # active history; global_messages remains append-only for debugging.
        message_checkpoints: list[list[dict[str, str]]] = [_copy_messages(active_messages)]
        # branch_attempt_checkpoints[i] is the consumed current-branch attempt
        # budget when generating from C_i. Rewind restores this value, refunding
        # attempts spent after the target checkpoint but not before it.
        branch_attempt_checkpoints: list[int] = [branch_attempts_used]

        while True:
            if total_play_turns >= step_budget:
                exhausted_reason = "step budget exhausted"
                break
            if segment_idx >= max_segments:
                exhausted_reason = "segment limit exhausted"
                break
            if total_llm_calls >= max_total_turns:
                exhausted_reason = "LLM turn limit exhausted"
                break
            if won:
                break

            segment_start_position = len(interaction_history)
            segment_steps: list[Step] = []
            segment_events: list[BranchEvent] = []
            last_action_valid = True
            invalid_reason = ""

            force_rewind = False
            force_rewind_reason = ""
            rewind_kind: Literal["forced", "model", "context_fold", ""] = ""
            rewind_target: int | None = None

            # --- One branch segment ---
            while True:
                if total_play_turns >= step_budget:
                    exhausted_reason = "step budget exhausted"
                    break
                if total_llm_calls >= max_total_turns:
                    exhausted_reason = "LLM turn limit exhausted"
                    break
                if won:
                    break

                current_position = len(interaction_history)

                if branch_attempts_used >= segment_max_turns:
                    force_rewind = True
                    rewind_kind = "forced"
                    force_rewind_reason = "forced rewind: current-branch attempt budget exhausted"
                    # Target is selected by the reflection prompt after the segment.
                    break

                obs_prompt = _build_observation_prompt(
                    observation=observation,
                    branch_attempts_used=branch_attempts_used,
                    game_position=current_position,
                    step_budget=step_budget,
                    budget_used=total_play_turns,
                    segment_max_turns=segment_max_turns,
                    actions_per_turn=actions_per_turn,
                    active_branch_events=segment_events,
                    action_is_valid=last_action_valid,
                    invalid_reason=invalid_reason,
                )
                active_messages.append(_message("user", obs_prompt))
                global_messages.append(_message("user", obs_prompt))
                messages = _copy_messages(active_messages)

                t_llm = time.perf_counter()
                try:
                    resp = await client.chat.completions.create(
                        model=config.model,
                        messages=messages,
                        **sampling,
                        timeout=120,
                    )
                    # from pprint import pprint
                    # pprint(messages)
                    # pprint(resp.choices[0].message.content)
                    # import pdb; pdb.set_trace()
                    llm_action_s += time.perf_counter() - t_llm
                    total_llm_calls += 1
                    total_play_turns += 1
                    branch_attempts_used += 1
                    content = resp.choices[0].message.content or ""
                except Exception as e:
                    failure = classify_llm_failure(e)
                    llm_action_s += time.perf_counter() - t_llm
                    total_llm_calls += 1
                    total_play_turns += 1
                    branch_attempts_used += 1
                    logger.warning(
                        "sokoban_rewind task %s segment %d turn %d: LLM failed: %s",
                        task.id,
                        segment_idx,
                        branch_attempts_used,
                        e,
                    )
                    segment_steps.append(
                        Step(
                            chat_completions=list(messages),
                            observation=obs_prompt,
                            model_response="",
                            action=failure.kind,
                            thought=failure.thought,
                        )
                    )
                    segment_events.append(
                        BranchEvent(
                            kind=failure.kind,
                            position_before=current_position,
                            position_after=current_position,
                            action=failure.kind,
                            outcome=failure.outcome,
                            observation_after=observation,
                        )
                    )
                    force_rewind = True
                    rewind_kind = "forced"
                    force_rewind_reason = failure.rewind_reason
                    # Target is selected by the reflection prompt after the segment.
                    break

                active_messages.append(_message("assistant", content))
                global_messages.append(_message("assistant", content))
                messages = _copy_messages(active_messages)

                finish_reason = resp.choices[0].finish_reason
                if finish_reason == "length":
                    segment_steps.append(
                        Step(
                            chat_completions=list(messages),
                            observation=obs_prompt,
                            model_response=content,
                            action="truncated",
                            thought=content,
                        )
                    )
                    segment_events.append(
                        BranchEvent(
                            kind="truncated",
                            position_before=current_position,
                            position_after=current_position,
                            action="response truncated",
                            outcome="token limit hit before a valid complete action",
                            observation_after=observation,
                            model_response=content,
                        )
                    )
                    force_rewind = True
                    rewind_kind = "forced"
                    force_rewind_reason = "forced rewind: response truncated by token length"
                    # Target is selected by the reflection prompt after the segment.
                    break

                command = parse_agent_command(content, actions_per_turn=actions_per_turn)

                if command.kind == "invalid":
                    invalid_reason = command.error or f"could not parse final action: {command.raw!r}"
                    last_action_valid = False
                    segment_steps.append(
                        Step(
                            chat_completions=list(messages),
                            observation=obs_prompt,
                            model_response=content,
                            action="invalid",
                            thought=content,
                        )
                    )
                    segment_events.append(
                        BranchEvent(
                            kind="invalid",
                            position_before=current_position,
                            position_after=current_position,
                            action=command.raw or "(empty)",
                            outcome=invalid_reason,
                            observation_after=observation,
                            model_response=content,
                        )
                    )
                    # Do not call env.step, do not save a checkpoint, do not advance C_k.
                    continue

                last_action_valid = True
                invalid_reason = ""

                if command.kind == "rewind":
                    target = command.rewind_to
                    if target is None or not (0 <= target < current_position):
                        invalid_reason = (
                            f"invalid rewind target {command.raw!r}; valid targets are "
                            f"{_format_checkpoint_range(current_position)}"
                        )
                        last_action_valid = False
                        segment_steps.append(
                            Step(
                                chat_completions=list(messages),
                                observation=obs_prompt,
                                model_response=content,
                                action="invalid_rewind",
                                thought=content,
                            )
                        )
                        segment_events.append(
                            BranchEvent(
                                kind="invalid_rewind",
                                position_before=current_position,
                                position_after=current_position,
                                action=command.raw,
                                outcome=invalid_reason,
                                observation_after=observation,
                                model_response=content,
                            )
                        )
                        continue

                    rewind_target = target
                    force_rewind = True
                    rewind_kind = "model"
                    force_rewind_reason = f"model rewind: requested rewind to C_{target}"
                    segment_steps.append(
                        Step(
                            chat_completions=list(messages),
                            observation=obs_prompt,
                            model_response=content,
                            action=f"rewind to C_{target}",
                            thought=content,
                        )
                    )
                    segment_events.append(
                        BranchEvent(
                            kind="model_rewind",
                            position_before=current_position,
                            position_after=current_position,
                            action=f"rewind to C_{target}",
                            outcome="model requested rollback",
                            observation_after=observation,
                            model_response=content,
                        )
                    )
                    break

                # --- Normal Sokoban move sequence ---
                assert command.kind == "move"
                actions = command.actions or []
                action_sequence_label = _format_action_sequence(actions)

                segment_steps.append(
                    Step(
                        chat_completions=list(messages),
                        observation=obs_prompt,
                        model_response=content,
                        action=action_sequence_label,
                        thought=content,
                    )
                )

                env_done_not_won = False
                step_start_position = len(interaction_history)
                step_start_observation = observation
                primitive_events: list[dict[str, str]] = []
                for action in actions:
                    if getattr(session, "step_budget_remaining", None) is not None:
                        remaining = int(session.step_budget_remaining)
                        if remaining <= 0:
                            exhausted_reason = "primitive step budget exhausted"
                            break
                    previous_observation = observation
                    action_label = _action_label(action)

                    try:
                        t_env = time.perf_counter()
                        result = await session.step(action)
                        env_step_s += time.perf_counter() - t_env
                    except Exception as e:
                        logger.warning(
                            "sokoban_rewind task %s: env.step failed during C_%d action batch at primitive action %s: %s",
                            task.id,
                            step_start_position,
                            action_label,
                            e,
                        )
                        segment_events.append(
                            BranchEvent(
                                kind="env_step_failed",
                                position_before=step_start_position,
                                position_after=step_start_position,
                                action=action_label,
                                outcome=str(e),
                                observation_after=previous_observation,
                                model_response=content,
                            )
                        )
                        force_rewind = True
                        rewind_kind = "forced"
                        force_rewind_reason = "forced rewind: environment step failed"
                        # Target is selected by the reflection prompt after the segment.
                        break

                    observation = result.observation
                    won = bool(result.won)
                    done = bool(result.done)
                    cum_reward += float(getattr(result, "reward", 0.0) or 0.0)

                    if won:
                        outcome = "won!"
                    elif done and not won:
                        outcome = "dead state (done=True, won=False)"
                    elif observation == previous_observation:
                        outcome = "no visible change (blocked or ineffective move)"
                    else:
                        outcome = "moved"

                    primitive_events.append(
                        {
                            "action": action_label,
                            "outcome": outcome,
                            "observation_before": previous_observation,
                            "observation_after": observation,
                        }
                    )

                    if won:
                        break
                    if done and not won:
                        env_done_not_won = True
                        break

                if primitive_events:
                    checkpoint_position = step_start_position + 1
                    checkpoint_id = await _save_checkpoint_for_position(session, checkpoint_position)
                    position_to_checkpoint_id.append(checkpoint_id)
                    cum_reward_checkpoints.append(cum_reward)
                    step_observations.append(observation)

                    if won:
                        step_outcome = "won!"
                    elif env_done_not_won:
                        step_outcome = "dead state (done=True, won=False)"
                    elif observation == step_start_observation:
                        step_outcome = "no visible change (blocked or ineffective move sequence)"
                    else:
                        step_outcome = "moved"

                    history_entry = {
                        "position_before": str(step_start_position),
                        "position_after": str(checkpoint_position),
                        "observation_before": step_start_observation,
                        "observation_after": observation,
                        "observation": observation,  # backward-compatible alias
                        "outcome": step_outcome,
                        "action": action_sequence_label,
                        "primitive_actions": ", ".join(event["action"] for event in primitive_events),
                        "primitive_outcomes": " | ".join(
                            f"{event['action']} => {event['outcome']}" for event in primitive_events
                        ),
                    }
                    interaction_history.append(history_entry)
                    message_checkpoints.append(_copy_messages(active_messages))
                    branch_attempt_checkpoints.append(branch_attempts_used)

                    primitive_outcomes = history_entry["primitive_outcomes"]
                    segment_events.append(
                        BranchEvent(
                            kind="env_step",
                            position_before=step_start_position,
                            position_after=checkpoint_position,
                            action=action_sequence_label,
                            outcome=f"{step_outcome}; primitives: {primitive_outcomes}",
                            observation_after=observation,
                            model_response=content,
                        )
                    )

                    # Development-time invariant guard. Keep this lightweight and non-fatal.
                    if len(step_observations) != len(interaction_history) + 1:
                        logger.warning(
                            "position invariant drift: observations=%d history=%d",
                            len(step_observations),
                            len(interaction_history),
                        )

                if won:
                    break

                if force_rewind:
                    break

                if env_done_not_won:
                    force_rewind = True
                    rewind_kind = "forced"
                    force_rewind_reason = "forced rewind: environment reached dead state (done=True, won=False)"
                    # Target is selected by the reflection prompt after the segment.
                    break

            # --- End of current segment inner loop ---

            if won:
                if segment_steps:
                    trajectories.append(
                        Trajectory(
                            name=_segment_traj_name(traj_grouping, segment_idx),
                            steps=segment_steps,
                            reward=None,
                            metadata={"role": PLAY_ROLE, "segment_idx": segment_idx, "cum_reward_at_end": cum_reward},
                        )
                    )
                break

            if exhausted_reason is not None:
                if segment_steps:
                    trajectories.append(
                        Trajectory(
                            name=_segment_traj_name(traj_grouping, segment_idx),
                            steps=segment_steps,
                            reward=None,
                            metadata={"role": PLAY_ROLE, "segment_idx": segment_idx, "cum_reward_at_end": cum_reward},
                        )
                    )
                break

            if not force_rewind:
                # Defensive fallback: the inner loop should normally end only via
                # win, exhaustion, or rewind trigger.
                exhausted_reason = exhausted_reason or "segment ended without rewind trigger"
                if segment_steps:
                    trajectories.append(
                        Trajectory(
                            name=_segment_traj_name(traj_grouping, segment_idx),
                            steps=segment_steps,
                            reward=None,
                            metadata={"role": PLAY_ROLE, "segment_idx": segment_idx, "cum_reward_at_end": cum_reward},
                        )
                    )
                break

            # If there is no room for another segment, keep the current trajectory
            # and stop. Reflection would not be used by any later attempt.
            if segment_idx + 1 >= max_segments:
                exhausted_reason = "segment limit exhausted"
                if segment_steps:
                    trajectories.append(
                        Trajectory(
                            name=_segment_traj_name(traj_grouping, segment_idx),
                            steps=segment_steps,
                            reward=None,
                            metadata={"role": PLAY_ROLE, "segment_idx": segment_idx, "cum_reward_at_end": cum_reward},
                        )
                    )
                break

            current_position = len(interaction_history)
            current_observation = observation
            rewind_to: int | None = rewind_target
            perform_env_rewind = True
            checkpoint_id_used: Any = None

            if rewind_kind == "forced" and rewind_to is None and current_position <= 0:
                rewind_to = current_position
                perform_env_rewind = False
                rewind_kind = "context_fold"
                force_rewind_reason = (
                    f"{force_rewind_reason}; no previous checkpoint exists, so fold context at C_0"
                )

            if rewind_to is None and rewind_kind != "forced":
                # Defensive guard. This should be unreachable, but avoids a hidden
                # system-selected C_0 if future code adds a new trigger path.
                exhausted_reason = "rewind requested but no checkpoint target was selected"
                if segment_steps:
                    trajectories.append(
                        Trajectory(
                            name=_segment_traj_name(traj_grouping, segment_idx),
                            steps=segment_steps,
                            reward=None,
                            metadata={"role": PLAY_ROLE, "segment_idx": segment_idx, "cum_reward_at_end": cum_reward},
                        )
                    )
                break

            if rewind_to is not None and perform_env_rewind and not (0 <= rewind_to < current_position):
                exhausted_reason = (
                    f"invalid selected rewind target C_{rewind_to}; "
                    f"valid targets are {_format_checkpoint_range(current_position)}"
                )
                if segment_steps:
                    trajectories.append(
                        Trajectory(
                            name=_segment_traj_name(traj_grouping, segment_idx),
                            steps=segment_steps,
                            reward=None,
                            metadata={"role": PLAY_ROLE, "segment_idx": segment_idx, "cum_reward_at_end": cum_reward},
                        )
                    )
                break

            if rewind_to is not None and not perform_env_rewind and rewind_to != current_position:
                exhausted_reason = "context-fold target must equal the current checkpoint"
                if segment_steps:
                    trajectories.append(
                        Trajectory(
                            name=_segment_traj_name(traj_grouping, segment_idx),
                            steps=segment_steps,
                            reward=None,
                            metadata={"role": PLAY_ROLE, "segment_idx": segment_idx, "cum_reward_at_end": cum_reward},
                        )
                    )
                break

            if segment_steps:
                trajectories.append(
                    Trajectory(
                        name=_segment_traj_name(traj_grouping, segment_idx),
                        steps=segment_steps,
                        reward=None,
                        metadata={"role": PLAY_ROLE, "segment_idx": segment_idx, "cum_reward_at_end": cum_reward},
                    )
                )

            rewind_to_obs = (
                step_observations[rewind_to]
                if rewind_to is not None and rewind_to < len(step_observations)
                else None
            )

            # The prior reflection (if any) lives at the prefix of the branch
            # origin we will reflect from: the model's target when known, else the
            # forced segment's start. Captured before truncation so the new
            # reflection can fold it into its branch history.
            prior_reflection_origin = rewind_to if rewind_to is not None else segment_start_position
            prior_reflection_at_target = (
                _find_trailing_reflection(message_checkpoints[prior_reflection_origin])
                if 0 <= prior_reflection_origin < len(message_checkpoints)
                else None
            )

            # --- Knowledge moves forward: compress failed branch into branch memory. ---
            branch_memory = ""
            reflect_content = ""
            reflect_prompt = ""
            reflect_rewind_target: int | None = None
            reflect_parse_error = ""
            if total_llm_calls < max_total_turns:
                t_reflect = time.perf_counter()
                total_llm_calls += 1
                total_reflection_turns += 1
                (
                    reflect_content,
                    branch_memory,
                    reflect_prompt,
                    reflect_rewind_target,
                    reflect_parse_error,
                ) = await _do_reflection(
                    client=client,
                    model=config.model,
                    sampling=sampling,
                    rewind_to=rewind_to,
                    current_position=current_position,
                    rewind_to_obs=rewind_to_obs,
                    current_observation=current_observation,
                    step_observations=step_observations,
                    interaction_history=interaction_history,
                    segment_events=segment_events,
                    reflection_history=reflection_history,
                    rewind_reason=force_rewind_reason,
                    task_id=task.id,
                    history_start=segment_start_position,
                    prior_reflection_at_target=prior_reflection_at_target,
                    action_only=action_only,
                )
                llm_reflection_s += time.perf_counter() - t_reflect
                if reflect_content:
                    reflection_history.append(_truncate_text(reflect_content.strip(), MAX_REFLECTION_CHARS))

                reflect_step = Step(
                    chat_completions=[
                        {"role": "user", "content": reflect_prompt},
                        {"role": "assistant", "content": reflect_content},
                    ],
                    observation=reflect_prompt,
                    model_response=reflect_content,
                    action=(
                        f"reflect_branch_memory_and_rewind_to_C_{reflect_rewind_target}"
                        if reflect_rewind_target is not None
                        else "reflect_branch_memory"
                    ),
                    thought=reflect_content,
                )
                trajectories.append(
                    Trajectory(
                        name=_reflect_traj_name(traj_grouping, segment_idx),
                        steps=[reflect_step],
                        reward=None,
                        metadata={"role": REFLECT_ROLE, "segment_idx": segment_idx, "cum_reward_at_trigger": cum_reward},
                    )
                )
            else:
                logger.warning(
                    "sokoban_rewind task %s: skipped reflection because LLM turn cap was reached",
                    task.id,
                )

            if perform_env_rewind:
                if rewind_to is None:
                    if reflect_rewind_target is None:
                        exhausted_reason = (
                            "reflection did not choose a valid rewind target: "
                            f"{reflect_parse_error or 'missing final rewind action'}"
                        )
                        break
                    rewind_to = reflect_rewind_target
                elif reflect_rewind_target is not None and reflect_rewind_target != rewind_to:
                    exhausted_reason = (
                        f"reflection target C_{reflect_rewind_target} did not match "
                        f"requested rewind target C_{rewind_to}"
                    )
                    break

                segment_events.append(
                    BranchEvent(
                        kind="reflection_rewind_target" if reflect_rewind_target is not None else "rewind_target",
                        position_before=current_position,
                        position_after=rewind_to,
                        action=f"rewind to C_{rewind_to}",
                        outcome=(
                            "reflection selected rollback checkpoint"
                            if reflect_rewind_target is not None
                            else "using existing rollback checkpoint"
                        ),
                        observation_after=step_observations[rewind_to],
                        model_response=reflect_content,
                    )
                )

            if not branch_memory:
                if rewind_to is None:
                    exhausted_reason = "rewind requested but no checkpoint target was selected"
                    break
                branch_memory = _fallback_branch_memory(
                    rewind_reason=force_rewind_reason,
                    rewind_to=rewind_to,
                    current_position=current_position,
                    interaction_history=interaction_history,
                    action_only=action_only,
                )

            branch_memory = _truncate_text(branch_memory, MAX_BRANCH_MEMORY_CHARS)
            if branch_memory:
                # Reflections are NOT injected globally. They are appended to the
                # rewind target's conversation prefix below, so only C_{rewind_to}
                # and its new descendants see this memory.
                branch_memory_records.append(
                    {
                        "segment": segment_idx,
                        "rewind_from": current_position,
                        "rewind_to": rewind_to,
                        "kind": rewind_kind or "forced",
                        "reason": force_rewind_reason,
                        "memory": branch_memory,
                    }
                )

            # --- Environment travels back, unless this is an explicit context fold. ---
            if perform_env_rewind:
                try:
                    rw_result, checkpoint_id_used = await _rewind_session_to_position(
                        session=session,
                        position_to_checkpoint_id=position_to_checkpoint_id,
                        position=rewind_to,
                    )
                    observation = rw_result.observation
                except ValueError as e:
                    # Do not silently fall back to C_0. If the model selected C_j,
                    # either we rewind to C_j or terminate with an explicit error.
                    logger.warning(
                        "sokoban_rewind task %s: rewind to C_%d failed: %s",
                        task.id,
                        rewind_to,
                        e,
                    )
                    exhausted_reason = f"rewind to C_{rewind_to} failed: {e}"
                    break
            else:
                checkpoint_id_used = (
                    position_to_checkpoint_id[rewind_to]
                    if rewind_to < len(position_to_checkpoint_id)
                    else "current"
                )
                observation = (
                    step_observations[rewind_to]
                    if rewind_to < len(step_observations)
                    else current_observation
                )

            if rewind_kind == "model":
                model_rewinds += 1
            elif rewind_kind == "context_fold":
                forced_context_folds += 1
            else:
                forced_rewinds += 1

            rewind_log.append(
                {
                    "segment": segment_idx,
                    "from": current_position,
                    "to": rewind_to,
                    "checkpoint_id_used": str(checkpoint_id_used),
                    "performed_env_rewind": perform_env_rewind,
                    "kind": rewind_kind or "forced",
                    "reason": force_rewind_reason,
                    "memory": branch_memory,
                }
            )
            global_messages.append(
                _message(
                    "system",
                    (
                        "[debug] rewind "
                        f"{rewind_kind or 'forced'} from C_{current_position} to C_{rewind_to}; "
                        f"performed_env_rewind={perform_env_rewind}; reason={force_rewind_reason}"
                    ),
                )
            )

            # Truncate active path and restore the active chat prefix for C_j.
            # This drops every checkpoint after C_{rewind_to}, so reflections that
            # were attached to abandoned-sibling checkpoints disappear with them.
            step_observations = step_observations[: rewind_to + 1]
            position_to_checkpoint_id = position_to_checkpoint_id[: rewind_to + 1]
            interaction_history = interaction_history[:rewind_to]
            message_checkpoints = message_checkpoints[: rewind_to + 1]
            branch_attempt_checkpoints = branch_attempt_checkpoints[: rewind_to + 1]
            cum_reward_checkpoints = cum_reward_checkpoints[: rewind_to + 1]
            cum_reward = cum_reward_checkpoints[rewind_to]
            active_messages = _copy_messages(message_checkpoints[rewind_to])
            branch_attempts_used = branch_attempt_checkpoints[rewind_to]

            # A previous rewind to this same checkpoint may have left its reflection
            # on the prefix; that prior memory is now folded into this reflection's
            # branch history, so drop it here to keep exactly one reflection riding
            # C_{rewind_to} instead of stacking on every re-rewind.
            _strip_trailing_reflections(active_messages)

            # Attach the reflection to C_{rewind_to}'s prefix. Re-snapshotting it
            # into message_checkpoints[rewind_to] means every future turn at C_j --
            # and every checkpoint created on the new branch (which snapshots
            # active_messages at save time) -- inherits this memory, while the
            # abandoned branch's reflections stay dropped by the truncation above.
            if branch_memory:
                reflection_msg = _message(
                    "user",
                    _format_reflection_injection(
                        branch_memory,
                        rewind_from=current_position,
                        rewind_to=rewind_to,
                    ),
                )
                active_messages.append(reflection_msg)
                global_messages.append(reflection_msg)
                message_checkpoints[rewind_to] = _copy_messages(active_messages)

            segment_idx += 1

        # --- Compute rewards: cross-segment backward discounting. ---
        _assign_cross_segment_rewards(
            trajectories=trajectories,
            won=won,
            traj_gamma=traj_gamma,
        )

        return Episode(
            trajectories=trajectories,
            metrics={
                "time/env_init_s": env_init_s,
                "time/env_step_s": env_step_s,
                "time/llm_action_s": llm_action_s,
                "time/llm_reflection_s": llm_reflection_s,
            },
            artifacts={
                "won": won,
                "segments": len([t for t in trajectories if (t.metadata or {}).get("role") == PLAY_ROLE]),
                "total_play_turns": total_play_turns,
                "model_step_budget_used": total_play_turns,
                "branch_attempts_used": branch_attempts_used,
                "branch_attempt_budget": segment_max_turns,
                "total_reflection_turns": total_reflection_turns,
                "total_llm_calls": total_llm_calls,
                "total_env_steps": session.total_steps,
                "step_budget": step_budget,
                "primitive_step_budget": primitive_step_budget,
                "max_segments": max_segments,
                "max_total_turns": max_total_turns,
                "exhausted_reason": exhausted_reason,
                "rewinds": len(rewind_log),
                "forced_rewinds": forced_rewinds,
                "model_rewinds": model_rewinds,
                "forced_context_folds": forced_context_folds,
                "branch_memories": branch_memory_records,
                "reflection_history": reflection_history,
                "rewind_log": rewind_log,
                "global_messages": global_messages,
                "final_game_position": len(interaction_history),
                "active_path_len": len(interaction_history),
                "checkpoint_map": [str(x) for x in position_to_checkpoint_id],
                "dim_room": dim_room,
                "num_boxes": num_boxes,
                "actions_per_turn": actions_per_turn,
                "mode": mode,
            },
            is_correct=won,
        )

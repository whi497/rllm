"""Rewind-choice MiniSweeper agent flow.

This mirrors the Sokoban rewind-choice flow semantics:
  - Environment state can rewind to a visible checkpoint C_k.
  - Knowledge only moves forward: failed branches are compressed into branch
    memories and injected into later attempts.
  - Forced rewind asks the model to choose the rollback checkpoint instead of
    unconditionally returning to the segment start.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
import textwrap
import time
from typing import Any, Literal

from env_service import create_env_session, parse_remark
from env_service.minesweeper import MineSweeperEnv
from openai import AsyncOpenAI

import rllm
from tbmf.flow_utils import classify_llm_failure
from rllm.types import AgentConfig, Episode, Step, Task, Trajectory

try:
    from ..prepare_minisweeper_data import LAMER_MINISWEEPER_CONFIG
    from .minisweeper_flow import parse_action
except (ImportError, ValueError):
    from prepare_minisweeper_data import LAMER_MINISWEEPER_CONFIG
    from minisweeper_flow import parse_action

logger = logging.getLogger(__name__)

DEFAULT_STEP_BUDGET = 60
DEFAULT_TRAJ_GAMMA = 0.7
DEFAULT_SEGMENT_MAX_TURNS = 20
DEFAULT_MAX_SEGMENTS = 6
DEFAULT_MAX_TOTAL_TURNS = 64

MAX_BRANCH_MEMORY_CHARS = 4800
MAX_REFLECTIONS_IN_CONTEXT = 3
MAX_REFLECTION_CHARS = 4800
MAX_BRANCH_HISTORY_CHARS = 24000
MAX_ACTIVE_BRANCH_EVENTS = 12
MAX_MODEL_RESPONSE_IN_HISTORY_CHARS = 900

_ACTION_TAG_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.IGNORECASE | re.DOTALL)
_REWIND_FULL_RE = re.compile(
    r"rewind\s+to\s+(?:C\s*[_-]?\s*)?(\d+)\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AgentCommand:
    kind: Literal["reveal", "rewind", "invalid"]
    cell: tuple[int, int] | None = None
    rewind_to: int | None = None
    raw: str = ""
    error: str = ""


@dataclass
class BranchEvent:
    kind: str
    position_before: int
    position_after: int
    action: str
    outcome: str
    observation_after: str
    model_response: str = ""


REWIND_SYSTEM_PROMPT = """\
You are an expert agent operating in the Minesweeper game with REWIND capability.
You will be given a {board_size} by {board_size} board, with {n_mines} hidden mines.
Rows and columns are indexed from 1 to {board_size}.

The task is guaranteed to be solvable. You must never conclude that the task is unsolvable, hopeless, or impossible, and you must never give up. Even if some branches appear unpromising or lead to dead ends, you can still learn from them, rewind when necessary, and continue exploring.
Your environment state may move backward through rewind, but your knowledge must keep moving forward. Use each failed attempt to refine your understanding, avoid repeating the same mistakes, and make better decisions in future branches until the task is solved.


# Cell States
- Unopened cells (?): hidden cells that may contain mines.
- Blank cells (.): opened safe cells with no neighboring mines.
- Numbered cells (1-8): opened safe cells showing how many neighboring cells contain mines.
- Mine cells (*): opened mine cells; revealing one loses the game.

# Goal
Reveal all Unopened and non-mine cells without hitting any mine.

# Reveal Rules
- Choose exactly ONE unopened cell (?) to reveal per model action step.
- Blank cells auto-cascade to reveal contiguous blanks and bordering numbered cells, you did not need to reveal the blank cells.
- Numbered cells reveal only that single cell.
- Mine cells end the current branch in failure.

# Checkpoints and Rewind Capability
You operate under two budgets, and you should manage them differently.


The first is a global action budget: {step_budget} model action steps for the entire episode. Each model action step reveals exactly one cell. This is a hard ceiling shared across the whole episode. You may use it for exploration, trial and error, and ultimately solving the task. This budget is accumulated globally and is never refunded by rewinding.

The second is a task-solve budget: {branch_attempt_budget} model action steps for making progress along the current active path. You should aim to solve the task within this allowance. Unlike the global action budget, this budget is tied to the checkpoint path. When you rewind to checkpoint C_j, the task-solve budget is rolled back to the amount already consumed at C_j. In other words, rewinding to C_j refunds the task-solve steps spent after C_j, while everything consumed up to C_j remains spent.

Use the global action budget and the task-solve budget together to balance exploration, learning from failed branches, trial and error, and direct progress toward solving the task.

Checkpoints track the current path as C_0, C_1, ..., C_k:
- C_0 is the initial board state.
- C_i is the state reached after i executed reveal actions on the current path.

Once you have executed at least one reveal action, you may travel back to any
earlier checkpoint:
- `<action>rewind to C_j</action>`, where 0 <= j < the current checkpoint index.
- `<action>rewind to j</action>` is equivalent.

Rewinding restores the environment to checkpoint C_j and rolls your branch attempt budget
back to its value at C_j, but what you learned on the abandoned branch is not lost — it
persists as memory. you can rewind when you find 
* a better plan to start at C_j 
* recognize the current branch is stuck(an action trap, an invalid loop, a dead end, or unproductive exploration) 
* You are in good path to success but find insufficient budget to make progress(the task-solve budget is exhausted) and want to move to current state in less turns and context.

# Response Format
- First reason step-by-step about numbered clues, candidate mines, and safe cells.
- Then reveal one unopened (?) cell(number cell and Blank cells is safe but make no progress): `<action>(row, col)</action>`.
- OR rewind to a previous checkpoint: `<action>rewind to C_j</action>`.
Only the final `<action>...</action>` tag will be executed.
"""


REWIND_REFLECT_PROMPT = """\
You are an expert Minesweeper player reflecting on a branch attempt.
Board: {board_size} by {board_size}, {n_mines} mines. Rows/columns indexed 1 to {board_size}.

# Cell States
- Unopened cells (?): hidden cells that may contain mines.
- Blank cells (.): opened safe cells with no neighboring mines.
- Numbered cells (1-8): opened safe cells showing how many neighboring cells contain mines.
- Mine cells (*): opened mine cells; revealing one loses the game.

# Reveal Rules
- Choose exactly ONE unopened cell (?) to reveal per model action step.
- Blank cells auto-cascade to reveal contiguous blanks and bordering numbered cells, you did not need to reveal the blank cells.
- Numbered cells reveal only that single cell.
- Mine cells end the current branch in failure.

# Core rollback semantics
The game was at step C_{rewind_from_step} when you decided to rewind back to step C_{rewind_to_step}.

Thus the env state returns to step C_{rewind_to_step}, Your goal is to push the knowledge state forward.
compress the branch attempt into a small branch memory that record what have you done that lead to what result,
what have you learned, what you should avoid and what you should do next.
Do not preserve the full trajectory. Preserve only information useful for future decisions
the core target is to help move closer to the target during the next new attempt or try explore other states.

# Context
{rewind_target_instruction}

# Rewind trigger
{rewind_reason}

# Relevant checkpoint states
{checkpoint_context}

# State at C_{rewind_from_step} before rewinding
{current_observation}

# Branch history from C_{rewind_to_step} to C_{rewind_from_step}
{history_after_rewind}

# Your task
Reflect on what went wrong and produce a concise branch memory for the next attempt based on the actions you have done and the environment feedback.
Focus on:
- revealed mine cells that should not be repeated;
- invalid/already-open/out-of-range reveals;
- numbered clue constraints and specific cells that are likely mines or safe;
- whether the branch should restart before or after a useful clue reveal;
- a concrete improved plan from the checkpoint you return to. you could move faster as you already know  some of the state and the knowledges.

Include a compact branch memory inside <remark> </remark> tags.
Use this structure, replacing C_j with the checkpoint you choose or confirm:
<remark>
# History Attempt from C_{rewind_to_step} to C_{rewind_from_step}
State at C_{rewind_to_step}: 
{rewind_to_observation}
State at C_{rewind_from_step}: 
{rewind_from_observation}

Actions I have done: ...
What's bad about the actions: ...
How to move forward: ...
Next new plan from C_j that can move faster and avoid the same mistakes: ...
</remark>
"""

# # Prior reflection outputs
# {reflection_history_context}



REWIND_REFLECT_PROMPT_WITH_CKPT_CHOICE = """\
You are an expert Minesweeper player reflecting on a branch attempt.
Board: {board_size} by {board_size}, {n_mines} mines. Rows/columns indexed 1 to {board_size}.

# Cell States
- Unopened cells (?): hidden cells that may contain mines.
- Blank cells (.): opened safe cells with no neighboring mines.
- Numbered cells (1-8): opened safe cells showing how many neighboring cells contain mines.
- Mine cells (*): opened mine cells; revealing one loses the game.

# Reveal Rules
- Choose exactly ONE unopened cell (?) to reveal per model action step.
- Blank cells auto-cascade to reveal contiguous blanks and bordering numbered cells, you did not need to reveal the blank cells.
- Numbered cells reveal only that single cell.
- Mine cells end the current branch in failure.

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


# Your task
Reflect on the branch attempt and produce a concise branch memory for the next new attempt direction from the checkpoint you return to.

Focus on:
- revealed mine cells that should not be repeated;
- invalid/already-open/out-of-range reveals;
- numbered clue constraints and specific cells that are likely mines or safe;
- whether the branch should restart before or after a useful clue reveal;
- a concrete improved plan from the checkpoint you return to. you could move faster as you already know  some of the state and the knowledges.

Include a compact branch memory inside <remark> </remark> tags.
Use this structure, replacing C_j with the checkpoint you choose or confirm:
<remark>
# History Attempt from C_{rewind_to_step} to C_{rewind_from_step}
State at C_{rewind_to_step}: 
{rewind_to_observation}
State at C_{rewind_from_step}: 
{rewind_from_observation}
Actions I have done: ...
What's bad about the actions: ...
How to move forward: ...
Next new plan from C_j that can move faster and avoid the same mistakes: ...
</remark>

End with exactly one final action tag selecting the checkpoint to restore:
<action>rewind to C_j</action>
"""
# # Prior reflection outputs
# {reflection_history_context}
def _copy_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    return [dict(message) for message in messages]


def _message(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


def _extract_checkpoint_id(obj: Any, fallback: Any = None) -> Any:
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
                "checkpoint id %r failed for visible C_%d; falling back to raw position id",
                checkpoint_id,
                position,
            )
            result = await session.rewind(position)
            return result, position
        raise


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


def _build_system_prompt(board_size: int, n_mines: int, step_budget: int, branch_attempt_budget: int) -> str:
    return REWIND_SYSTEM_PROMPT.format(
        board_size=board_size,
        n_mines=n_mines,
        step_budget=step_budget,
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
    for i, reflection in enumerate(reflection_history[-MAX_REFLECTIONS_IN_CONTEXT:], start=1):
        lines.append(f"Reflection #{i}: {_truncate_text(reflection, MAX_REFLECTION_CHARS)}")
    return "\n".join(lines)


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
    active_branch_events: list[BranchEvent] | None = None,
    action_is_valid: bool = True,
    invalid_reason: str = "",
) -> str:
    budget_remaining = max(0, step_budget - budget_used)
    branch_remaining = max(0, segment_max_turns - branch_attempts_used)
    segment_turn = branch_attempts_used + 1
    header = (
        "Initial observation at checkpoint C_0:"
        if game_position == 0 and budget_used == 0
        else f"Current observation at checkpoint C_{game_position} (segment turn {segment_turn}):"
    )

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
        "Choose exactly one unopened cell to reveal inside one final action tag, "
        "e.g. <action>(2, 3)</action>, or travel back: <action>rewind to C_j</action>."
    )


def _build_history_after_rewind(
    interaction_history: list[dict[str, str]],
    extra_events: list[BranchEvent] | None,
    start: int,
    end: int,
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
                    lines.append("  Model response excerpt:")
                    lines.append(
                        textwrap.indent(
                            _truncate_text(event.model_response, MAX_MODEL_RESPONSE_IN_HISTORY_CHARS),
                            "  ",
                        )
                    )

    history = "\n".join(lines).strip() if lines else "(no branch history)"
    return _truncate_text(history, MAX_BRANCH_HISTORY_CHARS)


def _fallback_branch_memory(
    rewind_reason: str,
    rewind_to: int,
    current_position: int,
    interaction_history: list[dict[str, str]],
) -> str:
    recent = _build_history_after_rewind(
        interaction_history=interaction_history,
        extra_events=None,
        start=rewind_to,
        end=current_position,
    )
    return _truncate_text(
        (
            f"Failure reason: {rewind_reason}\n"
            f"Useful facts learned: branch C_{rewind_to}->C_{current_position} failed.\n"
            f"Avoid repeating: {recent}\n"
            f"Plan from C_{rewind_to}: choose a different safe-cell deduction from this checkpoint."
        ),
        MAX_BRANCH_MEMORY_CHARS,
    )


def _extract_final_action_text(content: str) -> str:
    matches = _ACTION_TAG_RE.findall(content or "")
    if matches:
        return matches[-1].strip()
    lines = [line.strip() for line in (content or "").splitlines() if line.strip()]
    return lines[-1] if lines else ""


def parse_agent_command(content: str, board_size: int, observation: str | None = None) -> AgentCommand:
    # ``observation`` is accepted for debugger compatibility; this flow does not use it.
    raw = _extract_final_action_text(content)
    if not raw:
        return AgentCommand(kind="invalid", raw=raw, error="missing final <action> tag")

    rewind_match = _REWIND_FULL_RE.fullmatch(raw)
    if rewind_match:
        return AgentCommand(kind="rewind", rewind_to=int(rewind_match.group(1)), raw=raw)

    action = parse_action(f"<action>{raw}</action>", board_size=board_size)
    if action is None:
        return AgentCommand(kind="invalid", raw=raw, error="could not parse a valid in-bounds coordinate")
    return AgentCommand(kind="reveal", cell=action, raw=raw)


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
    board_size: int,
    n_mines: int,
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
) -> tuple[str, str, str, int | None, str]:
    def _obs_at(position: int) -> str:
        if step_observations is not None and 0 <= position < len(step_observations):
            return step_observations[position]
        return current_observation

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
        # Forced/choice path: the rollback target is not chosen yet. The branch
        # history is listed from where this segment began, and the model selects
        # the checkpoint to restore in its reflection response.
        prompt_history_start = (
            max(0, min(history_start, current_position)) if history_start is not None else 0
        )
        history_after = _build_history_after_rewind(
            interaction_history=interaction_history,
            extra_events=segment_events,
            start=prompt_history_start,
            end=current_position,
        )
        history_after = _fold_prior_reflection(history_after, prompt_history_start)
        checkpoint_context = _build_checkpoint_choice_context(
            step_observations=step_observations,
            current_position=current_position,
            fallback_observation=current_observation,
        )
        reflect_prompt = REWIND_REFLECT_PROMPT_WITH_CKPT_CHOICE.format(
            board_size=board_size,
            n_mines=n_mines,
            rewind_from_step=current_position,
            rewind_to_step=prompt_history_start,
            valid_targets=_format_checkpoint_range(current_position),
            checkpoint_context=checkpoint_context,
            current_observation=current_observation,
            history_start_step=prompt_history_start,
            history_after_rewind=history_after,
            reflection_history_context=_build_reflection_history_context(reflection_history),
            rewind_reason=rewind_reason,
            rewind_to_observation=_obs_at(prompt_history_start),
            rewind_from_observation=current_observation,
        )
    else:
        # Model rewind: the destination C_{rewind_to} is known. rewind_to_step is
        # literally that destination and the branch history spans C_{rewind_to} ->
        # C_{current_position}.
        history_after = _build_history_after_rewind(
            interaction_history=interaction_history,
            extra_events=segment_events,
            start=rewind_to,
            end=current_position,
        )
        history_after = _fold_prior_reflection(history_after, rewind_to)
        rewind_target_instruction = (
            f"The environment will resume from C_{rewind_to} after this reflection."
        )
        checkpoint_context = f"State at C_{rewind_to} where execution will resume:\n{rewind_to_obs or ''}"
        reflect_prompt = REWIND_REFLECT_PROMPT.format(
            board_size=board_size,
            n_mines=n_mines,
            rewind_from_step=current_position,
            rewind_to_step=rewind_to,
            rewind_target_instruction=rewind_target_instruction,
            checkpoint_context=checkpoint_context,
            current_observation=current_observation,
            history_after_rewind=history_after,
            reflection_history_context=_build_reflection_history_context(reflection_history),
            rewind_reason=rewind_reason,
            rewind_to_observation=rewind_to_obs or "",
            rewind_from_observation=current_observation,
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
        logger.warning("minisweeper_rewind_choice task %s: reflection failed: %s", task_id, e)
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


def _parse_suffix_int(name: str, prefix: str, default: int = -1) -> int:
    if not name.startswith(prefix):
        return default
    suffix = name[len(prefix):]
    try:
        return int(suffix)
    except ValueError:
        match = re.search(r"(\d+)$", name)
        return int(match.group(1)) if match else default


def _mean_trajectory_reward(trajectories: list[Trajectory], name_prefix: str) -> float:
    rewards = [
        float(t.reward)
        for t in trajectories
        if t.name.startswith(name_prefix) and t.reward is not None
    ]
    return sum(rewards) / len(rewards) if rewards else 0.0


def _assign_cross_segment_rewards(
    trajectories: list[Trajectory],
    won: bool,
    traj_gamma: float,
) -> None:
    play_items: list[tuple[int, int]] = []
    reflect_items: list[tuple[int, int]] = []
    for idx, traj in enumerate(trajectories):
        if traj.name.startswith("minisweeper_seg"):
            play_items.append((_parse_suffix_int(traj.name, "minisweeper_seg", idx), idx))
        elif traj.name.startswith("minisweeper_reflect"):
            reflect_items.append((_parse_suffix_int(traj.name, "minisweeper_reflect", idx), idx))

    play_items.sort(key=lambda x: x[0])
    n_play = len(play_items)
    if n_play > 0:
        play_rewards = [0.0] * n_play
        if won:
            play_rewards[-1] = 1.0

        discounted = [0.0] * n_play
        discounted[-1] = play_rewards[-1]
        for order in range(n_play - 2, -1, -1):
            discounted[order] = play_rewards[order] + traj_gamma * discounted[order + 1]

        for order, (_, traj_idx) in enumerate(play_items):
            trajectories[traj_idx].reward = discounted[order]

        for reflect_seg_id, traj_idx in reflect_items:
            next_order = None
            for order, (play_seg_id, _) in enumerate(play_items):
                if play_seg_id > reflect_seg_id:
                    next_order = order
                    break
            trajectories[traj_idx].reward = discounted[next_order] if next_order is not None else 0.0

    for traj in trajectories:
        if traj.reward is None:
            traj.reward = 0.0


@rllm.rollout(name="minisweeper_rewind_prefix")
async def minisweeper_rewind_choice_flow(task: Task, config: AgentConfig) -> Episode:
    """Drive MiniSweeper with explicit rollback and persistent branch memory."""
    meta = task.metadata or {}

    seed = int(meta.get("seed", LAMER_MINISWEEPER_CONFIG["env_seed"]))
    board_size = int(meta.get("board_size", LAMER_MINISWEEPER_CONFIG["board_size"]))
    n_mines = int(meta.get("n_mines", LAMER_MINISWEEPER_CONFIG["n_mines"]))
    board_type = meta.get("board_type", LAMER_MINISWEEPER_CONFIG["board_type"])
    mode = meta.get("mode", LAMER_MINISWEEPER_CONFIG["mode"])
    puzzle_state = meta.get("puzzle_state")

    max_env_steps = int(meta.get("max_steps", LAMER_MINISWEEPER_CONFIG["max_steps"]))
    step_budget = int(meta.get("step_budget", meta.get("model_step_budget", DEFAULT_STEP_BUDGET)))
    segment_max_turns = int(meta.get("segment_max_turns", DEFAULT_SEGMENT_MAX_TURNS))
    # max_segments is the single source of truth for how many play segments (and
    # thus rewinds: rewinds = segments - 1) an episode may run. Default 6.
    max_segments = int(meta.get("max_segments", DEFAULT_MAX_SEGMENTS))
    max_total_turns = int(meta.get("max_total_turns", max(DEFAULT_MAX_TOTAL_TURNS, step_budget + max_segments)))
    traj_gamma = float(meta.get("traj_gamma", DEFAULT_TRAJ_GAMMA))

    client = AsyncOpenAI(base_url=config.base_url, api_key="EMPTY")
    sampling = {k: v for k, v in config.sampling_params.items() if k != "top_k"}

    # The env-level step budget must NOT impose an extra limit on top of the
    # model budgets. Every model action step reveals one cell, and rewinds let the
    # model spend reveals across many branches without refunding env steps, so the
    # worst case is one reveal for every model turn the controller may issue:
    # max_segments * segment_max_turns. Derive a budget that comfortably covers it
    # (with margin) so only the model-facing budgets gate the episode.
    env_step_budget = int(
        meta.get(
            "env_step_budget",
            max(
                step_budget,
                max_total_turns,
                max_segments * segment_max_turns,
                max_env_steps,
            )
            + max_segments,
        )
    )
    session = await create_env_session(
        MineSweeperEnv,
        session_mode="local",
        step_budget=env_step_budget,
        board_size=board_size,
        n_mines=n_mines,
        board_type=board_type,
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

        initial_checkpoint_id = _extract_checkpoint_id(
            reset_info,
            fallback=_infer_checkpoint_id_from_session(session, fallback=0),
        )
        step_observations: list[str] = [observation]
        position_to_checkpoint_id: list[Any] = [initial_checkpoint_id]
        interaction_history: list[dict[str, str]] = []

        reflection_history: list[str] = []
        branch_memory_records: list[dict[str, Any]] = []
        rewind_log: list[dict[str, Any]] = []

        system_prompt = _build_system_prompt(board_size, n_mines, step_budget, segment_max_turns)

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
        message_checkpoints: list[list[dict[str, str]]] = [_copy_messages(active_messages)]
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
                    break

                obs_prompt = _build_observation_prompt(
                    observation=observation,
                    branch_attempts_used=branch_attempts_used,
                    game_position=current_position,
                    step_budget=step_budget,
                    budget_used=total_play_turns,
                    segment_max_turns=segment_max_turns,
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
                        "minisweeper_rewind_choice task %s segment %d turn %d: LLM failed: %s",
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
                    break

                command = parse_agent_command(content, board_size=board_size)
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

                assert command.kind == "reveal"
                row, col = command.cell or (-1, -1)
                action_label = f"({row}, {col})"
                step_start_position = len(interaction_history)
                step_start_observation = observation

                segment_steps.append(
                    Step(
                        chat_completions=list(messages),
                        observation=obs_prompt,
                        model_response=content,
                        action=action_label,
                        thought=content,
                    )
                )

                try:
                    t_env = time.perf_counter()
                    result = await session.step(("L", row, col))
                    env_step_s += time.perf_counter() - t_env
                except Exception as e:
                    logger.warning(
                        "minisweeper_rewind_choice task %s: env.step failed at C_%d reveal %s: %s",
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
                            observation_after=step_start_observation,
                            model_response=content,
                        )
                    )
                    force_rewind = True
                    rewind_kind = "forced"
                    force_rewind_reason = "forced rewind: environment step failed"
                    break

                observation = result.observation
                won = bool(result.won)
                done = bool(result.done)
                checkpoint_position = step_start_position + 1
                checkpoint_id = await _save_checkpoint_for_position(session, checkpoint_position)
                position_to_checkpoint_id.append(checkpoint_id)
                step_observations.append(observation)

                if won:
                    outcome = "won!"
                elif done and not won:
                    outcome = "mine hit (done=True, won=False)"
                elif observation == step_start_observation:
                    outcome = "no visible change (invalid, already open, or ineffective reveal)"
                else:
                    outcome = "safe reveal"

                history_entry = {
                    "position_before": str(step_start_position),
                    "position_after": str(checkpoint_position),
                    "observation_before": step_start_observation,
                    "observation_after": observation,
                    "observation": observation,
                    "outcome": outcome,
                    "action": action_label,
                }
                interaction_history.append(history_entry)
                message_checkpoints.append(_copy_messages(active_messages))
                branch_attempt_checkpoints.append(branch_attempts_used)
                segment_events.append(
                    BranchEvent(
                        kind="env_step",
                        position_before=step_start_position,
                        position_after=checkpoint_position,
                        action=action_label,
                        outcome=outcome,
                        observation_after=observation,
                        model_response=content,
                    )
                )

                if len(step_observations) != len(interaction_history) + 1:
                    logger.warning(
                        "position invariant drift: observations=%d history=%d",
                        len(step_observations),
                        len(interaction_history),
                    )

                if won:
                    break
                if done and not won:
                    force_rewind = True
                    rewind_kind = "forced"
                    force_rewind_reason = "forced rewind: mine hit (done=True, won=False)"
                    break

            if won:
                if segment_steps:
                    trajectories.append(Trajectory(name=f"minisweeper_seg{segment_idx}", steps=segment_steps, reward=None))
                break

            if exhausted_reason is not None:
                if segment_steps:
                    trajectories.append(Trajectory(name=f"minisweeper_seg{segment_idx}", steps=segment_steps, reward=None))
                break

            if not force_rewind:
                exhausted_reason = exhausted_reason or "segment ended without rewind trigger"
                if segment_steps:
                    trajectories.append(Trajectory(name=f"minisweeper_seg{segment_idx}", steps=segment_steps, reward=None))
                break

            if segment_idx + 1 >= max_segments:
                exhausted_reason = "segment limit exhausted"
                if segment_steps:
                    trajectories.append(Trajectory(name=f"minisweeper_seg{segment_idx}", steps=segment_steps, reward=None))
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
                exhausted_reason = "rewind requested but no checkpoint target was selected"
                if segment_steps:
                    trajectories.append(Trajectory(name=f"minisweeper_seg{segment_idx}", steps=segment_steps, reward=None))
                break

            if rewind_to is not None and perform_env_rewind and not (0 <= rewind_to < current_position):
                exhausted_reason = (
                    f"invalid selected rewind target C_{rewind_to}; "
                    f"valid targets are {_format_checkpoint_range(current_position)}"
                )
                if segment_steps:
                    trajectories.append(Trajectory(name=f"minisweeper_seg{segment_idx}", steps=segment_steps, reward=None))
                break

            if rewind_to is not None and not perform_env_rewind and rewind_to != current_position:
                exhausted_reason = "context-fold target must equal the current checkpoint"
                if segment_steps:
                    trajectories.append(Trajectory(name=f"minisweeper_seg{segment_idx}", steps=segment_steps, reward=None))
                break

            if segment_steps:
                trajectories.append(Trajectory(name=f"minisweeper_seg{segment_idx}", steps=segment_steps, reward=None))

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
                    board_size=board_size,
                    n_mines=n_mines,
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
                )
                llm_reflection_s += time.perf_counter() - t_reflect
                if reflect_content:
                    reflection_history.append(_truncate_text(reflect_content.strip(), MAX_REFLECTION_CHARS))

                trajectories.append(
                    Trajectory(
                        name=f"minisweeper_reflect{segment_idx}",
                        steps=[
                            Step(
                                chat_completions=[
                                    {"role": "user", "content": reflect_prompt},
                                    {"role": "assistant", "content": reflect_content},
                                ],
                                observation=reflect_prompt,
                                model_response=reflect_content,
                                action=(
                                    f"reflect_branch_memory_and_rewind_to_C_{reflect_rewind_target}"
                                    if reflect_rewind_target is not None
                                    else "reflect_branch_memory_invalid_rewind"
                                ),
                                thought=reflect_content,
                            )
                        ],
                        reward=None,
                    )
                )
            else:
                logger.warning(
                    "minisweeper_rewind_choice task %s: skipped reflection because LLM turn cap was reached",
                    task.id,
                )

            if perform_env_rewind:
                if rewind_to is None:
                    # Forced path: the reflection (choice template) must have
                    # selected the rollback checkpoint.
                    if reflect_rewind_target is None:
                        exhausted_reason = (
                            "reflection did not choose a valid rewind target: "
                            f"{reflect_parse_error or 'missing final rewind action'}"
                        )
                        break
                    rewind_to = reflect_rewind_target
                elif reflect_rewind_target is not None and reflect_rewind_target != rewind_to:
                    # Model rewind: the destination is already known. The known-target
                    # reflection template does not require a closing rewind action, so
                    # reflect_rewind_target=None is fine; only a mismatch is an error.
                    exhausted_reason = (
                        f"reflection target C_{reflect_rewind_target} did not match "
                        f"requested model rewind target C_{rewind_to}"
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

            if perform_env_rewind:
                try:
                    rw_result, checkpoint_id_used = await _rewind_session_to_position(
                        session=session,
                        position_to_checkpoint_id=position_to_checkpoint_id,
                        position=rewind_to,
                    )
                    observation = rw_result.observation
                except ValueError as e:
                    logger.warning(
                        "minisweeper_rewind_choice task %s: rewind to C_%d failed: %s",
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

            # Truncating back to C_{rewind_to} drops every later checkpoint, so
            # reflections attached to abandoned-sibling checkpoints disappear too.
            step_observations = step_observations[: rewind_to + 1]
            position_to_checkpoint_id = position_to_checkpoint_id[: rewind_to + 1]
            interaction_history = interaction_history[:rewind_to]
            message_checkpoints = message_checkpoints[: rewind_to + 1]
            branch_attempt_checkpoints = branch_attempt_checkpoints[: rewind_to + 1]
            active_messages = _copy_messages(message_checkpoints[rewind_to])
            branch_attempts_used = branch_attempt_checkpoints[rewind_to]

            # A previous rewind to this same checkpoint may have left its reflection
            # on the prefix; that prior memory is now folded into this reflection's
            # branch history, so drop it here to keep exactly one reflection riding
            # C_{rewind_to} instead of stacking on every re-rewind.
            _strip_trailing_reflections(active_messages)

            # Attach the reflection to C_{rewind_to}'s prefix and re-snapshot it, so
            # every future turn at C_j and every new descendant inherits this memory
            # while the abandoned branch's reflections stay dropped.
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

        _assign_cross_segment_rewards(
            trajectories=trajectories,
            won=won,
            traj_gamma=traj_gamma,
        )

        reflect_reward_avg = _mean_trajectory_reward(trajectories, "minisweeper_reflect")
        seg_reward_avg = _mean_trajectory_reward(trajectories, "minisweeper_seg")

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
                "segments": len([t for t in trajectories if t.name.startswith("minisweeper_seg")]),
                "total_play_turns": total_play_turns,
                "model_step_budget_used": total_play_turns,
                "branch_attempts_used": branch_attempts_used,
                "branch_attempt_budget": segment_max_turns,
                "total_reflection_turns": total_reflection_turns,
                "total_llm_calls": total_llm_calls,
                "total_env_steps": session.total_steps,
                "step_budget": step_budget,
                "env_step_budget": env_step_budget,
                "max_segments": max_segments,
                "max_total_turns": max_total_turns,
                "exhausted_reason": exhausted_reason,
                "rewinds": len(rewind_log),
                "forced_rewinds": forced_rewinds,
                "model_rewinds": model_rewinds,
                "forced_context_folds": forced_context_folds,
                "branch_memories": branch_memory_records,
                "reflection_history": reflection_history,
                "reflect_reward_avg": reflect_reward_avg,
                "seg_reward_avg": seg_reward_avg,
                "rewind_log": rewind_log,
                "global_messages": global_messages,
                "final_game_position": len(interaction_history),
                "active_path_len": len(interaction_history),
                "checkpoint_map": [str(x) for x in position_to_checkpoint_id],
                "board_size": board_size,
                "n_mines": n_mines,
                "board_type": board_type,
                "mode": mode,
            },
            is_correct=won,
        )

"""Rewind-capable MiniSweeper agent flow.

Travel Back to Move Forward version.

Core semantics:
  - Environment state can rewind to a previous checkpoint C_k.
  - Knowledge state only moves forward: before each rewind, the failed branch is
    compressed into a lightweight branch memory and injected into later context.

The model operates within a fixed environment step budget and can:
  1. Reveal a cell: `<action>(row, col)</action>`
  2. Rewind to a previous checkpoint: `<action>rewind to C_k</action>`
     The parser also accepts `<action>rewind to k</action>` for compatibility.

Forced rewind happens when:
  - Current segment turn limit is exceeded
  - LLM call failed or response was truncated by token length
  - Environment reaches done=True and won=False, e.g. a mine was hit

Model rewind happens when:
  - The model explicitly emits a valid rewind action.

Important invariants:
  - game_position == len(interaction_history)
  - C_0 is the initial board.
  - C_k is the board after k valid environment reveal actions on the current path.
  - session.total_steps is budget consumption and never decreases.
  - Rewind truncates the active environment path, but branch memories persist.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
import time
from typing import Any, Literal

from env_service import create_env_session, parse_remark
from env_service.minesweeper import MineSweeperEnv
from openai import AsyncOpenAI

import rllm
from rllm.types import AgentConfig, Episode, Step, Task, Trajectory

try:
    from ..prepare_minisweeper_data import LAMER_MINISWEEPER_CONFIG
except (ImportError, ValueError):
    from prepare_minisweeper_data import LAMER_MINISWEEPER_CONFIG

logger = logging.getLogger(__name__)

DEFAULT_STEP_BUDGET = 30
DEFAULT_TRAJ_GAMMA = 0.7
DEFAULT_SEGMENT_MAX_TURNS = 5
DEFAULT_MAX_SEGMENTS = 6
DEFAULT_MAX_TOTAL_TURNS = 100

MAX_BRANCH_MEMORIES_IN_CONTEXT = 3
MAX_BRANCH_MEMORY_CHARS = 1400
MAX_BRANCH_HISTORY_CHARS = 9000
MAX_ACTIVE_BRANCH_EVENTS = 8
MAX_MODEL_RESPONSE_IN_HISTORY_CHARS = 900

# --- Action parsing ---

_ACTION_TAG_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.IGNORECASE | re.DOTALL)
_REWIND_FULL_RE = re.compile(
    r"rewind\s+to\s+(?:C\s*[_-]?\s*)?(\d+)\s*$",
    re.IGNORECASE,
)
_COORD_FULL_RE = re.compile(r"\(?\s*(\d+)\s*,\s*(\d+)\s*\)?\s*$")


@dataclass(frozen=True)
class AgentCommand:
    kind: Literal["reveal", "rewind", "invalid"]
    row: int | None = None
    col: int | None = None
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

# Checkpoints and Rewind Capability
You have a total environment step budget of {step_budget} reveal actions.
The current path is indexed by checkpoints C_0, C_1, ..., C_k:
- C_0 is the initial board.
- C_i is the board after i valid reveal actions on the current path.

At any point after at least one reveal action, you can explicitly travel back:
- `<action>rewind to C_j</action>` where 0 <= j < current checkpoint index.
- `<action>rewind to j</action>` is also accepted.

When you rewind, the environment state returns to checkpoint C_j. However, the
knowledge gained from the failed branch persists as branch memory. Use rewind
when you realize the current branch is an action trap, invalid loop, dead end,
unproductive exploration, or when a previous assumption conflicts with new observations.

# Response Format
- First reason step-by-step about the current board state. Analyze numbered clues to deduce safe cells.
- Then choose ONE unopened cell to reveal: `<action>(row, col)</action>`
- OR rewind to a previous checkpoint: `<action>rewind to C_j</action>`
Only the final `<action>...</action>` tag will be executed.
"""

REWIND_REFLECT_PROMPT = """\
You are an expert Minesweeper player reflecting on a failed branch.
Board: {board_size} by {board_size}, {n_mines} mines. Rows/columns indexed 1 to {board_size}.

# Core rollback semantics
The environment will rewind from C_{rewind_from_step} to C_{rewind_to_step}.
The knowledge state must move forward: compress the failed branch into a small branch memory.
Do not preserve the full bad trajectory. Preserve only information useful for future decisions.

# Rewind trigger
{rewind_reason}

# Board at C_{rewind_to_step} where execution will resume
{rewind_to_observation}

# Board at C_{rewind_from_step} before rewinding
{current_observation}

# Branch history from C_{rewind_to_step} to C_{rewind_from_step}
{history_after_rewind}

# Your task
Reflect on what went wrong and produce a concise branch memory for the next attempt.
Focus on:
- failure reason or likely bad assumption;
- confirmed safe cells, confirmed mines, or clue constraints learned from this branch;
- invalid/risky actions that should not be repeated;
- a concrete improved plan from C_{rewind_to_step}.

End with a compact branch memory inside <remark> </remark> tags.
Use this structure:
<remark>
Failure reason: ...
Useful facts learned: ...
Avoid repeating: ...
Plan from C_{rewind_to_step}: ...
</remark>
"""


# --- Generic helpers ---


def _truncate_text(text: str, max_chars: int) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return "[truncated]\n" + text[-max_chars:]


def _build_system_prompt(board_size: int, n_mines: int, step_budget: int) -> str:
    return REWIND_SYSTEM_PROMPT.format(
        board_size=board_size,
        n_mines=n_mines,
        step_budget=step_budget,
    )


def _extract_checkpoint_id(obj: Any, fallback: Any = None) -> Any:
    """Best-effort extraction of a checkpoint id from env_service objects.

    Different env_service versions return checkpoint identifiers differently:
    an int, a dataclass-like object, a dict, or sometimes None. The rollout keeps
    an explicit position -> checkpoint_id map, but falls back to the visible
    position if the service uses identity checkpoint ids C_k == k.
    """
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
    """Save or identify the checkpoint for visible position C_position.

    If the environment auto-checkpoints on every step, save_checkpoint may be a
    no-op or may return the existing checkpoint. If it returns no id, we fall back
    to the visible position, matching env_service versions where checkpoint ids
    are exactly 0..N.
    """
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
    """Rewind env to visible path position C_position.

    Returns (rewind_result, checkpoint_id_used). Falls back to using the visible
    position directly for older env_service implementations.
    """
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


# --- Board/action parsing helpers ---


def _extract_final_action_text(content: str) -> str:
    """Return only the final action tag content.

    This avoids a common bug where reasoning text such as "do not rewind to 0"
    accidentally triggers a rewind. If no tag is present, inspect only the last
    non-empty line as a compatibility fallback.
    """
    matches = _ACTION_TAG_RE.findall(content or "")
    if matches:
        return matches[-1].strip()
    lines = [line.strip() for line in (content or "").splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _extract_board_grid(observation: str, board_size: int) -> list[list[str]] | None:
    """Best-effort parser for common text renderings of MiniSweeper boards.

    Returns None when the board format is unknown; callers should then avoid
    rejecting actions based on board-state parsing.
    """
    rows: list[list[str]] = []
    valid = {"?", ".", "*", "1", "2", "3", "4", "5", "6", "7", "8"}

    for raw_line in (observation or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # Tokenized form, e.g. "1 ? ? 2 .", "? ? 2 .", "1: ...", or
        # "Row 1: . . 1 1 1 .".
        parts = line.replace("|", " ").split()
        if parts:
            candidate = parts[:]
            expected_row_label = str(len(rows) + 1)
            # A row label may be rendered as "1", "1:", or "Row 1:". Strip the
            # optional leading "Row" word, then the "N"/"N:" index. Only an
            # explicit *numeric* index counts as a real row label; the bare
            # word "Row" alone must not (otherwise lines like
            # "Row labels: 1 2 3 4 5 6" would be mistaken for a board row).
            if candidate and candidate[0].lower() == "row":
                candidate = candidate[1:]
            had_row_label = False
            if candidate and candidate[0].rstrip(":") == expected_row_label:
                candidate = candidate[1:]
                had_row_label = True
            states = [p for p in candidate if p in valid]
            # A valid board row should contain at least one non-digit state
            # marker (?, ., *), which distinguishes it from a column header
            # like "1 2 3 4 5". A fully-revealed numbered row (e.g.
            # "Row 3: 1 1 2 2 2 1") has no such marker, so we additionally
            # accept rows that carried an explicit "Row N:"/"N:" label.
            has_cell_marker = any(p in {"?", ".", "*"} for p in candidate)
            if (has_cell_marker or had_row_label) and len(states) >= board_size:
                rows.append(states[:board_size])
                if len(rows) == board_size:
                    return rows
                continue

        # Compact form, e.g. "?????" or "1 ?????". Avoid treating
        # digit-only column headers as board rows.
        if not any(ch in line for ch in "?.*"):
            continue
        chars = [ch for ch in line if ch in valid]
        expected_row_label = str(len(rows) + 1)
        if len(chars) >= board_size + 1 and chars[0] == expected_row_label:
            chars = chars[1:]
        if len(chars) >= board_size:
            rows.append(chars[:board_size])
            if len(rows) == board_size:
                return rows

    return rows if len(rows) == board_size else None


def _is_unopened_cell(observation: str, row: int, col: int, board_size: int) -> bool | None:
    grid = _extract_board_grid(observation, board_size)
    if grid is None:
        return None
    return grid[row - 1][col - 1] == "?"


def parse_agent_command(content: str, board_size: int, observation: str | None = None) -> AgentCommand:
    raw = _extract_final_action_text(content)

    rewind_match = _REWIND_FULL_RE.fullmatch(raw)
    if rewind_match:
        return AgentCommand(kind="rewind", rewind_to=int(rewind_match.group(1)), raw=raw)

    coord_match = _COORD_FULL_RE.fullmatch(raw)
    if coord_match:
        row, col = int(coord_match.group(1)), int(coord_match.group(2))
        if not (1 <= row <= board_size and 1 <= col <= board_size):
            return AgentCommand(
                kind="invalid",
                raw=raw,
                error=f"coordinate ({row}, {col}) is outside the board",
            )
        if observation is not None:
            unopened = _is_unopened_cell(observation, row, col, board_size)
            if unopened is False:
                return AgentCommand(
                    kind="invalid",
                    raw=raw,
                    error=f"coordinate ({row}, {col}) is not an unopened '?' cell",
                )
        return AgentCommand(kind="reveal", row=row, col=col, raw=raw)

    return AgentCommand(
        kind="invalid",
        raw=raw,
        error="final action must be either `(row, col)` or `rewind to C_k`",
    )


# --- Context / reflection helpers ---


def _format_checkpoint_range(current_position: int) -> str:
    if current_position <= 0:
        return "none yet; reveal at least one cell before model-initiated rewind"
    return f"C_0..C_{current_position - 1}"


def _build_active_branch_context(events: list[BranchEvent]) -> str:
    if not events:
        return ""
    lines = ["\n# Recent events in the current branch:"]
    for event in events[-MAX_ACTIVE_BRANCH_EVENTS:]:
        if event.position_after != event.position_before:
            pos = f"C_{event.position_before} -> C_{event.position_after}"
        else:
            pos = f"C_{event.position_before}"
        lines.append(f"- {pos}: {event.kind} {event.action} => {event.outcome}")
    return "\n".join(lines) + "\n"


def _build_branch_memory_context(branch_memories: list[str] | None) -> str:
    if not branch_memories:
        return ""
    lines = ["\n# Branch memories from previous failed branches:"]
    recent = branch_memories[-MAX_BRANCH_MEMORIES_IN_CONTEXT:]
    for i, memory in enumerate(recent, start=1):
        lines.append(f"Memory #{i}: {_truncate_text(memory, MAX_BRANCH_MEMORY_CHARS)}")
    return "\n".join(lines) + "\n"


def _build_observation_prompt(
    observation: str,
    turn: int,
    game_position: int,
    step_budget: int,
    budget_used: int,
    branch_memories: list[str] | None = None,
    active_branch_events: list[BranchEvent] | None = None,
    action_is_valid: bool = True,
    invalid_reason: str = "",
) -> str:
    budget_remaining = max(0, step_budget - budget_used)

    if turn == 0 and game_position == 0:
        header = "The initial state of the game is:"
    else:
        header = f"Current observation at checkpoint C_{game_position} (segment turn {turn}):"

    retry = ""
    if not action_is_valid:
        reason = f" Reason: {invalid_reason}" if invalid_reason else ""
        retry = (
            "\nYour last command was invalid and was NOT executed in the environment."
            f"{reason}\nChoose one unopened cell (row, col), or rewind to a valid previous checkpoint.\n"
        )

    memory_context = _build_branch_memory_context(branch_memories)
    active_context = _build_active_branch_context(active_branch_events or [])

    return (
        f"{header}\n{observation}\n"
        f"{retry}"
        f"\nEnvironment budget used: {budget_used}/{step_budget} "
        f"(remaining: {budget_remaining}).\n"
        f"Current checkpoint: C_{game_position}.\n"
        f"Valid model rewind targets: {_format_checkpoint_range(game_position)}.\n"
        f"{memory_context}"
        f"{active_context}"
        "Choose one unopened cell to reveal: <action>(row, col)</action>, "
        "or travel back: <action>rewind to C_j</action>."
    )


def _build_history_after_rewind(
    interaction_history: list[dict[str, str]],
    extra_events: list[BranchEvent],
    start: int,
    end: int | None = None,
) -> str:
    """Build a reflection history for the branch being abandoned.

    Includes valid environment actions from the active path plus non-env events
    such as invalid commands, truncation, model rewind, or LLM failures.
    """
    if end is None:
        end = len(interaction_history)

    lines: list[str] = []

    for position in range(start, end):
        if position >= len(interaction_history):
            break
        entry = interaction_history[position]
        pos_before = int(entry.get("position_before", position))
        pos_after = int(entry.get("position_after", position + 1))
        lines.append(f"C_{pos_before} -> C_{pos_after}")
        lines.append(f"Action: reveal {entry.get('action', '(unknown)')}")
        lines.append(f"Outcome: {entry.get('outcome', '(unknown)')}")
        obs = entry.get("observation_after") or entry.get("observation") or ""
        if obs:
            lines.append("Observation after action:")
            lines.append(obs)
        lines.append("")

    non_env_events = [
        event
        for event in extra_events
        if event.kind != "env" and start <= event.position_before <= end
    ]
    if non_env_events:
        lines.append("Non-environment events in this branch:")
        for event in non_env_events:
            pos = f"C_{event.position_before}"
            if event.position_after != event.position_before:
                pos += f" -> C_{event.position_after}"
            lines.append(f"- {pos}: {event.kind} {event.action} => {event.outcome}")
            if event.model_response:
                lines.append("  Model response excerpt:")
                lines.append(_truncate_text(event.model_response, MAX_MODEL_RESPONSE_IN_HISTORY_CHARS))

    if not lines:
        result = "(no valid environment actions were taken in the failed branch)"
    else:
        result = "\n".join(lines)
    return _truncate_text(result, MAX_BRANCH_HISTORY_CHARS)


def _fallback_branch_memory(
    rewind_reason: str,
    rewind_to: int,
    current_position: int,
    interaction_history: list[dict[str, str]],
) -> str:
    learned: list[str] = []
    if 0 <= current_position - 1 < len(interaction_history):
        last = interaction_history[current_position - 1]
        action = last.get("action", "unknown")
        outcome = last.get("outcome", "unknown")
        if "mine" in outcome.lower():
            learned.append(f"The branch ended by revealing {action}, which hit a mine.")
        else:
            learned.append(f"Last valid reveal before rewind: {action} => {outcome}.")

    facts = " ".join(learned) if learned else "No reliable board fact was extracted automatically."
    return (
        f"Failure reason: {rewind_reason}.\n"
        f"Useful facts learned: {facts}\n"
        f"Avoid repeating: Do not blindly repeat the same failed branch from C_{rewind_to}.\n"
        f"Plan from C_{rewind_to}: Re-check numbered clues and choose a different safe-looking action."
    )


async def _do_reflection(
    client: AsyncOpenAI,
    model: str,
    sampling: dict[str, Any],
    board_size: int,
    n_mines: int,
    rewind_to: int,
    current_position: int,
    rewind_to_obs: str,
    current_observation: str,
    interaction_history: list[dict[str, str]],
    segment_events: list[BranchEvent],
    rewind_reason: str,
    task_id: str,
) -> tuple[str, str, str]:
    """Run reflection LLM call. Returns (full_response, branch_memory, prompt)."""
    history_after = _build_history_after_rewind(
        interaction_history=interaction_history,
        extra_events=segment_events,
        start=rewind_to,
        end=current_position,
    )

    reflect_prompt = REWIND_REFLECT_PROMPT.format(
        board_size=board_size,
        n_mines=n_mines,
        rewind_from_step=current_position,
        rewind_to_step=rewind_to,
        rewind_to_observation=rewind_to_obs,
        current_observation=current_observation,
        history_after_rewind=history_after,
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
        reflect_content = reflect_resp.choices[0].message.content or ""
    except Exception as e:
        logger.warning("minisweeper_rewind task %s: reflection failed: %s", task_id, e)
        reflect_content = ""

    branch_memory = parse_remark(reflect_content) if reflect_content else ""
    if not branch_memory and reflect_content:
        branch_memory = reflect_content.strip()
    branch_memory = _truncate_text(branch_memory, MAX_BRANCH_MEMORY_CHARS) if branch_memory else ""
    return reflect_content, branch_memory, reflect_prompt


# --- Reward assignment ---


def _parse_suffix_int(name: str, prefix: str, default: int = -1) -> int:
    if not name.startswith(prefix):
        return default
    suffix = name[len(prefix) :]
    try:
        return int(suffix)
    except ValueError:
        match = re.search(r"(\d+)$", name)
        return int(match.group(1)) if match else default


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

    # Never leak None rewards to downstream training.
    for traj in trajectories:
        if traj.reward is None:
            traj.reward = 0.0


# --- Main rollout ---


@rllm.rollout(name="minisweeper_rewind")
async def minisweeper_rewind_flow(task: Task, config: AgentConfig) -> Episode:
    """Drive MiniSweeper with explicit rollback and persistent branch memory."""
    meta = task.metadata or {}

    seed = int(meta.get("seed", LAMER_MINISWEEPER_CONFIG["env_seed"]))
    board_size = int(meta.get("board_size", LAMER_MINISWEEPER_CONFIG["board_size"]))
    n_mines = int(meta.get("n_mines", LAMER_MINISWEEPER_CONFIG["n_mines"]))
    board_type = meta.get("board_type", LAMER_MINISWEEPER_CONFIG["board_type"])
    mode = meta.get("mode", LAMER_MINISWEEPER_CONFIG["mode"])
    puzzle_state = meta.get("puzzle_state")

    step_budget = int(meta.get("step_budget", DEFAULT_STEP_BUDGET))
    segment_max_turns = int(
        meta.get("segment_max_turns", meta.get("max_turns", DEFAULT_SEGMENT_MAX_TURNS))
    )
    # max_segments is the single source of truth for how many play segments (and
    # thus rewinds: rewinds = segments - 1) an episode may run. Default 6.
    max_segments = int(meta.get("max_segments", DEFAULT_MAX_SEGMENTS))
    max_total_turns = int(meta.get("max_total_turns", DEFAULT_MAX_TOTAL_TURNS))
    traj_gamma = float(meta.get("traj_gamma", DEFAULT_TRAJ_GAMMA))

    client = AsyncOpenAI(base_url=config.base_url, api_key="EMPTY")
    sampling = {k: v for k, v in config.sampling_params.items() if k != "top_k"}

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

        # step_observations[i] = board observation at visible checkpoint C_i.
        step_observations: list[str] = [observation]

        # position_to_checkpoint_id[i] = env_service checkpoint id for visible C_i.
        # This avoids assuming env checkpoint ids always equal visible positions.
        position_to_checkpoint_id: list[Any] = [initial_checkpoint_id]

        # interaction_history[i] = valid env reveal taken from C_i to C_{i+1}.
        interaction_history: list[dict[str, str]] = []

        branch_memories: list[str] = []
        branch_memory_records: list[dict[str, Any]] = []
        rewind_log: list[dict[str, Any]] = []

        system_prompt = _build_system_prompt(board_size, n_mines, step_budget)

        won = False
        exhausted_reason: str | None = None
        segment_idx = 0
        total_play_turns = 0
        total_reflection_turns = 0
        total_llm_calls = 0
        forced_rewinds = 0
        model_rewinds = 0

        while True:
            budget_remaining = session.step_budget_remaining
            if budget_remaining is not None and budget_remaining <= 0:
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
            segment_turn = 0
            segment_steps: list[Step] = []
            segment_events: list[BranchEvent] = []
            last_action_valid = True
            invalid_reason = ""

            force_rewind = False
            force_rewind_reason = ""
            rewind_target: int | None = None

            # --- One branch segment ---
            while True:
                budget_remaining = session.step_budget_remaining
                if budget_remaining is not None and budget_remaining <= 0:
                    exhausted_reason = "step budget exhausted"
                    break
                if total_llm_calls >= max_total_turns:
                    exhausted_reason = "LLM turn limit exhausted"
                    break
                if won:
                    break

                current_position = len(interaction_history)

                if segment_turn >= segment_max_turns:
                    force_rewind = True
                    force_rewind_reason = "forced rewind: segment turn limit exceeded"
                    rewind_target = segment_start_position
                    break

                obs_prompt = _build_observation_prompt(
                    observation=observation,
                    turn=segment_turn,
                    game_position=current_position,
                    step_budget=step_budget,
                    budget_used=session.total_steps,
                    branch_memories=branch_memories,
                    active_branch_events=segment_events,
                    action_is_valid=last_action_valid,
                    invalid_reason=invalid_reason,
                )
                messages: list[dict[str, str]] = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": obs_prompt},
                ]

                try:
                    t_llm = time.perf_counter()
                    resp = await client.chat.completions.create(
                        model=config.model,
                        messages=messages,
                        **sampling,
                        timeout=120,
                    )
                    llm_action_s += time.perf_counter() - t_llm
                    total_llm_calls += 1
                    total_play_turns += 1
                    content = resp.choices[0].message.content or ""
                except Exception as e:
                    llm_action_s += time.perf_counter() - t_llm if "t_llm" in locals() else 0.0
                    total_llm_calls += 1
                    total_play_turns += 1
                    logger.warning(
                        "minisweeper_rewind task %s segment %d turn %d: LLM failed: %s",
                        task.id,
                        segment_idx,
                        segment_turn,
                        e,
                    )
                    segment_turn += 1
                    segment_steps.append(
                        Step(
                            chat_completions=list(messages),
                            observation=obs_prompt,
                            model_response="",
                            action="llm_failed",
                            thought=f"LLM call failed: {e}",
                        )
                    )
                    segment_events.append(
                        BranchEvent(
                            kind="llm_failed",
                            position_before=current_position,
                            position_after=current_position,
                            action="llm_failed",
                            outcome=str(e),
                            observation_after=observation,
                        )
                    )
                    force_rewind = True
                    force_rewind_reason = "forced rewind: LLM call failed"
                    rewind_target = segment_start_position
                    break

                messages.append({"role": "assistant", "content": content})
                segment_turn += 1

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
                    force_rewind_reason = "forced rewind: response truncated by token length"
                    rewind_target = segment_start_position
                    break

                command = parse_agent_command(content, board_size=board_size, observation=observation)

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
                    force_rewind_reason = f"model rewind: requested rewind to C_{target}"
                    model_rewinds += 1
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

                # --- Normal reveal action ---
                assert command.kind == "reveal"
                assert command.row is not None and command.col is not None

                row, col = command.row, command.col
                action_label = f"({row}, {col})"
                segment_steps.append(
                    Step(
                        chat_completions=list(messages),
                        observation=obs_prompt,
                        model_response=content,
                        action=action_label,
                        thought=content,
                    )
                )

                previous_observation = observation
                previous_position = len(interaction_history)

                try:
                    t_env = time.perf_counter()
                    result = await session.step(("L", row, col))
                    env_step_s += time.perf_counter() - t_env
                except Exception as e:
                    logger.warning(
                        "minisweeper_rewind task %s: env.step failed at C_%d action %s: %s",
                        task.id,
                        previous_position,
                        action_label,
                        e,
                    )
                    segment_events.append(
                        BranchEvent(
                            kind="env_step_failed",
                            position_before=previous_position,
                            position_after=previous_position,
                            action=action_label,
                            outcome=str(e),
                            observation_after=previous_observation,
                            model_response=content,
                        )
                    )
                    force_rewind = True
                    force_rewind_reason = "forced rewind: environment step failed"
                    rewind_target = segment_start_position
                    break

                observation = result.observation
                won = bool(result.won)
                done = bool(result.done)

                checkpoint_position = previous_position + 1
                checkpoint_id = await _save_checkpoint_for_position(session, checkpoint_position)
                position_to_checkpoint_id.append(checkpoint_id)
                step_observations.append(observation)

                outcome = "mine hit!" if (done and not won) else ("won!" if won else "safe")
                history_entry = {
                    "position_before": str(previous_position),
                    "position_after": str(checkpoint_position),
                    "observation_before": previous_observation,
                    "observation_after": observation,
                    "observation": observation,  # backward-compatible alias
                    "outcome": outcome,
                    "action": action_label,
                }
                interaction_history.append(history_entry)

                segment_events.append(
                    BranchEvent(
                        kind="env",
                        position_before=previous_position,
                        position_after=checkpoint_position,
                        action=action_label,
                        outcome=outcome,
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

                if done and not won:
                    force_rewind = True
                    force_rewind_reason = "forced rewind: environment reached dead state (done=True, won=False)"
                    rewind_target = segment_start_position
                    forced_rewinds += 1
                    break

            # --- End of current segment inner loop ---

            if won:
                if segment_steps:
                    trajectories.append(
                        Trajectory(
                            name=f"minisweeper_seg{segment_idx}",
                            steps=segment_steps,
                            reward=None,
                        )
                    )
                break

            if exhausted_reason is not None:
                if segment_steps:
                    trajectories.append(
                        Trajectory(
                            name=f"minisweeper_seg{segment_idx}",
                            steps=segment_steps,
                            reward=None,
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
                            name=f"minisweeper_seg{segment_idx}",
                            steps=segment_steps,
                            reward=None,
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
                            name=f"minisweeper_seg{segment_idx}",
                            steps=segment_steps,
                            reward=None,
                        )
                    )
                break

            if force_rewind_reason.startswith("forced rewind") and not (
                "dead state" in force_rewind_reason
            ):
                forced_rewinds += 1

            current_position = len(interaction_history)
            rewind_to = segment_start_position if rewind_target is None else rewind_target
            rewind_to = max(0, min(rewind_to, current_position))

            if segment_steps:
                trajectories.append(
                    Trajectory(
                        name=f"minisweeper_seg{segment_idx}",
                        steps=segment_steps,
                        reward=None,
                    )
                )

            rewind_to_obs = (
                step_observations[rewind_to]
                if rewind_to < len(step_observations)
                else initial_observation
            )
            current_observation = observation

            # --- Knowledge moves forward: compress failed branch into branch memory. ---
            branch_memory = ""
            reflect_content = ""
            reflect_prompt = ""
            if total_llm_calls < max_total_turns:
                t_reflect = time.perf_counter()
                total_llm_calls += 1
                total_reflection_turns += 1
                reflect_content, branch_memory, reflect_prompt = await _do_reflection(
                    client=client,
                    model=config.model,
                    sampling=sampling,
                    board_size=board_size,
                    n_mines=n_mines,
                    rewind_to=rewind_to,
                    current_position=current_position,
                    rewind_to_obs=rewind_to_obs,
                    current_observation=current_observation,
                    interaction_history=interaction_history,
                    segment_events=segment_events,
                    rewind_reason=force_rewind_reason,
                    task_id=task.id,
                )
                llm_reflection_s += time.perf_counter() - t_reflect

                reflect_step = Step(
                    chat_completions=[
                        {"role": "user", "content": reflect_prompt},
                        {"role": "assistant", "content": reflect_content},
                    ],
                    observation=reflect_prompt,
                    model_response=reflect_content,
                    action="reflect_branch_memory",
                    thought=reflect_content,
                )
                trajectories.append(
                    Trajectory(
                        name=f"minisweeper_reflect{segment_idx}",
                        steps=[reflect_step],
                        reward=None,
                    )
                )
            else:
                logger.warning(
                    "minisweeper_rewind task %s: skipped reflection because LLM turn cap was reached",
                    task.id,
                )

            if not branch_memory:
                branch_memory = _fallback_branch_memory(
                    rewind_reason=force_rewind_reason,
                    rewind_to=rewind_to,
                    current_position=current_position,
                    interaction_history=interaction_history,
                )

            branch_memory = _truncate_text(branch_memory, MAX_BRANCH_MEMORY_CHARS)
            if branch_memory:
                branch_memories.append(branch_memory)
                branch_memory_records.append(
                    {
                        "segment": segment_idx,
                        "rewind_from": current_position,
                        "rewind_to": rewind_to,
                        "reason": force_rewind_reason,
                        "memory": branch_memory,
                    }
                )

            # --- Environment travels back. ---
            try:
                rw_result, checkpoint_id_used = await _rewind_session_to_position(
                    session=session,
                    position_to_checkpoint_id=position_to_checkpoint_id,
                    position=rewind_to,
                )
                observation = rw_result.observation
            except ValueError as e:
                logger.warning(
                    "minisweeper_rewind task %s: rewind to C_%d failed: %s; falling back to C_0",
                    task.id,
                    rewind_to,
                    e,
                )
                rw_result, checkpoint_id_used = await _rewind_session_to_position(
                    session=session,
                    position_to_checkpoint_id=position_to_checkpoint_id,
                    position=0,
                )
                observation = rw_result.observation
                rewind_to = 0

            rewind_log.append(
                {
                    "segment": segment_idx,
                    "from": current_position,
                    "to": rewind_to,
                    "checkpoint_id_used": str(checkpoint_id_used),
                    "reason": force_rewind_reason,
                    "memory": branch_memory,
                }
            )

            # Truncate active path. Branch memories intentionally persist.
            step_observations = step_observations[: rewind_to + 1]
            position_to_checkpoint_id = position_to_checkpoint_id[: rewind_to + 1]
            interaction_history = interaction_history[:rewind_to]

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
                "segments": len([t for t in trajectories if t.name.startswith("minisweeper_seg")]),
                "total_play_turns": total_play_turns,
                "total_reflection_turns": total_reflection_turns,
                "total_llm_calls": total_llm_calls,
                "total_env_steps": session.total_steps,
                "step_budget": step_budget,
                "max_segments": max_segments,
                "max_total_turns": max_total_turns,
                "exhausted_reason": exhausted_reason,
                "rewinds": len(rewind_log),
                "forced_rewinds": forced_rewinds,
                "model_rewinds": model_rewinds,
                "branch_memories": branch_memory_records,
                "rewind_log": rewind_log,
                "final_game_position": len(interaction_history),
                "active_path_len": len(interaction_history),
                "checkpoint_map": [str(x) for x in position_to_checkpoint_id],
                "board_size": board_size,
                "n_mines": n_mines,
                "mode": mode,
            },
            is_correct=won,
        )

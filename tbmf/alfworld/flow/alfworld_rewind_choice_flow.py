"""Replay-based rewind-choice ALFWorld flow."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import logging
import os
import re
import time
from typing import Any, Literal

from env_service import create_env_session, parse_remark
from env_service.alfworld import AlfWorldEnv
from openai import AsyncOpenAI

import rllm
from rllm.types import AgentConfig, Episode, Step, Task, Trajectory
from tbmf.flow_utils import classify_llm_failure

try:
    from .alfworld_flow import SYSTEM_PROMPT, parse_action
except (ImportError, ValueError):
    from alfworld_flow import SYSTEM_PROMPT, parse_action

try:
    from ..milestone_loader import MilestoneLoader
except (ImportError, ValueError):
    try:
        from milestone_loader import MilestoneLoader
    except (ImportError, ValueError):
        MilestoneLoader = None

logger = logging.getLogger(__name__)

DEFAULT_SEGMENT_MAX_TURNS = 12
DEFAULT_MAX_SEGMENTS = 8
DEFAULT_MAX_TOTAL_TURNS = 80
DEFAULT_TRAJ_GAMMA = 0.7
MAX_BRANCH_MEMORIES_IN_CONTEXT = 4
MAX_BRANCH_MEMORY_CHARS = 1800
MAX_BRANCH_HISTORY_CHARS = 18000

_ACTION_RE = re.compile(r"```action\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_GENERIC_RE = re.compile(r"```(?!action\b)\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_REWIND_RE = re.compile(r"rewind\s+to\s+(?:C\s*[_-]?\s*)?(\d+)\s*$", re.IGNORECASE)

# Expert-milestone matching threshold (matches LaMer alfworld's 0.85).
_MILESTONE_THRESHOLD = 0.85


def _gamefile_to_trial_id(gamefile_path: str | None) -> str | None:
    """Extract trial_id from a game.tw-pddl path for milestone lookup.

    Path format: .../task_type-Object-Location-N/trial_T<ts>/game.tw-pddl
    trial_id format: task_type-Object-Location-N_trial_T<ts> (matches alfworld.json ids).
    """
    if not gamefile_path:
        return None
    parts = gamefile_path.replace("\\", "/").split("/")
    try:
        game_idx = parts.index("game.tw-pddl") if "game.tw-pddl" in parts else len(parts) - 1
        trial_part = parts[game_idx - 1]
        task_part = parts[game_idx - 2]
        return f"{task_part}_{trial_part}"
    except (IndexError, ValueError):
        return None


def _count_milestones(path_actions: list[str], milestones: list[str] | None) -> int:
    """Greedy in-order fuzzy match of executed actions against expert milestones.

    Mirrors LaMer's match: advance a monotone pointer ``k`` whenever the next
    executed action matches the current expected milestone with SequenceMatcher
    ratio >= 0.85. Returns how many expert sub-goals were reproduced in order.
    """
    if not milestones:
        return 0
    k = 0
    for action in path_actions:
        if k >= len(milestones):
            break
        if SequenceMatcher(None, action, milestones[k]).ratio() >= _MILESTONE_THRESHOLD:
            k += 1
    return k


@dataclass(frozen=True)
class AgentCommand:
    kind: Literal["action", "rewind", "invalid"]
    action: str | None = None
    rewind_to: int | None = None
    raw: str = ""
    error: str = ""


def _message(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


def _copy_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    return [dict(message) for message in messages]


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return "[truncated]\n" + text[-max_chars:]


def _extract_action_text(response: str) -> str:
    matches = _ACTION_RE.findall(response or "")
    if matches:
        return matches[-1].strip()
    matches = _GENERIC_RE.findall(response or "")
    if matches:
        return matches[-1].strip()
    lines = [line.strip() for line in (response or "").splitlines() if line.strip()]
    return lines[-1] if lines else ""


def parse_agent_command(response: str) -> AgentCommand:
    raw = _extract_action_text(response)
    if not raw:
        return AgentCommand(kind="invalid", raw=raw, error="missing action block")
    rewind_match = _REWIND_RE.fullmatch(raw)
    if rewind_match:
        return AgentCommand(kind="rewind", rewind_to=int(rewind_match.group(1)), raw=raw)
    action = parse_action(response)
    if not action:
        return AgentCommand(kind="invalid", raw=raw, error="missing valid action")
    return AgentCommand(kind="action", action=action, raw=raw)


def _format_checkpoint_range(current_position: int) -> str:
    if current_position <= 0:
        return "(none)"
    if current_position == 1:
        return "C_0"
    return f"C_0..C_{current_position - 1}"


def _format_commands(commands: list[str]) -> str:
    return "\n".join(f"  - {cmd}" for cmd in commands)


def _branch_memory_context(branch_memories: list[str]) -> str:
    if not branch_memories:
        return ""
    lines = ["# Branch memories from previous failed branches:"]
    for idx, memory in enumerate(branch_memories[-MAX_BRANCH_MEMORIES_IN_CONTEXT:], start=1):
        lines.append(f"Memory #{idx}: {_truncate(memory, MAX_BRANCH_MEMORY_CHARS)}")
    return "\n\n" + "\n".join(lines)


def _build_user_prompt(
    observation: str,
    admissible_commands: list[str],
    current_position: int,
    total_turns: int,
    step_budget: int,
    branch_turns: int,
    segment_max_turns: int,
    branch_memories: list[str],
    *,
    action_is_valid: bool = True,
    invalid_reason: str = "",
) -> str:
    commands_str = _format_commands(admissible_commands)
    remaining = max(step_budget - total_turns, 0)
    branch_remaining = max(segment_max_turns - branch_turns, 0)
    retry = ""
    if not action_is_valid:
        retry = (
            "\nYour previous response did not contain a valid executable action. "
            f"{invalid_reason}\n"
        )
    prompt = (
        f"Current observation at checkpoint C_{current_position}:\n{observation}\n"
        f"{retry}\n"
        f"Admissible commands:\n{commands_str}\n\n"
        f"Global action budget used: {total_turns}/{step_budget} "
        f"(remaining: {remaining}).\n"
        f"Current-branch budget used: {branch_turns}/{segment_max_turns} "
        f"(remaining: {branch_remaining}).\n"
        f"Valid rewind targets: {_format_checkpoint_range(current_position)}."
        f"{_branch_memory_context(branch_memories)}\n\n"
        "Choose exactly one environment action in ```action ... ```, "
        "or travel back with ```action\nrewind to C_j\n```."
    )
    return prompt


def _build_history(history: list[dict[str, Any]], start: int, end: int) -> str:
    lines: list[str] = []
    for entry in history[max(0, start):max(0, end)]:
        lines.extend(
            [
                f"C_{entry['position_before']} -> C_{entry['position_after']}",
                f"Action: {entry['action']}",
                f"Outcome: {entry['outcome']}",
                "Observation before:",
                entry["observation_before"],
                "Observation after:",
                entry["observation_after"],
                "",
            ]
        )
    return _truncate("\n".join(lines).strip() or "(no branch history)", MAX_BRANCH_HISTORY_CHARS)


REFLECT_PROMPT = """\
You are reflecting on a failed ALFWorld branch.

# Rewind trigger
{rewind_reason}

# Current checkpoint
The branch is currently at C_{current_position}.

# Valid rewind targets
{valid_targets}

# Branch history
{branch_history}

Write a compact memory that helps the next attempt avoid the failed branch.
Include wrong locations, wrong object choices, missed open/take/put/state-change steps,
and useful facts learned from observations.

Return memory inside <remark>...</remark>, then end with exactly one action block:
```action
rewind to C_j
```
"""


def _parse_rewind_target(content: str, current_position: int) -> tuple[int | None, str]:
    raw = _extract_action_text(content)
    match = _REWIND_RE.fullmatch(raw)
    if not match:
        return None, f"final action {raw!r} is not a rewind command"
    target = int(match.group(1))
    if not (0 <= target < current_position):
        return None, f"target C_{target} invalid; valid targets are {_format_checkpoint_range(current_position)}"
    return target, ""


async def _new_session(game_file: str, max_steps: int):
    session = await create_env_session(AlfWorldEnv, session_mode="ray", game_file=game_file, max_steps=max_steps)
    observation, info = await session.reset()
    return session, observation, info


async def _replay_to(
    game_file: str,
    max_steps: int,
    actions: list[str],
    position: int,
) -> tuple[Any, str, dict[str, Any], float, float]:
    t0 = time.perf_counter()
    session, observation, info = await _new_session(game_file, max_steps)
    env_init_s = time.perf_counter() - t0
    env_step_s = 0.0
    for action in actions[:position]:
        t_step = time.perf_counter()
        result = await session.step(action)
        env_step_s += time.perf_counter() - t_step
        observation = result.observation
        info = result.info
        if result.done:
            break
    return session, observation, info, env_init_s, env_step_s


async def _reflect(
    client: AsyncOpenAI,
    model: str,
    sampling: dict[str, Any],
    current_position: int,
    rewind_reason: str,
    history: list[dict[str, Any]],
    history_start: int,
    task_id: str,
) -> tuple[str, str, int | None, str, str]:
    prompt = REFLECT_PROMPT.format(
        rewind_reason=rewind_reason,
        current_position=current_position,
        valid_targets=_format_checkpoint_range(current_position),
        branch_history=_build_history(history, history_start, current_position),
    )
    messages = [{"role": "user", "content": prompt}]
    try:
        resp = await client.chat.completions.create(model=model, messages=messages, **sampling, timeout=120)
        content = resp.choices[0].message.content or ""
    except Exception as e:
        logger.warning("alfworld_rewind_choice task %s: reflection failed: %s", task_id, e)
        content = ""
    memory = parse_remark(content) if content else ""
    if not memory and content:
        memory = content.strip()
    memory = _truncate(memory, MAX_BRANCH_MEMORY_CHARS) if memory else ""
    target, parse_error = _parse_rewind_target(content, current_position) if content else (None, "empty reflection")
    return content, memory, target, parse_error, prompt


def _assign_rewards(trajectories: list[Trajectory], won: bool, gamma: float) -> None:
    play = [(int(re.search(r"(\d+)$", traj.name).group(1)), idx) for idx, traj in enumerate(trajectories) if traj.name.startswith("alfworld_seg")]
    play.sort()
    if play:
        discounted = [0.0] * len(play)
        if won:
            discounted[-1] = 1.0
        for i in range(len(play) - 2, -1, -1):
            discounted[i] = gamma * discounted[i + 1]
        for order, (_, idx) in enumerate(play):
            trajectories[idx].reward = discounted[order]
    for traj in trajectories:
        if traj.reward is None:
            traj.reward = 0.0


@rllm.rollout(name="alfworld_rewind_choice")
async def alfworld_rewind_choice_flow(task: Task, config: AgentConfig) -> Episode:
    meta = task.metadata or {}
    game_file = meta.get("game_file")
    if not game_file:
        raise ValueError("Task metadata must include 'game_file'")

    max_steps = int(meta.get("max_steps", 50))
    step_budget = int(meta.get("step_budget", max_steps))
    segment_max_turns = int(meta.get("segment_max_turns", DEFAULT_SEGMENT_MAX_TURNS))
    max_segments = int(meta.get("max_segments", DEFAULT_MAX_SEGMENTS))
    max_total_turns = int(meta.get("max_total_turns", max(DEFAULT_MAX_TOTAL_TURNS, step_budget + max_segments)))
    traj_gamma = float(meta.get("traj_gamma", DEFAULT_TRAJ_GAMMA))
    task_type = meta.get("task_type", "unknown")
    uid = config.session_uid or task.id

    # Expert-trajectory milestones for this game instance (used only when the
    # caller selects the milestone-diff reflection reward; harmless otherwise).
    milestone_trial_id = _gamefile_to_trial_id(game_file)
    expert_milestones: list[str] | None = None
    if MilestoneLoader is not None and milestone_trial_id is not None:
        try:
            expert_milestones = MilestoneLoader().get_milestones(milestone_trial_id)
        except Exception as e:  # noqa: BLE001 - milestone data is best-effort
            logger.warning("alfworld_rewind_choice %s: milestone load failed: %s", uid, e)
            expert_milestones = None
    total_milestones = len(expert_milestones) if expert_milestones else 0

    client = AsyncOpenAI(base_url=config.base_url, api_key="EMPTY")
    sampling = {k: v for k, v in config.sampling_params.items() if k != "top_k"}

    env_init_s = 0.0
    env_step_s = 0.0
    llm_action_s = 0.0
    llm_reflection_s = 0.0
    trajectories: list[Trajectory] = []
    branch_memories: list[str] = []
    branch_memory_records: list[dict[str, Any]] = []
    rewind_log: list[dict[str, Any]] = []
    interaction_history: list[dict[str, Any]] = []
    path_actions: list[str] = []
    path_observations: list[str] = []
    path_infos: list[dict[str, Any]] = []
    global_messages: list[dict[str, str]] = [_message("system", SYSTEM_PROMPT)]

    t0 = time.perf_counter()
    logger.info("alfworld_rewind_choice rollout %s: env init start game=%s", uid, os.path.basename(game_file))
    session, observation, info = await _new_session(game_file, max_steps)
    env_init_s += time.perf_counter() - t0
    path_observations.append(observation)
    path_infos.append(info)
    admissible_commands = info.get("admissible_commands", [])

    won = False
    exhausted_reason: str | None = None
    total_play_turns = 0
    total_llm_calls = 0
    segment_idx = 0
    forced_rewinds = 0
    model_rewinds = 0
    context_folds = 0
    last_action: str | None = None

    try:
        while True:
            if won:
                break
            if total_play_turns >= step_budget:
                exhausted_reason = "step budget exhausted"
                break
            if total_llm_calls >= max_total_turns:
                exhausted_reason = "LLM turn limit exhausted"
                break
            if segment_idx >= max_segments:
                exhausted_reason = "segment limit exhausted"
                break

            segment_start = len(interaction_history)
            segment_steps: list[Step] = []
            segment_turns = 0
            active_messages = [_message("system", SYSTEM_PROMPT)]
            last_valid = True
            invalid_reason = ""
            force_rewind = False
            rewind_reason = ""
            rewind_target: int | None = None
            rewind_kind = "forced"

            while True:
                if won or total_play_turns >= step_budget or total_llm_calls >= max_total_turns:
                    break
                current_position = len(interaction_history)
                if segment_turns >= segment_max_turns:
                    force_rewind = True
                    rewind_reason = "forced rewind: current-branch attempt budget exhausted"
                    rewind_kind = "forced"
                    break

                obs_prompt = _build_user_prompt(
                    observation,
                    admissible_commands,
                    current_position,
                    total_play_turns,
                    step_budget,
                    segment_turns,
                    segment_max_turns,
                    branch_memories,
                    action_is_valid=last_valid,
                    invalid_reason=invalid_reason,
                )
                active_messages.append(_message("user", obs_prompt))
                global_messages.append(_message("user", obs_prompt))
                messages = _copy_messages(active_messages)

                t_llm = time.perf_counter()
                try:
                    resp = await client.chat.completions.create(
                        model=config.model, messages=messages, **sampling, timeout=120,
                    )
                    content = resp.choices[0].message.content or ""
                except Exception as e:
                    failure = classify_llm_failure(e)
                    llm_action_s += time.perf_counter() - t_llm
                    total_llm_calls += 1
                    total_play_turns += 1
                    segment_turns += 1
                    segment_steps.append(Step(
                        chat_completions=messages,
                        observation=obs_prompt,
                        model_response="",
                        action=failure.kind,
                        thought=failure.thought,
                    ))
                    force_rewind = True
                    rewind_reason = failure.rewind_reason
                    rewind_kind = "forced"
                    break

                llm_action_s += time.perf_counter() - t_llm
                total_llm_calls += 1
                total_play_turns += 1
                segment_turns += 1
                active_messages.append(_message("assistant", content))
                global_messages.append(_message("assistant", content))
                messages = _copy_messages(active_messages)

                if resp.choices[0].finish_reason == "length":
                    segment_steps.append(Step(
                        chat_completions=messages,
                        observation=obs_prompt,
                        model_response=content,
                        action="truncated",
                        thought=content,
                    ))
                    force_rewind = True
                    rewind_reason = "forced rewind: response truncated by token length"
                    rewind_kind = "forced"
                    break

                command = parse_agent_command(content)
                if command.kind == "invalid":
                    last_valid = False
                    invalid_reason = command.error
                    segment_steps.append(Step(
                        chat_completions=messages,
                        observation=obs_prompt,
                        model_response=content,
                        action="invalid",
                        thought=content,
                    ))
                    continue

                if command.kind == "rewind":
                    target = command.rewind_to
                    if target is None or not (0 <= target < current_position):
                        last_valid = False
                        invalid_reason = f"invalid rewind target {command.raw!r}; valid targets are {_format_checkpoint_range(current_position)}"
                        segment_steps.append(Step(
                            chat_completions=messages,
                            observation=obs_prompt,
                            model_response=content,
                            action="invalid_rewind",
                            thought=content,
                        ))
                        continue
                    force_rewind = True
                    rewind_reason = f"model rewind: requested rewind to C_{target}"
                    rewind_target = target
                    rewind_kind = "model"
                    segment_steps.append(Step(
                        chat_completions=messages,
                        observation=obs_prompt,
                        model_response=content,
                        action=f"rewind to C_{target}",
                        thought=content,
                    ))
                    break

                assert command.action is not None
                action = command.action
                last_action = action
                previous_observation = observation
                previous_commands = list(admissible_commands)

                t_env = time.perf_counter()
                result = await session.step(action)
                env_step_s += time.perf_counter() - t_env

                observation = result.observation
                won = bool(result.won)
                done = bool(result.done)
                admissible_commands = result.info.get("admissible_commands", [])

                if won:
                    outcome = "won"
                elif done:
                    outcome = "done without success"
                elif "Nothing happens" in observation:
                    outcome = "no effect"
                else:
                    outcome = "advanced"

                segment_steps.append(Step(
                    chat_completions=messages,
                    observation=obs_prompt,
                    model_response=content,
                    action=action,
                    thought=content,
                    reward=1.0 if won else 0.0,
                    done=done,
                    metadata={"won": won, "outcome": outcome},
                ))
                next_position = current_position + 1
                interaction_history.append({
                    "position_before": current_position,
                    "position_after": next_position,
                    "action": action,
                    "outcome": outcome,
                    "observation_before": previous_observation,
                    "observation_after": observation,
                    "admissible_before": previous_commands,
                    "admissible_after": list(admissible_commands),
                })
                path_actions.append(action)
                path_observations.append(observation)
                path_infos.append(result.info)

                if won:
                    break
                if done:
                    force_rewind = True
                    rewind_reason = "forced rewind: environment reached done=True without success"
                    rewind_kind = "forced"
                    break

            if segment_steps:
                # Milestone count along the active path at this segment's end.
                # Milestones are monotone along path_actions, so the path-end count
                # equals the segment max; a win is credited the full expert length
                # (mirrors LaMer alfworld). Stored for the milestone-diff reflection
                # reward; ignored by the default cum_reward source.
                seg_milestone = (
                    total_milestones if won else _count_milestones(path_actions, expert_milestones)
                )
                trajectories.append(Trajectory(
                    name=f"alfworld_seg{segment_idx}",
                    steps=segment_steps,
                    reward=None,
                    metadata={"milestone_at_end": seg_milestone},
                ))
            if won or total_play_turns >= step_budget or total_llm_calls >= max_total_turns:
                if not won and exhausted_reason is None:
                    exhausted_reason = "budget exhausted"
                break
            if not force_rewind:
                exhausted_reason = exhausted_reason or "segment ended without rewind trigger"
                break
            if segment_idx + 1 >= max_segments:
                exhausted_reason = "segment limit exhausted"
                break

            current_position = len(interaction_history)
            if current_position <= 0:
                rewind_to = 0
                context_folds += 1
            else:
                t_reflect = time.perf_counter()
                reflect_content, memory, selected_target, parse_error, reflect_prompt = await _reflect(
                    client,
                    config.model,
                    sampling,
                    current_position,
                    rewind_reason,
                    interaction_history,
                    segment_start,
                    task.id,
                )
                llm_reflection_s += time.perf_counter() - t_reflect
                total_llm_calls += 1
                if memory:
                    branch_memories.append(memory)
                branch_memory_records.append({
                    "segment": segment_idx,
                    "reason": rewind_reason,
                    "memory": memory,
                    "reflection_parse_error": parse_error,
                })
                trajectories.append(Trajectory(
                    name=f"alfworld_reflect{segment_idx}",
                    steps=[Step(
                        chat_completions=[{"role": "user", "content": reflect_prompt}, {"role": "assistant", "content": reflect_content}],
                        observation=reflect_prompt,
                        model_response=reflect_content,
                        action=f"rewind to C_{selected_target}" if selected_target is not None else "reflect",
                        thought=reflect_content,
                    )],
                    reward=None,
                ))
                rewind_to = selected_target if selected_target is not None else rewind_target
                if rewind_to is None:
                    rewind_to = max(0, min(segment_start, current_position - 1))

            rewind_to = max(0, min(rewind_to, current_position))
            if rewind_kind == "model":
                model_rewinds += 1
            else:
                forced_rewinds += 1

            await session.close()
            path_actions = path_actions[:rewind_to]
            interaction_history = interaction_history[:rewind_to]
            path_observations = path_observations[:rewind_to + 1]
            path_infos = path_infos[:rewind_to + 1]
            session, observation, info, replay_init_s, replay_step_s = await _replay_to(
                game_file, max_steps, path_actions, rewind_to,
            )
            env_init_s += replay_init_s
            env_step_s += replay_step_s
            admissible_commands = info.get("admissible_commands", [])
            if len(path_observations) <= rewind_to:
                path_observations.append(observation)
                path_infos.append(info)
            else:
                path_observations[rewind_to] = observation
                path_infos[rewind_to] = info

            rewind_log.append({
                "segment": segment_idx,
                "kind": rewind_kind,
                "from": current_position,
                "to": rewind_to,
                "reason": rewind_reason,
            })
            segment_idx += 1
    finally:
        await session.close()

    _assign_rewards(trajectories, won, traj_gamma)
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
            "success": won,
            "turns": total_play_turns,
            "total_play_turns": total_play_turns,
            "total_llm_calls": total_llm_calls,
            "segments": segment_idx + 1,
            "total_milestones": total_milestones,
            "has_milestones": expert_milestones is not None,
            "rewinds": len(rewind_log),
            "forced_rewinds": forced_rewinds,
            "model_rewinds": model_rewinds,
            "context_folds": context_folds,
            "rewind_log": rewind_log,
            "branch_memories": branch_memories,
            "branch_memory_records": branch_memory_records,
            "exhausted_reason": exhausted_reason,
            "last_action": last_action,
            "task_type": task_type,
            "env_backend": "ray_replay",
            "global_messages": global_messages,
        },
        is_correct=won,
    )

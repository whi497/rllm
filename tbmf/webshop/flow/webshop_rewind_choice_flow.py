"""Replay-based rewind-choice WebShop flow."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
import time
from typing import Any, Literal

from env_service import create_env_session, parse_remark
from env_service.webshop import WebShopEnv
from openai import AsyncOpenAI

import rllm
from rllm.types import AgentConfig, Episode, Step, Task, Trajectory
from tbmf.flow_utils import classify_llm_failure

try:
    from ..prepare_webshop_data import LAMER_WEBSHOP_CONFIG
    from .webshop_flow import (
        _ACTION_RE,
        _GENERIC_RE,
        _build_system_prompt,
        _build_user_prompt,
        _strip_think_block,
        _valid_action,
        load_world_model_summary,
        parse_action,
    )
except (ImportError, ValueError):
    from prepare_webshop_data import LAMER_WEBSHOP_CONFIG
    from webshop_flow import (
        _ACTION_RE,
        _GENERIC_RE,
        _build_system_prompt,
        _build_user_prompt,
        _strip_think_block,
        _valid_action,
        load_world_model_summary,
        parse_action,
    )

logger = logging.getLogger(__name__)

DEFAULT_SEGMENT_MAX_TURNS = 10
DEFAULT_MAX_SEGMENTS = 8
DEFAULT_MAX_TOTAL_TURNS = 80
DEFAULT_TRAJ_GAMMA = 0.7
MAX_BRANCH_MEMORIES_IN_CONTEXT = 4
MAX_BRANCH_MEMORY_CHARS = 1800
MAX_BRANCH_HISTORY_CHARS = 18000

_REWIND_RE = re.compile(r"rewind\s+to\s+(?:C\s*[_-]?\s*)?(\d+)\s*$", re.IGNORECASE)


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


def parse_agent_command(response: str, has_search_bar: bool) -> AgentCommand:
    raw = _extract_action_text(response)
    if not raw:
        return AgentCommand(kind="invalid", raw=raw, error="missing action block")
    rewind_match = _REWIND_RE.fullmatch(raw)
    if rewind_match:
        return AgentCommand(kind="rewind", rewind_to=int(rewind_match.group(1)), raw=raw)
    action = parse_action(response, has_search_bar=has_search_bar)
    if not action:
        return AgentCommand(kind="invalid", raw=raw, error="missing valid WebShop action")
    return AgentCommand(kind="action", action=action, raw=raw)


def _format_checkpoint_range(current_position: int) -> str:
    if current_position <= 0:
        return "(none)"
    if current_position == 1:
        return "C_0"
    return f"C_0..C_{current_position - 1}"


def _branch_memory_context(branch_memories: list[str]) -> str:
    if not branch_memories:
        return ""
    lines = ["\n\n## Branch memories from previous failed branches"]
    for idx, memory in enumerate(branch_memories[-MAX_BRANCH_MEMORIES_IN_CONTEXT:], start=1):
        lines.append(f"Memory #{idx}: {_truncate(memory, MAX_BRANCH_MEMORY_CHARS)}")
    return "\n".join(lines)


def _build_rewind_user_prompt(
    observation: str,
    instruction: str,
    available_actions: dict,
    current_position: int,
    total_turns: int,
    step_budget: int,
    branch_turns: int,
    segment_max_turns: int,
    branch_memories: list[str],
    *,
    use_available_actions: bool,
    action_is_valid: bool,
) -> str:
    prompt = _build_user_prompt(
        observation,
        instruction,
        available_actions,
        turn=total_turns,
        max_turns=step_budget,
        use_available_actions=use_available_actions,
        action_is_valid=action_is_valid,
    )
    prompt += (
        f"\n\nCurrent checkpoint: C_{current_position}.\n"
        f"Current-branch budget used: {branch_turns}/{segment_max_turns}.\n"
        f"Valid rewind targets: {_format_checkpoint_range(current_position)}.\n"
        "You may also travel back with exactly:\n```action\nrewind to C_j\n```"
        f"{_branch_memory_context(branch_memories)}"
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
                f"Task score: {entry['task_score']}",
                "Observation before:",
                entry["observation_before"],
                "Observation after:",
                entry["observation_after"],
                "",
            ]
        )
    return _truncate("\n".join(lines).strip() or "(no branch history)", MAX_BRANCH_HISTORY_CHARS)


REFLECT_PROMPT = """\
You are reflecting on a failed WebShop branch.

# Shopping task
{instruction}

# Rewind trigger
{rewind_reason}

# Current checkpoint
The branch is currently at C_{current_position}.

# Valid rewind targets
{valid_targets}

# Branch history
{branch_history}

Write a compact memory that helps the next attempt avoid the failed branch.
Include wrong search queries, wrong products, missing options, wrong clicks,
and useful product/page facts.

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


async def _new_session(meta: dict[str, Any]):
    session = await create_env_session(
        WebShopEnv,
        session_mode="ray_pool",
        observation_mode=str(meta["observation_mode"]),
        max_steps=int(meta["max_steps"]),
        num_products=meta["num_products"],
        session_id=int(meta["session_id"]),
        seed=int(meta["seed"]),
        file_path=meta.get("file_path"),
        attr_path=meta.get("attr_path"),
        human_goals=bool(meta["human_goals"]),
    )
    observation, info = await session.reset()
    return session, observation, info


async def _replay_to(
    meta: dict[str, Any],
    actions: list[str],
    position: int,
) -> tuple[Any, str, dict[str, Any], float, float, float, bool, bool]:
    t0 = time.perf_counter()
    session, observation, info = await _new_session(meta)
    env_init_s = time.perf_counter() - t0
    env_step_s = 0.0
    reward = 0.0
    done = False
    won = False
    for action in actions[:position]:
        t_step = time.perf_counter()
        result = await session.step(action)
        env_step_s += time.perf_counter() - t_step
        observation = result.observation
        info = result.info
        reward = result.reward
        done = result.done
        won = result.won
        if done:
            break
    return session, observation, info, env_init_s, env_step_s, reward, done, won


async def _reflect(
    client: AsyncOpenAI,
    model: str,
    sampling: dict[str, Any],
    instruction: str,
    current_position: int,
    rewind_reason: str,
    history: list[dict[str, Any]],
    history_start: int,
    task_id: str,
) -> tuple[str, str, int | None, str, str]:
    prompt = REFLECT_PROMPT.format(
        instruction=instruction,
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
        logger.warning("webshop_rewind_choice task %s: reflection failed: %s", task_id, e)
        content = ""
    memory = parse_remark(content) if content else ""
    if not memory and content:
        memory = content.strip()
    memory = _truncate(memory, MAX_BRANCH_MEMORY_CHARS) if memory else ""
    target, parse_error = _parse_rewind_target(content, current_position) if content else (None, "empty reflection")
    return content, memory, target, parse_error, prompt


def _assign_rewards(trajectories: list[Trajectory], won: bool, gamma: float) -> None:
    play = [(int(re.search(r"(\d+)$", traj.name).group(1)), idx) for idx, traj in enumerate(trajectories) if traj.name.startswith("webshop_seg")]
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


@rllm.rollout(name="webshop_rewind_choice")
async def webshop_rewind_choice_flow(task: Task, config: AgentConfig) -> Episode:
    meta = task.metadata or {}
    flow_meta = {
        "session_id": int(meta.get("session_id")),
        "max_steps": int(meta.get("max_steps", LAMER_WEBSHOP_CONFIG["max_steps"])),
        "max_turns": int(meta.get("max_turns", meta.get("max_steps", LAMER_WEBSHOP_CONFIG["max_steps"]))),
        "seed": int(meta.get("seed", LAMER_WEBSHOP_CONFIG["env_seed"])),
        "observation_mode": str(meta.get("observation_mode", LAMER_WEBSHOP_CONFIG["observation_mode"])),
        "num_products": meta.get("num_products", LAMER_WEBSHOP_CONFIG["num_products"]),
        "human_goals": bool(meta.get("human_goals", LAMER_WEBSHOP_CONFIG["human_goals"])),
        "file_path": meta.get("file_path"),
        "attr_path": meta.get("attr_path"),
    }
    step_budget = int(meta.get("step_budget", flow_meta["max_turns"]))
    segment_max_turns = int(meta.get("segment_max_turns", DEFAULT_SEGMENT_MAX_TURNS))
    max_segments = int(meta.get("max_segments", DEFAULT_MAX_SEGMENTS))
    max_total_turns = int(meta.get("max_total_turns", max(DEFAULT_MAX_TOTAL_TURNS, step_budget + max_segments)))
    traj_gamma = float(meta.get("traj_gamma", DEFAULT_TRAJ_GAMMA))
    use_available_actions = bool(meta.get("use_available_actions", True))
    use_accumulate_history = bool(meta.get("use_accumulate_history", True))
    use_accumulate_thinking = bool(meta.get("use_accumulate_thinking", True))
    world_model_summary = load_world_model_summary(meta.get("world_model_file"))

    client = AsyncOpenAI(base_url=config.base_url, api_key="EMPTY")
    sampling = {k: v for k, v in config.sampling_params.items() if k != "top_k"}
    system_prompt = _build_system_prompt(world_model_summary)

    t0 = time.perf_counter()
    session, observation, info = await _new_session(flow_meta)
    env_init_s = time.perf_counter() - t0
    env_step_s = 0.0
    llm_action_s = 0.0
    llm_reflection_s = 0.0
    available_actions = info.get("available_actions", {})
    instruction = info.get("instruction", task.instruction if isinstance(task.instruction, str) else "")

    trajectories: list[Trajectory] = []
    interaction_history: list[dict[str, Any]] = []
    path_actions: list[str] = []
    branch_memories: list[str] = []
    branch_memory_records: list[dict[str, Any]] = []
    rewind_log: list[dict[str, Any]] = []
    global_messages: list[dict[str, str]] = [_message("system", system_prompt)]
    won = False
    done = False
    final_reward = 0.0
    task_score = 0.0
    total_play_turns = 0
    total_llm_calls = 0
    segment_idx = 0
    forced_rewinds = 0
    model_rewinds = 0
    context_folds = 0
    last_action: str | None = None
    exhausted_reason: str | None = None

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
            active_messages = [_message("system", system_prompt)]
            last_valid = True
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

                obs_prompt = _build_rewind_user_prompt(
                    observation,
                    instruction,
                    available_actions,
                    current_position,
                    total_play_turns,
                    step_budget,
                    segment_turns,
                    segment_max_turns,
                    branch_memories,
                    use_available_actions=use_available_actions,
                    action_is_valid=last_valid,
                )
                active_messages.append(_message("user", obs_prompt))
                global_messages.append(_message("user", obs_prompt))
                model_messages = _copy_messages(active_messages if use_accumulate_history else [active_messages[0], active_messages[-1]])

                t_llm = time.perf_counter()
                try:
                    resp = await client.chat.completions.create(
                        model=config.model, messages=model_messages, **sampling, timeout=120,
                    )
                    raw_content = resp.choices[0].message.content or ""
                except Exception as e:
                    failure = classify_llm_failure(e)
                    llm_action_s += time.perf_counter() - t_llm
                    total_llm_calls += 1
                    total_play_turns += 1
                    segment_turns += 1
                    segment_steps.append(Step(
                        chat_completions=model_messages,
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
                assistant_content = raw_content if use_accumulate_thinking else _strip_think_block(raw_content)
                active_messages.append(_message("assistant", assistant_content))
                global_messages.append(_message("assistant", assistant_content))
                step_messages = list(model_messages) + [_message("assistant", assistant_content)]

                if resp.choices[0].finish_reason == "length":
                    segment_steps.append(Step(
                        chat_completions=step_messages,
                        observation=obs_prompt,
                        model_response=assistant_content,
                        action="truncated",
                        thought=assistant_content,
                    ))
                    force_rewind = True
                    rewind_reason = "forced rewind: response truncated by token length"
                    rewind_kind = "forced"
                    break

                command = parse_agent_command(raw_content, has_search_bar=bool(available_actions.get("has_search_bar", False)))
                if command.kind == "invalid":
                    last_valid = False
                    segment_steps.append(Step(
                        chat_completions=step_messages,
                        observation=obs_prompt,
                        model_response=assistant_content,
                        action="invalid",
                        thought=assistant_content,
                    ))
                    continue

                if command.kind == "rewind":
                    target = command.rewind_to
                    if target is None or not (0 <= target < current_position):
                        last_valid = False
                        segment_steps.append(Step(
                            chat_completions=step_messages,
                            observation=obs_prompt,
                            model_response=assistant_content,
                            action="invalid_rewind",
                            thought=assistant_content,
                        ))
                        continue
                    force_rewind = True
                    rewind_reason = f"model rewind: requested rewind to C_{target}"
                    rewind_target = target
                    rewind_kind = "model"
                    segment_steps.append(Step(
                        chat_completions=step_messages,
                        observation=obs_prompt,
                        model_response=assistant_content,
                        action=f"rewind to C_{target}",
                        thought=assistant_content,
                    ))
                    break

                assert command.action is not None
                action = command.action
                action_is_valid = _valid_action(action, available_actions)
                if not action_is_valid:
                    last_valid = False
                    segment_steps.append(Step(
                        chat_completions=step_messages,
                        observation=obs_prompt,
                        model_response=assistant_content,
                        action="invalid",
                        thought=assistant_content,
                        metadata={"invalid_action": action},
                    ))
                    continue

                last_action = action
                previous_observation = observation
                previous_actions = dict(available_actions)
                t_env = time.perf_counter()
                result = await session.step(action)
                env_step_s += time.perf_counter() - t_env
                observation = result.observation
                final_reward = result.reward
                won = result.won
                done = result.done
                task_score = float(result.info.get("task_score", final_reward))
                available_actions = result.info.get("available_actions", {})

                if won:
                    outcome = "won"
                elif done:
                    outcome = "done without success"
                elif observation == previous_observation:
                    outcome = "no visible change"
                else:
                    outcome = "advanced"

                segment_steps.append(Step(
                    chat_completions=step_messages,
                    observation=obs_prompt,
                    model_response=assistant_content,
                    action=action,
                    thought=assistant_content,
                    reward=final_reward,
                    done=done,
                    metadata={
                        "won": won,
                        "task_score": task_score,
                        "action_is_valid": action_is_valid,
                        "outcome": outcome,
                    },
                ))
                current_position = len(interaction_history)
                interaction_history.append({
                    "position_before": current_position,
                    "position_after": current_position + 1,
                    "action": action,
                    "outcome": outcome,
                    "task_score": task_score,
                    "observation_before": previous_observation,
                    "observation_after": observation,
                    "available_actions_before": previous_actions,
                    "available_actions_after": available_actions,
                })
                path_actions.append(action)
                if won:
                    break
                if done:
                    force_rewind = True
                    rewind_reason = "forced rewind: environment reached done=True without success"
                    rewind_kind = "forced"
                    break

            if segment_steps:
                trajectories.append(Trajectory(name=f"webshop_seg{segment_idx}", steps=segment_steps, reward=None))
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
                    instruction,
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
                    name=f"webshop_reflect{segment_idx}",
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
            session, observation, info, replay_init_s, replay_step_s, final_reward, done, won = await _replay_to(
                flow_meta, path_actions, rewind_to,
            )
            env_init_s += replay_init_s
            env_step_s += replay_step_s
            available_actions = info.get("available_actions", {})
            instruction = info.get("instruction", instruction)
            task_score = float(info.get("task_score", final_reward))
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
            "task_score": task_score,
            "reward": final_reward,
            "turns": total_play_turns,
            "env_steps": len(path_actions),
            "total_play_turns": total_play_turns,
            "total_llm_calls": total_llm_calls,
            "segments": segment_idx + 1,
            "rewinds": len(rewind_log),
            "forced_rewinds": forced_rewinds,
            "model_rewinds": model_rewinds,
            "context_folds": context_folds,
            "rewind_log": rewind_log,
            "branch_memories": branch_memories,
            "branch_memory_records": branch_memory_records,
            "exhausted_reason": exhausted_reason,
            "last_action": last_action,
            "session_id": flow_meta["session_id"],
            "instruction": instruction,
            "max_steps": flow_meta["max_steps"],
            "max_turns": flow_meta["max_turns"],
            "observation_mode": flow_meta["observation_mode"],
            "num_products": flow_meta["num_products"],
            "human_goals": flow_meta["human_goals"],
            "env_backend": "ray_pool_replay",
            "global_messages": global_messages,
        },
        is_correct=won,
    )

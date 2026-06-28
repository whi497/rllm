#!/usr/bin/env python3
"""Comprehensive eval-trajectory visualizer / audit tool.

A single self-contained Flask app that serves a web UI over a TBMF eval run
directory (the kind produced by ``tbmf.eval_flows``):

  <run_dir>/
    summary.json
    trajectories.jsonl
    episodes/<variant>/episode_*.json

It exposes a small JSON API and a one-page client that lets you:
  - see the per-flow x env dashboard (win-rate + signal averages),
  - browse every task per variant (won/lost, key signals),
  - replay a trajectory step-by-step with the rendered game board, the model's
    full reasoning, and the parsed action,
  - for the rewind flow: inspect the segment/checkpoint structure, every rewind
    (forced vs model) with its reason, and the injected branch memories.

Run::

    python3 rllm/tbmf/viz/traj_viewer.py \
        --run-dir eval_outputs/three_flows_full/20260622_020221 \
        --port 5230

Then open the printed URL.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from flask import Flask, abort, jsonify, send_from_directory

app = Flask(__name__)
RUN_DIR: Path = Path(".")


# ----------------------------------------------------------------------------
# Data access helpers
# ----------------------------------------------------------------------------

def _episodes_root() -> Path:
    return RUN_DIR / "episodes"


def _variants() -> list[str]:
    root = _episodes_root()
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def _episode_files(variant: str) -> list[Path]:
    d = _episodes_root() / variant
    if not d.is_dir():
        return []
    return sorted(d.glob("episode_*.json"))


@lru_cache(maxsize=4096)
def _load_episode(variant: str, fname: str) -> dict[str, Any]:
    path = _episodes_root() / variant / fname
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _load_summary() -> dict[str, Any]:
    path = RUN_DIR / "summary.json"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------
# Per-step extraction: recover the board observation + action for each turn.
#
# The eval dataset does not store ``observation`` separately; the rendered game
# state lives inside the conversation. For each Step, the board the model saw is
# the LAST ``user`` message in that step's ``chat_completions`` (the observation
# prompt right before the assistant reply). We slice out the board grid and the
# budget header so the UI can show a compact per-turn view.
# ----------------------------------------------------------------------------

_ACTION_TAG_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.IGNORECASE | re.DOTALL)
# Optional row label: bare "0:" / "12:" (sokoban) or "Row 3:" (minesweeper).
_ROW_LABEL_RE = re.compile(r"^\s*(?:row\s*)?\d+\s*:\s*", re.IGNORECASE)
# Glyphs that make up a board cell grid.
_BOARD_GLYPHS = set("#_.*?XOP√S0123456789 |-")


def _last_user_content(chat: list[dict[str, str]] | None) -> str:
    if not chat:
        return ""
    for msg in reversed(chat):
        if msg.get("role") == "user":
            return msg.get("content", "") or ""
    return ""


def _is_board_line(ln: str) -> bool:
    """A board row: optional ``Row N:`` / ``N:`` label, then only cell glyphs,
    and containing at least one real cell marker (not just digits/spaces)."""
    body = _ROW_LABEL_RE.sub("", ln).strip()
    if not body:
        return False
    if any(ch not in _BOARD_GLYPHS for ch in body):
        return False
    return any(ch in body for ch in "#_.*?XOP√")


def _extract_board(observation_text: str) -> str:
    """Pull the contiguous board-grid block out of an observation prompt.

    The board is the first run of consecutive lines that look like a grid
    (glyph rows, optionally row-labelled). Returns "" if none found.
    """
    if not observation_text:
        return ""
    block: list[str] = []
    started = False
    for ln in observation_text.splitlines():
        if _is_board_line(ln):
            block.append(ln.rstrip())
            started = True
        elif started:
            break  # board block ended
    return "\n".join(block).strip()


def _final_action_text(model_response: str) -> str:
    matches = _ACTION_TAG_RE.findall(model_response or "")
    if matches:
        return matches[-1].strip()
    return ""


def _step_view(step: dict[str, Any]) -> dict[str, Any]:
    """Compact per-step record for the UI."""
    chat = step.get("chat_completions") or []
    obs = _last_user_content(chat)
    model_response = step.get("model_response") or step.get("thought") or ""
    return {
        "action": step.get("action", ""),
        "final_action": _final_action_text(model_response),
        "done": bool(step.get("done", False)),
        "reward": step.get("reward", 0.0),
        "board": _extract_board(obs),
        "observation": obs,
        "model_response": model_response,
        "is_reflection": (step.get("action") == "reflect")
        or (len(chat) == 2 and chat and chat[0].get("role") == "user"),
        "n_messages": len(chat),
    }


def _trajectory_view(traj: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": traj.get("name", ""),
        "reward": traj.get("reward"),
        "n_steps": len(traj.get("steps") or []),
        "signals": traj.get("signals") or {},
        "steps": [_step_view(s) for s in (traj.get("steps") or [])],
    }


def _episode_index(variant: str) -> list[dict[str, Any]]:
    """Lightweight list of all episodes in a variant (for the task list)."""
    rows: list[dict[str, Any]] = []
    for i, path in enumerate(_episode_files(variant)):
        ep = _load_episode(variant, path.name)
        art = ep.get("artifacts") or {}
        task = ep.get("task") or {}
        rows.append(
            {
                "idx": ep.get("eval_idx", i),
                "fname": path.name,
                "task_id": task.get("uid") or ep.get("id"),
                "won": bool(art.get("won", ep.get("is_correct", False))),
                "is_correct": bool(ep.get("is_correct", False)),
                "termination": ep.get("termination_reason"),
                # headline signals depending on family
                "turns": art.get("turns") or art.get("turns_total") or art.get("total_play_turns"),
                "env_steps": art.get("env_steps") or art.get("env_steps_total") or art.get("total_env_steps"),
                "rewinds": art.get("rewinds"),
                "segments": art.get("segments"),
                "episodes_played": art.get("episodes_played"),
                "episode_rewards": art.get("episode_rewards"),
                "solution_length": task.get("solution_length"),
            }
        )
    return rows


# ----------------------------------------------------------------------------
# API
# ----------------------------------------------------------------------------

@app.route("/api/meta")
def api_meta():
    summary = _load_summary()
    return jsonify(
        {
            "run_dir": str(RUN_DIR),
            "model": summary.get("model"),
            "base_url": summary.get("base_url"),
            "max_examples": summary.get("max_examples"),
            "variants": _variants(),
            "runs": summary.get("runs", {}),
        }
    )


@app.route("/api/episodes/<variant>")
def api_episodes(variant: str):
    if variant not in _variants():
        abort(404)
    return jsonify(_episode_index(variant))


@app.route("/api/episode/<variant>/<fname>")
def api_episode(variant: str, fname: str):
    if variant not in _variants():
        abort(404)
    if not re.fullmatch(r"episode_[\w.\-]+\.json", fname):
        abort(400)
    try:
        ep = _load_episode(variant, fname)
    except FileNotFoundError:
        abort(404)
    art = ep.get("artifacts") or {}
    return jsonify(
        {
            "id": ep.get("id"),
            "variant": variant,
            "eval_idx": ep.get("eval_idx"),
            "is_correct": ep.get("is_correct"),
            "termination_reason": ep.get("termination_reason"),
            "task": ep.get("task") or {},
            "artifacts": art,
            "metrics": ep.get("metrics") or {},
            "trajectories": [_trajectory_view(t) for t in (ep.get("trajectories") or [])],
        }
    )


@app.route("/")
def index():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "index.html")


@app.route("/favicon.ico")
def favicon():
    return ("", 204)


def main() -> None:
    global RUN_DIR
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, help="Path to an eval run directory containing summary.json + episodes/")
    ap.add_argument("--port", type=int, default=5230)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    RUN_DIR = Path(args.run_dir).resolve()
    if not (RUN_DIR / "episodes").is_dir():
        raise SystemExit(f"No episodes/ under {RUN_DIR} — is this an eval run dir?")

    print(f"Serving trajectory viewer for: {RUN_DIR}")
    print(f"  variants: {_variants()}")
    print(f"  open: http://localhost:{args.port}/")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()

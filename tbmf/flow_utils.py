"""Shared helpers for TBMF rollout flows."""

from __future__ import annotations

from dataclasses import dataclass


_CONTEXT_ERROR_PATTERNS = (
    "context length",
    "maximum context",
    "max context",
    "context window",
    "maximum number of tokens",
    "maximum token",
    "prompt is too long",
    "input is too long",
    "too many tokens",
    "token limit",
    "length limit",
    "exceeds the model",
    "exceeded model",
)


@dataclass(frozen=True)
class LLMFailure:
    kind: str
    thought: str
    outcome: str
    rewind_reason: str


def classify_llm_failure(error: BaseException) -> LLMFailure:
    """Classify LLM call failures for trajectory and rewind logging."""

    detail = str(error)
    lowered = detail.lower()
    if any(pattern in lowered for pattern in _CONTEXT_ERROR_PATTERNS):
        return LLMFailure(
            kind="context_length_exceeded",
            thought=f"LLM context length exceeded: {detail}",
            outcome=f"context length exceeded: {detail}",
            rewind_reason="forced rewind: LLM context length exceeded",
        )
    return LLMFailure(
        kind="llm_failed",
        thought=f"LLM call failed: {detail}",
        outcome=detail,
        rewind_reason="forced rewind: LLM call failed",
    )

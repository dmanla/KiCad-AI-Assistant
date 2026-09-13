"""Unified stream-lifecycle event queue core (wx-free).

The background LLM thread is the *sole producer* of turn-internal events
(text_start / text_chunk / text_end / tool_call / turn_end) and enqueues
them in occurrence order into a FIFO deque owned by the panel; the panel's
50 ms timer is the *sole consumer*, which drains the deque and applies each
event through :func:`apply_stream_event`, then renders once.

Because the FIFO order is the only source of truth, a ``text_end`` is always
applied before the ``tool_call`` / ``turn_end`` events that follow it.  The
race where an end-of-turn notification (``reload_kicad``) finalised a draft
that was still being filled by the 50 ms text pipeline is therefore
structurally impossible, and draft finalisation is driven by the explicit
``text_end`` marker instead of by heuristics about pending text.

This module intentionally imports nothing from wx / webview / IO so the
state machine can be unit-tested without a display.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# Event payloads, produced in this order by the background thread:
#   {"type": "text_start"}                              ordering marker (new LLM call) -> reset reasoning
#   {"type": "text_chunk", "content": str}              streamed content
#   {"type": "text_end"}                                response text complete -> finalise draft
#   {"type": "reasoning_chunk", "content": str}         model "thinking" delta (live, ephemeral)
#   {"type": "usage", "input_tokens": int, "output_tokens": int, "total_tokens": int}
#                                                       provider-reported token usage for one LLM call
#   {"type": "context_estimate", "used_tokens": int, "limit_tokens": int}
#                                                       local context-window estimate for one LLM call
#   {"type": "tool_call", "name": str, "args": dict, "result": Any}
#   {"type": "turn_end", "reply": str}                  turn complete (takes over _on_reply)
#   {"type": "status", "text": str, "color_hex": str}   transient system notice (e.g. compacted history)
#
# The panel tags every event with the generation of the turn it belongs to
# (``_gen``) and drops events of older turns before calling apply_stream_event.

AI_ENTRY_TYPES = ("user", "ai")


def make_ai_entry(text: str, timestamp: str) -> dict[str, Any]:
    """Build a permanent AI conversation entry (rendering format unchanged)."""
    return {"type": "ai", "text": text, "timestamp": timestamp}


@dataclass
class TurnState:
    """Scalar turn state after applying one event."""

    pending: str = ""
    tool_calls_made: bool = False
    turn_had_text: bool = False
    delta_chars: int = 0
    # Live model "thinking" text for the active LLM call (ephemeral, never
    # persisted); reset by the ``text_start`` marker of the next call.
    pending_reasoning: str = ""
    # Token/context counters surfaced by the panel's meta bar.
    used_tokens: int = 0
    limit_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    # Mutation flags for the caller's single merged render pass.
    draft_changed: bool = False
    entries_changed: bool = False
    reasoning_changed: bool = False
    meta_changed: bool = False


def apply_stream_event(
    *,
    pending: str,
    entries: list,
    tool_calls_made: bool,
    turn_had_text: bool,
    delta_chars: int,
    cancelled: bool,
    evt: dict[str, Any],
    timestamp: Callable[[], str],
    pending_reasoning: str = "",
    used_tokens: int = 0,
    limit_tokens: int = 0,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> TurnState:
    """Apply one lifecycle event to the turn state (pure; ``entries`` mutated).

    Cancellation: text events are dropped once ``cancelled`` is set (the
    draft keeps whatever was already consumed, the turn-end finalises it);
    ``tool_call`` cards are still inserted so the user sees what the agent
    was doing; ``turn_end`` still closes the turn out.

    Returns the full updated scalar state plus mutation flags.
    """
    base = TurnState(
        pending=pending,
        tool_calls_made=tool_calls_made,
        turn_had_text=turn_had_text,
        delta_chars=delta_chars,
        pending_reasoning=pending_reasoning,
        used_tokens=used_tokens,
        limit_tokens=limit_tokens,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    etype = evt.get("type")
    if etype == "text_start":
        # A new LLM call begins: drop the previous call's reasoning preview
        # (it is done) so the meta bar/thinking block reflects the live call.
        if pending_reasoning:
            base.pending_reasoning = ""
            base.reasoning_changed = True
        return base  # otherwise an ordering marker only

    if etype == "text_chunk":
        if cancelled:
            return base
        content = evt.get("content") or ""
        base.pending = pending + content
        base.turn_had_text = True
        base.delta_chars = delta_chars + len(content)
        base.draft_changed = True
        return base

    if etype == "reasoning_chunk":
        content = evt.get("content") or ""
        if content:
            base.pending_reasoning = pending_reasoning + content
            base.reasoning_changed = True
        return base

    if etype == "usage":
        base.input_tokens = int(evt.get("input_tokens") or 0)
        base.output_tokens = int(evt.get("output_tokens") or 0)
        # Provider-reported input tokens are the most accurate context fill.
        if base.input_tokens > 0:
            base.used_tokens = base.input_tokens
        if evt.get("limit_tokens"):
            base.limit_tokens = int(evt["limit_tokens"])
        base.meta_changed = True
        return base

    if etype == "context_estimate":
        if evt.get("used_tokens") is not None:
            base.used_tokens = int(evt.get("used_tokens") or 0)
        if evt.get("limit_tokens"):
            base.limit_tokens = int(evt["limit_tokens"])
        base.meta_changed = True
        return base

    if etype == "text_end":
        if cancelled or not pending:
            return base
        entries.append(make_ai_entry(pending, timestamp()))
        base.pending = ""
        base.entries_changed = True
        return base

    if etype == "tool_call":
        entries.append(
            {
                "type": "tool_call",
                "name": evt.get("name", "?"),
                "args": evt.get("args"),
                "result": evt.get("result"),
            }
        )
        base.tool_calls_made = True
        base.entries_changed = True
        return base

    if etype == "turn_end":
        if pending:  # defensive: text_end normally finalises the draft first
            entries.append(make_ai_entry(pending, timestamp()))
            base.pending = ""
            base.entries_changed = True
        reply = evt.get("reply") or ""
        if not turn_had_text and reply:
            # Non-streamed turn (LLM/framework error, iteration cap, or a
            # cancelled turn whose text never reached the draft): the reply
            # is the only answer text this turn produced.  Mirrors the old
            # ``was_streamed=False`` branch of _on_reply.
            entries.append(make_ai_entry(reply, timestamp()))
            base.entries_changed = True
        return base

    if etype == "status":
        entries.append(
            {
                "type": "status",
                "text": evt.get("text", ""),
                "color_hex": evt.get("color_hex", "#1E1E1E"),
            }
        )
        base.entries_changed = True
        return base

    return base  # unknown event type — ignore

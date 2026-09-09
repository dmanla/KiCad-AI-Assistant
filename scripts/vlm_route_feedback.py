#!/usr/bin/env python3
"""
VLM-feedback routing experiment driver (issue #124).

Closes the loop between a vision LLM and the existing A* router:

    render board state -> ask VLM (image + prompt) -> parse feedback
    -> auto_route_pair -> success: write to PCB, render new state, next round
                       -> failure: feed RouteFailure back, VLM adjusts, retry

Feedback dimensions (all within existing ``RouteRequest`` knobs):
    * routing order  — which pad pair to connect next
    * layer choice   — ``layer_hint`` for thru-hole pads (SMD layers are fixed)
    * retry strategy — layer hint / pair choice after a RouteFailure

Output protocol expected from the VLM (one line each):

    route: <ref_a>.<pad_a> -> <ref_b>.<pad_b>
    layer: <F.Cu|B.Cu|auto>          (thru-hole only; ignored for SMD)
    reason: <one sentence>

On failure the prompt is extended with:

    last_error: <RouteFailure message>
    advice: <layer / order / give-up recommendation>

Usage:
    export LARK_LLM_BASE_URL, LARK_LLM_MODEL, LARK_LLM_API_KEY
    python scripts/vlm_route_feedback.py                  # default test board
    python scripts/vlm_route_feedback.py --pcb X.kicad_pcb --rounds 10
    python scripts/vlm_route_feedback.py --dry-run        # no LLM: fixed order

Dependencies: matplotlib, shapely, sexpdata (repo already requires them).
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import json
import os
import sys
import tempfile
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# Make the repo root importable (script lives in scripts/).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from kcaa.router.router import (  # noqa: E402
    RouteFailure,
    RouteRequest,
    auto_route_pair,
)
from kcaa.tools.pcb_routing_tools import (  # noqa: E402
    _segment_to_sexp,
    _via_to_sexp,
)
from kcaa.utils.pcb_sexp_utils import load_pcb, save_pcb  # noqa: E402

DEFAULT_PCB = os.path.join(
    _REPO_ROOT, "tests", "integration", "fixtures", "test_routing_board.kicad_pcb"
)

# Layer hops the router may use for multi-layer routes on the experiment
# board (F.Cu top, B.Cu bottom, In1.Cu inner).  Pass --via-pairs to override.
DEFAULT_VIA_PAIRS: tuple[tuple[str, str], ...] = (("F.Cu", "B.Cu"), ("B.Cu", "In1.Cu"))

_SYSTEM_PROMPT = """You are a PCB routing planner. You will be shown the current
state of a printed circuit board as an image. Copper pads are drawn as
rectangles labelled with their reference designator and pad number
(e.g. "R1.1"); existing routed tracks are drawn as lines between pads.

Your job: decide what to route next. Reply with exactly three lines:

  route: <ref_a>.<pad_a> -> <ref_b>.<pad_b>
  layer: <F.Cu|B.Cu|auto>
  reason: <one short sentence>

Rules:
- Connect one pad pair that is still unconnected (from the asked list).
- If you are given a last_error and asked for advice, either change the
  layer (thru-hole only), pick a different pair, or give up on that pair.
- Pads on different nets cannot be connected: only connect pairs from the
  asked list.
- Do not invent pad numbers that were not shown."""  # noqa: E501

_NET_COLORS = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e", "#8c564b"]


def _sym(value: Any) -> str:
    """Return the string form of a sexpdata Symbol or plain string.

    sexpdata.Symbol subclasses str but overrides ``__eq__``, so
    ``Symbol('pad') == 'pad'`` is False.  Always compare via this helper.
    """
    return str(value)


def _net_color(net: str) -> str:
    idx = abs(hash(net)) % len(_NET_COLORS)
    return _NET_COLORS[idx]


@dataclass
class VLMFeedback:
    """Parsed semantic feedback from the VLM."""

    route_a: str | None = None
    route_b: str | None = None
    layer: str | None = None
    reason: str = ""


def parse_feedback(text: str) -> VLMFeedback:
    """Parse the VLM's reply into structured feedback.

    Accepts the documented ``route:`` / ``layer:`` / ``reason:`` lines
    regardless of surrounding prose.  Returns a :class:`VLMFeedback` with
    ``None`` pads when no ``route:`` line is found.
    """
    fb = VLMFeedback()
    for line in text.splitlines():
        stripped = line.strip()
        low = stripped.lower()
        if low.startswith("route:"):
            value = stripped[len("route:") :].strip()
            parts = [p.strip() for p in value.replace("->", "→").replace("→", " ").split()]
            # Only pad specs carry a dot ("R1.1"); drop separators ("-", "to").
            parts = [p for p in parts if "." in p]
            if len(parts) >= 2:
                fb.route_a = parts[0]
                fb.route_b = parts[1]
        elif low.startswith("layer:"):
            fb.layer = stripped[len("layer:") :].strip()
        elif low.startswith("reason:"):
            fb.reason = stripped[len("reason:") :].strip()
    return fb


def parse_pad(spec: str) -> tuple[str, str]:
    """Split ``R1.1`` into ``("R1", "1")``.  Raises ValueError on bad input."""
    if "." not in spec:
        raise ValueError(f"expected '<ref>.<pad>', got {spec!r}")
    ref, pad = spec.rsplit(".", 1)
    ref = ref.strip()
    pad = pad.strip()
    if not ref or not pad:
        raise ValueError(f"empty ref/pad in {spec!r}")
    return ref, pad


@dataclass
class PairSpec:
    """One pad pair to route, plus the net that joins them."""

    ref_a: str
    pad_a: str
    ref_b: str
    pad_b: str
    net: str
    via_pairs: tuple[tuple[str, str], ...] | None = None
    attempts: int = 0
    status: str = "pending"  # pending | done | failed

    @property
    def key(self) -> str:
        return f"{self.ref_a}.{self.pad_a}-{self.ref_b}.{self.pad_b}"

    @property
    def description(self) -> str:
        return f"{self.ref_a}.{self.pad_a} -> {self.ref_b}.{self.pad_b} (net {self.net})"


class VLMClient:
    """Minimal OpenAI-compatible chat client for the experiment.

    Sends one image + text prompt and returns the assistant text.  Uses
    stdlib ``urllib`` only, configured from environment variables so the
    script needs no config-file plumbing:

    - ``LARK_LLM_BASE_URL``  (default https://api.openai.com/v1)
    - ``LARK_LLM_MODEL``     (default gpt-4o)
    - ``LARK_LLM_API_KEY``   (default empty)
    """

    def __init__(self) -> None:
        self.base_url = os.environ.get("LARK_LLM_BASE_URL", "https://api.openai.com/v1")
        self.model = os.environ.get("LARK_LLM_MODEL", "gpt-4o")
        self.api_key = os.environ.get("LARK_LLM_API_KEY", "")

    def ask(self, png_bytes: bytes, user_prompt: str) -> str:
        """Send a vision request; return the assistant's text reply."""
        import urllib.request

        b64 = base64.b64encode(png_bytes).decode("ascii")
        content: list[dict[str, Any]] = [
            {"type": "text", "text": user_prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            },
        ]
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            "max_tokens": 512,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url = self.base_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"

        req = urllib.request.Request(  # nosec B310 -- user-configured LLM endpoint
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:  # noqa: S310
                body = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(f"VLM request failed: {exc}") from exc

        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected VLM response: {body!r}") from exc


def _footprint_ref(fp: list) -> str | None:
    for sub in fp:
        if (
            isinstance(sub, list)
            and len(sub) >= 3
            and _sym(sub[0]) == "property"
            and sub[1] == "Reference"
        ):
            return sub[2]
    return None


def _footprint_place(fp: list) -> tuple[float, float, float]:
    """World (x, y, rotation) of a footprint, from its (at ...) node."""
    for sub in fp:
        if isinstance(sub, list) and len(sub) >= 3 and _sym(sub[0]) == "at":
            try:
                x = float(sub[1])
                y = float(sub[2])
            except (TypeError, ValueError):
                return (0.0, 0.0, 0.0)
            rot = float(sub[3]) if len(sub) >= 4 else 0.0
            return (x, y, rot)
    return (0.0, 0.0, 0.0)


def _pad_net(pad: list) -> str | None:
    for sub in pad:
        if isinstance(sub, list) and len(sub) >= 1 and _sym(sub[0]) == "net":
            if len(sub) >= 3:
                return str(sub[2])
            if len(sub) >= 2:
                return str(sub[1])
    return None


def collect_nets(pcb_path: str) -> dict[str, list[tuple[str, str]]]:
    """Map net name -> list of (ref, pad) belonging to it.

    Reads pads straight from the S-expression so we do not depend on the
    router's internals to enumerate what could be routed.
    """
    data = load_pcb(pcb_path)
    nets: dict[str, list[tuple[str, str]]] = {}
    for node in data:
        if not isinstance(node, list) or len(node) < 2:
            continue
        if _sym(node[0]) != "footprint":
            continue
        ref = _footprint_ref(node)
        if ref is None:
            continue
        for sub in node:
            if not isinstance(sub, list) or len(sub) < 2:
                continue
            if _sym(sub[0]) != "pad":
                continue
            pad_num = sub[1] if isinstance(sub[1], str) else str(sub[1])
            net = _pad_net(sub)
            if net:
                nets.setdefault(net, []).append((ref, pad_num))
    return nets


def routable_pairs(
    pcb_path: str, via_pairs: tuple[tuple[str, str], ...] | None = None
) -> list[PairSpec]:
    """Enumerate connectable pad pairs: every pair within a net with 2+ pads."""
    nets = collect_nets(pcb_path)
    pairs: list[PairSpec] = []
    for net, pads in sorted(nets.items()):
        pads_sorted = sorted(pads)
        for i in range(len(pads_sorted)):
            for j in range(i + 1, len(pads_sorted)):
                pairs.append(
                    PairSpec(
                        ref_a=pads_sorted[i][0],
                        pad_a=pads_sorted[i][1],
                        ref_b=pads_sorted[j][0],
                        pad_b=pads_sorted[j][1],
                        net=net,
                        via_pairs=via_pairs,
                    )
                )
    return pairs


def route_pair(pcb_path: str, pair: PairSpec, layer_hint: str | None = None) -> str | None:
    """Route one pair in place.  Returns None on success, error string on failure."""
    req = RouteRequest(
        pcb_path=pcb_path,
        ref_a=pair.ref_a,
        pad_a=pair.pad_a,
        ref_b=pair.ref_b,
        pad_b=pair.pad_b,
        net=pair.net,
        layer_hint=layer_hint,
        via_pairs=pair.via_pairs or DEFAULT_VIA_PAIRS,
    )
    try:
        result = auto_route_pair(req)
    except (RouteFailure, ValueError, RuntimeError) as exc:
        return str(exc)

    data = load_pcb(pcb_path)
    for seg in result.segments:
        data.append(_segment_to_sexp(seg))
    for via in result.vias:
        data.append(_via_to_sexp(via))
    try:
        save_pcb(pcb_path, data)
    except OSError as exc:
        return f"failed to write PCB: {exc}"
    return None


def _node_coord(node: list, name: str) -> tuple[float, float] | None:
    for sub in node:
        if isinstance(sub, list) and len(sub) >= 3 and _sym(sub[0]) == name:
            try:
                return float(sub[1]), float(sub[2])
            except (TypeError, ValueError):
                return None
    return None


def _pad_geometry(pad: list, fp_at: tuple[float, float], fp_rot: float) -> Any:
    """Return a shapely box for a pad node in world coordinates, or None.

    KiCad stores pad ``(at ...)`` / ``(size ...)`` in the *footprint's*
    local frame; the pad centre must be rotated by the footprint rotation
    and translated by the footprint origin to land on the board.
    """
    from math import cos, radians, sin

    from shapely.geometry import box

    at = _node_coord(pad, "at")
    size = None
    for sub in pad:
        if isinstance(sub, list) and len(sub) >= 3 and _sym(sub[0]) == "size":
            size = (float(sub[1]), float(sub[2]))
    if at is None or size is None:
        return None

    # Pad rotation (rare) rotates the pad within the footprint frame.
    pad_rot = 0.0
    for sub in pad:
        if isinstance(sub, list) and len(sub) >= 4 and _sym(sub[0]) == "at":
            try:
                pad_rot = float(sub[3])
            except (TypeError, ValueError):
                pad_rot = 0.0

    total_rot = pad_rot + fp_rot
    w, h = size
    tha = radians(total_rot)
    c, s = cos(tha), sin(tha)
    # Local pad centre, then rotate CCW (KiCad world is Y-down so CCW on
    # screen == CW in math coords — the board file convention) and translate.
    lx, ly = at[0], at[1]
    corners = [
        (lx - w / 2, ly - h / 2),
        (lx + w / 2, ly - h / 2),
        (lx + w / 2, ly + h / 2),
        (lx - w / 2, ly + h / 2),
    ]
    xs = [fp_at[0] + x * c - y * s for x, y in corners]
    ys = [fp_at[1] + x * s + y * c for x, y in corners]
    return box(min(xs), min(ys), max(xs), max(ys))


def _pad_center(data: list, ref: str, pad_num: str) -> tuple[float, float] | None:
    """World centre of a pad (footprint origin + rotated local centre)."""
    for node in data:
        if not isinstance(node, list) or len(node) < 2:
            continue
        if _sym(node[0]) != "footprint":
            continue
        if _footprint_ref(node) != ref:
            continue
        fp_at = _footprint_place(node)
        for sub in node:
            if not isinstance(sub, list) or len(sub) < 2:
                continue
            if _sym(sub[0]) != "pad":
                continue
            pnum = sub[1] if isinstance(sub[1], str) else str(sub[1])
            if pnum != pad_num:
                continue
            geo = _pad_geometry(sub, (fp_at[0], fp_at[1]), fp_at[2])
            if geo is None:
                return None
            c = geo.centroid
            return (c.x, c.y)
    return None


def _draw_pad(ax, geo: Any, net: str) -> None:
    color = _net_color(net)
    x, y = geo.bounds[0], geo.bounds[1]
    w = geo.bounds[2] - geo.bounds[0]
    h = geo.bounds[3] - geo.bounds[1]
    ax.add_patch(
        mpatches.Rectangle(
            (x, y), w, h, facecolor=color, edgecolor="black", linewidth=0.5, zorder=4
        )
    )


def _draw_line_node(ax, node: list) -> None:
    start = _node_coord(node, "start")
    end = _node_coord(node, "end")
    if start is None or end is None:
        return
    if "Edge.Cuts" in [str(s) for s in node] or "F.CrtYd" in [str(s) for s in node]:
        ax.plot([start[0], end[0]], [start[1], end[1]], "k-", linewidth=0.6, alpha=0.5)
    else:
        ax.plot([start[0], end[0]], [start[1], end[1]], "g-", linewidth=1.2)


def _draw_zone(ax, zone: list) -> None:
    pts: list[tuple[float, float]] = []
    for sub in zone:
        if not isinstance(sub, list) or _sym(sub[0]) != "polygon":
            continue
        for pts_node in sub:
            if not isinstance(pts_node, list) or _sym(pts_node[0]) != "pts":
                continue
            for xy in pts_node[1:]:
                if isinstance(xy, list) and len(xy) >= 3 and _sym(xy[0]) == "xy":
                    pts.append((float(xy[1]), float(xy[2])))
    if len(pts) >= 3:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.fill(xs, ys, facecolor="#cccccc", edgecolor="#888888", alpha=0.4, hatch="//", zorder=2)


def render_board_snapshot(pcb_path: str, out_path: str, pairs: list[PairSpec]) -> None:
    """Render an annotated board snapshot: pads labelled ref.pad, nets coloured.

    Unrouted pairs are drawn as dashed lines between the two pad centres so
    the VLM can see exactly what still needs connecting.
    """
    data = load_pcb(pcb_path)

    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    ax.set_aspect("equal")
    ax.set_title("PCB current state — pads labelled ref.pad, dashed lines = still to route")

    pad_shapes: list[tuple[Any, str, str]] = []
    for node in data:
        if not isinstance(node, list) or len(node) < 2:
            continue
        head = _sym(node[0])
        if head in ("segment", "gr_line"):
            _draw_line_node(ax, node)
        elif head == "footprint":
            ref = _footprint_ref(node)
            if ref is None:
                continue
            fp_at = _footprint_place(node)
            for sub in node:
                if not isinstance(sub, list) or len(sub) < 2:
                    continue
                if _sym(sub[0]) != "pad":
                    continue
                geo = _pad_geometry(sub, (fp_at[0], fp_at[1]), fp_at[2])
                if geo is None:
                    continue
                pad_num = sub[1] if isinstance(sub[1], str) else str(sub[1])
                net = _pad_net(sub) or "?"
                center = geo.centroid
                _draw_pad(ax, geo, net)
                ax.annotate(
                    f"{ref}.{pad_num}",
                    xy=(center.x, center.y),
                    xytext=(center.x + 0.35, center.y + 0.35),
                    fontsize=7,
                    color="black",
                    zorder=5,
                )
                pad_shapes.append((geo, net, ref))
        elif head == "zone":
            _draw_zone(ax, node)

    # Dashed lines for pairs still to route (drawn on pad centres).
    for pair in pairs:
        if pair.status == "done":
            continue
        a = _pad_center(data, pair.ref_a, pair.pad_a)
        b = _pad_center(data, pair.ref_b, pair.pad_b)
        if a is None or b is None:
            continue
        ax.plot(
            [a[0], b[0]],
            [a[1], b[1]],
            "r--",
            linewidth=1.0,
            alpha=0.7,
            zorder=3,
        )

    # Keep the board's own bounds; fall back to pad bounds.
    xmin = ymin = 1e9
    xmax = ymax = -1e9
    for geo, _net, _ref in pad_shapes:
        bounds = geo.bounds
        xmin = min(xmin, bounds[0])
        ymin = min(ymin, bounds[1])
        xmax = max(xmax, bounds[2])
        ymax = max(ymax, bounds[3])
    if xmin < xmax:
        margin = 2.0
        ax.set_xlim(xmin - margin, xmax + margin)
        ax.set_ylim(ymin - margin, ymax + margin)
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.invert_yaxis()  # KiCad PCB convention: +Y down.
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def build_prompt(pairs: list[PairSpec], last_error: str | None) -> str:
    """Compose the text prompt describing what remains to route."""
    pending = [p for p in pairs if p.status == "pending"]
    lines = ["Current board state is in the image.", "Still to connect:"]
    for p in pending:
        lines.append(f"  - {p.description}")
    if last_error:
        lines += [
            "",
            "The last routing attempt failed with:",
            f"  last_error: {last_error}",
            "advice: change layer (thru-hole only), pick a different pair first,",
            "        or give up on this pair.",
        ]
    lines.append("Reply with your chosen route, layer and a one-line reason.")
    return "\n".join(lines)


def run_experiment(
    pcb_path: str,
    client: VLMClient | None,
    rounds: int,
    work_dir: str,
    dry_run: bool = False,
    via_pairs: tuple[tuple[str, str], ...] | None = None,
) -> dict[str, Any]:
    """Run the feedback loop, writing snapshots into *work_dir*.

    Returns a metrics dict describing what happened.
    """
    pairs = routable_pairs(pcb_path, via_pairs=via_pairs)
    if not pairs:
        raise ValueError(f"no routable pad pairs found in {pcb_path}")

    metrics: dict[str, Any] = {
        "pcb": pcb_path,
        "pairs_total": len(pairs),
        "pairs_done": 0,
        "pairs_failed": 0,
        "rounds": 0,
        "attempts": 0,
        "feedback_errors": 0,
        "per_pair": {},
    }

    os.makedirs(work_dir, exist_ok=True)
    last_error: str | None = None

    for rnd in range(1, rounds + 1):
        pending = [p for p in pairs if p.status == "pending"]
        if not pending:
            break
        snapshot = os.path.join(work_dir, f"round_{rnd:02d}.png")
        render_board_snapshot(pcb_path, snapshot, pairs)
        metrics["rounds"] = rnd

        if dry_run or client is None:
            # No feedback: try pairs in fixed order, no layer hint.
            pick = pending[0]
            layer_hint = None
        else:
            prompt = build_prompt(pairs, last_error)
            with open(snapshot, "rb") as fh:
                png_bytes = fh.read()
            try:
                reply = client.ask(png_bytes, prompt)
            except RuntimeError as exc:
                metrics["feedback_errors"] += 1
                metrics["last_error"] = str(exc)
                last_error = str(exc)
                continue

            fb = parse_feedback(reply)
            if not fb.route_a or not fb.route_b:
                metrics["feedback_errors"] += 1
                last_error = f"VLM reply unparseable: {reply!r}"
                metrics["last_error"] = last_error
                continue
            try:
                ref_a, pad_a = parse_pad(fb.route_a)
                ref_b, pad_b = parse_pad(fb.route_b)
            except ValueError as exc:
                metrics["feedback_errors"] += 1
                last_error = f"bad pad spec in reply: {exc}"
                metrics["last_error"] = last_error
                continue

            cand = None
            for p in pending:
                if {p.ref_a, p.pad_a} == {ref_a, pad_a} and {p.ref_b, p.pad_b} == {ref_b, pad_b}:
                    cand = p
                    break
            if cand is None:
                metrics["feedback_errors"] += 1
                last_error = (
                    f"VLM chose {fb.route_a} -> {fb.route_b} which is not a pending pair "
                    f"({[p.key for p in pending]}); pick one of those."
                )
                metrics["last_error"] = last_error
                continue
            pick = cand
            layer_hint = fb.layer if fb.layer and fb.layer.lower() != "auto" else None

        metrics["attempts"] += 1
        pick.attempts += 1
        error = route_pair(pcb_path, pick, layer_hint)
        if error is None:
            pick.status = "done"
            metrics["pairs_done"] += 1
            last_error = None
            metrics.pop("last_error", None)
        else:
            pick.status = "failed"
            metrics["pairs_failed"] += 1
            last_error = error
            metrics["last_error"] = last_error
        metrics["per_pair"][pick.key] = {
            "net": pick.net,
            "attempts": pick.attempts,
            "status": pick.status,
            "layer_hint": layer_hint,
        }

    # Final snapshot regardless of completion.
    render_board_snapshot(pcb_path, os.path.join(work_dir, "final.png"), pairs)

    metrics["pairs_remaining"] = len([p for p in pairs if p.status == "pending"])
    return metrics


def _parse_via_pairs(text: str) -> tuple[tuple[str, str], ...]:
    """Parse 'F.Cu:B.Cu,B.Cu:In1.Cu' into (('F.Cu','B.Cu'),('B.Cu','In1.Cu'))."""
    out: list[tuple[str, str]] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise argparse.ArgumentTypeError(f"expected 'layerA:layerB', got {token!r}")
        a, b = token.split(":", 1)
        out.append((a.strip(), b.strip()))
    if not out:
        raise argparse.ArgumentTypeError("via pairs list is empty")
    return tuple(out)


def main(argv: list[str] | None = None) -> int:
    doc = __doc__ or ""
    parser = argparse.ArgumentParser(description=doc.splitlines()[0])
    parser.add_argument("--pcb", default=DEFAULT_PCB, help=".kicad_pcb file (default: test board)")
    parser.add_argument("--rounds", type=int, default=8, help="max feedback rounds (default 8)")
    parser.add_argument("--work-dir", default=None, help="where to write snapshots/metrics")
    parser.add_argument(
        "--via-pairs",
        default=None,
        type=_parse_via_pairs,
        help="allowed via layer hops, comma-separated pairs (default: F.Cu:B.Cu,B.Cu:In1.Cu)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="no VLM: connect pairs in fixed order (control arm)",
    )
    parser.add_argument("--json", default=None, help="write metrics JSON to this path")
    args = parser.parse_args(argv)

    if not os.path.exists(args.pcb):
        print(f"error: PCB not found: {args.pcb}", file=sys.stderr)
        return 2
    pro_hint = os.path.join(
        os.path.dirname(args.pcb), os.path.basename(args.pcb)[:-10] + ".kicad_pro"
    )
    if not os.path.exists(pro_hint):
        print(
            "warning: no sibling .kicad_pro found; router will fail width/clearance lookup",
            file=sys.stderr,
        )

    work_dir = args.work_dir or tempfile.mkdtemp(prefix="vlm_route_")
    client = None if args.dry_run else VLMClient()

    try:
        metrics = run_experiment(
            args.pcb,
            client,
            args.rounds,
            work_dir,
            dry_run=args.dry_run,
            via_pairs=args.via_pairs,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"snapshots: {work_dir}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())

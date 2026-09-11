"""
Self-contained PCB board renderer (no kicad-cli dependency).

Renders a composite image of a KiCad board with the KiCad default theme
(dark background): courtyards, copper layers, board edge, silkscreen text,
and — when requested — the green ratsnest of user-specified pads that are
not yet routed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import io
import math
import os
from typing import Any

import matplotlib

matplotlib.use("Agg")
from fastmcp import Context, FastMCP
from fastmcp.utilities.types import Image
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from kcaa.utils.pcb_sexp_utils import load_pcb

# KiCad default theme approximations (same palette as scripts/vlm_route_feedback.py).
_KICAD_LAYER_COLORS = {
    "F.Cu": "#E31A1C",
    "B.Cu": "#1E5BC6",
    "In1.Cu": "#2E9E44",
    "In2.Cu": "#E6A71D",
    "In3.Cu": "#9B59B6",
    "In4.Cu": "#0FA3B1",
    "Edge.Cuts": "#F2DA57",
}
_BG_COLOR = "#17181D"
_SILK_COLOR = "#E8E8E8"
_COURTYARD_COLOR = "#A9C940"
_RATSNEST_COLOR = "#1BE41B"
_PAD_ALPHA = 0.9

# Render order (bottom to top).
_Z_COURTYARD = 1
_Z_COPPER_BOTTOM = 2
_Z_COPPER_TOP = 3
_Z_EDGE = 4
_Z_SILK = 5
_Z_RATSNEST = 6

_PT_PER_MM = 72.0 / 25.4


def _sym(value: Any) -> str:
    """String form of a sexpdata Symbol or plain string.

    ``sexpdata.Symbol`` overrides ``__eq__`` so it never equals a ``str``;
    always compare via this helper.
    """
    return str(value)


def _layer_color(layer: str) -> str:
    return _KICAD_LAYER_COLORS.get(layer, "#9A9A9A")


def _node_coord(node: list, name: str) -> tuple[float, float] | None:
    for sub in node:
        if isinstance(sub, list) and len(sub) >= 3 and _sym(sub[0]) == name:
            try:
                return float(sub[1]), float(sub[2])
            except (TypeError, ValueError):
                return None
    return None


def _is_courtyard(layer: str) -> bool:
    return "CrtYd" in layer or "Courtyard" in layer


def _is_silk(layer: str) -> bool:
    return "SilkS" in layer


def _is_edge(layer: str) -> bool:
    return layer == "Edge.Cuts"


def _layer_of(node: list) -> str | None:
    for sub in node:
        if isinstance(sub, list) and len(sub) >= 2 and _sym(sub[0]) == "layer":
            return str(sub[1])
    return None


def _copper_layers(data: list) -> list[str]:
    """Copper layers present on the board, in KiCad stack order."""
    layers: list[str] = []
    for node in data:
        if not isinstance(node, list) or _sym(node[0]) != "layers":
            continue
        for entry in node[1:]:
            if isinstance(entry, list) and len(entry) >= 3:
                name = str(entry[1])
                if name.endswith(".Cu"):
                    layers.append(name)
    return layers


def _wpt(fp: tuple[float, float, float], x: float, y: float) -> tuple[float, float]:
    """Transform a footprint-local point to world coordinates."""
    fx, fy, rot = fp
    if rot:
        a = math.radians(rot)
        c, s = math.cos(a), math.sin(a)
        x, y = x * c - y * s, x * s + y * c
    return (fx + x, fy + y)


@dataclass
class Pad:
    """A footprint pad in world coordinates."""

    ref: str
    number: str
    net: str | None
    center: tuple[float, float]
    copper_layers: list[str]
    shape: Any


@dataclass
class BoardData:
    copper_layers: list[str] = field(default_factory=list)
    pads: list[Pad] = field(default_factory=list)
    tracks: list[dict] = field(default_factory=list)
    bodies: list[dict] = field(default_factory=list)  # courtyard/silk shapes
    texts: list[dict] = field(default_factory=list)  # silkscreen labels
    edges: list[dict] = field(default_factory=list)  # Edge.Cuts graphics
    routed_nets: set[str] = field(default_factory=set)


def _build_pad_shape(pad: list, fp: tuple[float, float, float]) -> Any | None:
    """Build a matplotlib patch for a pad at world coordinates.

    The patch is centered on the pad center (KiCad pads rotate about their
    center; matplotlib's Rectangle angle rotates about its lower-left
    anchor, so rotated rects are built explicitly from rotated corners).
    """
    at = _node_coord(pad, "at")
    size = _node_coord(pad, "size")
    if at is None or size is None:
        return None
    w, h = size
    pad_rot = 0.0
    for sub in pad:
        if isinstance(sub, list) and _sym(sub[0]) == "at" and len(sub) >= 4:
            try:
                pad_rot = float(sub[3])
            except (TypeError, ValueError):
                pad_rot = 0.0
    total_rot = pad_rot + fp[2]
    pad_shape = str(pad[3]) if len(pad) > 3 else ("rect" if len(pad) < 3 else str(pad[2]))

    # Custom pads carry (primitives ...) geometry; fall back to the size box.
    primitives = None
    for sub in pad:
        if isinstance(sub, list) and _sym(sub[0]) == "primitives":
            primitives = sub
            break
    if primitives is not None:
        pts: list[tuple[float, float]] = []
        for prim in primitives[1:]:
            if not isinstance(prim, list) or not prim:
                continue
            kind = _sym(prim[0])
            if kind == "gr_poly":
                for sub in prim:
                    if isinstance(sub, list) and _sym(sub[0]) == "pts":
                        for xy in sub[1:]:
                            if isinstance(xy, list) and len(xy) >= 3 and _sym(xy[0]) == "xy":
                                try:
                                    pts.append((float(xy[1]), float(xy[2])))
                                except (TypeError, ValueError):
                                    pass
        if pts:
            # Primitive coords are in the pad's local frame; rotate by pad
            # rotation, translate by the pad position, then footprint transform.
            ra = math.radians(pad_rot)
            c, s = math.cos(ra), math.sin(ra)
            world = [_wpt(fp, at[0] + x * c - y * s, at[1] + x * s + y * c) for x, y in pts]
            return mpatches.Polygon(world, closed=True)

    center = _wpt(fp, at[0], at[1])
    if pad_shape in ("circle", "roundrect"):
        radius = max(w, h) / 2
        return mpatches.Circle(center, radius)
    if pad_shape == "oval":
        return mpatches.FancyBboxPatch(
            (center[0] - w / 2, center[1] - h / 2),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={min(w, h) / 2:.4f}",
        )
    # rect / trapezoid / custom shapes: rotate the corners about the center.
    ra = math.radians(total_rot)
    c, s = math.cos(ra), math.sin(ra)
    corners = [
        (-w / 2, -h / 2),
        (w / 2, -h / 2),
        (w / 2, h / 2),
        (-w / 2, h / 2),
    ]
    world = [(center[0] + x * c - y * s, center[1] + x * s + y * c) for x, y in corners]
    return mpatches.Polygon(world, closed=True)


def _pad_net(pad: list) -> str | None:
    """Net name of a pad (KiCad 10 name-only; KiCad 8 ``(net N "X")``)."""
    for sub in pad:
        if isinstance(sub, list) and len(sub) >= 2 and _sym(sub[0]) == "net":
            if len(sub) >= 3 and isinstance(sub[1], int):
                return str(sub[2])
            return str(sub[1])
    return None


def _pad_copper_layers(pad: list, all_copper: list[str]) -> list[str]:
    """Copper layers a pad occupies; ``*.Cu`` expands to the full stack."""
    out: list[str] = []
    for sub in pad:
        if isinstance(sub, list) and _sym(sub[0]) == "layers":
            for l in sub[1:]:
                if not isinstance(l, str):
                    continue
                if l == "*.Cu":
                    out.extend(all_copper)
                elif l in all_copper:
                    out.append(l)
    return list(dict.fromkeys(out))


def _parse_text(
    node: list, fp: tuple[float, float, float], ref: str = "", value: str = ""
) -> dict | None:
    """Parse a silk text from an ``fp_text`` or ``property`` node."""
    if len(node) < 3:
        return None
    head = _sym(node[0])
    if head == "property":
        pname = str(node[1])
        if pname not in ("Reference", "Value"):
            return None
        label = str(node[2]) if len(node) > 2 else ""
        for sub in node:
            if isinstance(sub, list) and _sym(sub[0]) == "hide":
                return None
    else:  # fp_text
        kind = _sym(node[1])
        if kind not in ("reference", "value", "user"):
            return None
        label = str(node[2]) if len(node) > 2 else ""
        if label == "${REFERENCE}":
            label = ref
        elif label == "${VALUE}":
            label = value
    layer = _layer_of(node)
    if layer is None or not _is_silk(layer):
        return None
    at = _node_coord(node, "at")
    if at is None:
        return None
    rot = 0.0
    for sub in node:
        if isinstance(sub, list) and _sym(sub[0]) == "at" and len(sub) >= 4:
            try:
                rot = float(sub[3])
            except (TypeError, ValueError):
                pass
    size = 1.0
    bold = False
    for sub in node:
        if isinstance(sub, list) and _sym(sub[0]) == "effects":
            for e in sub:
                if isinstance(e, list) and _sym(e[0]) == "font":
                    for f in e[1:]:
                        if isinstance(f, list) and _sym(f[0]) == "size":
                            try:
                                size = float(f[1])
                            except (TypeError, ValueError):
                                pass
                        if isinstance(f, str) and f == "bold":
                            bold = True
    return {
        "kind": "text",
        "label": label,
        "at": _wpt(fp, at[0], at[1]),
        "rot": rot,
        "size": size,
        "bold": bold,
    }


def _parse_shape(shape: list, fp: tuple[float, float, float]) -> dict | None:
    kind = _sym(shape[0])
    if kind not in (
        "fp_line",
        "fp_rect",
        "fp_circle",
        "fp_poly",
        "fp_arc",
        "gr_line",
        "gr_rect",
        "gr_circle",
        "gr_poly",
        "gr_arc",
    ):
        return None
    layer = _layer_of(shape)
    if layer is None:
        return None
    entry: dict[str, Any] = {"kind": kind, "layer": layer, "fp": fp}
    start = _node_coord(shape, "start")
    end = _node_coord(shape, "end")
    mid = _node_coord(shape, "mid")
    if start:
        entry["start"] = _wpt(fp, start[0], start[1])
    if end:
        entry["end"] = _wpt(fp, end[0], end[1])
    if mid:
        entry["mid"] = _wpt(fp, mid[0], mid[1])
    if kind in ("fp_poly", "gr_poly"):
        pts = []
        for sub in shape:
            if isinstance(sub, list) and _sym(sub[0]) == "pts":
                for xy in sub[1:]:
                    if isinstance(xy, list) and len(xy) >= 3 and _sym(xy[0]) == "xy":
                        try:
                            pts.append(_wpt(fp, float(xy[1]), float(xy[2])))
                        except (TypeError, ValueError):
                            pass
        if not pts:
            return None
        entry["pts"] = pts
    return entry


def parse_board(pcb_path: str) -> BoardData:
    """Parse a .kicad_pcb file into a BoardData model."""
    data = load_pcb(pcb_path)
    board = BoardData()
    board.copper_layers = _copper_layers(data)
    if not board.copper_layers:
        board.copper_layers = ["F.Cu", "B.Cu"]

    for node in data:
        if not isinstance(node, list) or not node:
            continue
        kind = _sym(node[0])

        if kind == "footprint":
            fp_at = (0.0, 0.0, 0.0)
            for sub in node:
                if isinstance(sub, list) and _sym(sub[0]) == "at":
                    try:
                        fp_at = (
                            float(sub[1]),
                            float(sub[2]),
                            float(sub[3]) if len(sub) > 3 else 0.0,
                        )
                    except (TypeError, ValueError):
                        fp_at = (0.0, 0.0, 0.0)
                    break
            ref = "?"
            value = ""
            for sub in node:
                if isinstance(sub, list) and len(sub) >= 3 and _sym(sub[0]) == "property":
                    if _sym(sub[1]) == "Reference":
                        ref = str(sub[2])
                    elif _sym(sub[1]) == "Value":
                        value = str(sub[2])
            for sub in node:
                if not isinstance(sub, list) or not sub:
                    continue
                sk = _sym(sub[0])
                if sk == "pad":
                    at = _node_coord(sub, "at")
                    if at is None:
                        continue
                    board.pads.append(
                        Pad(
                            ref=ref,
                            number=str(sub[1]) if len(sub) > 1 else "?",
                            net=_pad_net(sub),
                            center=_wpt(fp_at, at[0], at[1]),
                            copper_layers=_pad_copper_layers(sub, board.copper_layers) or ["F.Cu"],
                            shape=_build_pad_shape(sub, fp_at),
                        )
                    )
                elif sk in ("fp_line", "fp_rect", "fp_circle", "fp_poly", "fp_arc"):
                    entry = _parse_shape(sub, fp_at)
                    if entry is None:
                        continue
                    layer = entry["layer"]
                    if _is_edge(layer):
                        board.edges.append(entry)
                    elif _is_courtyard(layer) or _is_silk(layer):
                        board.bodies.append(entry)
                elif sk in ("fp_text", "property"):
                    t = _parse_text(sub, fp_at, ref, value)
                    if t is not None:
                        board.texts.append(t)
            continue

        if kind == "segment":
            start = _node_coord(node, "start")
            end = _node_coord(node, "end")
            layer = _layer_of(node)
            if start is None or end is None or layer is None:
                continue
            width = 0.25
            net = None
            for sub in node:
                if isinstance(sub, list) and _sym(sub[0]) == "width":
                    try:
                        width = float(sub[1])
                    except (TypeError, ValueError):
                        pass
                if isinstance(sub, list) and _sym(sub[0]) == "net" and len(sub) >= 2:
                    net = str(sub[1])
            if _is_edge(layer):
                board.edges.append(
                    {
                        "kind": "gr_line",
                        "layer": layer,
                        "start": start,
                        "end": end,
                        "fp": (0.0, 0.0, 0.0),
                    }
                )
            else:
                board.tracks.append({"layer": layer, "start": start, "end": end, "width": width})
                if net:
                    board.routed_nets.add(net)
            continue

        if kind == "via":
            for sub in node:
                if isinstance(sub, list) and _sym(sub[0]) == "net" and len(sub) >= 2:
                    board.routed_nets.add(str(sub[1]))
                    break
            continue

        if kind in ("gr_line", "gr_rect", "gr_circle", "gr_poly", "gr_arc"):
            entry = _parse_shape(node, (0.0, 0.0, 0.0))
            if entry is None:
                continue
            layer = entry["layer"]
            if _is_edge(layer):
                board.edges.append(entry)
            elif _is_courtyard(layer) or _is_silk(layer):
                board.bodies.append(entry)
            continue

    return board


def _mst(
    points: list[tuple[float, float]],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Prim MST over points; returns segment list."""
    segs: list[tuple[tuple[float, float], tuple[float, float]]] = []
    if len(points) < 2:
        return segs
    in_tree = [0]
    rest = list(range(1, len(points)))
    while rest:
        best: tuple[float, int, int] | None = None
        for i in rest:
            for j in in_tree:
                d2 = (points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 2
                if best is None or d2 < best[0]:
                    best = (d2, j, i)
        if best is None:
            raise RuntimeError("MST iteration failed to extend tree")
        _, j, i = best
        segs.append((points[j], points[i]))
        in_tree.append(i)
        rest.remove(i)
    return segs


def _bounds(
    board: BoardData, ratsnest: list[tuple[tuple[float, float], tuple[float, float]]]
) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for p in board.pads:
        xs.append(p.center[0])
        ys.append(p.center[1])
    for seg in board.tracks:
        xs += [seg["start"][0], seg["end"][0]]
        ys += [seg["start"][1], seg["end"][1]]
    for e in board.edges:
        if "start" in e:
            xs.append(e["start"][0])
            ys.append(e["start"][1])
        if "end" in e:
            xs.append(e["end"][0])
            ys.append(e["end"][1])
        if "mid" in e:
            xs.append(e["mid"][0])
            ys.append(e["mid"][1])
        for p in e.get("pts", []):
            xs.append(p[0])
            ys.append(p[1])
    for a, b in ratsnest:
        xs += [a[0], b[0]]
        ys += [a[1], b[1]]
    for t in board.texts:
        xs.append(t["at"][0])
        ys.append(t["at"][1])
    if not xs:
        return (0.0, 0.0, 100.0, 100.0)
    pad = 2.0
    return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)


def _draw_shape(ax, entry: dict, color: str, lw: float, alpha: float, zorder: int) -> None:
    """Draw a shape entry (already in world coordinates)."""
    kind = entry["kind"]
    if kind in ("fp_line", "gr_line"):
        ax.plot(
            [entry["start"][0], entry["end"][0]],
            [entry["start"][1], entry["end"][1]],
            color=color,
            linewidth=lw,
            alpha=alpha,
            zorder=zorder,
        )
    elif kind in ("fp_rect", "gr_rect"):
        s, e = entry["start"], entry["end"]
        ax.add_patch(
            mpatches.Rectangle(
                s,
                e[0] - s[0],
                e[1] - s[1],
                fill=False,
                edgecolor=color,
                linewidth=lw,
                alpha=alpha,
                zorder=zorder,
            )
        )
    elif kind in ("fp_circle", "gr_circle"):
        c = entry["start"]
        e = entry["end"]
        ax.add_patch(
            mpatches.Circle(
                c,
                math.hypot(e[0] - c[0], e[1] - c[1]),
                fill=False,
                edgecolor=color,
                linewidth=lw,
                alpha=alpha,
                zorder=zorder,
            )
        )
    elif kind in ("fp_poly", "gr_poly"):
        pts = entry.get("pts", [])
        if len(pts) >= 2:
            ax.add_patch(
                mpatches.Polygon(
                    pts,
                    closed=True,
                    fill=False,
                    edgecolor=color,
                    linewidth=lw,
                    alpha=alpha,
                    zorder=zorder,
                )
            )
    elif kind in ("fp_arc", "gr_arc"):
        _draw_arc(ax, entry, color, lw, alpha, zorder)


def _draw_arc(ax, entry: dict, color: str, lw: float, alpha: float, zorder: int) -> None:
    """Draw a KiCad arc (start/mid/end on the circle) as sampled polyline."""
    s, m, e = entry["start"], entry["mid"], entry["end"]
    (x1, y1), (x2, y2), (x3, y3) = s, m, e
    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-12:
        ax.plot([s[0], e[0]], [s[1], e[1]], color=color, linewidth=lw, alpha=alpha, zorder=zorder)
        return
    ux = (
        (x1 * x1 + y1 * y1) * (y2 - y3)
        + (x2 * x2 + y2 * y2) * (y3 - y1)
        + (x3 * x3 + y3 * y3) * (y1 - y2)
    ) / d
    uy = (
        (x1 * x1 + y1 * y1) * (x3 - x2)
        + (x2 * x2 + y2 * y2) * (x1 - x3)
        + (x3 * x3 + y3 * y3) * (x2 - x1)
    ) / d
    r = math.hypot(x1 - ux, y1 - uy)
    t1 = math.atan2(y1 - uy, x1 - ux)
    t2 = math.atan2(y3 - uy, x3 - ux)
    tm = math.atan2(y2 - uy, x2 - ux)
    # Sweep so the arc passes through mid.
    while tm < t1:
        tm += 2 * math.pi
    while t2 < t1:
        t2 += 2 * math.pi
    if tm > t2:
        t2 = tm
    n = max(8, int(abs(t2 - t1) / (math.pi / 90)))
    ts = [t1 + (t2 - t1) * i / n for i in range(n + 1)]
    pts = [(ux + r * math.cos(t), uy + r * math.sin(t)) for t in ts]
    ax.plot(
        [p[0] for p in pts],
        [p[1] for p in pts],
        color=color,
        linewidth=lw,
        alpha=alpha,
        zorder=zorder,
    )


def render_board(
    pcb_path: str,
    connect_pads: list[str] | None = None,
    dpi: int | None = None,
) -> tuple[list[str], bytes, dict[str, Any]]:
    """Render a board to (report_lines, png_bytes, report_dict).

    ``dpi`` scales the PNG resolution directly (line widths and font sizes
    are in mm units, so a higher dpi gives a sharper image of the same
    layout).  Defaults to 200.

    connect_pads: optional list of ``ref.pad`` specs (e.g. ``["J1.2", "J2.2"]``)
    to draw ratsnest lines for — green, only for nets that are not yet routed.
    """
    if dpi is None:
        dpi = 200
    board = parse_board(pcb_path)

    # Resolve requested pads to nets (name-based in KiCad 10).
    requested: dict[str, list[tuple[float, float]]] = {}
    missing: list[str] = []
    for spec in connect_pads or []:
        ref, _, num = spec.partition(".")
        found = None
        for p in board.pads:
            if p.ref == ref and p.number == num:
                found = p
                break
        if found is None:
            missing.append(spec)
            continue
        if found.net:
            requested.setdefault(found.net, []).append(found.center)

    ratsnest: list[tuple[tuple[float, float], tuple[float, float]]] = []
    pending_nets: list[str] = []
    routed_reported: list[str] = []
    for net, pts in sorted(requested.items()):
        if len(pts) < 2:
            continue
        if net in board.routed_nets:
            routed_reported.append(net)
            continue
        pending_nets.append(net)
        ratsnest.extend(_mst(pts))

    # --- figure ---
    xmin, ymin, xmax, ymax = _bounds(board, ratsnest)
    w_mm, h_mm = xmax - xmin, ymax - ymin
    fig_w_in = w_mm / 25.4
    # Scale dpi so the output is sharp at any board size (min 1600px wide).
    if w_mm >= 0.1:
        dpi = max(dpi, int(1600 / fig_w_in))
    fig, ax = plt.subplots(figsize=(fig_w_in, h_mm / 25.4), dpi=dpi)
    ax.set_facecolor(_BG_COLOR)
    fig.patch.set_facecolor(_BG_COLOR)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.invert_yaxis()  # KiCad PCB convention: +Y down.
    ax.set_aspect("equal")
    ax.axis("off")

    # 1. Courtyards (bottom of visual stack).
    for b in board.bodies:
        if not _is_courtyard(b["layer"]):
            continue
        _draw_shape(ax, b, _COURTYARD_COLOR, 0.5, 0.6, _Z_COURTYARD)

    # 2. Copper: bottom layer first, then top layers.
    copper_bottom = board.copper_layers[-1] if len(board.copper_layers) > 1 else None
    for p in board.pads:
        if p.shape is None:
            continue
        zbase = _Z_COPPER_BOTTOM if (copper_bottom in p.copper_layers) else _Z_COPPER_TOP
        p.shape.set_facecolor(_layer_color(p.copper_layers[0]))
        p.shape.set_edgecolor("black")
        p.shape.set_linewidth(0.3)
        p.shape.set_alpha(_PAD_ALPHA)
        p.shape.set_zorder(zbase)
        ax.add_patch(p.shape)
    for seg in board.tracks:
        layer = seg["layer"]
        z = _Z_COPPER_BOTTOM if copper_bottom == layer else _Z_COPPER_TOP
        ax.plot(
            [seg["start"][0], seg["end"][0]],
            [seg["start"][1], seg["end"][1]],
            color=_layer_color(layer),
            linewidth=max(seg["width"] * _PT_PER_MM, 0.5),
            solid_capstyle="round",
            zorder=z,
        )

    # 3. Board edge (Edge.Cuts).
    for e in board.edges:
        _draw_shape(ax, e, _layer_color("Edge.Cuts"), 1.0, 1.0, _Z_EDGE)

    # 4. Silkscreen text (top of the visual stack).
    for t in board.texts:
        ax.text(
            t["at"][0],
            t["at"][1],
            t["label"],
            rotation=t["rot"],
            fontsize=t["size"] * _PT_PER_MM,
            color=_SILK_COLOR,
            ha="center",
            va="center",
            fontweight="bold" if t["bold"] else "normal",
            zorder=_Z_SILK,
            path_effects=[pe.withStroke(linewidth=1.0, foreground=_BG_COLOR)],
        )

    # 5. Ratsnest on top.
    for a, b in ratsnest:
        ax.plot(
            [a[0], b[0]],
            [a[1], b[1]],
            color=_RATSNEST_COLOR,
            linewidth=2.4,
            alpha=0.95,
            zorder=_Z_RATSNEST,
            path_effects=[pe.withStroke(linewidth=3.0, foreground=_BG_COLOR)],
        )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=_BG_COLOR, dpi=dpi)
    plt.close(fig)

    report: dict[str, Any] = {
        "pads": len(board.pads),
        "copper_layers": board.copper_layers,
        "connect_pads_requested": len(connect_pads or []),
        "missing_pads": missing,
        "pending_nets": pending_nets,
        "routed_nets": routed_reported,
    }
    lines = [
        f"Rendered {os.path.basename(pcb_path)}: {len(board.pads)} pads, "
        f"{len(board.tracks)} tracks; copper layers: {', '.join(board.copper_layers)}."
    ]
    if connect_pads:
        lines.append(
            f"connect_pads: missing={missing or 'none'}; "
            f"pending (unrouted) nets={pending_nets or 'none'}; "
            f"already routed nets={routed_reported or 'none'}."
        )
    return lines, buf.getvalue(), report


def register_render_board_tools(mcp: FastMCP) -> None:
    """Register self-contained board rendering tools with the MCP server."""

    @mcp.tool()
    async def export_pcb_layer_image(
        pcb_path: str,
        connect_pads: list[str] | None = None,
        output_dir: str | None = None,
        ctx: Context | None = None,
    ) -> tuple[str, Image]:
        """Render a KiCad PCB to a PNG composite image (no kicad-cli needed).

        The default composite shows courtyards, copper layers (F.Cu red,
        B.Cu blue), the board edge and silkscreen reference designators on a
        dark KiCad-style background.  When ``connect_pads`` is provided (e.g.
        ``["J1.2", "J2.2"]``) the *unrouted* nets joining those pads are
        drawn as green ratsnest lines so the model can see exactly which pads
        still need to be connected.

        Args:
            pcb_path: Path to the .kicad_pcb file.
            connect_pads: Optional list of ``REF.PAD`` specs to check.  Nets
                with copper already present are reported as routed; the rest
                are drawn as green ratsnest.
            output_dir: Optional directory to write the PNG to.
            ctx: FastMCP context for progress reporting.

        Returns:
            A text report plus the PNG image.
        """
        lines, png, report = render_board(pcb_path, connect_pads=connect_pads)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(pcb_path))[0]
            with open(os.path.join(output_dir, f"{base}.png"), "wb") as f:
                f.write(png)
        return "\n".join(lines) + f"\nreport={report}", Image(data=png, format="png")

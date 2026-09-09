"""
Unit tests for scripts/vlm_route_feedback.py (issue #124).

Covers the pure logic that the experiment script depends on: feedback
parsing, pad-spec parsing, net/pair enumeration, prompt building and the
round driver with a stubbed VLM client (no network, no real LLM).
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import shutil
import sys

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "vlm_route_feedback.py"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_spec = importlib.util.spec_from_file_location("vlm_route_feedback", _SCRIPT)
assert _spec is not None and _spec.loader is not None
vrf = importlib.util.module_from_spec(_spec)
sys.modules["vlm_route_feedback"] = vrf
_spec.loader.exec_module(vrf)

_FIXTURE_DIR = _REPO_ROOT / "tests" / "integration" / "fixtures"
_BOARD_FIXTURE = _FIXTURE_DIR / "test_routing_board.kicad_pcb"
_PRO_FIXTURE = _FIXTURE_DIR / "test_routing_board.kicad_pro"


@pytest.fixture
def board_copy(tmp_path: Path) -> str:
    """Copy the routing fixture to a writable temp dir, keeping the .pro."""
    pcb = tmp_path / "board.kicad_pcb"
    shutil.copy(_BOARD_FIXTURE, pcb)
    shutil.copy(_PRO_FIXTURE, tmp_path / "board.kicad_pro")
    return str(pcb)


@pytest.fixture
def pair_vcc() -> vrf.PairSpec:
    return vrf.PairSpec(ref_a="R1", pad_a="1", ref_b="C1", pad_b="1", net="VCC")


# ---------------------------------------------------------------------------
# parse_feedback
# ---------------------------------------------------------------------------


def test_parse_feedback_full_protocol() -> None:
    fb = vrf.parse_feedback("route: R1.1 -> C1.1\nlayer: F.Cu\nreason: shortest distance")
    assert fb.route_a == "R1.1"
    assert fb.route_b == "C1.1"
    assert fb.layer == "F.Cu"
    assert fb.reason == "shortest distance"


def test_parse_feedback_tolerates_prose() -> None:
    fb = vrf.parse_feedback(
        "Looking at the board:\n  route: R1.1 -> C1.1\n  layer: auto\nI chose this pair."
    )
    assert fb.route_a == "R1.1"
    assert fb.route_b == "C1.1"
    assert fb.layer == "auto"


def test_parse_feedback_missing_route_returns_none_pads() -> None:
    fb = vrf.parse_feedback("I cannot see any pads.")
    assert fb.route_a is None
    assert fb.route_b is None


def test_parse_feedback_arrow_variants() -> None:
    for arrow in ["->", "→", "-"]:
        fb = vrf.parse_feedback(f"route: R1.1 {arrow} C1.1")
        assert (fb.route_a, fb.route_b) == ("R1.1", "C1.1"), arrow


# ---------------------------------------------------------------------------
# parse_pad
# ---------------------------------------------------------------------------


def test_parse_pad_splits_ref_and_number() -> None:
    assert vrf.parse_pad("R1.1") == ("R1", "1")


def test_parse_pad_strips_whitespace() -> None:
    assert vrf.parse_pad(" R1.1 ") == ("R1", "1")


def test_parse_pad_rejects_missing_dot() -> None:
    with pytest.raises(ValueError):
        vrf.parse_pad("R11")


def test_parse_pad_rejects_empty_parts() -> None:
    with pytest.raises(ValueError):
        vrf.parse_pad("R1.")


# ---------------------------------------------------------------------------
# Pair enumeration against the real fixture
# ---------------------------------------------------------------------------


def test_routable_pairs_on_fixture(board_copy: str) -> None:
    pairs = vrf.routable_pairs(board_copy)
    keys = {p.key for p in pairs}
    # Fixture nets: VCC (R1.1, C1.1), GND (R1.2, D1.1), NET_A (C1.2 only).
    assert keys == {"C1.1-R1.1", "D1.1-R1.2"}
    for p in pairs:
        assert p.net in ("VCC", "GND")


def test_pad_center_world_coords(board_copy: str) -> None:
    data = vrf.load_pcb(board_copy)
    c = vrf._pad_center(data, "R1", "1")
    assert c is not None
    assert abs(c[0] - 29.5) < 1e-6  # R1 at (30,30), pad 1 local (-0.5, 0)
    assert abs(c[1] - 30.0) < 1e-6


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------


def test_build_prompt_lists_pending_pairs() -> None:
    pending = vrf.PairSpec(ref_a="R1", pad_a="1", ref_b="C1", pad_b="1", net="VCC")
    done = vrf.PairSpec(ref_a="R1", pad_a="2", ref_b="D1", pad_b="1", net="GND", status="done")
    text = vrf.build_prompt([pending, done], last_error=None)
    assert "R1.1 -> C1.1" in text
    assert "R1.2 -> D1.1" not in text
    assert "last_error" not in text


def test_build_prompt_appends_error_on_failure() -> None:
    pairs = [vrf.PairSpec(ref_a="R1", pad_a="1", ref_b="C1", pad_b="1", net="VCC")]
    text = vrf.build_prompt(pairs, last_error="path blocked")
    assert "last_error: path blocked" in text
    assert "advice:" in text


# ---------------------------------------------------------------------------
# run_experiment with a stubbed VLM client
# ---------------------------------------------------------------------------


class StubVLM:
    """Fake VLM: returns scripted replies in order, records prompts."""

    def __init__(self, replies: list[str], exhausted: str | None = None) -> None:
        self.replies = list(replies)
        self.exhausted = exhausted
        self.prompts: list[str] = []

    def ask(self, png_bytes: bytes, user_prompt: str) -> str:
        self.prompts.append(user_prompt)
        assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n", "expected a PNG payload"
        if not self.replies:
            if self.exhausted is None:
                raise RuntimeError("no more scripted replies")
            return self.exhausted
        return self.replies.pop(0)


def test_experiment_routes_with_stub_vlm(board_copy: str, tmp_path: Path) -> None:
    client = StubVLM(
        [
            "route: C1.1 -> R1.1\nlayer: auto\nreason: direct pair\n",
            "route: D1.1 -> R1.2\nlayer: auto\nreason: cross-layer net\n",
        ]
    )
    metrics = vrf.run_experiment(
        board_copy, client, rounds=8, work_dir=str(tmp_path), dry_run=False
    )
    assert metrics["pairs_total"] == 2
    assert metrics["pairs_done"] == 2
    assert metrics["pairs_failed"] == 0
    assert metrics["feedback_errors"] == 0
    assert metrics["per_pair"]["C1.1-R1.1"]["status"] == "done"
    assert metrics["per_pair"]["D1.1-R1.2"]["status"] == "done"
    assert (tmp_path / "round_01.png").exists()
    assert (tmp_path / "final.png").exists()


def test_experiment_failure_feeds_back_and_recovers(board_copy: str, tmp_path: Path) -> None:
    # First reply names a non-pending pair -> feedback_error; second is valid.
    # Once replies are exhausted the stub converges on the remaining pair, so
    # the loop runs out of pending work and exits cleanly.
    client = StubVLM(
        [
            "route: U1.1 -> R1.1\nlayer: auto\nreason: wrong pair\n",
            "route: C1.1 -> R1.1\nlayer: auto\nreason: corrected\n",
        ],
        exhausted="route: D1.1 -> R1.2\nlayer: auto\nreason: remaining\n",
    )
    metrics = vrf.run_experiment(
        board_copy, client, rounds=8, work_dir=str(tmp_path), dry_run=False
    )
    assert metrics["feedback_errors"] == 1
    assert metrics["pairs_done"] == 2
    assert metrics["pairs_remaining"] == 0
    assert "last_error" not in metrics  # recovery path clears the error


def test_experiment_unparseable_reply_counts_error(board_copy: str, tmp_path: Path) -> None:
    client = StubVLM(
        ["I see resistors and a capacitor.\n"],
        exhausted="route: C1.1 -> R1.1\nlayer: auto\nreason: fallback\n",
    )
    # rounds=2: round 1 is unparseable, round 2 consumes the fallback reply.
    metrics = vrf.run_experiment(
        board_copy, client, rounds=2, work_dir=str(tmp_path), dry_run=False
    )
    assert metrics["feedback_errors"] == 1
    assert metrics["pairs_done"] == 1  # fallback routed the first pair (VCC)
    assert metrics["pairs_remaining"] == 1  # GND pair left untried
    assert "last_error" not in metrics  # cleared once routing succeeds


# ---------------------------------------------------------------------------
# dry-run (fixed-order control arm)
# ---------------------------------------------------------------------------


def test_dry_run_routes_in_fixed_order(board_copy: str, tmp_path: Path) -> None:
    metrics = vrf.run_experiment(board_copy, None, rounds=4, work_dir=str(tmp_path), dry_run=True)
    assert metrics["pairs_total"] == 2
    assert metrics["pairs_done"] == 2
    assert metrics["pairs_failed"] == 0


# ---------------------------------------------------------------------------
# argparse: --via-pairs parsing
# ---------------------------------------------------------------------------


def test_parse_via_pairs() -> None:
    assert vrf._parse_via_pairs("F.Cu:B.Cu,B.Cu:In1.Cu") == (
        ("F.Cu", "B.Cu"),
        ("B.Cu", "In1.Cu"),
    )


def test_parse_via_pairs_rejects_bad_token() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        vrf._parse_via_pairs("F.Cu-B.Cu")

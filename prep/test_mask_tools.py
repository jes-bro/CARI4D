#!/usr/bin/env python3
"""Tests for the mask visualization / trimming helpers in run_sam3_masks.py.

Run:  python3 -m pytest prep/test_mask_tools.py -q

These cover the pure logic only -- run finding, the timeline strip, bbox and
zoom geometry, and mask-area counting. Whether a rendered frame LOOKS right is
for a human to judge; what is tested here is what a human cannot eyeball, e.g.
that a single lost frame is not averaged into a healthy-looking timeline.

run_sam3_masks imports torch / cv2 / sam3 at module scope, which are not needed
by any of these helpers, so they are stubbed before import. cv2 is stubbed only
for the drawing calls the pure helpers never make.
"""
import sys
import types

import numpy as np
import pytest

# --- stub the heavy imports so the module can be loaded without a GPU env ----
for name in ("torch", "cv2", "imageio", "h5py"):
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)
sys.modules["torch"].cuda = types.SimpleNamespace(device_count=lambda: 0)
sam3 = types.ModuleType("sam3")
sam3_mb = types.ModuleType("sam3.model_builder")
sam3_mb.build_sam3_video_predictor = lambda **kw: None
sys.modules["sam3"] = sam3
sys.modules["sam3.model_builder"] = sam3_mb

import os  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_sam3_masks as R  # noqa: E402


# ------------------------------------------------------------- mask handling

def test_missing_mask_reads_as_empty_not_error():
    """A dropped track is an absent key OR an empty mask; both mean 'lost'."""
    assert R._mask_or_empty({}, 3, (4, 5)).shape == (4, 5)
    assert not R._mask_or_empty({}, 3, (4, 5)).any()


def test_mask_areas_counts_pixels_over_both_dicts():
    shape = (2, 3)
    hm = {0: np.ones(shape, bool), 1: np.zeros(shape, bool)}
    om = {0: np.zeros(shape, bool)}                       # frame 1 absent entirely
    per, obj = R.mask_areas(hm, om, 2, shape)
    assert list(per) == [6, 0]
    assert list(obj) == [0, 0]


# --------------------------------------------------------------- run finding

def test_runs_basic_and_edges():
    assert R.find_tracked_runs(np.array([0, 1, 1, 0, 1], bool)) == [(1, 2), (4, 4)]
    assert R.find_tracked_runs(np.array([1, 1, 0, 1, 1], bool)) == [(0, 1), (3, 4)]
    assert R.find_tracked_runs(np.ones(4, bool)) == [(0, 3)]
    assert R.find_tracked_runs(np.zeros(4, bool)) == []


def test_gap_tolerance_bridges_only_short_dropouts():
    good = np.array([1, 1, 0, 1, 1], bool)
    assert R.find_tracked_runs(good, 0) == [(0, 1), (3, 4)]
    assert R.find_tracked_runs(good, 1) == [(0, 4)]
    longer = np.array([1, 1, 0, 0, 0, 1], bool)
    assert R.find_tracked_runs(longer, 2) == [(0, 1), (5, 5)]
    assert R.find_tracked_runs(longer, 3) == [(0, 5)]


def test_bridged_run_never_ends_on_a_dropout():
    """Trailing lost frames must not be padded onto the end of a run."""
    assert R.find_tracked_runs(np.array([1, 1, 0, 0], bool), 5) == [(0, 1)]


def test_longest_run_is_selectable_by_rank():
    good = np.array([1, 1, 0, 1, 1, 1, 1, 0, 1], bool)
    runs = sorted(R.find_tracked_runs(good), key=lambda r: -(r[1] - r[0] + 1))
    assert runs[0] == (3, 6) and runs[-1] == (8, 8)


# ---------------------------------------------------------------- geometry

def test_bbox_none_when_empty_and_clamped_at_edges():
    assert R.object_bbox(np.zeros((5, 5), bool)) is None
    m = np.zeros((20, 20), bool)
    m[8:12, 9:13] = True
    assert R.object_bbox(m, pad=3) == (6, 5, 16, 15)
    m2 = np.zeros((20, 20), bool)
    m2[0, 0] = True
    assert R.object_bbox(m2, pad=5) == (0, 0, 6, 6)


def test_zoom_inset_magnifies_and_is_square():
    img = np.zeros((40, 40, 3), np.uint8)
    img[10:14, 10:14] = 200
    ins = R.zoom_inset(img, (10, 10, 14, 14), size=40)
    assert ins.shape == (40, 40, 3)
    assert ins[0, 0].tolist() == [200, 200, 200]


def test_zoom_inset_degenerate_bbox_is_none():
    assert R.zoom_inset(np.zeros((8, 8, 3), np.uint8), (2, 2, 2, 2), 20) is None


# ---------------------------------------------------------------- timeline

def test_strip_colours_and_shape():
    g = R.timeline_strip(np.ones(50, bool), width=25, height=7)
    assert g.shape == (7, 25, 3)
    assert set(map(tuple, g.reshape(-1, 3))) == {(60, 190, 90)}
    r = R.timeline_strip(np.zeros(50, bool), width=25)
    assert set(map(tuple, r.reshape(-1, 3))) == {(150, 30, 30)}


def test_single_dropout_survives_downsampling():
    """The bar must not average a lost frame away into a green column."""
    good = np.ones(100, bool)
    good[50] = False
    strip = R.timeline_strip(good, width=10)
    assert (150, 30, 30) in {tuple(strip[0, i]) for i in range(10)}


def test_cursor_drawn_and_moves():
    good = np.ones(100, bool)
    a = R.timeline_strip(good, width=50, cursor=0)
    b = R.timeline_strip(good, width=50, cursor=99)
    assert (255, 220, 0) in set(map(tuple, a[0]))
    assert not np.array_equal(a, b)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

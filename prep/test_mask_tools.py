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


# ------------------------------------------------------ instance selection

def _det(*masks):
    """A fake SAM3 detection output holding the given (H, W) bool masks."""
    return {"out_binary_masks": np.stack(masks), "out_obj_ids": np.arange(len(masks))}


def _blob(shape, y0, y1, x0, x1):
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


def test_select_instances_keeps_each_slot_by_overlap():
    """Two dancers, new chunk: each slot re-finds its own person, whatever SAM3's order."""
    shape = (20, 40)
    a_prev, b_prev = _blob(shape, 0, 20, 0, 15), _blob(shape, 0, 20, 25, 40)
    # SAM3 lists B first this chunk, slightly shifted
    out = _det(_blob(shape, 0, 20, 26, 40), _blob(shape, 0, 20, 1, 16))
    masks, ids = R.select_instances(out, [a_prev, b_prev])
    assert ids == [1, 0]
    assert masks[0][:, 5].all() and masks[1][:, 30].all()


def test_select_instances_greedy_settles_the_best_pair_first():
    """When one instance overlaps both slots, the slot it overlaps MORE gets it."""
    shape = (10, 30)
    a_prev, b_prev = _blob(shape, 0, 10, 0, 10), _blob(shape, 0, 10, 8, 20)
    big = _blob(shape, 0, 10, 6, 22)          # overlaps A by 4 cols, B by 12
    other = _blob(shape, 0, 10, 0, 3)         # overlaps A by 3
    masks, ids = R.select_instances(out := _det(big, other), [a_prev, b_prev])
    assert ids == [1, 0], "B keeps the big one, A falls back to the other"


def test_select_instances_fills_unmatched_slots_by_area():
    """First chunk (no history): slots are filled largest-first, no slot left empty."""
    shape = (10, 30)
    small, large = _blob(shape, 0, 3, 0, 3), _blob(shape, 0, 10, 10, 30)
    masks, ids = R.select_instances(_det(small, large), [None, None])
    assert ids == [1, 0]


def test_select_instances_fewer_detections_than_slots():
    """One dancer visible: the other slot is None, not a duplicate."""
    shape = (10, 10)
    masks, ids = R.select_instances(_det(_blob(shape, 0, 10, 0, 5)), [None, None])
    assert ids == [0, None] and masks[1] is None
    masks, ids = R.select_instances({"out_binary_masks": np.zeros((0, 10, 10), bool),
                                     "out_obj_ids": np.array([])}, [None, None])
    assert ids == [None, None]


def test_select_single_mask_is_the_one_slot_case(monkeypatch):
    """The old function's contract survives: overlap, then near-person, then largest."""
    shape = (32, 32)
    monkeypatch.setattr(R.cv2, "dilate", lambda m, k: m, raising=False)
    far_big, near_small = _blob(shape, 0, 32, 0, 16), _blob(shape, 20, 24, 20, 24)
    person = _blob(shape, 18, 26, 18, 26)
    m, i = R.select_single_mask(_det(far_big, near_small), person_mask=person)
    assert i == 1, "near the person beats largest"
    m, i = R.select_single_mask(_det(far_big, near_small), ref_mask=far_big)
    assert i == 0, "overlap with the previous mask beats everything"
    m, i = R.select_single_mask(_det(far_big, near_small))
    assert i == 0, "largest when there is nothing else to go on"


def test_tracked_frames_needs_everyone_present():
    """No object: only the people count. Two people: both must be there."""
    shape = (2, 2)
    on, off = np.ones(shape, bool), np.zeros(shape, bool)
    h1 = {0: on, 1: on, 2: off}
    h2 = {0: on, 1: off, 2: on}
    good, per, obj, per2 = R.tracked_frames(h1, {}, 3, shape, has_object=False)
    assert list(good) == [True, True, False] and per2 is None
    good, per, obj, per2 = R.tracked_frames(h1, {}, 3, shape, human2_masks=h2, has_object=False)
    assert list(good) == [True, False, False] and list(per2) == [4, 0, 4]
    good, *_ = R.tracked_frames(h1, {0: on}, 3, shape, has_object=True)
    assert list(good) == [True, False, False]


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

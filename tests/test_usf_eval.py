"""Unit tests of the UrbanSARFloods evaluation helpers (analysis/usf_eval.py); no data download."""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("usf_eval", Path(__file__).parents[1] / "analysis" / "usf_eval.py")
usf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(usf)


def test_confusion_excludes_other_flood_class():
    gt = np.array([0, 0, 1, 1, 2, 2, 255], np.uint8)
    pred = np.array([1, 0, 1, 0, 1, 0, 1], bool)
    valid = np.ones(7, bool)
    # open flood: the urban pixels and the GT no-data pixel are ignored
    assert usf.confusion(pred, gt, valid, 1) == (1, 1, 1, 1)
    assert usf.confusion(pred, gt, valid, 2) == (1, 1, 1, 1)
    valid[0] = False
    assert usf.confusion(pred, gt, valid, 1) == (1, 0, 1, 1)


def test_scores():
    s = usf.scores(tp=6, fp=2, fn=4)
    assert s["precision"] == pytest.approx(0.75)
    assert s["recall"] == pytest.approx(0.6)
    assert s["f1"] == pytest.approx(2 * 6 / (12 + 6))
    assert s["iou"] == pytest.approx(0.5)
    assert all(math.isnan(v) for v in usf.scores(0, 0, 0).values())


def _stats(coh_values, db):
    """Synthetic per-band statistics in the format of band_stats()."""
    stats = {}
    for b, (lo, hi) in enumerate(coh_values, 1):
        stats[b] = {"min": 0.0, "max": 1.0, "median": lo, "mean_by_gt": {"0": lo, "1": lo, "2": hi}}
    for b, (med, d1) in enumerate(db, 5):
        stats[b] = {"min": -60.0, "max": 6.99, "median": med, "mean_by_gt": {"0": med, "1": med + d1, "2": med}}
    return stats


def test_identify_bands_matches_data_layout():
    # coherence: bands 1/2 high over urban flood (pre-event), 3/4 low (co-event)
    coh = [(0.3, 0.65), (0.38, 0.7), (0.28, 0.28), (0.32, 0.4)]
    # intensity: 5/6 pre (VH, VV), 7/8 post (VH, VV) darkening over open flood
    db = [(-13.5, -0.8), (-7.0, -0.5), (-13.2, -8.3), (-7.0, -9.3)]
    mapping, reasons = usf.identify_bands(_stats(coh, db))
    assert {k: mapping[k] for k in usf.EXPECTED_BANDS} == usf.EXPECTED_BANDS
    assert any("post-event" in r for r in reasons)


def test_identify_bands_detects_swapped_order():
    db = [(-13.2, -8.3), (-7.0, -9.3), (-13.5, -0.8), (-7.0, -0.5)]  # post pair first
    coh = [(0.3, 0.65), (0.38, 0.7), (0.28, 0.28), (0.32, 0.4)]
    mapping, _ = usf.identify_bands(_stats(coh, db))
    assert mapping["int_post_VV"] == 6 and mapping["int_pre_VV"] == 8


def test_error_map_codes():
    gt = np.array([0, 0, 1, 1, 2, 2, 255], np.uint8)
    pred = np.array([0, 1, 1, 0, 1, 0, 1], bool)
    e = usf.error_map(pred, gt, np.ones(7, bool))
    assert e.tolist() == [1, 6, 2, 4, 3, 5, 0]

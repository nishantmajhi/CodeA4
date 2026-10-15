"""
Stage 1+2 against synthetic sheets with known ground truth.

What these tests do prove: that line-finding, line-to-exercise alignment and
character alignment survive perspective, uneven lighting, blur, noise, drifting
baselines, organic wrapping and letters written touching.

What they do not prove: behaviour on real paper and real pens — ink bleed,
ruled lines, crossings-out, genuinely joined cursive. The thresholds below are
therefore floors chosen to catch regressions, not claims about accuracy in the
field.
"""
import json

import pytest

from pipeline.exercise import INK_LINES
from pipeline.imaging import CaptureError, find_line_bands, prepare_sheet
from pipeline.segmentation import align_lines, segment_photo


def _truth_by_line(path):
    by_line = {}
    for record in json.load(open(path)):
        by_line.setdefault(record["written_line"], []).append(record)
    for values in by_line.values():
        values.sort(key=lambda r: r["x"])
    return by_line


def test_every_written_line_is_found(sheets):
    """
    Line detection has to be exact, not merely close. Missing one line shifts
    every following line onto the wrong expected text, so a single miss
    corrupts the whole sheet rather than costing one line.
    """
    for image, truth_path in sheets:
        truth = _truth_by_line(truth_path)
        mask, _gray = prepare_sheet(image)
        bands = find_line_bands(mask)
        assert len(bands) == len(truth), f"{image}: band count must match exactly"


def test_lines_map_onto_the_right_exercise_text(sheets):
    """Each expected line must claim the bands actually written for it,
    including the over-long line that wraps onto several."""
    for image, truth_path in sheets:
        truth = _truth_by_line(truth_path)
        mask, _gray = prepare_sheet(image)
        bands = find_line_bands(mask)
        assignment = align_lines(mask, bands)

        assert len(assignment) == len(INK_LINES)
        for line_index, band_indices in enumerate(assignment):
            for band in band_indices:
                written = truth[band]
                assert written[0]["line"] == line_index, (
                    f"band {band} assigned to exercise line {line_index}, "
                    f"but was written for line {written[0]['line']}"
                )


def test_wrapped_line_spans_several_bands(sheets):
    """The exercise's deliberately long line must be observed wrapping — that
    is the only place we learn how this person breaks a line."""
    image, _truth = sheets[0]
    mask, _gray = prepare_sheet(image)
    assignment = align_lines(mask, find_line_bands(mask))
    assert max(len(bands) for bands in assignment) > 1


def test_character_labels_are_mostly_correct(sheets, segmented):
    """
    Characters must land on the right ink. Some error is expected where
    letters were written touching and had to be cut apart; the floor guards
    against a regression that would quietly poison the glyph sets.
    """
    correct = total = 0
    for (_image, truth_path), result in zip(sheets, segmented):
        truth = _truth_by_line(truth_path)
        by_band = {}
        for instance in result["instances"]:
            by_band.setdefault(instance["band"], []).append(instance)

        for band, predictions in by_band.items():
            predictions.sort(key=lambda i: i["actual_xy"][0])
            for prediction, actual in zip(predictions, truth.get(band, [])):
                total += 1
                correct += prediction["char"] == actual["char"]

    accuracy = correct / total
    assert accuracy > 0.90, f"character labelling fell to {accuracy:.1%}"


def test_almost_every_character_is_extracted(segmented):
    for result in segmented:
        meta = result["meta"]
        assert meta["matched_chars"] >= 0.98 * meta["expected_chars"]


def test_instances_carry_what_later_stages_need(segmented):
    for instance in segmented[0]["instances"][:50]:
        assert instance["path"].startswith("M")
        assert instance["bitmap"]
        assert len(instance["expected_xy"]) == 2
        assert len(instance["actual_xy"]) == 2
        assert instance["advance_em"] > 0


def test_measurements_are_physically_plausible(segmented):
    """Guards against a units mix-up, which would otherwise surface only as
    output rendered at absurd size."""
    for result in segmented:
        meta = result["meta"]
        assert 1.5 < meta["x_height_mm"] < 8.0
        assert 0.0 < meta["letter_gap_em"] < 0.5
        assert meta["word_gap_em"] > meta["letter_gap_em"]
        assert 1.0 < meta["line_spacing_em"] < 3.5


def test_unreadable_photo_raises_a_message_for_the_user(tmp_path):
    """
    A failure the user can fix must arrive as advice, not a stack trace — the
    capture screen shows this text verbatim.
    """
    import numpy as np
    import cv2

    blank = np.full((1200, 850, 3), 255, dtype=np.uint8)
    path = str(tmp_path / "blank.jpg")
    cv2.imwrite(path, blank)

    with pytest.raises(CaptureError) as excinfo:
        segment_photo(path)
    assert len(str(excinfo.value)) > 30


def test_the_width_prior_is_actually_fitted_to_the_hand(segmented):
    """
    `calibrate_widths` exists because the generic width prior can be wrong
    about this writer in two independent ways -- scale, and how uniform the
    hand is -- and the alignment DP scores purely on width, so a systematic
    error makes merging two letters cost about the same as matching them
    correctly.

    It had never run. Its evidence was lines whose component count already
    equalled their character count, needing thirty such samples; one or two of
    the sixteen lines on a sheet are ever 1:1, because a single merge or split
    anywhere disqualifies the whole line. It returned the identity fallback on
    every sheet this project has been given -- and would have returned it most
    surely on the hand it was written for, since a hand the prior fits badly
    produces fewer 1:1 lines, not more.
    """
    for result in segmented:
        fit = result["meta"]["width_fit"]
        assert (fit["alpha"], fit["beta"]) != (0.0, 1.0), (
            "the width calibration fell back to the raw prior")
        assert 0.0 <= fit["beta"] <= 2.6
        assert -0.2 <= fit["alpha"] <= 1.0


def test_the_width_fit_recovers_a_hand_that_is_narrower_than_the_prior():
    """A synthetic sheet whose ink is a known function of the prior: the fit
    has to come back with that function."""
    from pipeline.exercise import INK_LINES
    from pipeline.segmentation import calibrate_widths, width_prior

    alpha, beta, em_px = 0.05, 0.6, 50.0
    per_band, assignment = [], []
    for index, line in enumerate(INK_LINES):
        chars = [c for c in line if c != " "]
        cursor = 0
        comps = []
        for char in chars:
            width = (alpha + beta * width_prior(char)) * em_px
            comps.append({"x0": int(cursor), "x1": int(cursor + width),
                          "y0": 0, "y1": 20, "area": 10})
            cursor += width + 4
        per_band.append(comps)
        assignment.append([index])

    _widths, (got_alpha, got_beta) = calibrate_widths(per_band, assignment, em_px)
    assert got_beta == pytest.approx(beta, abs=0.05)
    assert got_alpha == pytest.approx(alpha, abs=0.03)


def test_the_width_fit_falls_back_rather_than_fitting_nonsense():
    """Too few usable lines has to mean the plain prior, not a fit through
    two points."""
    from pipeline.segmentation import calibrate_widths, width_prior

    per_band = [[{"x0": 0, "x1": 10, "y0": 0, "y1": 10, "area": 5}]]
    _widths, fit = calibrate_widths(per_band, [[0]], 50.0)
    assert fit == (0.0, 1.0)
    widths, _fit = calibrate_widths([], [], 50.0)
    assert widths["a"] == width_prior("a")

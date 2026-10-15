"""
The image front-end, on synthetic masks rather than photographs.

`tests/test_segmentation.py` exercises this whole chain on generated sheets,
which is the right test for accuracy and the wrong one for behaviour: when it
fails it says a band count was wrong, not which of six stages produced it.
These are the individual stages, on input whose answer is arithmetic.
"""
import cv2
import numpy as np
import pytest

from pipeline.imaging import (CaptureError, PAGE_H, PAGE_W, _order_quad,
                              binarize, clear_page_border, estimate_skew,
                              find_line_bands, mm_to_px, prepare_sheet,
                              px_to_mm, rotate)


def _blank(height=600, width=800):
    return np.zeros((height, width), dtype=np.uint8)


def _bar(mask, top, bottom, left=100, right=700):
    mask[top:bottom, left:right] = 255
    return mask


# ------------------------------------------------------------------- units

def test_millimetres_and_pixels_round_trip():
    for mm in (1.0, 3.0, 18.0, 210.0):
        assert px_to_mm(mm_to_px(mm)) == pytest.approx(mm)


def test_the_page_canvas_is_a4_at_the_declared_dpi():
    """Every downstream measurement -- x-height in mm, drift amplitude in em
    -- depends on one page pixel being a known physical size."""
    assert px_to_mm(PAGE_W) == pytest.approx(210.0, abs=0.3)
    assert px_to_mm(PAGE_H) == pytest.approx(297.0, abs=0.3)


# ------------------------------------------------------------- page corners

def test_corners_are_ordered_clockwise_from_the_top_left():
    for rotation in range(4):
        points = np.array([[10, 10], [200, 20], [190, 300], [5, 290]],
                          dtype=np.float32)
        rolled = np.roll(points, rotation, axis=0)
        ordered = _order_quad(rolled)
        assert ordered[0].tolist() == [10, 10]     # top left
        assert ordered[1].tolist() == [200, 20]    # top right
        assert ordered[2].tolist() == [190, 300]   # bottom right
        assert ordered[3].tolist() == [5, 290]     # bottom left


# --------------------------------------------------------------- line bands

def test_each_written_line_becomes_one_band():
    mask = _blank()
    for top in (100, 200, 300, 400):
        _bar(mask, top, top + 30)
    bands = find_line_bands(mask)
    assert len(bands) == 4
    for (top, bottom), expected in zip(bands, (100, 200, 300, 400)):
        assert abs(top - expected) <= 3
        assert bottom > top


def test_a_blank_page_has_no_bands():
    assert find_line_bands(_blank()) == []


def test_a_line_holding_one_character_is_still_a_line():
    """
    A closing brace on its own has a tiny projection next to a line of prose.
    A threshold relative to the busiest line drops it, and dropping one line
    puts every following line onto the wrong expected text -- so the whole
    sheet is corrupted rather than one line lost.
    """
    mask = _blank()
    _bar(mask, 100, 130, left=100, right=700)
    _bar(mask, 200, 230, left=100, right=700)
    _bar(mask, 300, 330, left=100, right=112)      # a lone '}'
    _bar(mask, 400, 430, left=100, right=700)
    assert len(find_line_bands(mask)) == 4


def test_a_letter_and_its_own_ascender_are_one_line():
    """A detached dot or ascender must rejoin its line rather than becoming
    one of its own."""
    mask = _blank()
    _bar(mask, 100, 104, left=300, right=310)      # the dot of an 'i'
    _bar(mask, 112, 140, left=100, right=700)      # the line it belongs to
    _bar(mask, 250, 280, left=100, right=700)
    assert len(find_line_bands(mask)) == 2


def test_two_touching_lines_are_split_apart():
    """
    Handwriting on unruled paper regularly has a descender from one line
    overlapping an ascender from the next, so the projection never reaches
    zero between them. Left merged, every following line is off by one.
    """
    mask = _blank(height=900)
    for top in (100, 200, 300):
        _bar(mask, top, top + 40)
    _bar(mask, 400, 440)
    _bar(mask, 441, 480)                            # touching the one above
    for top in (550, 650):
        _bar(mask, top, top + 40)
    bands = find_line_bands(mask)
    assert len(bands) == 7


def test_no_band_is_empty_or_inverted():
    """Bands index into the mask, and a band whose top is past its bottom
    yields no components -- which shifts every band index after it."""
    rng = np.random.default_rng(5)
    for _ in range(60):
        mask = _blank(height=900)
        top = 60
        while top < 820:
            height = int(rng.integers(12, 70))
            _bar(mask, top, top + height, left=int(rng.integers(60, 200)),
                 right=int(rng.integers(400, 780)))
            top += height + int(rng.integers(1, 40))
        for band_top, band_bottom in find_line_bands(mask):
            assert 0 <= band_top <= band_bottom < mask.shape[0]


# -------------------------------------------------------------- page border

def test_the_page_edge_is_blanked():
    """Perspective correction drags a sliver of the desk into the frame, and
    adaptive thresholding reads it as ink at x~0, which wrecks the left
    margin measurement."""
    mask = _blank()
    mask[:, :5] = 255                                # the desk
    mask[300:330, 100:700] = 255                     # a line of writing
    cleared = clear_page_border(mask)
    assert cleared[:, :5].sum() == 0
    assert cleared[300:330, 100:700].sum() > 0


def test_clearing_a_zero_margin_changes_nothing():
    mask = _bar(_blank(), 100, 130)
    assert np.array_equal(clear_page_border(mask, margin_mm=0.0), mask)


# -------------------------------------------------------------- binarizing

def test_ink_becomes_white_and_paper_black():
    gray = np.full((400, 400), 240, dtype=np.uint8)
    gray[100:300, 150:156] = 40                      # a pen stroke
    mask = binarize(gray)
    assert mask[200, 152] == 255
    assert mask[20, 20] == 0


def test_a_shadow_is_not_ink_however_dark_it_is():
    """
    Illumination is flattened by dividing out a heavily blurred copy, so an
    area much larger than the blur kernel becomes its own idea of paper. That
    is the point -- the dark half of a page under a hand is paper, not a
    stroke -- and it is worth pinning down, because it also means this stage
    cannot be checked with a big solid rectangle.
    """
    gray = np.full((400, 400), 240, dtype=np.uint8)
    gray[100:300, 100:300] = 40                      # a shadow, not a pen
    assert binarize(gray)[200, 200] == 0


def test_a_stroke_that_fades_in_the_middle_stays_one_blob():
    """
    Where a pen bears down the ink is far darker than the thin fast middle of
    the same stroke. A margin tight enough to reject paper also rejects that
    middle, and the stroke breaks into two components -- which the segmenter
    hands to two different characters, sliding every label after it onto the
    wrong ink.
    """
    gray = np.full((300, 300), 235, dtype=np.uint8)
    gray[100:200, 100:110] = 30                      # a firm stroke
    gray[100:200, 110:130] = 150                     # the same stroke, faint
    gray[100:200, 130:140] = 30
    mask = binarize(gray)
    count, _labels = cv2.connectedComponents((mask > 0).astype(np.uint8))
    assert count - 1 == 1, "the stroke came back in pieces"


def test_isolated_faint_speckle_is_not_ink():
    gray = np.full((300, 300), 235, dtype=np.uint8)
    gray[150:160, 150:160] = 25                      # real ink
    rng = np.random.default_rng(3)
    for _ in range(40):
        y, x = rng.integers(10, 280, 2)
        gray[y, x] = 190                             # a faint speck
    mask = binarize(gray)
    assert mask[150:160, 150:160].sum() > 0
    assert mask.sum() / 255 < 400


def test_an_uneven_gradient_does_not_decide_what_is_ink():
    """A phone photo always has one, and a single global threshold either
    loses faint ink on the dark side or floods the bright side."""
    _ys, xs = np.mgrid[0:400, 0:400]
    gray = (250 - 0.25 * xs).astype(np.int16)
    for x in (60, 330):
        gray[180:220, x:x + 8] -= 170                # the same pen, both ends
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    mask = binarize(gray)
    assert mask[200, 63] == 255, "ink lost on the bright side"
    assert mask[200, 333] == 255, "ink lost on the dark side"


# -------------------------------------------------------------------- skew

def test_a_straight_page_is_reported_as_straight():
    mask = _blank(height=800)
    for top in range(100, 700, 100):
        _bar(mask, top, top + 25)
    assert abs(estimate_skew(mask)) < 0.3


def test_a_tilted_page_is_measured_and_the_tilt_is_removed():
    mask = _blank(height=800)
    for top in range(100, 700, 100):
        _bar(mask, top, top + 25)
    tilted = rotate(mask, -2.0)
    measured = estimate_skew(tilted)
    assert measured == pytest.approx(2.0, abs=0.5)
    assert abs(estimate_skew(rotate(tilted, measured))) < 0.5


# ------------------------------------------------------- whole-sheet errors

def test_a_photo_that_is_not_an_image_is_refused_with_advice(tmp_path):
    path = tmp_path / "not-a-photo.jpg"
    path.write_bytes(b"this is not a JPEG")
    with pytest.raises(CaptureError) as excinfo:
        prepare_sheet(str(path))
    assert "photo" in str(excinfo.value).lower()


def test_a_blank_sheet_says_it_found_no_writing(tmp_path):
    path = str(tmp_path / "blank.jpg")
    cv2.imwrite(path, np.full((1200, 850, 3), 255, dtype=np.uint8))
    with pytest.raises(CaptureError) as excinfo:
        prepare_sheet(path)
    assert "writing" in str(excinfo.value).lower()


def test_a_photo_that_is_almost_all_ink_says_it_is_too_dark(tmp_path):
    rng = np.random.default_rng(1)
    noise = rng.integers(0, 90, (1200, 850, 3), dtype=np.uint8)
    path = str(tmp_path / "dark.jpg")
    cv2.imwrite(path, noise)
    with pytest.raises(CaptureError) as excinfo:
        prepare_sheet(path)
    assert len(str(excinfo.value)) > 30


def test_every_capture_error_is_written_for_the_user(tmp_path):
    """
    These strings are shown verbatim in the capture step's error box, so each
    has to say what the user can do differently rather than what failed.
    """
    path = tmp_path / "bad.jpg"
    path.write_bytes(b"nope")
    with pytest.raises(CaptureError) as excinfo:
        prepare_sheet(str(path))
    message = str(excinfo.value)
    assert message[0].isupper() and message.rstrip().endswith(".")
    assert "None" not in message and "Traceback" not in message

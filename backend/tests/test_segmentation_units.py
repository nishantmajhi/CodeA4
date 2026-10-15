"""
Stage 2's decisions, separated from the photograph they normally arrive with.

The whole method rests on knowing exactly what the user wrote, which turns
recognition into alignment: never "what is this blob", only "which blobs are
which known characters". That alignment is two dynamic programs and a handful
of clean-up passes, and each has a failure that is silent -- the character
count stays right while the ink behind each label slides one place along.

Everything here is built from component boxes directly, so a failure names the
pass that produced it.
"""
import numpy as np
import pytest

from pipeline.exercise import INK_LINES
from pipeline.imaging import CaptureError
from pipeline.segmentation import (_absorb_fragments, _margin_offsets,
                                   _measure_gaps, _merge_stacked,
                                   _split_component, align_chars, align_lines,
                                   alignment_confidence, width_prior)


def comp(x0, x1, y0=0, y1=40, band=0):
    return {"x0": x0, "x1": x1, "y0": y0, "y1": y1,
            "area": (x1 - x0) * (y1 - y0), "band_index": band}


# ------------------------------------------------------------ width priors

def test_every_character_the_exercise_asks_for_has_a_prior():
    """The alignment DP scores on width alone. A character on the generic
    default is a character the DP cannot tell from any other."""
    from pipeline.segmentation import _DEFAULT_WIDTH, _WIDTH_PRIOR

    written = {c for line in INK_LINES for c in line if c != " "}
    assert written <= set(_WIDTH_PRIOR), (
        f"no width for {sorted(written - set(_WIDTH_PRIOR))}")
    assert _DEFAULT_WIDTH > 0


def test_priors_order_narrow_and_wide_letters_correctly():
    for narrow, wide in [("l", "m"), ("i", "w"), ("I", "W"), (".", "8"),
                         ("'", "@")]:
        assert width_prior(narrow) < width_prior(wide), f"{narrow} vs {wide}"


def test_a_measured_table_overrides_the_prior():
    assert width_prior("a", {"a": 0.9}) == 0.9
    assert width_prior("b", {"a": 0.9}) == width_prior("b")


# ------------------------------------------------------- line-level alignment

def _bands_for(mask_width=1600):
    """One band per exercise line, wide enough to look like writing."""
    return [(i * 60, i * 60 + 40) for i in range(len(INK_LINES))]


def _mask_for(bands, width=1600):
    mask = np.zeros((bands[-1][1] + 60, width), dtype=np.uint8)
    for index, (top, bottom) in enumerate(bands):
        length = int(width * 0.9 * (0.4 + 0.6 * (index % 3) / 2))
        mask[top:bottom, 40:40 + length] = 255
    return mask


def test_too_few_lines_is_reported_as_something_the_user_can_fix():
    bands = [(0, 40), (60, 100)]
    mask = _mask_for(bands)
    with pytest.raises(CaptureError) as excinfo:
        align_lines(mask, bands)
    message = str(excinfo.value)
    assert str(len(INK_LINES)) in message
    assert "photo" in message.lower() or "sheet" in message.lower()


def test_far_too_many_lines_is_refused_rather_than_aligned():
    bands = [(i * 20, i * 20 + 10) for i in range(len(INK_LINES) * 4)]
    with pytest.raises(CaptureError):
        align_lines(_mask_for(bands), bands)


def test_one_band_per_line_is_matched_one_to_one():
    bands = _bands_for()
    assignment = align_lines(_mask_for(bands), bands)
    assert assignment == [[i] for i in range(len(INK_LINES))]


def test_every_line_gets_at_least_one_band():
    """A line assigned nothing has no ink, and every character on it is then
    attributed to whatever came next."""
    bands = _bands_for() + [(len(INK_LINES) * 60, len(INK_LINES) * 60 + 40)]
    assignment = align_lines(_mask_for(bands), bands)
    assert len(assignment) == len(INK_LINES)
    assert all(assignment)
    used = [b for group in assignment for b in group]
    assert used == sorted(used) == list(range(len(bands)))


# ------------------------------------------------- component clean-up passes

def test_a_dot_is_merged_into_the_letter_below_it():
    """An 'i' and its dot are two connected components and one character."""
    merged = _merge_stacked([comp(100, 112, y0=0, y1=8),
                             comp(100, 112, y0=20, y1=60)])
    assert len(merged) == 1
    assert (merged[0]["y0"], merged[0]["y1"]) == (0, 60)


def test_the_two_bars_of_an_equals_sign_are_one_character():
    merged = _merge_stacked([comp(100, 140, y0=20, y1=26),
                             comp(100, 140, y0=34, y1=40)])
    assert len(merged) == 1


def test_letters_side_by_side_are_not_merged():
    merged = _merge_stacked([comp(100, 140), comp(150, 190)])
    assert len(merged) == 2


def test_a_broken_off_sliver_rejoins_the_stroke_it_touches():
    """
    A thin diagonal -- the middle of a 'W', the join of an 'N' -- can come
    back as its own tiny component. Left alone the DP gives it a whole
    character, pushing every label after it onto the wrong ink.
    """
    absorbed = _absorb_fragments([comp(100, 140), comp(140, 146),
                                  comp(200, 240), comp(260, 300)])
    assert len(absorbed) == 3
    assert absorbed[0]["x1"] == 146


def test_punctuation_with_air_around_it_is_left_alone():
    """A full stop is told from a fragment by contact, not by size."""
    kept = _absorb_fragments([comp(100, 140), comp(160, 166),
                              comp(200, 240), comp(260, 300)])
    assert len(kept) == 4


def test_absorbing_never_loses_ink():
    rng = np.random.default_rng(2)
    for _ in range(200):
        comps, cursor = [], 0
        for _ in range(int(rng.integers(2, 9))):
            width = int(rng.integers(3, 50))
            comps.append(comp(cursor, cursor + width))
            cursor += width + int(rng.integers(-4, 30))
        out = _absorb_fragments(comps)
        assert out
        assert min(c["x0"] for c in out) == min(c["x0"] for c in comps)
        assert max(c["x1"] for c in out) == max(c["x1"] for c in comps)
        assert [c["x0"] for c in out] == sorted(c["x0"] for c in out)


# -------------------------------------------------------------- cutting apart

def test_touching_letters_are_cut_where_the_ink_is_thinnest():
    """A true minimum in ink density is where the pen was thinnest, which is
    a better cut than the expected width alone."""
    mask = np.zeros((60, 300), dtype=np.uint8)
    mask[10:50, 100:148] = 255
    mask[10:50, 152:200] = 255                   # a gap at 148..152
    spans = _split_component(comp(100, 200, y0=10, y1=49), ["a", "b"], mask)
    assert len(spans) == 2
    assert 144 <= spans[0][1] <= 156


def test_a_component_too_narrow_to_cut_is_divided_evenly():
    mask = np.zeros((20, 20), dtype=np.uint8)
    spans = _split_component(comp(0, 3, y0=0, y1=10), ["a", "b"], mask)
    assert len(spans) == 2
    assert spans[0][0] == 0 and spans[-1][1] == 3


def test_cuts_stay_inside_the_component_and_in_order():
    rng = np.random.default_rng(7)
    mask = (rng.random((60, 400)) > 0.6).astype(np.uint8) * 255
    for parts in (2, 3, 4):
        spans = _split_component(comp(50, 350, y0=5, y1=55),
                                 list("abcd")[:parts], mask)
        assert len(spans) == parts
        assert spans[0][0] == 50 and spans[-1][1] == 350
        for (_a0, a1), (b0, _b1) in zip(spans, spans[1:]):
            assert a1 == b0


# ------------------------------------------------- character-level alignment

def _even_comps(text, width=40, gap=8, space=40):
    comps, cursor = [], 0
    for char in text:
        if char == " ":
            cursor += space
            continue
        comps.append(comp(cursor, cursor + width))
        cursor += width + gap
    return comps


def test_one_component_per_character_is_matched_one_to_one():
    text = "abc def"
    result = align_chars(_even_comps(text), text, em_px=80.0)
    assert len(result) == len("abcdef")
    assert all(kind == "comps" and len(payload) == 1
               for kind, payload in result)


def test_two_letters_in_one_blob_are_split_apart():
    text = "ab"
    comps = [comp(0, 80)]
    result = align_chars(comps, text, em_px=80.0)
    assert [kind for kind, _ in result] == ["split", "split"]
    assert [payload[1] for _kind, payload in result] == [0, 1]


def test_a_letter_in_two_blobs_is_merged():
    text = "m"
    comps = [comp(0, 30), comp(31, 64)]
    result = align_chars(comps, text, em_px=80.0)
    assert len(result) == 1
    kind, payload = result[0]
    assert kind == "comps" and len(payload) == 2


def test_a_character_is_never_carved_across_a_word_space():
    """
    A space is a pen lift, so two characters on either side of one cannot
    share a blob of ink. It was consistently the cheapest wrong answer
    available, and forbidding it outright was the single biggest improvement
    to alignment accuracy.
    """
    text = "a b"
    result = align_chars(_even_comps(text), text, em_px=80.0)
    for kind, payload in result:
        if kind == "split":
            shared, _index, total = payload
            assert total == 1, "a blob was shared across a space"


def test_alignment_of_nothing_is_empty_not_an_error():
    assert align_chars([], "abc", em_px=80.0) == []
    assert align_chars(_even_comps("abc"), "", em_px=80.0) == []


def test_every_character_is_accounted_for():
    """The DP has to produce exactly one entry per non-space character; one
    short and every label after it is wrong."""
    rng = np.random.default_rng(4)
    for _ in range(80):
        text = "".join(rng.choice(list("abcdefgmwil "), rng.integers(3, 18)))
        if not text.replace(" ", ""):
            continue
        comps = _even_comps(text, width=int(rng.integers(20, 60)))
        result = align_chars(comps, text, em_px=80.0)
        if result:
            assert len(result) == len(text.replace(" ", ""))


# -------------------------------------------------------------- measurements

def _instance(char, x, width, band=0):
    return {"char": char, "band": band, "advance_em": width,
            "actual_xy": [x, 100.0], "expected_xy": [x, 100.0],
            "bounds": [0.0, -0.5, width, 0.0], "on_baseline": True}


def test_letter_and_word_gaps_separate():
    """A space leaves no ink, so its width can only be inferred from the gaps
    around it."""
    instances, x = [], 0.0
    for index in range(40):
        instances.append(_instance("a", x, 0.4))
        x += 0.4 + (0.35 if index % 6 == 5 else 0.05)
    letter_gap, word_gap = _measure_gaps(instances, em_px=1.0)
    assert letter_gap == pytest.approx(0.05, abs=0.03)
    assert word_gap > letter_gap + 0.1


def test_too_few_gaps_falls_back_to_a_sensible_default():
    letter_gap, word_gap = _measure_gaps([_instance("a", 0, 0.4)], em_px=1.0)
    assert 0 < letter_gap < word_gap


def test_confidence_notices_crops_that_are_the_wrong_size():
    """
    A count of matched characters says nothing about whether each got the
    *right* ink: an alignment shifted one blob to the left still matches
    every character. Size is what shows.
    """
    good = [_instance("a", i * 0.5, 0.4) for i in range(40)]
    assert alignment_confidence(good, em_px=1.0)[0] > 0.95

    bad = good + [_instance("n", 30.0, 0.08) for _ in range(10)]
    confidence, suspect = alignment_confidence(bad, em_px=1.0)
    assert confidence < 0.9
    assert "n" in suspect


def test_confidence_of_nothing_is_zero_not_an_error():
    assert alignment_confidence([], em_px=1.0) == (0.0, [])


def test_margin_offsets_ignore_the_exercise_indentation():
    """
    The exercise indents four lines by four spaces. Measured against one
    nominal margin that reads as a large rightward drift the writer never
    made, so lines are compared against others with the same indent.
    """
    assignment = [[i] for i in range(len(INK_LINES))]
    instances = []
    for index, line in enumerate(INK_LINES):
        indent = len(line) - len(line.lstrip(" "))
        for k in range(8):
            inst = _instance("a", 100.0 + 30 * indent + k, 0.4, band=index)
            inst["expected_xy"] = [100.0 + k, 100.0]
            instances.append(inst)

    offsets = _margin_offsets(assignment, instances, em_px=50.0)
    assert len(offsets) == len(INK_LINES)
    assert max(abs(dx) for _band, dx in offsets) < 0.05


def test_margin_offsets_still_see_a_real_drift():
    assignment = [[i] for i in range(len(INK_LINES))]
    instances = []
    for index, _line in enumerate(INK_LINES):
        for k in range(8):
            inst = _instance("a", 100.0 + 12 * index + k, 0.4, band=index)
            inst["expected_xy"] = [100.0 + k, 100.0]
            instances.append(inst)

    offsets = _margin_offsets(assignment, instances, em_px=50.0)
    assert max(dx for _band, dx in offsets) > 1.0
    assert min(dx for _band, dx in offsets) < -1.0


def test_a_wrapped_continuation_line_carries_no_margin_information():
    """Where a continuation starts is decided by where the previous line ran
    out of room, not by the writer choosing a margin."""
    assignment = [[0, 1]] + [[i] for i in range(2, len(INK_LINES) + 1)]
    instances = []
    for band in range(len(INK_LINES) + 1):
        for k in range(8):
            inst = _instance("a", 400.0 if band == 1 else 100.0 + k, 0.4,
                             band=band)
            inst["expected_xy"] = [100.0 + k, 100.0]
            instances.append(inst)

    offsets = _margin_offsets(assignment, instances, em_px=50.0)
    assert 1 not in [band for band, _dx in offsets]

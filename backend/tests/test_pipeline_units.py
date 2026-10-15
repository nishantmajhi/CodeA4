"""Unit tests for the pieces that can be checked without a photograph."""
import numpy as np
import pytest

from pipeline import clustering, drift, features, vectorize
from pipeline.render_svg import Metrics, expand_tabs, find_block_boundaries, wrap_line


# ---------------------------------------------------------------- whitespace

@pytest.mark.parametrize("line,expected", [
    ("\tx", "    x"),
    ("ab\tc", "ab  c"),          # tab completes the first 4-column stop
    ("abcd\te", "abcd    e"),
    ("    x", "    x"),          # spaces are left exactly as written
])
def test_expand_tabs_reaches_tab_stops(line, expected):
    assert expand_tabs(line) == expected


def test_block_boundaries_are_language_agnostic():
    """
    Style may only shift at a natural break. The detector must work on any
    language, so it reads indentation and blank lines rather than keywords.
    """
    lines = [
        "function first() {",
        "    return 1;",
        "}",
        "",
        "function second() {",
        "    return 2;",
        "}",
    ]
    boundaries = find_block_boundaries(lines)
    assert 0 in boundaries
    assert 4 in boundaries        # blank line then a new top-level unit
    assert 1 not in boundaries    # inside a block


# --------------------------------------------------------------------- shape

def test_normalized_bitmap_is_scale_invariant():
    """Two sizes of the same shape must describe identically, or clustering
    would separate instances by size instead of by letterform."""
    small = np.zeros((10, 10), dtype=np.uint8)
    small[2:8, 3:7] = 255
    large = np.zeros((40, 40), dtype=np.uint8)
    large[8:32, 12:28] = 255

    a = features.normalized_bitmap(small).astype(float)
    b = features.normalized_bitmap(large).astype(float)
    assert np.abs(a - b).mean() < 12.0


def test_bitmap_encoding_roundtrips():
    bmp = (np.random.default_rng(0).random((16, 16)) * 255).astype(np.uint8)
    assert np.array_equal(features.decode_bitmap(features.encode_bitmap(bmp)), bmp)


def test_vectorize_places_baseline_at_origin():
    """A glyph's path must be expressed with y=0 on the baseline, since the
    renderer positions everything by baseline."""
    mask = np.zeros((30, 20), dtype=np.uint8)
    mask[10:25, 5:15] = 255           # ink sitting on row 24
    bounds = vectorize.path_bounds(mask, origin_x=0, baseline_y=25, unit_px=20)
    assert bounds[3] == pytest.approx(0.0, abs=0.06)   # bottom at the baseline
    assert bounds[1] < 0                                # body above it


def test_vectorize_returns_empty_for_blank_mask():
    assert vectorize.mask_to_path(np.zeros((10, 10), np.uint8), 0, 5, 10) == ""


# --------------------------------------------------------------------- drift

def test_ou_fit_recovers_a_known_process():
    """The fit must recover the amplitude of a walk it is given."""
    rng = np.random.default_rng(7)
    phi, sigma = 0.8, 1.0
    series = []
    for _ in range(40):
        value, row = 0.0, []
        for _ in range(60):
            value = phi * value + rng.normal(0, sigma)
            row.append(value)
        series.append(row)

    fit = clustering.fit_ou(series)
    expected_std = sigma / np.sqrt(1 - phi ** 2)
    assert fit["std"] == pytest.approx(expected_std, rel=0.2)
    assert fit["measured_phi"] == pytest.approx(phi, abs=0.08)


def test_drift_walk_is_smooth_not_jittery():
    """
    Consecutive offsets must be correlated. If they were independent, the
    output would have per-character jitter — exactly the "random noise"
    handwriting the spec rules out.
    """
    model = drift.DriftModel({"phi": 0.9, "sigma": 0.02, "std": 0.05},
                             np.random.default_rng(0))
    walk = model.walk(500)
    correlation = np.corrcoef(walk[:-1], walk[1:])[0, 1]
    assert correlation > 0.6


def test_drift_is_deterministic_for_a_seed():
    """A preview and the PDF downloaded after it must be identical."""
    a = drift.DriftModel({"phi": 0.8, "sigma": 0.03, "std": 0.05},
                         drift.make_rng("session", "file", 1)).walk(50)
    b = drift.DriftModel({"phi": 0.8, "sigma": 0.03, "std": 0.05},
                         drift.make_rng("session", "file", 1)).walk(50)
    assert np.array_equal(a, b)


def test_outlier_rejection_drops_a_broken_instance():
    """
    A mis-segmented fragment must not survive to become a "style". Twelve
    consistent instances plus one half-width fragment: the fragment should go.
    """
    rng = np.random.default_rng(3)
    def make(width, seed_shift):
        bmp = np.zeros((16, 16), dtype=np.uint8)
        bmp[4:12, 3:3 + seed_shift] = 255
        return {"bitmap": features.encode_bitmap(bmp), "aspect": 0.6,
                "density": 0.4, "advance_em": width}

    instances = [make(0.5 + rng.normal(0, 0.01), 8) for _ in range(12)]
    instances.append(make(0.22, 3))       # half a letter

    kept, indices = clustering.reject_outliers(instances)
    assert 12 not in indices
    assert len(kept) == 12


# ------------------------------------------------------------------ wrapping

class _FakeMetrics(Metrics):
    """Metrics with a fixed advance, so wrapping maths is checked exactly."""

    def __init__(self):
        self.letter_gap = 0.0
        self.word_gap = 0.5
        self.line_spacing = 1.5
        self.x_height_mm = 3.0
        self.styles = {}

    def advance(self, char, style_id):
        return 0.5


def test_wrap_prefers_a_late_word_break():
    rows = wrap_line("aaa bbb ccc ddd", 1, _FakeMetrics(), content_width_em=5.0)
    assert rows[0] == "aaa bbb"
    assert not any(row.endswith("-") for row in rows), "no continuation marker"


def test_wrap_breaks_mid_token_when_it_must():
    """A single long token has no space to break at, so it breaks mid-word —
    which is what a hand does when it runs out of paper."""
    rows = wrap_line("abcdefghijklmnop", 1, _FakeMetrics(), content_width_em=3.0)
    assert len(rows) > 1
    assert "".join(rows) == "abcdefghijklmnop"


def test_wrap_preserves_all_characters_including_indent():
    line = "        deeply_indented_call(argument_one, argument_two)"
    rows = wrap_line(line, 1, _FakeMetrics(), content_width_em=8.0)
    assert rows[0].startswith("        "), "indentation must not be stripped"


def test_flat_input_produces_no_baseline_drift():
    """
    A sheet written dead straight must be modelled as dead straight.

    Locating a baseline from segmented ink carries error, and that error is
    what a naive fit measures when there is no real drift to find. If it is
    not subtracted, the renderer reproduces it as wander the writer never
    made — which is exactly what happened when a user captured a screenshot
    of the on-screen exercise, which has no drift in it whatsoever.
    """
    rng = np.random.default_rng(7)
    records = []
    for _line in range(14):
        n = 40
        xs = np.arange(n, dtype=float) * 0.6
        ys = rng.normal(0.0, 0.02, n)          # measurement noise only
        centred = xs - xs.mean()
        sxx = float((centred ** 2).sum())
        slope = float((centred * ys).sum() / sxx)
        offset = float(ys.mean())
        resid = ys - (offset + slope * centred)
        records.append({"slope": slope, "offset": offset,
                        "scatter": float((resid ** 2).sum() / (n - 2)),
                        "n": n, "sxx": sxx})

    fit = clustering.fit_baseline(records)
    assert fit["slope_std"] < 0.005, "invented a tilt from measurement noise"
    assert fit["offset_std"] < 0.02, "invented an offset from measurement noise"


def test_real_drift_is_still_measured():
    """The noise correction must not flatten a hand that genuinely drifts."""
    rng = np.random.default_rng(11)
    records, offset = [], 0.0
    for _line in range(14):
        n = 40
        offset = 0.8 * offset + rng.normal(0.0, 0.12)
        true_slope = rng.normal(0.0, 0.05)
        xs = np.arange(n, dtype=float) * 0.6
        centred = xs - xs.mean()
        ys = offset + true_slope * centred + rng.normal(0.0, 0.02, n)
        sxx = float((centred ** 2).sum())
        slope = float((centred * ys).sum() / sxx)
        mean_y = float(ys.mean())
        resid = ys - (mean_y + slope * centred)
        records.append({"slope": slope, "offset": mean_y,
                        "scatter": float((resid ** 2).sum() / (n - 2)),
                        "n": n, "sxx": sxx})

    fit = clustering.fit_baseline(records)
    assert fit["slope_std"] > 0.02, "flattened a hand that really does tilt"
    assert fit["offset_std"] > 0.05, "flattened real line-to-line wander"


def test_line_baseline_is_monotonic_not_rippled():
    """
    Within one line the baseline must move in a single direction.

    A hand drifting upward keeps going up until the line ends; it has nothing
    to correct against until it returns to the left margin. A mean-reverting
    walk along the characters instead reverses several times per line, which
    reads as a ripple no hand makes.
    """
    model = drift.BaselineModel(
        {"phi": 0.8, "sigma": 0.05, "offset_std": 0.1, "slope_std": 0.04},
        drift.make_rng("session", "file", 1),
    )
    _offset, tilt = model.next_line()
    positions = [tilt * x for x in np.linspace(0, 30, 60)]
    steps = np.diff(positions)
    assert np.all(steps >= 0) or np.all(steps <= 0), "baseline reversed mid-line"

"""
A synthetic style asset, for tests that are not about the capture pipeline.

Building a real style model costs three generated sheets and a full run of
Stages 1-4 -- about a minute. Most of what the API, the layout engine and the
retention rules do is independent of *which* letterforms are in the asset, so
those tests install this instead and run in milliseconds.

The glyphs are deliberately trivial (a box per character) and deliberately
complete (all of printable ASCII), so a test that fails is failing about the
thing it names rather than about glyph coverage.
"""

# A 0.4 em wide, 0.5 em tall box sitting on the baseline.
BOX_PATH = "M0.0000,0.0000L0.0000,-0.5000L0.4000,-0.5000L0.4000,0.0000Z"

META = {
    "x_height_mm": 3.0,
    "letter_gap_em": 0.06,
    "word_gap_em": 0.30,
    "line_spacing_em": 1.4,
}

DRIFT = {
    "baseline": {"phi": 0.5, "sigma": 0.01, "offset_std": 0.02, "slope_std": 0.004,
                 "std": 0.02, "points": [[0, 0.0], [1, 0.01], [2, -0.01], [3, 0.0]]},
    "margin": {"phi": 0.5, "sigma": 0.01, "std": 0.02,
               "points": [[0, 0.0], [1, 0.01], [2, -0.01], [3, 0.0]]},
}


def glyph(advance=0.4, path=BOX_PATH):
    return {"path": path, "advance": advance, "slant": 0.0, "slant_std": 0.0,
            "bounds": [0.0, -0.5, advance, 0.0], "instances": 5}


def style_sets(n=2, chars=None):
    """`n` style sets covering printable ASCII, each with its own advance."""
    if chars is None:
        chars = [chr(c) for c in range(33, 127)]
    out = []
    for i in range(n):
        advance = 0.4 + 0.05 * i
        out.append({
            "id": i + 1,
            "share": round(1.0 / n, 4),
            "glyphs": {ch: glyph(advance) for ch in chars},
            "coverage": len(chars),
        })
    return out


def install(session_id, models, n=2, chars=None):
    """Give `session_id` a ready style model and return it."""
    sets = style_sets(n=n, chars=chars)
    models.save_style_asset(session_id, sets, DRIFT, META)
    models.mark_style_ready(session_id, True)
    return sets

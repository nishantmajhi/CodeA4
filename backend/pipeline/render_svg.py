"""
Spec Stage 6 — Flow-consistent rendering
Spec Stage 7 — Pagination & line wrapping

Takes parsed source lines and the writer's style assets and composes finished
A4 pages as SVG, with every character placed as a real vector glyph.

Three commitments from the spec drive the design:

* **Whitespace is preserved exactly** (§3). Indentation is never normalized
  and nothing is reformatted. Tabs are the one thing that must be interpreted
  rather than copied, because a tab is a request to reach a column, not a
  fixed amount of ink — see `expand_tabs`.
* **Style is consistent within a flow** (§4 Stage 6). One style set is used
  for a whole block and may only change at a natural break. Switching per
  character or per line would produce exactly the "random noise" handwriting
  the spec rejects in §2.
* **Long lines wrap the way a hand does** (§3). No continuation marker, no
  hyphen; the writing simply carries on at the margin of the next line.

Layout is measured in **em**, the writer's own unit from Stage 2, and only
converted to points when the SVG is emitted. That keeps the whole engine
independent of how large this particular person writes.
"""
import numpy as np

from pipeline import drift
from pipeline.glyph_assets import resolve_glyph

# A4 in PostScript points, which is what cairosvg and the PDF both use.
A4_WIDTH_PT = 595.276
A4_HEIGHT_PT = 841.890
MM_PT = 72.0 / 25.4

MARGIN_LEFT_MM = 18.0
MARGIN_RIGHT_MM = 14.0
MARGIN_TOP_MM = 20.0
MARGIN_BOTTOM_MM = 18.0

# Minimum spacing between baselines, in em. The measured spacing is used when
# it is larger; this floor stops a cramped exercise sheet from producing
# output whose ascenders and descenders collide.
MIN_LINE_SPACING_EM = 1.35

TAB_WIDTH = 4

# The x-height an em corresponds to (must match segmentation.X_HEIGHT_PER_EM).
X_HEIGHT_PER_EM = 0.5


# --------------------------------------------------------------------------
# Text preparation
# --------------------------------------------------------------------------

def expand_tabs(line, tab_width=TAB_WIDTH):
    """
    Replace tabs with the spaces needed to reach the next tab stop.

    A tab cannot be drawn. It means "advance to the next column", and columns
    only exist in a monospaced grid, so it has to be resolved into spaces
    before proportional layout. Doing it here, in character columns, keeps the
    *visual* indentation the author intended: a tab-indented block and a
    four-space-indented block line up, which is what the writer would have
    done by hand.
    """
    out = []
    column = 0
    for ch in line:
        if ch == "\t":
            advance = tab_width - (column % tab_width)
            out.append(" " * advance)
            column += advance
        else:
            out.append(ch)
            column += 1
    return "".join(out)


# --------------------------------------------------------------------------
# Style blocks (Stage 6)
# --------------------------------------------------------------------------

def find_block_boundaries(lines):
    """
    Find the natural breaks in a source file where the style may shift.

    Deliberately language-agnostic — the product accepts any source file, so
    keying off Python's `def` (as the scaffold's placeholder did) would simply
    fail on everything else. Structure is read from layout instead, which
    holds across C, Java, Go, JavaScript and the rest:

    * a blank line followed by content, which is how every language separates
      top-level units;
    * a return to zero indentation after indented lines, which is a block
      closing and the next one opening.

    Returns the set of line indices that begin a new block.
    """
    boundaries = {0}
    previous_indent = 0
    previous_blank = False

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            previous_blank = True
            continue

        indent = len(line) - len(line.lstrip(" "))
        if previous_blank and indent == 0:
            boundaries.add(i)
        elif indent == 0 and previous_indent > 0:
            boundaries.add(i)

        previous_indent = indent
        previous_blank = False

    return boundaries


def assign_block_styles(lines, style_ids, shares, rng):
    """
    Give every line a style id, constant within a block.

    Styles are drawn in proportion to how much of the writer's samples each
    one represents, so the output is dominated by the way they usually write,
    with their less common forms appearing occasionally — which is what the
    clustering actually measured.
    """
    boundaries = find_block_boundaries(lines)
    weights = np.array([max(s, 1e-6) for s in shares], dtype=float)
    weights = weights / weights.sum()

    assignments = []
    current = style_ids[0]
    for i in range(len(lines)):
        if i in boundaries:
            current = int(rng.choice(style_ids, p=weights))
        assignments.append(current)
    return assignments


# --------------------------------------------------------------------------
# Measurement and wrapping (Stage 7)
# --------------------------------------------------------------------------

class Metrics:
    """Everything needed to measure text in this writer's hand, in em."""

    def __init__(self, style_sets, meta):
        self.styles = {s["id"]: s for s in style_sets}
        self.letter_gap = float(meta.get("letter_gap_em", 0.06))
        self.word_gap = float(meta.get("word_gap_em", 0.30))
        self.line_spacing = max(float(meta.get("line_spacing_em", 1.4)),
                                MIN_LINE_SPACING_EM)
        self.x_height_mm = float(meta.get("x_height_mm", 3.0))

    def em_pt(self):
        """One em in points, from how large this person actually writes."""
        return (self.x_height_mm / X_HEIGHT_PER_EM) * MM_PT

    def glyphs(self, style_id):
        style = self.styles.get(style_id) or next(iter(self.styles.values()))
        return style["glyphs"]

    def advance(self, char, style_id):
        """How far the pen moves after writing `char`, in em."""
        if char == " ":
            return self.word_gap
        glyph, _resolved, _exact = resolve_glyph(self.glyphs(style_id), char)
        if glyph is None:
            return self.word_gap
        return glyph["advance"] + self.letter_gap

    def measure(self, text, style_id):
        return sum(self.advance(c, style_id) for c in text)


def wrap_line(line, style_id, metrics, content_width_em):
    """
    Break one source line into as many written lines as it needs.

    Wrapping prefers the last word break in the final quarter of the line, and
    otherwise breaks mid-word — which is what actually happens by hand when a
    long token runs out of room. No hyphen and no continuation marker is
    added, per spec §3: on paper there is nothing to mark, the writing just
    continues.
    """
    if not line:
        return [""]

    rows = []
    remaining = line
    while remaining:
        width = 0.0
        cut = len(remaining)
        for i, ch in enumerate(remaining):
            step = metrics.advance(ch, style_id)
            if width + step > content_width_em and i > 0:
                cut = i
                break
            width += step
        else:
            rows.append(remaining)
            break

        # Prefer a word break, but only a late one — breaking early would
        # leave a visibly short line that no one writing by hand would produce.
        window_start = max(1, int(cut * 0.75))
        space_at = remaining.rfind(" ", window_start, cut)
        if space_at > 0:
            rows.append(remaining[:space_at])
            remaining = remaining[space_at + 1:]
        else:
            rows.append(remaining[:cut])
            remaining = remaining[cut:]

    return rows or [""]


def layout_file(filename, lines, metrics, style_sets, seed_parts):
    """
    Wrap and paginate one file.

    Returns a list of page dicts, each
    ``{"filename_header", "page_num", "lines": [{"text", "style_id"}]}``.
    """
    style_ids = [s["id"] for s in style_sets]
    shares = [s["share"] for s in style_sets]
    rng = drift.make_rng(*seed_parts, "styles")

    expanded = [expand_tabs(line) for line in lines]
    line_styles = assign_block_styles(expanded, style_ids, shares, rng)

    content_width_em = _content_width_pt() / metrics.em_pt()

    written = []
    for text, style_id in zip(expanded, line_styles):
        for row in wrap_line(text, style_id, metrics, content_width_em):
            written.append({"text": row, "style_id": style_id})

    return _paginate(filename, written, metrics)


def _content_width_pt():
    return A4_WIDTH_PT - (MARGIN_LEFT_MM + MARGIN_RIGHT_MM) * MM_PT


def _content_height_pt():
    return A4_HEIGHT_PT - (MARGIN_TOP_MM + MARGIN_BOTTOM_MM) * MM_PT


def _paginate(filename, written, metrics):
    """Split written lines across pages, leaving room for the header."""
    line_pt = metrics.line_spacing * metrics.em_pt()
    header_pt = metrics.em_pt() * 2.6      # header line plus the rule under it

    first_page_rows = max(1, int((_content_height_pt() - header_pt) // line_pt))
    later_page_rows = max(1, int(_content_height_pt() // line_pt))

    pages = []
    index = 0
    while index < len(written) or not pages:
        rows = first_page_rows if not pages else later_page_rows
        chunk = written[index:index + rows]
        pages.append({
            "filename_header": filename if not pages else None,
            "page_num": len(pages) + 1,
            "lines": chunk,
        })
        index += rows
        if index >= len(written):
            break
    return pages


# --------------------------------------------------------------------------
# SVG composition (Stage 6)
# --------------------------------------------------------------------------

def compose_page_svg(page, metrics, drift_models, seed_parts):
    """
    Compose one page as SVG, placing every character's own vector glyph.

    Each glyph is positioned at a computed (x, y, rotation): x from the
    accumulated advances of this writer's letterforms, y from the line's
    baseline offset by the synthesized baseline drift, and rotation from the
    spread of slants measured across their samples. The line's whole baseline
    is additionally displaced by the margin drift for that row, so the left
    edge of the page wanders down the page the way theirs does.
    """
    em = metrics.em_pt()
    line_pt = metrics.line_spacing * em
    left = MARGIN_LEFT_MM * MM_PT
    top = MARGIN_TOP_MM * MM_PT

    rng = drift.make_rng(*seed_parts, page["page_num"], "glyphs")
    margin_walk = drift_models["margin"].walk(len(page["lines"]) + 1)

    parts = []
    y = top + em

    if page.get("filename_header"):
        parts.append(_render_run(page["filename_header"], page["lines"][0]["style_id"]
                                 if page["lines"] else 1,
                                 metrics, left, y, em, rng, drift_models, scale=1.15))
        rule_y = y + em * 0.55
        parts.append(
            f'<line x1="{left:.2f}" y1="{rule_y:.2f}" '
            f'x2="{A4_WIDTH_PT - MARGIN_RIGHT_MM * MM_PT:.2f}" y2="{rule_y:.2f}" '
            f'stroke="#111" stroke-width="0.8"/>'
        )
        y += em * 2.6

    for row_index, row in enumerate(page["lines"]):
        if row["text"].strip():
            offset, tilt = drift_models["baseline"].next_line()
            x = left + float(margin_walk[row_index]) * em
            parts.append(_render_run(row["text"], row["style_id"], metrics,
                                     x, y + offset * em, em, rng, drift_models,
                                     tilt=tilt))
        y += line_pt

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{A4_WIDTH_PT:.2f}" height="{A4_HEIGHT_PT:.2f}" '
        f'viewBox="0 0 {A4_WIDTH_PT:.2f} {A4_HEIGHT_PT:.2f}">'
        f'<rect width="100%" height="100%" fill="#ffffff"/>'
        f'{"".join(parts)}'
        f'</svg>'
    )


def _render_run(text, style_id, metrics, x, baseline_y, em, rng, drift_models,
                scale=1.0, tilt=0.0):
    """
    Place one run of text, returning an SVG group.

    `tilt` is the line's slope in em per em of horizontal travel. It is
    applied against distance actually covered, so the baseline moves steadily
    in one direction for the whole line — which is what a drifting hand does
    — rather than reversing several times along it.
    """
    glyphs = metrics.glyphs(style_id)
    x_start = x

    marks = []
    for char in text:
        if char == " ":
            x += metrics.advance(char, style_id) * em * scale
            continue

        glyph, _resolved, _exact = resolve_glyph(glyphs, char)
        if glyph is None:
            x += metrics.word_gap * em * scale
            continue

        dy = tilt * (x - x_start)
        size = em * scale
        # The glyph already carries the slant of the instance it was traced
        # from, so only the *spread* of slants is re-applied here — applying
        # the mean again would tilt every character twice.
        jitter = float(rng.normal(0.0, min(glyph.get("slant_std", 0.0), 4.0)))
        rotation = f' rotate({jitter:.2f})' if abs(jitter) > 0.05 else ""

        marks.append(
            f'<path d="{glyph["path"]}" '
            f'transform="translate({x:.2f},{baseline_y + dy:.2f}) '
            f'scale({size:.3f}){rotation}"/>'
        )
        x += (glyph["advance"] + metrics.letter_gap) * em * scale

    return f'<g fill="#111111" fill-rule="evenodd">{"".join(marks)}</g>'


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def render_file_to_pages(filename, lines, style_sets, drift_params, meta,
                         session_id="session", file_index=0):
    """
    Args:
        filename: original source filename, used for the handwritten header.
        lines: output of pipeline.code_parser.parse_source().
        style_sets: Stage 4 output.
        drift_params: the "drift" branch of Stage 3 output.
        meta: capture metadata (writing size and spacing).
        session_id: mixed into the render seed so output is reproducible.
        file_index: this file's position in the render. Also part of the
            seed, because a filename is not unique within one render — two
            files picked from different folders arrive under the same name —
            and two files with the same seed come out with byte-identical
            wander, which is the repeated wave this module exists to avoid.
            The position is stable for a given upload, so output stays
            reproducible.

    Returns:
        list[dict]: one per page, each with its composed SVG under "svg".
    """
    metrics = Metrics(style_sets, meta)
    seed_parts = (session_id, file_index, filename)

    pages = layout_file(filename, lines, metrics, style_sets, seed_parts)

    rng = drift.make_rng(*seed_parts, "drift")
    drift_models = {
        "baseline": drift.BaselineModel((drift_params or {}).get("baseline"), rng),
        "margin": drift.DriftModel((drift_params or {}).get("margin"), rng),
    }

    for page in pages:
        page["file"] = filename
        page["svg"] = compose_page_svg(page, metrics, drift_models, seed_parts)
    return pages


def sample_svg(style_sets, meta, style_id, text, pad_pt=6.0):
    """
    Render a short sample of one style set, tightly cropped.

    Used by the style cards so the UI shows the same layout engine the pages
    are produced with, rather than a second, approximate one. Drift is left
    out deliberately: the card is there to show the *letterforms*, and a
    wandering baseline in a one-line swatch reads as a rendering fault.
    """
    if not any(s["id"] == style_id for s in style_sets):
        return None

    metrics = Metrics(style_sets, meta)
    em = metrics.em_pt()
    glyphs = metrics.glyphs(style_id)

    marks = []
    x = pad_pt
    top = 0.0
    bottom = 0.0
    for char in text:
        if char == " ":
            x += metrics.word_gap * em
            continue
        glyph, _resolved, _exact = resolve_glyph(glyphs, char)
        if glyph is None:
            x += metrics.word_gap * em
            continue
        bounds = glyph.get("bounds") or [0, -1, 0, 0]
        top = min(top, bounds[1] * em)
        bottom = max(bottom, bounds[3] * em)
        marks.append(
            f'<path d="{glyph["path"]}" '
            f'transform="translate({x:.2f},0) scale({em:.3f})"/>'
        )
        x += (glyph["advance"] + metrics.letter_gap) * em

    width = x + pad_pt
    height = (bottom - top) + 2 * pad_pt
    baseline = -top + pad_pt

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width:.1f}" height="{height:.1f}" '
        f'viewBox="0 0 {width:.1f} {height:.1f}">'
        f'<g fill="#111111" fill-rule="evenodd" '
        f'transform="translate(0,{baseline:.2f})">{"".join(marks)}</g>'
        f'</svg>'
    )

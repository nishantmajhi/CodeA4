"""
Spec Stage 1 — Guided handwriting capture (comparison against known text)
Spec Stage 2 — Character instance extraction

The whole point of the guided exercise is that we know *exactly* what the
user wrote. That turns segmentation from an open recognition problem into an
alignment problem: we never have to guess which character a blob is, only
which blobs correspond to which known characters. Everything here leans on
that.

The chain is:

1. ``imaging.prepare_sheet`` — perspective-correct, flatten lighting,
   binarize and deskew the photo into canonical A4 page coordinates.
2. Find written line bands, and align them to the exercise's expected lines.
   One expected line may occupy two written lines, because the exercise
   deliberately contains a line too long to fit — the writer wraps it by hand
   (spec §4 Stage 1), and that wrap is itself a thing we want to observe.
3. Within each line, cut the ink into components and align that sequence to
   the expected characters with dynamic programming, allowing components to
   be merged (an 'i' and its dot) or split (two letters written touching).
4. Fit a nominal straight-line page grid — regular left margin, regular line
   spacing — and record each character's offset from where it *should* have
   landed. That offset is the raw material for Stage 3's drift model.
5. Emit one instance per character occurrence: its shape descriptor, its
   vector outline, its measured slant and its positional error.

Nothing is averaged or discarded at this stage, per spec §4 Stage 2.
"""
import cv2
import numpy as np

from pipeline import features, vectorize
from pipeline.exercise import INK_LINES
from pipeline.imaging import (
    CaptureError, find_line_bands, mm_to_px, prepare_sheet,
)

# Relative advance widths, in em, used only as a *prior* for alignment — the
# real widths are measured from the writer's own hand afterwards. These just
# need to get the ordering right (an 'i' is narrow, an 'm' is wide) so the
# alignment DP can tell which blob is which.
_WIDTH_PRIOR = {
    " ": 0.30,
    # Lowercase. An x-height letter is roughly as wide as it is tall, and
    # x-height is 0.5 em by the convention above.
    "a": 0.48, "b": 0.48, "c": 0.44, "d": 0.48, "e": 0.44, "f": 0.30,
    "g": 0.48, "h": 0.48, "i": 0.20, "j": 0.22, "k": 0.45, "l": 0.20,
    "m": 0.80, "n": 0.48, "o": 0.48, "p": 0.48, "q": 0.48, "r": 0.32,
    "s": 0.38, "t": 0.30, "u": 0.48, "v": 0.45, "w": 0.70, "x": 0.45,
    "y": 0.45, "z": 0.42,
    # Uppercase. Capitals are consistently wider than lowercase in every
    # hand; leaving them on the generic default made every capital look
    # twice as wide as it is to the alignment DP.
    "A": 0.62, "B": 0.58, "C": 0.58, "D": 0.60, "E": 0.52, "F": 0.50,
    "G": 0.60, "H": 0.60, "I": 0.22, "J": 0.40, "K": 0.58, "L": 0.48,
    "M": 0.80, "N": 0.62, "O": 0.64, "P": 0.54, "Q": 0.64, "R": 0.58,
    "S": 0.50, "T": 0.55, "U": 0.60, "V": 0.58, "W": 0.85, "X": 0.58,
    "Y": 0.55, "Z": 0.52,
    # Digits are near-tabular in most hands.
    "0": 0.50, "1": 0.34, "2": 0.48, "3": 0.48, "4": 0.50, "5": 0.48,
    "6": 0.48, "7": 0.46, "8": 0.48, "9": 0.48,
    # Punctuation and symbols.
    "|": 0.18, "!": 0.20, ".": 0.20, ",": 0.20,
    "'": 0.18, "`": 0.20, ":": 0.20, ";": 0.20, "-": 0.30, "(": 0.26,
    ")": 0.26, "[": 0.26, "]": 0.26, "{": 0.30, "}": 0.30, "/": 0.30,
    "\\": 0.30,
    "%": 0.75, "@": 0.85, "&": 0.65, "#": 0.60, "~": 0.55, "^": 0.45,
    "_": 0.55, "=": 0.55, "+": 0.55, "<": 0.55, ">": 0.55, "*": 0.40,
    "\"": 0.32, "?": 0.45, "$": 0.50,
}
_DEFAULT_WIDTH = 0.50

# Bounds on the width-prior calibration (see calibrate_widths). Wide enough to
# absorb a genuinely condensed or sprawling hand — and a monospaced one, whose
# beta is near zero because none of the prior's proportions apply — tight
# enough that a badly misread sheet cannot rescale the prior into nonsense.
MAX_WIDTH_SCALE = 2.60
# The prior is an advance and the fit is against ink, so the constant term is
# genuinely a little negative for most hands.
MIN_WIDTH_OFFSET = -0.20
# Below this many unwrapped lines there is not enough to fit two parameters.
MIN_WIDTH_FIT_LINES = 6

# Characters whose ink sits on the baseline (no descender), used for fitting
# where the baseline actually is.
_BASELINE_CHARS = set("abcdehiklmnorstuvwxzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
# Characters whose height *is* the x-height, used to measure writing size.
_XHEIGHT_CHARS = set("acemnorsuvwxz")

# An em is defined as twice the writer's x-height. The exact constant is
# arbitrary but must be the same everywhere: glyph paths, advance widths and
# render sizing are all expressed in these units.
X_HEIGHT_PER_EM = 0.5

# The most written lines a single expected line may occupy. The exercise's
# over-long line wraps once at a small hand and twice at a large one.
MAX_BANDS_PER_LINE = 3

# Costs steering the character-alignment DP. Merges and splits are real but
# should be a last resort: the DP can otherwise trade a spurious merge against
# a spurious split, keeping the character count right while shifting every
# label in between onto the wrong ink.
MERGE_PENALTY = 0.45
SPLIT_PENALTY = 0.50
# Pooling several fragments and re-cutting them into several characters is
# the most permissive move the DP has, so it is charged more than either a
# merge or a split and capped at three components to three characters.
RECUT_PENALTY = 0.70
RECUT_STEP = 0.12
MAX_RECUT = 3
# Whether a wide gap coincides with an expected space is a strong, reliable
# cue, since spaces are the one thing the writer marks unambiguously.
SPACE_CUE_PENALTY = 0.30
# How strongly to insist that a line which claims more than one written line
# actually ran into the right margin on all but its last.
WRAP_MARGIN_PENALTY = 0.60


def width_prior(ch, widths=None):
    """
    Expected advance width for a character, in em.

    `widths` is an optional measured table that overrides the prior. The prior
    only has to get the ordering roughly right; the measured table is what
    makes the cuts between touching letters land in the right place.
    """
    if widths and ch in widths:
        return widths[ch]
    return _WIDTH_PRIOR.get(ch, _DEFAULT_WIDTH)


def _width_equations(per_band, assignment, em_px):
    """
    One equation per written line: (character count, summed prior, measured
    ink), in em.

    The evidence used to be characters from lines whose component count
    already equalled their character count, which sounds like the cleanest
    possible sample and is in fact almost never available. A single merge or
    split anywhere in a line disqualifies the whole line, and on real ink
    there is always one: an 'i' and its dot, a ':' in two pieces, two letters
    written touching. One or two of the sixteen lines on a sheet qualify, and
    thirty samples were required, so the fit never ran — least of all on a
    hand the prior fits badly, which produces *fewer* clean lines, not more.

    A whole line is usable without matching anything inside it. We know which
    characters are on it, and the total inked width of its components is
    measurable whether or not the letters within it were cut correctly. That
    gives a linear equation per line in the same two unknowns, from every line
    on the sheet rather than from the rare one.

    Lines that wrap are skipped, because where the writer broke them decides
    which characters landed on which band, and that is not known yet.
    """
    rows = []
    for line_idx, band_indices in enumerate(assignment):
        if line_idx >= len(INK_LINES) or len(band_indices) != 1:
            continue
        comps = per_band[band_indices[0]]
        chars = [c for c in INK_LINES[line_idx] if c != " "]
        if not comps or not chars:
            continue
        ink = sum(c["x1"] - c["x0"] for c in comps) / em_px
        rows.append((len(chars), sum(width_prior(c) for c in chars), ink))
    return rows


def _solve_width_fit(rows):
    """Least squares for (alpha, beta) in ``ink = alpha * n + beta * prior``."""
    design = np.array([[n, p] for n, p, _ in rows], dtype=float)
    measured = np.array([ink for _n, _p, ink in rows], dtype=float)
    solution, *_ = np.linalg.lstsq(design, measured, rcond=None)
    return float(solution[0]), float(solution[1]), design, measured


def calibrate_widths(per_band, assignment, em_px):
    """
    Fit the generic width prior onto this writer's actual hand.

    Two things can be wrong with the prior, and both wreck the alignment DP,
    which scores purely on width:

    * **Scale.** How wide a hand sets its letters for a given x-height varies
      enormously. A systematic error of even 1.5x makes merging two letters
      into one cost about the same as matching them correctly, and the
      alignment starts trading spurious merges against spurious splits.
    * **Shape.** The prior assumes proportional writing, where 'W' is four
      times the width of 'l'. Some hands — and any monospaced source — are
      far flatter than that, and rescaling cannot fix a wrong *ratio*: it
      leaves every wide capital still over-predicted and every narrow letter
      under-predicted.

    So rather than one scale factor, fit ``ink = alpha + beta * prior`` over
    the sheet's unwrapped lines. beta carries the scale, alpha carries how
    uniform the hand is — a fully monospaced sheet fits beta near zero and
    alpha at its constant advance, a sprawling proportional hand fits alpha
    near zero and beta at its scale. Both come out of the same fit.

    alpha is allowed to be slightly negative: the prior is an *advance*, which
    includes the space either side of the ink, while what is measured here is
    the ink, and the difference is a small constant per character.

    Returns a widths table, falling back to the plain prior when there is not
    enough evidence to fit anything.
    """
    all_chars = {c for line in INK_LINES for c in line if c != " "}
    plain = ({ch: width_prior(ch) for ch in all_chars}, (0.0, 1.0))

    rows = _width_equations(per_band, assignment, em_px)
    if len(rows) < MIN_WIDTH_FIT_LINES:
        return plain

    alpha, beta, design, measured = _solve_width_fit(rows)

    # One round of outlier rejection: a line read badly enough to throw its
    # ink total out is one line, and there are only a dozen or so of them.
    residual = np.abs(design @ np.array([alpha, beta]) - measured)
    limit = 2.5 * float(np.median(residual)) + 1e-9
    kept = [row for row, r in zip(rows, residual) if r <= limit]
    if MIN_WIDTH_FIT_LINES <= len(kept) < len(rows):
        alpha, beta, _design, _measured = _solve_width_fit(kept)

    beta = float(np.clip(beta, 0.0, MAX_WIDTH_SCALE))
    alpha = float(np.clip(alpha, MIN_WIDTH_OFFSET, 1.0))

    # A degenerate fit — both terms collapsed — means the evidence was junk.
    if alpha + beta * _DEFAULT_WIDTH < 0.05:
        return plain

    widths = {ch: max(alpha + beta * width_prior(ch), 0.04) for ch in all_chars}
    return widths, (round(alpha, 4), round(beta, 4))


# --------------------------------------------------------------------------
# Line-level alignment
# --------------------------------------------------------------------------

def _band_right_edges(mask, bands):
    """Rightmost inked column of each band."""
    edges = []
    for top, bottom in bands:
        cols = np.flatnonzero(mask[top:bottom + 1].any(axis=0))
        edges.append(float(cols[-1]) if cols.size else 0.0)
    return edges


def _band_ink_widths(mask, bands):
    """Total inked width of each band, used to match bands to expected lines."""
    widths = []
    for top, bottom in bands:
        cols = mask[top:bottom + 1].any(axis=0)
        widths.append(float(cols.sum()))
    return widths


def align_lines(mask, bands):
    """
    Map expected exercise lines onto detected written bands.

    Each expected line consumes between one and ``MAX_BANDS_PER_LINE``
    consecutive bands — more than one when the writer ran past the margin and
    wrapped. The exercise's deliberately over-long line can easily take three
    written lines at a large hand, so the DP has to allow for that; capping it
    at two silently misaligns every line after it.

    Solved as a small DP over (expected line, band) minimizing the mismatch
    between how much ink a line should contain and how much the assigned
    bands do contain.

    Returns a list, one entry per expected line, of the band indices it owns.
    """
    n_lines, n_bands = len(INK_LINES), len(bands)
    if n_bands < n_lines:
        raise CaptureError(
            f"Only {n_bands} lines of writing were found, but the exercise has "
            f"{n_lines}. Check the whole sheet is in the photo and that no "
            "lines are cut off at the edges."
        )
    if n_bands > MAX_BANDS_PER_LINE * n_lines:
        raise CaptureError(
            "This photo has far more lines of writing than the exercise. Make "
            "sure only the exercise sheet is in frame, on a plain background."
        )

    widths = _band_ink_widths(mask, bands)
    rights = _band_right_edges(mask, bands)
    page_right = max(rights) if rights else 1.0

    prior = [sum(width_prior(c) for c in line) for line in INK_LINES]
    total_prior = sum(prior) or 1.0
    total_width = sum(widths) or 1.0
    expected_width = [p / total_prior * total_width for p in prior]

    INF = float("inf")
    dp = np.full((n_lines + 1, n_bands + 1), INF)
    back = np.zeros((n_lines + 1, n_bands + 1), dtype=int)
    dp[0][0] = 0.0

    for i in range(1, n_lines + 1):
        for j in range(1, n_bands + 1):
            for take in range(1, MAX_BANDS_PER_LINE + 1):
                if j - take < 0:
                    continue
                prev = dp[i - 1][j - take]
                if prev == INF:
                    continue
                got = sum(widths[j - take:j])
                cost = abs(got - expected_width[i - 1]) / total_width

                # How much ink a line holds is a weak cue on its own — a short
                # wrapped tail looks much like a short line. The decisive
                # evidence is *why* a line wrapped: it ran into the right
                # margin. So every band a line claims except its last must
                # itself reach the margin. Without this, a six-character tail
                # gets read as the next line of the exercise and every line
                # after it is off by one.
                for band in range(j - take, j - 1):
                    reached = rights[band] / page_right
                    cost += WRAP_MARGIN_PENALTY * max(0.0, 1.0 - reached)

                cost += 0.06 * (take - 1)
                if prev + cost < dp[i][j]:
                    dp[i][j] = prev + cost
                    back[i][j] = take

    if dp[n_lines][n_bands] == INF:
        raise CaptureError(
            "The lines on this sheet could not be matched to the exercise. "
            "Please photograph the sheet again, square-on and in full."
        )

    assignment = []
    j = n_bands
    for i in range(n_lines, 0, -1):
        take = back[i][j]
        assignment.append(list(range(j - take, j)))
        j -= take
    assignment.reverse()
    return assignment


# --------------------------------------------------------------------------
# Component extraction and character alignment
# --------------------------------------------------------------------------

def _components(mask, band, min_area):
    """Connected components within a band, left to right."""
    top, bottom = band
    sub = mask[top:bottom + 1]
    n, labels, stats, _ = cv2.connectedComponentsWithStats((sub > 0).astype(np.uint8), 8)
    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        comps.append({
            "x0": int(x), "x1": int(x + w), "y0": int(top + y), "y1": int(top + y + h),
            "area": int(area),
        })
    comps.sort(key=lambda c: c["x0"])
    return comps


def _merge_stacked(comps, overlap_frac=0.55):
    """
    Pre-merge components that sit directly above/below each other.

    Dots on 'i' and 'j', the two bars of '=', the pieces of ':' and '"' are
    separate connected components but one character. Merging them here keeps
    the character-alignment DP from having to explain them as spurious extra
    blobs.
    """
    merged = []
    for comp in comps:
        placed = False
        for m in merged:
            lo = max(m["x0"], comp["x0"])
            hi = min(m["x1"], comp["x1"])
            overlap = hi - lo
            narrow = min(m["x1"] - m["x0"], comp["x1"] - comp["x0"])
            if narrow > 0 and overlap > overlap_frac * narrow:
                m["x0"] = min(m["x0"], comp["x0"])
                m["x1"] = max(m["x1"], comp["x1"])
                m["y0"] = min(m["y0"], comp["y0"])
                m["y1"] = max(m["y1"], comp["y1"])
                m["area"] += comp["area"]
                placed = True
                break
        if not placed:
            merged.append(dict(comp))
    merged.sort(key=lambda c: c["x0"])
    return merged


def _absorb_fragments(comps, min_frac=0.40, area_frac=0.28):
    """
    Absorb slivers of a letter that broke apart back into their neighbour.

    A thin diagonal — the middle of a 'W', the join of an 'N' — can come back
    from the binarizer as its own tiny component sitting flush against the
    stroke it belongs to. Left alone it is one more blob for the alignment DP
    to explain, and because the DP matches on width it will cheerfully hand
    that sliver a whole character, pushing every label after it onto the wrong
    ink.

    A fragment is distinguished from real punctuation by *contact*: a full
    stop or a comma has clear air either side of it, whereas a broken-off
    piece of a letter touches or overlaps the box it came from. Only touching
    slivers are absorbed, so punctuation survives untouched.
    """
    if len(comps) < 2:
        return [dict(c) for c in comps]

    widths = np.array([c["x1"] - c["x0"] for c in comps], dtype=float)
    areas = np.array([c["area"] for c in comps], dtype=float)
    median_w = float(np.median(widths))
    median_a = float(np.median(areas))
    if median_w <= 0:
        return [dict(c) for c in comps]

    out = [dict(c) for c in comps]
    changed = True
    while changed and len(out) > 1:
        changed = False
        for i, comp in enumerate(out):
            width = comp["x1"] - comp["x0"]
            if width >= min_frac * median_w or comp["area"] >= area_frac * median_a:
                continue

            # Gap to each neighbour; negative means the boxes overlap.
            candidates = []
            if i > 0:
                candidates.append((comp["x0"] - out[i - 1]["x1"], i - 1))
            if i + 1 < len(out):
                candidates.append((out[i + 1]["x0"] - comp["x1"], i + 1))
            candidates = [(gap, j) for gap, j in candidates if gap <= 0]
            if not candidates:
                continue

            _gap, j = min(candidates)
            host = out[j]
            host["x0"] = min(host["x0"], comp["x0"])
            host["x1"] = max(host["x1"], comp["x1"])
            host["y0"] = min(host["y0"], comp["y0"])
            host["y1"] = max(host["y1"], comp["y1"])
            host["area"] += comp["area"]
            out.pop(i)
            changed = True
            break

    out.sort(key=lambda c: c["x0"])
    return out


def _split_component(comp, chars, mask, widths=None):
    """
    Cut one component into len(chars) pieces because characters were written
    touching. Cut points are the deepest vertical-projection minima near where
    the expected character widths say the boundary should fall — a true
    minimum in ink density is where the pen was thinnest.
    """
    x0, x1 = comp["x0"], comp["x1"]
    n_parts = len(chars)
    sub = mask[comp["y0"]:comp["y1"] + 1, x0:x1]
    span = x1 - x0
    if sub.size == 0 or span < n_parts * 2:
        edges = np.linspace(x0, x1, n_parts + 1).astype(int)
        return [(int(edges[i]), int(edges[i + 1])) for i in range(n_parts)]

    profile = (sub > 0).sum(axis=0).astype(np.float64)
    parts = np.array([width_prior(c, widths) for c in chars], dtype=np.float64)
    fracs = np.cumsum(parts / parts.sum())[:-1]

    # Search a generous window around each expected boundary and score
    # candidate columns by how little ink they cut through, with only a soft
    # pull toward the expected position. Weighting the other way round — a
    # narrow window centred on the prior — is what cuts a round letter in
    # half when the width prior is slightly off for this hand.
    cuts = []
    part_width = span / n_parts
    window_half = max(2, int(0.45 * part_width))
    depth_scale = max(profile.max(), 1.0)
    for frac in fracs:
        target = int(round(frac * span))
        lo, hi = max(1, target - window_half), min(profile.size - 1, target + window_half)
        if hi <= lo:
            cuts.append(x0 + target)
            continue
        cols = np.arange(lo, hi)
        score = profile[lo:hi] / depth_scale + 0.35 * np.abs(cols - target) / window_half
        cuts.append(x0 + int(cols[int(np.argmin(score))]))

    edges = [x0] + sorted(cuts) + [x1]
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_parts)]


def align_chars(comps, text, em_px, widths=None):
    """
    Align a left-to-right component sequence to the known characters of a line.

    Returns a list, one entry per non-space character, of ``(kind, payload)``
    where kind is "comps" (one or more components form this character) or
    "split" (this character shares a component with its neighbours, given as
    ``(component, index, count)``).

    The DP allows 1:1, many:1 (merge) and 1:many (split) matches. Cost is the
    mismatch between measured ink width and the expected width prior, plus
    small penalties that keep merges and splits rare — they should only be
    chosen when the widths genuinely demand it.
    """
    chars = [c for c in text if c != " "]
    # Whether a space precedes each non-space character, used as a soft cue:
    # a wide gap should coincide with a space.
    space_before = []
    seen_space = False
    for c in text:
        if c == " ":
            seen_space = True
        else:
            space_before.append(seen_space)
            seen_space = False

    m, n = len(comps), len(chars)
    if m == 0 or n == 0:
        return []

    # Where the gap between two blobs is wide enough to be a word space.
    # A fixed threshold in em mis-reads a hand that sets its words tightly,
    # so it is taken from the line itself: we know exactly how many spaces
    # this line contains, so the widest that many gaps are the spaces, and
    # the threshold sits between them and the widest gap that is not.
    gaps = []
    for a_comp, b_comp in zip(comps, comps[1:]):
        if a_comp.get("band_index") != b_comp.get("band_index"):
            continue          # a wrap, not a space
        gaps.append((b_comp["x0"] - a_comp["x1"]) / em_px)
    n_spaces = sum(1 for flag in space_before[1:] if flag)
    space_threshold = 0.28
    if gaps and 0 < n_spaces < len(gaps):
        ordered = sorted(gaps, reverse=True)
        space_threshold = 0.5 * (ordered[n_spaces - 1] + ordered[n_spaces])
    space_threshold = float(np.clip(space_threshold, 0.10, 0.80))

    INF = float("inf")
    dp = np.full((m + 1, n + 1), INF)
    back = {}
    dp[0][0] = 0.0

    max_merge, max_split = 3, 3
    for i in range(m + 1):
        for j in range(n + 1):
            if dp[i][j] == INF:
                continue
            for a in range(1, max_merge + 1):          # components consumed
                for b in range(1, max_split + 1):      # characters consumed
                    if a > 1 and b > 1 and (a > MAX_RECUT or b > MAX_RECUT):
                        continue
                    if i + a > m or j + b > n:
                        continue
                    # A word space is a pen lift, so it cannot fall inside a
                    # single blob of ink. Carving two characters out of one
                    # component across a space is therefore not merely
                    # unlikely but impossible, and forbidding it outright
                    # removes the cheapest way for the alignment to go wrong.
                    if b > 1 and any(space_before[j + 1:j + b]):
                        continue

                    group = comps[i:i + a]
                    got = (group[-1]["x1"] - group[0]["x0"]) / em_px
                    want = sum(width_prior(c, widths) for c in chars[j:j + b])
                    cost = abs(got - want)

                    if a > 1 and b > 1:
                        # Several fragments spanning several characters: pool
                        # the ink and re-cut it. This is the only move that
                        # can recover letters written touching *and* broken
                        # apart by the binarizer, where neither a plain merge
                        # nor a plain split lands the boundaries correctly.
                        # It subsumes both, so it is charged once and dearly
                        # rather than charged as a merge and a split at once.
                        cost += RECUT_PENALTY + RECUT_STEP * (a + b - 2)
                    elif a > 1:
                        # Merging is right for stacked marks, wrong for two
                        # separate letters — charge for the horizontal gap.
                        for p, q in zip(group, group[1:]):
                            cost += MERGE_PENALTY + max(0.0, (q["x0"] - p["x1"]) / em_px)
                    elif b > 1:
                        cost += SPLIT_PENALTY * (b - 1)

                    if i > 0:
                        gap = (comps[i]["x0"] - comps[i - 1]["x1"]) / em_px
                        if (gap > space_threshold) != space_before[j]:
                            cost += SPACE_CUE_PENALTY
                    if a > 1:
                        # Likewise, a merged group must not straddle a space.
                        for p, q in zip(group, group[1:]):
                            if (q["x0"] - p["x1"]) / em_px > space_threshold:
                                cost += SPACE_CUE_PENALTY

                    if dp[i][j] + cost < dp[i + a][j + b]:
                        dp[i + a][j + b] = dp[i][j] + cost
                        back[(i + a, j + b)] = (i, j, a, b)

    if dp[m][n] == INF:
        return []

    steps = []
    node = (m, n)
    while node != (0, 0):
        if node not in back:
            return []
        i, j, a, b = back[node]
        steps.append((i, j, a, b))
        node = (i, j)
    steps.reverse()

    out = []
    for i, j, a, b in steps:
        if b == 1:
            out.append(("comps", comps[i:i + a]))
            continue
        group = comps[i:i + a]
        if a == 1:
            shared = group[0]
        else:
            # Pool the fragments into one box and let the usual split logic
            # cut it, so downstream code never has to know the difference.
            shared = {
                **group[0],
                "x0": min(c["x0"] for c in group),
                "x1": max(c["x1"] for c in group),
                "y0": min(c["y0"] for c in group),
                "y1": max(c["y1"] for c in group),
                "area": sum(c["area"] for c in group),
            }
        for k in range(b):
            out.append(("split", (shared, k, b)))
    return out


# --------------------------------------------------------------------------
# Page geometry
# --------------------------------------------------------------------------

def _robust_line_fit(xs, ys):
    """Least-squares line with one round of outlier rejection."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if len(xs) < 2:
        return 0.0, float(ys.mean()) if len(ys) else 0.0
    slope, intercept = np.polyfit(xs, ys, 1)
    resid = ys - (slope * xs + intercept)
    mad = float(np.median(np.abs(resid - np.median(resid)))) or 1.0
    keep = np.abs(resid) < 3.0 * mad
    if 2 <= keep.sum() < len(xs):
        slope, intercept = np.polyfit(xs[keep], ys[keep], 1)
    return float(slope), float(intercept)


def _band_geometry(matched, n_bands):
    """Per written line: fitted baseline, x-height, left ink edge."""
    geom = {}
    for bi in range(n_bands):
        entries = [e for e in matched if e["band_index"] == bi]
        if not entries:
            geom[bi] = None
            continue

        base_pts = [(0.5 * (e["box"][0] + e["box"][2]), e["box"][3])
                    for e in entries if e["char"] in _BASELINE_CHARS]
        if len(base_pts) >= 2:
            slope, intercept = _robust_line_fit([p[0] for p in base_pts],
                                                [p[1] for p in base_pts])
        else:
            slope = 0.0
            intercept = float(np.median([e["box"][3] for e in entries]))

        xh = [e["box"][3] - e["box"][1] for e in entries if e["char"] in _XHEIGHT_CHARS]
        geom[bi] = {
            "slope": slope,
            "intercept": intercept,
            "x_height": float(np.median(xh)) if xh else None,
            "left": min(e["box"][0] for e in entries),
        }
    return geom


def _baseline_at(g, x):
    return g["slope"] * x + g["intercept"]


def _page_grid(geom):
    """
    Derive the *ideal* page the writer was aiming at: one regular left margin
    and one regular line spacing. Actual positions are compared against this
    grid — that is what "expected position on a straight line" means in spec
    §4 Stage 1.
    """
    live = [(bi, g) for bi, g in sorted(geom.items()) if g]
    if not live:
        raise CaptureError("No usable lines were found on this sheet.")

    x_heights = [g["x_height"] for _, g in live if g["x_height"]]
    x_height = float(np.median(x_heights)) if x_heights else mm_to_px(3.0)
    em_px = max(x_height / X_HEIGHT_PER_EM, 1.0)

    nominal_left = float(np.median([g["left"] for _, g in live]))
    baselines = [(bi, _baseline_at(g, nominal_left)) for bi, g in live]

    # Map each inked band onto its row on the page. Band indices count only
    # the bands that have ink in them, so a blank line in the exercise makes
    # two consecutive bands sit two rows apart while their indices differ by
    # one. Regressing position against the index therefore averages single
    # and double gaps together and over-estimates the line pitch — which then
    # shows up as every line being progressively further from where it was
    # "expected", i.e. as baseline drift the writer never made.
    #
    # Gaps between bands are near-integer multiples of one true pitch, so the
    # pitch is recovered from the gaps themselves and the rows counted off in
    # units of it.
    rows = {}
    if len(baselines) >= 2:
        ys_all = np.array([b[1] for b in baselines], dtype=float)
        gaps = np.diff(ys_all)
        gaps = gaps[gaps > 1.0]
        if gaps.size:
            base = float(np.percentile(gaps, 25))
            multiples = np.maximum(np.round(gaps / max(base, 1e-6)), 1.0)
            pitch = float(np.median(gaps / multiples))
        else:
            pitch = mm_to_px(8.0)

        row = 0.0
        rows[baselines[0][0]] = 0.0
        for (bi, y), (_prev_bi, prev_y) in zip(baselines[1:], baselines[:-1]):
            row += max(round((y - prev_y) / max(pitch, 1e-6)), 1)
            rows[bi] = row

        idx = np.array([rows[b[0]] for b in baselines], dtype=float)
        ys = ys_all
        spacing, offset = np.polyfit(idx, ys, 1)
    else:
        rows[baselines[0][0]] = 0.0
        spacing, offset = mm_to_px(8.0), baselines[0][1]

    return em_px, float(spacing), nominal_left, float(offset), rows


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def segment_photo(photo_path):
    """
    Args:
        photo_path: path to the uploaded photo for one capture round.

    Returns:
        dict with:
            "instances": one entry per character occurrence, each carrying the
                character, its expected and actual position, its shape
                descriptor, its vector outline and its measured slant.
            "meta": per-sheet measurements (em size, line spacing, margin)
                that Stage 3 and Stage 4 need.

    Raises:
        CaptureError: with a message written for the user, when the sheet
            cannot be read or does not match the exercise.
    """
    mask, _gray = prepare_sheet(photo_path)
    bands = find_line_bands(mask)
    if not bands:
        raise CaptureError(
            "No lines of writing were found on this sheet. Check the photo is "
            "in focus and the page fills the frame."
        )

    assignment = align_lines(mask, bands)

    min_area = int(mm_to_px(0.35) ** 2)
    per_band = []
    for band in bands:
        comps = _merge_stacked(_components(mask, band, min_area))
        per_band.append(_absorb_fragments(comps))

    heights = [c["y1"] - c["y0"] for comps in per_band for c in comps]
    if not heights:
        raise CaptureError(
            "No writing could be isolated on this sheet. Try again with more "
            "even lighting and a darker pen."
        )
    # Rough em for the alignment DP. Median component height is dominated by
    # x-height letters, so this lands close enough to drive width costs.
    em_px_guess = float(np.median(heights)) / X_HEIGHT_PER_EM * 0.8

    # Two passes. The first uses the generic width prior, which is only good
    # enough to get the alignment right. Where two letters were written
    # touching, though, the cut between them is placed by those widths, and a
    # prior that is slightly wrong for this hand puts the cut inside a letter
    # — producing a 'c' crop that has the neighbouring stem attached and looks
    # like a 'd'. So the first pass's cleanly matched characters are measured
    # to build this writer's own width table, and the alignment and cuts are
    # then redone with it.
    # Before either pass, bring the generic prior onto this hand's own scale.
    # Without this the first pass is scored against widths that can be out by
    # a factor of two, and it is the first pass that decides which ink each
    # character gets — so the second pass only measures the first one's
    # mistakes back to itself.
    calibrated, width_fit = calibrate_widths(per_band, assignment, em_px_guess)

    matched = _match_characters(per_band, assignment, mask, em_px_guess, calibrated)
    measured = _measure_widths(matched, em_px_guess)
    # Measured widths win where we have them; the calibrated prior covers the
    # characters that appear too rarely to measure.
    combined = {**calibrated, **measured}
    matched = _match_characters(per_band, assignment, mask, em_px_guess, combined)

    geom = _band_geometry(matched, len(bands))
    em_px, line_spacing, nominal_left, baseline_offset, band_rows = _page_grid(geom)

    instances = _build_instances(matched, mask, geom, em_px, nominal_left,
                                 line_spacing, baseline_offset, combined, band_rows)

    expected_total = sum(len(line.replace(" ", "")) for line in INK_LINES)
    if len(instances) < 0.5 * expected_total:
        raise CaptureError(
            "Too little of the exercise could be read from this sheet. Retake "
            "the photo with the whole page in frame and in focus."
        )

    # Margin drift is a property of *written lines*, not of characters: it is
    # where each line was started relative to where it was meant to start. It
    # cannot be derived from per-character residuals, because a wrapped
    # continuation line begins mid-sentence and its first character is
    # legitimately far from the margin.
    band_offsets = _margin_offsets(assignment, instances, em_px)
    letter_gap, word_gap = _measure_gaps(instances, em_px)
    confidence, suspect = alignment_confidence(instances, em_px)

    return {
        "instances": instances,
        "meta": {
            "em_px": round(em_px, 3),
            "x_height_mm": round(em_px * X_HEIGHT_PER_EM / mm_to_px(1.0), 3),
            "line_spacing_em": round(line_spacing / em_px, 4),
            "nominal_left_px": round(nominal_left, 2),
            "letter_gap_em": letter_gap,
            "word_gap_em": word_gap,
            "band_offsets": band_offsets,
            "written_lines": len(bands),
            "expected_lines": len(INK_LINES),
            "matched_chars": len(instances),
            "expected_chars": expected_total,
            "width_fit": {"alpha": width_fit[0], "beta": width_fit[1]},
            "alignment_confidence": confidence,
            "suspect_chars": suspect,
        },
    }


def _resolve_splits(matched, mask, widths=None):
    """Turn 'this character shares a component' markers into real sub-boxes."""
    groups = {}
    for entry in matched:
        if entry["split"] is None:
            continue
        comp, _k, _total = entry["split"]
        key = (comp["x0"], comp["y0"], comp["x1"], comp["y1"])
        groups.setdefault(key, []).append(entry)

    for entries in groups.values():
        entries.sort(key=lambda e: e["split"][1])
        comp = entries[0]["split"][0]
        spans = _split_component(comp, [e["char"] for e in entries], mask, widths)
        for entry, (sx0, sx1) in zip(entries, spans):
            entry["box"] = (sx0, comp["y0"], max(sx1, sx0 + 1), comp["y1"])


def alignment_confidence(instances, em_px):
    """
    How plausible the ink-to-character assignment looks, in [0, 1].

    A count of matched characters says nothing about whether each character
    got the *right* ink: an alignment that shifts every label one blob to the
    left still matches every character. What does show up is size — if a
    character's crop is a quarter the width of the other crops on the sheet,
    it is a sliver, and if it is triple, it has swallowed its neighbour.

    Returns (confidence, suspect_chars) where confidence is the fraction of
    instances whose crop width is within a plausible band of the sheet's own
    median, and suspect_chars lists the characters most often outside it.
    """
    if not instances or em_px <= 0:
        return 0.0, []

    widths = np.array([i["advance_em"] for i in instances], dtype=float)
    median_w = float(np.median(widths))
    if median_w <= 0:
        return 0.0, []

    ok = (widths > 0.30 * median_w) & (widths < 2.60 * median_w)
    suspect = {}
    for inst, good in zip(instances, ok):
        if not good:
            suspect[inst["char"]] = suspect.get(inst["char"], 0) + 1

    worst = sorted(suspect.items(), key=lambda kv: -kv[1])[:12]
    return round(float(ok.mean()), 4), [ch for ch, _n in worst]


def _measure_gaps(instances, em_px):
    """
    Measure how far apart this writer sets their letters and their words.

    Both come from one population: the whitespace between the right edge of
    each character and the left edge of the next, in em. The two separate
    cleanly — most gaps fall between letters inside a word, and a much smaller
    number, the spaces, are markedly wider. Rather than deciding case by case
    which gaps are spaces, the two are read off as a low and a high percentile
    of the distribution, which stays stable even when a few gaps are distorted
    by a mis-segmentation.

    This matters because a space leaves no ink, so its width can never be
    observed directly — it can only be inferred from the gaps around it.

    Returns (letter_gap_em, word_gap_em).
    """
    by_band = {}
    for inst in instances:
        by_band.setdefault(inst["band"], []).append(inst)

    gaps = []
    for band_instances in by_band.values():
        band_instances.sort(key=lambda i: i["actual_xy"][0])
        for cur, nxt in zip(band_instances, band_instances[1:]):
            gap = (nxt["actual_xy"][0] - cur["actual_xy"][0]) / em_px - cur["advance_em"]
            if gap > -0.15:      # ignore overlaps left by a bad split
                gaps.append(gap)

    if len(gaps) < 10:
        return 0.08, 0.30

    spacing = np.array(gaps, dtype=float)
    letter_gap = max(float(np.percentile(spacing, 30)), 0.01)
    word_gap = max(float(np.percentile(spacing, 93)), letter_gap + 0.10)
    return round(letter_gap, 4), round(word_gap, 4)


def _match_characters(per_band, assignment, mask, em_px, widths=None):
    """Align every expected character to ink, returning matched boxes."""
    matched = []
    for line_idx, band_indices in enumerate(assignment):
        text = INK_LINES[line_idx]
        # A wrapped line's characters run across its bands in order, so align
        # against the concatenated component sequence and attribute each
        # character back to whichever band its ink lives in.
        comps = []
        for bi in band_indices:
            for c in per_band[bi]:
                comps.append({**c, "band_index": bi})

        alignment = align_chars(comps, text, em_px, widths)
        if not alignment:
            raise CaptureError(
                f"Line {line_idx + 1} of the sheet could not be matched to the "
                "exercise text. Check it was copied exactly, then photograph "
                "the sheet again."
            )

        chars = [c for c in text if c != " "]
        for ch, (kind, payload) in zip(chars, alignment):
            if kind == "comps":
                group = payload
                matched.append({
                    "line": line_idx, "band_index": group[0]["band_index"],
                    "char": ch, "split": None,
                    "box": (min(c["x0"] for c in group), min(c["y0"] for c in group),
                            max(c["x1"] for c in group), max(c["y1"] for c in group)),
                })
            else:
                comp, k, total = payload
                matched.append({
                    "line": line_idx, "band_index": comp["band_index"],
                    "char": ch, "split": (comp, k, total),
                    "box": (comp["x0"], comp["y0"], comp["x1"], comp["y1"]),
                })

    _resolve_splits(matched, mask, widths)
    return matched


def _measure_widths(matched, em_px):
    """
    Build this writer's own advance-width table from the first pass.

    Only characters that were matched to whole components are used. A
    character carved out of a shared component has the width the cut gave it,
    so measuring those would just feed the prior's mistakes back in as if they
    were evidence.
    """
    observed = {}
    for entry in matched:
        if entry["split"] is not None:
            continue
        x0, _y0, x1, _y1 = entry["box"]
        observed.setdefault(entry["char"], []).append((x1 - x0) / em_px)

    widths = {}
    for char, values in observed.items():
        if len(values) >= 3:
            widths[char] = round(float(np.median(values)), 4)
    return widths


def _margin_offsets(assignment, instances, em_px):
    """
    Per written line, how far its left edge sits from where that line was
    meant to start, in em.

    Two things make the naive measurement — leftmost ink minus a nominal
    margin — unusable:

    * **Indentation.** The exercise indents lines by four spaces, which would
      read as a large rightward drift. Rather than assuming a width for a
      space, which we never observe directly because a space leaves no ink,
      lines are grouped by how far they are indented and measured against the
      median of their own group.
    * **Noise.** A single character's left edge depends on which character it
      happens to be — a 'T' and an 'i' start differently — and that variation
      is as large as the drift being measured. Taking the median residual over
      every character in the line averages it away.

    Wrapped continuation lines are excluded. Where a continuation starts is
    decided by where the previous line ran out of room, not by the writer
    choosing a margin, so it carries no information about margin drift.
    """
    first_band = {}
    for line_idx, band_indices in enumerate(assignment):
        if band_indices:
            leading = len(INK_LINES[line_idx]) - len(INK_LINES[line_idx].lstrip(" "))
            first_band[band_indices[0]] = leading

    residuals = {}
    for inst in instances:
        if inst["band"] in first_band:
            residuals.setdefault(inst["band"], []).append(
                inst["actual_xy"][0] - inst["expected_xy"][0]
            )

    groups = {}
    for band, values in residuals.items():
        groups.setdefault(first_band[band], []).append((band, float(np.median(values))))

    offsets = []
    for entries in groups.values():
        reference = float(np.median([r for _band, r in entries]))
        for band, r in entries:
            offsets.append([band, round((r - reference) / em_px, 4)])

    offsets.sort()
    return offsets


def _build_instances(matched, mask, geom, em_px, nominal_left, line_spacing,
                     baseline_offset, widths=None, band_rows=None):
    """Cut, describe and vectorize every matched character."""
    by_line = {}
    for e in matched:
        by_line.setdefault(e["line"], []).append(e)

    instances = []
    for line_idx, line_entries in sorted(by_line.items()):
        line_entries.sort(key=lambda e: (e["band_index"], e["box"][0]))
        text = INK_LINES[line_idx]

        # Where each character sits along the line if it were written
        # perfectly straight, in em, measured from the start of the line.
        prior_positions = []
        cursor = 0.0
        for ch in text:
            if ch == " ":
                cursor += width_prior(" ", widths)
                continue
            prior_positions.append(cursor)
            cursor += width_prior(ch, widths)

        # The reference is laid out per *written* line, not per source line.
        # When the writer wraps, the second written line starts back at the
        # margin, so treating the source line as one long straight run would
        # put every character after the wrap hundreds of pixels from where it
        # really is — and that error would then be read as drift.
        by_band = {}
        for idx, entry in enumerate(line_entries):
            by_band.setdefault(entry["band_index"], []).append((idx, entry))

        for band_index, band_entries in sorted(by_band.items()):
            first_idx = band_entries[0][0]
            origin = prior_positions[first_idx] if first_idx < len(prior_positions) else 0.0

            # Fit this hand's own advance scale for the band, so a systematic
            # mismatch between the width prior and this writer does not
            # masquerade as horizontal drift.
            scale = 1.0
            if len(band_entries) > 1:
                last_idx = band_entries[-1][0]
                if last_idx < len(prior_positions):
                    prior_span = prior_positions[last_idx] - origin
                    measured_span = (band_entries[-1][1]["box"][0]
                                     - band_entries[0][1]["box"][0])
                    if prior_span > 0:
                        scale = float(np.clip(
                            measured_span / (prior_span * em_px), 0.6, 1.7))

            for idx, entry in band_entries:
                if idx >= len(prior_positions):
                    continue
                inst = _describe_instance(
                    entry, mask, geom, em_px, line_idx,
                    expected_x=nominal_left + (prior_positions[idx] - origin) * em_px * scale,
                    expected_y=line_spacing * (band_rows or {}).get(
                        entry["band_index"], entry["band_index"]) + baseline_offset,
                )
                if inst:
                    instances.append(inst)

    return instances


def _describe_instance(entry, mask, geom, em_px, line_idx, expected_x, expected_y):
    """Cut, measure and vectorize one matched character."""
    g = geom.get(entry["band_index"])
    if not g:
        return None
    x0, y0, x1, y1 = entry["box"]
    crop = mask[y0:y1 + 1, x0:x1 + 1]
    if crop.size == 0 or crop.max() == 0:
        return None

    baseline_y = _baseline_at(g, 0.5 * (x0 + x1))
    path = vectorize.mask_to_path(crop, origin_x=0, baseline_y=baseline_y - y0,
                                  unit_px=em_px)
    if not path:
        return None

    w, h = (x1 - x0), (y1 - y0)
    # The character's *own* contact with the baseline is what carries the
    # within-line wander. Reporting the fitted line here instead would be
    # circular — the fit is a straight line through these very points, so it
    # smooths away the signal Stage 3 is trying to measure. Characters with
    # descenders have no baseline contact to measure, so they are flagged and
    # excluded from the drift fit.
    on_baseline = entry["char"] in _BASELINE_CHARS

    return {
        "char": entry["char"],
        "line": line_idx,
        "band": entry["band_index"],
        "on_baseline": on_baseline,
        "expected_xy": [round(float(expected_x), 2), round(float(expected_y), 2)],
        "actual_xy": [round(float(x0), 2),
                      round(float(y1) if on_baseline else float(baseline_y), 2)],
        "advance_em": round(w / em_px, 4),
        "aspect": round(float(w) / max(h, 1), 4),
        "density": round(float((crop > 0).mean()), 4),
        "slant": round(features.slant_angle(crop), 2),
        "bitmap": features.encode_bitmap(features.normalized_bitmap(crop)),
        "path": path,
        "bounds": [round(v, 4) for v in
                   vectorize.path_bounds(crop, 0, baseline_y - y0, em_px)],
    }

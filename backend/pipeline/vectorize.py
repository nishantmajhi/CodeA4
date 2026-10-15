"""
Bitmap ink -> SVG outline path (spec Stage 4).

Per spec §5 no font file is generated: a glyph is just an SVG path string,
composed directly at render time. This module produces those paths.

Output convention, which the renderer depends on:

* Units are **em**, where 1 em is the writer's own measured em size (derived
  from their x-height in ``segmentation``). A glyph is therefore resolution-
  independent and can be placed at any print size.
* The origin is the glyph's **baseline at its left edge**: x=0 is the left of
  the inked area, y=0 is the baseline, y increases downward (SVG convention).
  So a descender has positive y, an ascender negative.
* The path traces the **outline of the ink** and is filled, not stroked.
  Interior counters (the hole in an 'o', 'e', 'a') come back as separate
  subpaths and are cut out with ``fill-rule="evenodd"``.
"""
import cv2
import numpy as np

# Contours smaller than this fraction of the glyph's ink area are dropped as
# speckle rather than kept as a stray dot.
MIN_CONTOUR_AREA_FRAC = 0.012


def _smooth_path(points, fillet):
    """
    Turn a polygon into a compact, smooth SVG path.

    Each corner is rounded by a small fillet: the path runs straight along
    the edge, turns through a short quadratic with the vertex as its control
    point, then continues straight. `fillet` is the radius, in the same units
    as `points`, and is additionally capped at half of each adjoining edge so
    a short edge cannot be swallowed by the corners at either end of it.

    The cap is the whole point. Rounding a vertex all the way to the
    midpoints of its two edges — which is what an uncapped fillet does —
    leaves no straight run at all, so a stem traced as a thin rectangle comes
    back as an ellipse and every letter inflates into a blob. Pen strokes are
    smooth at the corners and straight in between, and only a bounded fillet
    gives both.
    """
    n = len(points)
    if n < 3:
        return ""

    def sub(a, b):
        return (a[0] - b[0], a[1] - b[1])

    def norm(v):
        length = (v[0] ** 2 + v[1] ** 2) ** 0.5
        return length, (v[0] / length, v[1] / length) if length > 1e-9 else (0.0, 0.0)

    entries, exits = [], []
    for i in range(n):
        prev_pt, pt, next_pt = points[i - 1], points[i], points[(i + 1) % n]
        len_in, dir_in = norm(sub(pt, prev_pt))
        len_out, dir_out = norm(sub(next_pt, pt))
        radius = min(fillet, 0.5 * len_in, 0.5 * len_out)
        if radius <= 1e-9:
            entries.append(pt)
            exits.append(pt)
            continue
        entries.append((pt[0] - dir_in[0] * radius, pt[1] - dir_in[1] * radius))
        exits.append((pt[0] + dir_out[0] * radius, pt[1] + dir_out[1] * radius))

    parts = [f"M{exits[0][0]:.4f},{exits[0][1]:.4f}"]
    for i in range(1, n + 1):
        k = i % n
        parts.append(f"L{entries[k][0]:.4f},{entries[k][1]:.4f}")
        if entries[k] != exits[k]:
            parts.append(
                f"Q{points[k][0]:.4f},{points[k][1]:.4f} "
                f"{exits[k][0]:.4f},{exits[k][1]:.4f}"
            )
    parts.append("Z")
    return "".join(parts)


# Contour simplification tolerance and corner fillet, both in em. Expressing
# them in em rather than pixels is what keeps a glyph looking the same
# whatever resolution the sheet was photographed at — a fixed pixel tolerance
# is gentle on a large scan and destroys a small one.
SIMPLIFY_EM = 0.012
FILLET_EM = 0.016


# Glyph crops are resampled to about this height before tracing.
TRACE_HEIGHT_PX = 140
MAX_TRACE_SCALE = 8


def _upsample(mask, origin_x, baseline_y, unit_px):
    """Resample a glyph crop up to tracing resolution, anti-aliased."""
    binary = (mask > 0).astype(np.uint8) * 255
    height = binary.shape[0]
    if height <= 0:
        return binary, origin_x, baseline_y, unit_px

    scale = int(np.clip(round(TRACE_HEIGHT_PX / height), 1, MAX_TRACE_SCALE))
    if scale <= 1:
        return binary, origin_x, baseline_y, unit_px

    big = cv2.resize(binary, None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_CUBIC)
    # Blur then threshold at the half-way grey: the interpolated edge carries
    # the sub-pixel position of the boundary, and this reads it back out.
    ksize = 2 * (scale // 2) + 1
    if ksize > 1:
        big = cv2.GaussianBlur(big, (ksize, ksize), 0)
    big = (big > 127).astype(np.uint8)
    return big, origin_x * scale, baseline_y * scale, unit_px * scale


def mask_to_path(mask, origin_x, baseline_y, unit_px, simplify_px=None):
    """
    Args:
        mask: binary crop containing exactly one character's ink.
        origin_x: column (in `mask` coordinates) that maps to x=0.
        baseline_y: row (in `mask` coordinates) that maps to y=0.
        unit_px: pixels per em.
        simplify_px: Douglas-Peucker tolerance before smoothing.

    Returns:
        SVG path data string in em units, or "" if the mask holds no ink.
    """
    if unit_px <= 0:
        return ""

    # Trace at higher resolution than the sheet was photographed at. A
    # character on an A4 sheet is only a few dozen pixels across, and a
    # contour traced at that size is a staircase whose steps are a
    # substantial fraction of a stroke width — no amount of downstream
    # smoothing recovers the curve, it only rounds the staircase. Upsampling
    # with interpolation and re-thresholding puts the edge back where the
    # ink's grey levels say it was, sub-pixel, before it is traced.
    mask, origin_x, baseline_y, unit_px = _upsample(
        mask, origin_x, baseline_y, unit_px
    )

    if simplify_px is None:
        simplify_px = max(SIMPLIFY_EM * unit_px, 0.5)
    contours, hierarchy = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return ""

    total_area = sum(cv2.contourArea(c) for c in contours) or 1.0
    subpaths = []
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_CONTOUR_AREA_FRAC * total_area:
            continue
        approx = cv2.approxPolyDP(cnt, simplify_px, True)
        pts = approx.reshape(-1, 2).astype(np.float64)
        if len(pts) < 3:
            continue
        pts[:, 0] = (pts[:, 0] - origin_x) / unit_px
        pts[:, 1] = (pts[:, 1] - baseline_y) / unit_px
        sub = _smooth_path([tuple(p) for p in pts], FILLET_EM)
        if sub:
            subpaths.append(sub)

    return "".join(subpaths)


def path_bounds(mask, origin_x, baseline_y, unit_px):
    """Ink bounding box in em units: (x_min, y_min, x_max, y_max)."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        float((xs.min() - origin_x) / unit_px),
        float((ys.min() - baseline_y) / unit_px),
        float((xs.max() + 1 - origin_x) / unit_px),
        float((ys.max() + 1 - baseline_y) / unit_px),
    )

"""
Shared OpenCV front-end for Stage 1/2 (spec §4).

Everything here turns a *photograph of a sheet of paper* into a clean,
deskewed, binary ink mask in a canonical coordinate system, plus the line
bands within it. It is deliberately separate from ``segmentation.py`` so the
image-conditioning half can be tested (and tuned against real photos) apart
from the character-alignment half.

Coordinate system after ``prepare_sheet``: the page is warped to portrait A4
at ``PAGE_DPI``, so 1 px is a known physical size and every downstream
measurement (x-height, advance widths, drift amplitude) is in real units
rather than "pixels of whatever photo the user happened to take".
"""
import cv2
import numpy as np

# A4 at 200 dpi. High enough that a 3 mm x-height is ~24 px — plenty of
# detail to vectorize a glyph outline from — without making per-photo
# processing slow.
PAGE_DPI = 200
MM_PER_INCH = 25.4
PAGE_W_MM, PAGE_H_MM = 210.0, 297.0
PAGE_W = int(round(PAGE_W_MM / MM_PER_INCH * PAGE_DPI))   # 1654
PAGE_H = int(round(PAGE_H_MM / MM_PER_INCH * PAGE_DPI))   # 2339


class CaptureError(Exception):
    """
    Raised when a photo cannot be processed into usable samples.

    The message is shown to the user more or less verbatim in the capture
    step's error box, so it must say what *they* can do differently — not
    what failed internally.
    """


def px_to_mm(px):
    return px / PAGE_DPI * MM_PER_INCH


def mm_to_px(mm):
    return mm / MM_PER_INCH * PAGE_DPI


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise CaptureError(
            "That file could not be read as an image. Try photographing the "
            "sheet again, or upload a JPEG or PNG."
        )
    return img


def find_page_quad(img):
    """
    Locate the sheet of paper within the photo.

    Returns the four corners (as float32, in tl/tr/br/bl order) or None if no
    convincing quadrilateral is found — in which case the caller falls back to
    treating the whole frame as the page, which is correct for an already
    cropped scan.
    """
    h, w = img.shape[:2]
    scale = 900.0 / max(h, w)
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # The page is a bright region on a darker surround. Otsu on the blurred
    # image finds it more reliably than edge detection, which tends to latch
    # onto the ruled text block instead of the paper border.
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    frame_area = small.shape[0] * small.shape[1]
    best = None
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        area = cv2.contourArea(cnt)
        if area < 0.25 * frame_area:
            break
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            best = approx
            break

    if best is None:
        return None
    if cv2.contourArea(best) > 0.985 * frame_area:
        # The "page" is the whole frame — i.e. the photo is already cropped to
        # the sheet. Warping would be a no-op that only costs resampling.
        return None
    return _order_quad(best.reshape(4, 2).astype(np.float32) / scale)


def _order_quad(pts):
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([
        pts[np.argmin(s)],   # tl
        pts[np.argmin(d)],   # tr
        pts[np.argmax(s)],   # br
        pts[np.argmax(d)],   # bl
    ], dtype=np.float32)


def warp_to_page(img, quad):
    """Perspective-correct the detected sheet onto a canonical A4 canvas."""
    dst = np.array([
        [0, 0], [PAGE_W - 1, 0], [PAGE_W - 1, PAGE_H - 1], [0, PAGE_H - 1]
    ], dtype=np.float32)
    m = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(img, m, (PAGE_W, PAGE_H), flags=cv2.INTER_CUBIC)


def flatten_illumination(gray):
    """
    Remove the smooth brightness gradient a phone photo always has (shadow of
    the hand, uneven desk lighting) by dividing out a heavily blurred copy.

    Without this, a single global threshold either loses faint ink on the dark
    side of the page or floods the bright side with noise.
    """
    background = cv2.medianBlur(gray, 61)
    background = np.maximum(background, 1)
    flat = (gray.astype(np.float32) / background.astype(np.float32)) * 200.0
    return np.clip(flat, 0, 255).astype(np.uint8)


# Adaptive threshold settings for confidently-inked pixels. C is the margin,
# in grey levels, by which a pixel must beat its local mean.
STRONG_BLOCK, STRONG_C = 41, 12
# How far below the flattened paper level a pixel must fall to count as
# possible ink. flatten_illumination normalises paper to a known level, so
# this can be a single global fraction rather than another local mean.
WEAK_PAPER_FRAC = 0.70


def binarize(gray):
    """
    Return a uint8 mask where ink is 255 and paper is 0.

    Thresholding is done twice and combined by hysteresis, the way Canny
    handles edges: a strict pass finds pixels that are certainly ink, a
    lenient pass finds every pixel that might be, and a lenient pixel is kept
    only if it belongs to a blob that the strict pass also lit up.

    A single strict threshold cannot cope with the range of darkness within
    one stroke. Where a pen bears down — the start of a stroke, a serif, a
    junction — ink is far darker than the thin fast middle of the same
    stroke, and a margin tight enough to reject paper also rejects that thin
    middle. The stroke then breaks into two blobs with a hole where the
    letter's waist should be, which is much worse than losing it entirely:
    the segmenter sees two components, hands them to two different
    characters, and every label after it slides onto the wrong ink.

    A single lenient threshold instead floods the mask with paper texture.
    Hysteresis gets both — faint ink survives when it is continuous with ink
    we are sure about, while an isolated faint speck, which is what noise
    looks like, does not.
    """
    flat = flatten_illumination(gray)

    strong = cv2.adaptiveThreshold(
        flat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        blockSize=STRONG_BLOCK, C=STRONG_C,
    )
    # Drop pepper noise but keep thin pen strokes: opening with a 2x2 kernel
    # removes isolated specks without eating a 3 px pen line. Done on the
    # strict pass only, so it cannot re-break a stroke that hysteresis is
    # about to reconnect.
    strong = cv2.morphologyEx(strong, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    paper = float(np.percentile(flat, 90))
    weak = (flat < WEAK_PAPER_FRAC * paper).astype(np.uint8)

    return _hysteresis(strong, weak)


def _hysteresis(strong, weak):
    """Keep the connected components of `weak` that contain `strong` ink."""
    if not strong.any():
        return np.zeros_like(strong)

    count, labels = cv2.connectedComponents(weak, connectivity=8)
    if count <= 1:
        return (strong > 0).astype(np.uint8) * 255

    seeded = np.unique(labels[strong > 0])
    seeded = seeded[seeded != 0]
    if seeded.size == 0:
        return (strong > 0).astype(np.uint8) * 255

    keep = np.zeros(count, dtype=bool)
    keep[seeded] = True
    return (keep[labels]).astype(np.uint8) * 255


def clear_page_border(mask, margin_mm=4.0):
    """
    Blank a strip around the page edge.

    Perspective-correcting a photo almost always drags a sliver of the darker
    surface the page is resting on into the frame, and adaptive thresholding
    reads that sliver as ink. A stray component at x~0 quietly wrecks the
    left-margin measurement, so anything within a few millimetres of the edge
    is discarded — nobody writes there anyway.
    """
    m = int(mm_to_px(margin_mm))
    if m <= 0:
        return mask
    out = mask.copy()
    out[:m, :] = 0
    out[-m:, :] = 0
    out[:, :m] = 0
    out[:, -m:] = 0
    return out


def estimate_skew(mask, max_deg=7.0, step=0.25):
    """
    Find the small residual rotation left after perspective correction.

    Scored by the variance of the horizontal projection profile: text lines
    are maximally "peaky" in projection when they are exactly horizontal.
    That is more robust on handwriting than Hough lines, which need long
    straight strokes that handwriting does not have.
    """
    small = cv2.resize(mask, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    best_angle, best_score = 0.0, -1.0
    h, w = small.shape
    center = (w / 2, h / 2)
    for angle in np.arange(-max_deg, max_deg + 1e-9, step):
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        rot = cv2.warpAffine(small, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        profile = rot.sum(axis=1, dtype=np.float64)
        score = float(np.var(profile))
        if score > best_score:
            best_angle, best_score = float(angle), score
    return best_angle


def rotate(img, angle, border_value=0):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)


def prepare_sheet(path):
    """
    Photo path -> (ink_mask, gray) in canonical page coordinates.

    Raises CaptureError with a user-facing message when the photo is unusable.
    """
    img = load_image(path)
    quad = find_page_quad(img)
    if quad is not None:
        img = warp_to_page(img, quad)
    else:
        img = cv2.resize(img, (PAGE_W, PAGE_H), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = clear_page_border(binarize(gray))

    ink_fraction = float(mask.mean()) / 255.0
    if ink_fraction < 0.001:
        raise CaptureError(
            "We could not find any writing on this sheet. Check that the page "
            "is in focus and fills the frame, then take the photo again."
        )
    if ink_fraction > 0.35:
        raise CaptureError(
            "This photo is too dark or too blurry to separate the writing from "
            "the page. Try again in brighter, even light without a shadow "
            "across the sheet."
        )

    angle = estimate_skew(mask)
    if abs(angle) > 0.2:
        gray = rotate(gray, angle, border_value=255)
        mask = clear_page_border(binarize(gray))

    return mask, gray


def find_line_bands(mask, min_height_px=None, gap_px=None):
    """
    Split the ink mask into horizontal bands, one per written line.

    Returns a list of (top, bottom) row indices. Uses a smoothed horizontal
    projection so that the gaps *within* a line (between words, or above the
    body of a short letter) do not split it, while true inter-line gaps do.
    """
    if min_height_px is None:
        min_height_px = int(mm_to_px(1.6))
    if gap_px is None:
        gap_px = int(mm_to_px(1.2))

    profile = mask.sum(axis=1, dtype=np.float64) / 255.0
    if profile.max() <= 0:
        return []

    # Smooth over roughly a stroke width so single-pixel dropouts don't count
    # as a gap between lines.
    k = max(3, int(mm_to_px(0.5)) | 1)
    kernel = np.ones(k) / k
    smooth = np.convolve(profile, kernel, mode="same")

    # An absolute threshold, not one scaled to the page's busiest lines: a
    # line holding a single character ("}" on its own) has a tiny projection
    # compared to a full line of prose, and a relative threshold silently
    # drops it — which then throws the whole line-to-exercise alignment out by
    # one. A couple of ink pixels per row is enough to call a row inked;
    # speckle is already gone (morphological opening in `binarize`) and any
    # band too short to be writing is filtered out below.
    threshold = 2.0
    on = smooth > threshold

    bands = []
    start = None
    for i, v in enumerate(on):
        if v and start is None:
            start = i
        elif not v and start is not None:
            bands.append([start, i - 1])
            start = None
    if start is not None:
        bands.append([start, len(on) - 1])

    # Merge bands separated by less than a real inter-line gap (this rejoins a
    # line with its own ascenders/descenders when they happen to be detached).
    merged = []
    for band in bands:
        if merged and band[0] - merged[-1][1] <= gap_px:
            merged[-1][1] = band[1]
        else:
            merged.append(band)

    bands = [(t, b) for t, b in merged if (b - t + 1) >= min_height_px]
    return _split_tall_bands(smooth, bands)


def _split_tall_bands(profile, bands, tolerance=1.65):
    """
    Split bands that are really two touching lines.

    Handwriting on unruled paper regularly has a descender from one line
    overlapping an ascender from the next, so the projection never returns to
    zero between them and they come out as a single band. Left alone this
    throws every subsequent line off by one, which is fatal — the whole method
    depends on knowing which expected line each band is.

    A band more than `tolerance` times the median band height is cut at the
    lightest rows near the estimated internal boundaries.
    """
    if len(bands) < 3:
        return bands

    heights = np.array([b - t + 1 for t, b in bands], dtype=float)
    median_h = float(np.median(heights))
    if median_h <= 0:
        return bands

    out = []
    for top, bottom in bands:
        height = bottom - top + 1
        parts = int(round(height / median_h))
        if height <= tolerance * median_h or parts < 2:
            out.append((top, bottom))
            continue

        cuts = []
        for k in range(1, parts):
            target = top + int(round(k * height / parts))
            half = max(2, int(0.22 * height / parts))
            lo, hi = max(top + 1, target - half), min(bottom, target + half)
            if hi <= lo:
                cuts.append(target)
            else:
                cuts.append(lo + int(np.argmin(profile[lo:hi])))

        edges = [top] + sorted(cuts) + [bottom]
        for i in range(len(edges) - 1):
            lo = edges[i] if i == 0 else edges[i] + 1
            out.append((lo, edges[i + 1]))

    return out

"""
Development tool — generates a synthetic "photograph of a handwritten sheet".

This is NOT part of the product pipeline. It exists because the CV stages
(segmentation, clustering, drift fitting) cannot be developed or regression-
tested without input, and real handwriting samples are not checked into the
repo. The generator deliberately reproduces the properties the pipeline is
supposed to discover:

* **Per-character modes** — each character is drawn from one of several
  variant renderings with fixed probabilities, so clustering has real modes
  to find rather than noise around a single shape. The three faces have to
  differ by more than one face varies from itself under the wobble below, or
  the property is claimed rather than provided; see FONT_CANDIDATES.
* **Mean-reverting drift** — the baseline wanders within a line and the left
  margin wanders down the page, both as Ornstein-Uhlenbeck walks, so the
  drift fit has real structure to recover.
* **Organic wrapping** — the deliberately over-long exercise line runs past
  the right margin and is wrapped onto a second written line, exactly as a
  human would, so line alignment has to handle 1 expected line -> 2 written
  lines.
* **Photo degradation** — perspective, an illumination gradient, blur and
  sensor noise, so the imaging front-end is exercised rather than handed a
  clean render.

Because the same generator seed always produces the same sheet, tests can
assert on exact recovery numbers.

Usage:
    python tools/make_sample_sheet.py out.jpg --seed 1
"""
import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.exercise import INK_LINES                   # noqa: E402
from pipeline.imaging import PAGE_W, PAGE_H, mm_to_px      # noqa: E402

# Three font faces stand in for three ways of forming each letter. They are
# only a fixture device — the real system extracts modes from one person's
# hand, it does not mix typefaces.
#
# **They have to be three genuinely different designs.** These were once three
# humanist sans faces, and the difference between them was smaller in the
# 16x16 normalized bitmap than the per-instance wobble applied below — so the
# modes were not recoverable even before the sheet was photographed, let alone
# segmented. Clustering therefore had nothing real to find, separated the
# well-segmented instances from the badly-segmented ones instead, and style 2
# came out as a broken alphabet that the Style step reported as "41% of your
# samples". Nothing caught it, because the only test in the area asserted that
# the style sets *differ* — which clusters of noise satisfy.
#
# tests/test_style_quality.py now measures cluster purity against the modes
# recorded here, which is the check that would have. Keep the three groups a
# sans, a serif and something as unlike both as the machine has.
# Each inner list is one mode, in order of preference, so the same three
# distinguishable faces are found on a Linux CI box and on a Windows desktop.
# They must be three *different* designs: the modes are what Stage 3 clusters,
# and three copies of one face gives it nothing to find.
FONT_CANDIDATES = [
    # Sans.
    ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
     "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
     "C:/Windows/Fonts/verdana.ttf", "C:/Windows/Fonts/arial.ttf"],
    # Serif.
    ["/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
     "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
     "C:/Windows/Fonts/times.ttf", "C:/Windows/Fonts/georgia.ttf"],
    # Monospace, which is as unlike a proportional sans and a serif as a
    # commonly installed face gets. A script face separates the modes a little
    # better still, but none is standard on a Linux box, and a fixture that
    # behaves differently depending on what fonts a machine happens to have is
    # worse than one that is slightly harder.
    ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
     "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
     "C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/cour.ttf"],
]
MODE_WEIGHTS = [0.5, 0.32, 0.18]

EM_PX = 46                      # ~5.8 mm em -> ~3 mm x-height, natural size
LINE_H = int(mm_to_px(8.4))
LEFT_MARGIN = int(mm_to_px(18))
RIGHT_MARGIN = PAGE_W - int(mm_to_px(14))
TOP_MARGIN = int(mm_to_px(24))


class MissingFonts(Exception):
    """Raised when this machine has none of the fixture faces installed."""


def _load_fonts():
    """
    Load one scalable face per mode.

    Falling back to ``ImageFont.load_default()`` was worse than failing: it is
    a bitmap face fixed at about 11 px, so every glyph came out a fifth of the
    intended size and all three modes were the *same* face. The sheet still
    rendered, and the failure surfaced four stages later as "Only 12 lines of
    writing were found", which says nothing about fonts. Say so here instead.
    """
    fonts = []
    for group in FONT_CANDIDATES:
        for path in group:
            if os.path.exists(path):
                fonts.append(ImageFont.truetype(path, EM_PX))
                break
        else:
            raise MissingFonts(
                "none of these fixture fonts is installed: " + ", ".join(group)
            )
    return fonts


def _ou_walk(n, theta, sigma, rng):
    """Ornstein-Uhlenbeck walk: wanders, then pulls back toward zero."""
    out = np.zeros(n)
    x = 0.0
    for i in range(n):
        x += -theta * x + rng.normal(0.0, sigma)
        out[i] = x
    return out


def _render_glyph(ch, font, rng):
    """
    Draw one character and distort it.

    Returns (alpha_image, advance, base_row) where `base_row` is the row in
    the returned image that the glyph's baseline sits on. Anchoring on the
    baseline rather than the ascender box matters: the three fixture fonts
    have different ascents, so top-anchoring would make baseline position
    correlate with mode and hand the drift fit a signal that isn't drift.
    """
    pad = EM_PX
    size = EM_PX * 3
    box = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(box)
    base_row = pad + EM_PX
    draw.text((pad, base_row), ch, fill=255, font=font, anchor="ls")

    arr = np.array(box)
    if arr.max() == 0:
        return None, int(EM_PX * 0.5), base_row

    # Per-instance wobble: a small rotation and shear, plus a low-frequency
    # elastic warp — the within-mode variation a real hand has.
    h, w = arr.shape
    angle = rng.normal(0, 2.2)
    shear = rng.normal(0, 0.06)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, rng.normal(1.0, 0.035))
    m[0, 1] += shear
    arr = cv2.warpAffine(arr, m, (w, h), flags=cv2.INTER_LINEAR)

    grid = 6
    dx = cv2.resize(rng.normal(0, 1.1, (grid, grid)).astype(np.float32), (w, h),
                    interpolation=cv2.INTER_CUBIC)
    dy = cv2.resize(rng.normal(0, 1.1, (grid, grid)).astype(np.float32), (w, h),
                    interpolation=cv2.INTER_CUBIC)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    arr = cv2.remap(arr, xs + dx, ys + dy, interpolation=cv2.INTER_LINEAR)

    advance = int(round(font.getlength(ch) * rng.normal(1.0, 0.03)))
    return arr, max(advance, 4), base_row


def render_sheet(seed=0):
    """
    Render the exercise as an ideal (undegraded) page image, ink dark on white.

    Returns (page_gray, truth) where `truth` records, for every glyph drawn,
    the exercise line it came from, the character, its intended baseline
    position and the mode used — the ground truth tests measure recovery
    against.
    """
    rng = np.random.default_rng(seed)
    fonts = _load_fonts()
    page = np.zeros((PAGE_H, PAGE_W), dtype=np.float32)
    truth = []

    max_lines = len(INK_LINES) * 2 + 8
    # Margin drift: one slow mean-reverting walk down the whole page.
    # ~1.5 mm of accumulated wander, which is a modest but realistic amount
    # for unruled paper.
    margin_drift = _ou_walk(max_lines, theta=0.10, sigma=7.0, rng=rng)

    written_line = 0
    for li, text in enumerate(INK_LINES):
        x = LEFT_MARGIN + margin_drift[written_line]
        y = TOP_MARGIN + written_line * LINE_H
        # Baseline drift: a faster walk across the line, resampled per char.
        baseline = _ou_walk(len(text) + 8, theta=0.22, sigma=0.9, rng=rng)

        for ci, ch in enumerate(text):
            if ch == " ":
                x += EM_PX * 0.30
                continue

            mode = int(rng.choice(len(fonts), p=MODE_WEIGHTS))
            glyph, advance, base_row = _render_glyph(ch, fonts[mode], rng)
            if glyph is None:
                x += advance
                continue

            base_y = y + baseline[ci]
            gy = int(round(base_y)) - base_row
            gx = int(round(x))
            gh, gw = glyph.shape
            y0, x0 = max(0, gy), max(0, gx)
            y1, x1 = min(PAGE_H, gy + gh), min(PAGE_W, gx + gw)
            if y1 > y0 and x1 > x0:
                patch = glyph[y0 - gy:y1 - gy, x0 - gx:x1 - gx]
                page[y0:y1, x0:x1] = np.maximum(page[y0:y1, x0:x1], patch)
                truth.append({
                    "line": li, "written_line": written_line, "char": ch,
                    "x": float(x), "baseline_y": float(base_y), "mode": mode,
                })

            x += advance

            # Organic wrap: once the pen runs past the right margin the writer
            # simply continues on the next line — no hyphen, no marker. The
            # exercise has one line authored to force this, but any line that
            # overflows gets the same treatment, as it would by hand.
            if x > RIGHT_MARGIN and ci < len(text) - 1:
                written_line += 1
                x = LEFT_MARGIN + margin_drift[written_line]
                y = TOP_MARGIN + written_line * LINE_H
                baseline = _ou_walk(len(text) + 8, theta=0.22, sigma=0.9, rng=rng)

        written_line += 1

    ink = np.clip(page, 0, 255).astype(np.uint8)
    return 255 - ink, truth  # dark ink on white paper


def degrade(page_gray, seed=0):
    """Turn a clean render into something that looks photographed."""
    rng = np.random.default_rng(seed + 9001)
    h, w = page_gray.shape

    img = cv2.cvtColor(page_gray, cv2.COLOR_GRAY2BGR)

    # Paper on a darker surface, so page detection has a border to find.
    pad = int(0.07 * w)
    canvas = np.full((h + 2 * pad, w + 2 * pad, 3), 70, dtype=np.uint8)
    canvas[pad:pad + h, pad:pad + w] = img

    # Mild perspective, as if the phone was not perfectly square to the page.
    ch, cw = canvas.shape[:2]
    jitter = 0.018 * cw
    src = np.float32([[pad, pad], [pad + w, pad], [pad + w, pad + h], [pad, pad + h]])
    dst = src + rng.uniform(-jitter, jitter, src.shape).astype(np.float32)
    m = cv2.getPerspectiveTransform(src, dst)
    canvas = cv2.warpPerspective(canvas, m, (cw, ch), borderValue=(70, 70, 70))

    # Uneven lighting: a soft diagonal gradient plus a broad shadow.
    ys, xs = np.mgrid[0:ch, 0:cw].astype(np.float32)
    grad = 1.0 - 0.28 * (xs / cw) - 0.16 * (ys / ch)
    shadow = 1.0 - 0.20 * np.exp(-((xs - 0.75 * cw) ** 2 + (ys - 0.15 * ch) ** 2)
                                 / (2 * (0.30 * cw) ** 2))
    canvas = np.clip(canvas.astype(np.float32) * (grad * shadow)[..., None], 0, 255)

    canvas = cv2.GaussianBlur(canvas, (3, 3), 0.8)
    canvas += rng.normal(0, 3.2, canvas.shape)
    return np.clip(canvas, 0, 255).astype(np.uint8)


def make_sheet(out_path, seed=0, clean=False, truth_path=None):
    page, truth = render_sheet(seed=seed)
    img = cv2.cvtColor(page, cv2.COLOR_GRAY2BGR) if clean else degrade(page, seed=seed)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if truth_path:
        import json
        with open(truth_path, "w") as f:
            json.dump(truth, f)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", help="output image path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clean", action="store_true",
                    help="skip photo degradation (flat scan)")
    ap.add_argument("--truth", help="also write ground-truth JSON here")
    args = ap.parse_args()
    print(make_sheet(args.out, seed=args.seed, clean=args.clean,
                     truth_path=args.truth))


if __name__ == "__main__":
    main()

"""
Shape descriptors for Stage 3 clustering.

Each extracted character instance is reduced to a small fixed-size descriptor
so instances of the same character can be compared. Two things matter here:

* The descriptor has to be **scale- and position-normalized**, or clustering
  would separate instances by how big they happen to be rather than by how
  they are formed. Aspect ratio is preserved (an 'l' is not stretched into a
  square) and kept as an explicit feature instead.
* It has to be **small enough to store per instance**. A sheet yields ~530
  instances and there are three sheets, so a fat descriptor would bloat
  ``chars.json`` fast. A 16x16 uint8 bitmap is 256 bytes, stored base64.
"""
import base64

import cv2
import numpy as np

GRID = 16


def normalized_bitmap(mask):
    """
    Fit a binary glyph mask into a GRID x GRID box, preserving aspect ratio
    and centring it. Returns a uint8 array (0..255).
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros((GRID, GRID), dtype=np.uint8)

    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    scale = (GRID - 2) / max(h, w)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(crop.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_AREA)

    out = np.zeros((GRID, GRID), dtype=np.uint8)
    y0 = (GRID - nh) // 2
    x0 = (GRID - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = resized
    return out


def encode_bitmap(bmp):
    return base64.b64encode(bmp.astype(np.uint8).tobytes()).decode("ascii")


def decode_bitmap(data):
    raw = base64.b64decode(data)
    return np.frombuffer(raw, dtype=np.uint8).reshape(GRID, GRID)


def slant_angle(mask):
    """
    Estimate the glyph's slant in degrees from its second-order image moments.

    This is the per-instance rotation the renderer later re-applies (spec
    Stage 6 asks for a computed rotation per glyph); measuring it here means
    the rotation applied at render time comes from the writer's own hand
    rather than being invented noise.
    """
    m = cv2.moments(mask.astype(np.uint8), binaryImage=True)
    if m["m00"] == 0:
        return 0.0
    mu20 = m["mu20"] / m["m00"]
    mu02 = m["mu02"] / m["m00"]
    mu11 = m["mu11"] / m["m00"]
    if abs(mu20 - mu02) < 1e-9 and abs(mu11) < 1e-9:
        return 0.0
    theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    deg = float(np.degrees(theta))
    # Map onto a sensible +/-45 range; the principal axis is direction-free.
    while deg > 45:
        deg -= 90
    while deg < -45:
        deg += 90
    return deg


def feature_vector(instance):
    """
    Build the clustering feature vector for one stored instance dict.

    Bitmap pixels dominate the vector (they carry the letterform), with
    aspect ratio and ink density appended as two explicit, comparably scaled
    features so that, for example, a narrow and a wide way of writing the
    same letter separate even when their normalized bitmaps look alike.
    """
    bmp = decode_bitmap(instance["bitmap"]).astype(np.float32) / 255.0
    extras = np.array([
        instance.get("aspect", 1.0),
        instance.get("density", 0.0),
    ], dtype=np.float32)
    return np.concatenate([bmp.ravel(), extras * 2.0])


def feature_matrix(instances):
    return np.vstack([feature_vector(i) for i in instances])

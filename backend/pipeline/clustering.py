"""
Spec Stage 3 — Style clustering & drift modeling

Two independent things are learned from the pooled Stage 2 output of all
three capture rounds.

**Shape clustering.** For each character, its instances are clustered into a
small number of natural modes. The count is chosen from the data rather than
fixed: silhouette score decides whether a character genuinely has two or
three ways of being formed, or only one. This is the core claim in spec §2 —
handwriting is a few variants per letter, not one shape plus noise, and not
unlimited randomness — so guessing k would beg the question.

**Drift modeling.** The baseline wanders vertically within a line and the
left margin wanders horizontally down the page, both correcting *gradually*.
That behaviour is exactly an Ornstein-Uhlenbeck process: pulled back toward
the mean in proportion to how far it has strayed, with fresh noise each step.
Fitting one gives two interpretable numbers — how strongly the writer
self-corrects (theta) and how much they wander per step (sigma) — which the
renderer resamples to generate *new* drift with the same character, rather
than replaying the captured wobble verbatim.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from pipeline import features

# Bounds on what can be an instance of a character at all, relative to the
# median of that character's own instances. Unlike the MAD filter below these
# are absolute limits rather than a measure of spread: no hand writes one 'n'
# at two thirds the width of its other 'n's, so an instance that narrow is a
# cut that landed inside a letter, and one that wide has swallowed its
# neighbour. Asymmetric on purpose — the two failures are not the same size.
MIN_PLAUSIBLE_WIDTH = 0.65
MAX_PLAUSIBLE_WIDTH = 1.60
# ...and how far off its character's usual seating an instance's ink may sit,
# in em. A crop that lost the bottom of its letter, or gained the top of the
# line below, shows up here and nowhere else.
MAX_SEATING_DEVIATION = 0.20

# Below this many instances of a character there is not enough evidence to
# claim it has more than one mode.
MIN_INSTANCES_FOR_CLUSTERING = 6
# A cluster smaller than this is a stray or a mis-segmentation, not a mode.
MIN_CLUSTER_SIZE = 3
# ...and it must also account for at least this share of the character.
MIN_CLUSTER_FRACTION = 0.18
# Silhouette below this means the split is not well separated; keep one mode.
MIN_SILHOUETTE = 0.16
MAX_MODES_PER_CHAR = 4

# How many internally consistent style sets to build (spec Stage 4's "N").
MIN_STYLES, MAX_STYLES = 2, 4

# Bounds on the fitted autocorrelation of the drift walks. See fit_ou.
#
# The floor differs by series because a "step" means something different in
# each. A margin step is a whole written line, and consecutive lines really
# are only loosely related. A baseline step is a single character, and a hand
# does not re-choose its height letter by letter — the baseline of a word
# drifts as one, so its autocorrelation is necessarily high. Sharing one low
# floor between them was giving every character an independent vertical
# offset, which reads as letters bouncing rather than a line that wanders.
PHI_FLOOR_MARGIN = 0.35
MIN_LINE_SAMPLES = 6
PHI_CEILING = 0.97
PHI_FLOOR = PHI_FLOOR_MARGIN          # kept for callers that do not specify


# --------------------------------------------------------------------------
# Shape clustering
# --------------------------------------------------------------------------

def reject_implausible(instances):
    """
    Drop instances that cannot be this character at all.

    The MAD filter below measures spread, which is the right tool for deciding
    whether an instance is unusual. It is the wrong tool for deciding whether
    an instance is *possible*: when a third of a character's instances were
    cut badly in the same way, the spread widens to accommodate them and they
    survive. k-means then gives them a cluster of their own, that cluster
    becomes a style, and the Style step shows the user a broken alphabet and
    tells them it is 41% of how they write. This is the failure the module
    docstring for `reject_outliers` describes, and measuring spread does not
    prevent it.

    So this filter is a flat plausibility bound against the character's own
    median, applied first: too narrow to be the letter, too wide to be only
    the letter, or sitting too far off the baseline its own kind sit on.

    At least half the instances always survive, because a filter that can
    empty a character is worse than one that lets something through.

    Returns (kept_instances, kept_indices).
    """
    if len(instances) < 4:
        return list(instances), list(range(len(instances)))

    widths = np.array([i.get("advance_em", 0.0) for i in instances], dtype=float)
    seating = np.array([(i.get("bounds") or [0.0, 0.0, 0.0, 0.0])[3]
                        for i in instances], dtype=float)
    median_width = float(np.median(widths))
    median_seat = float(np.median(seating))
    if median_width <= 0:
        return list(instances), list(range(len(instances)))

    keep = (
        (widths >= MIN_PLAUSIBLE_WIDTH * median_width)
        & (widths <= MAX_PLAUSIBLE_WIDTH * median_width)
        & (np.abs(seating - median_seat) <= MAX_SEATING_DEVIATION)
    )
    if keep.sum() < max(4, int(0.5 * len(instances))):
        return list(instances), list(range(len(instances)))

    indices = [int(i) for i in np.flatnonzero(keep)]
    return [instances[i] for i in indices], indices


def reject_outliers(instances, keep_min=0.6, threshold=3.0):
    """
    Drop instances whose shape or size does not look like the others.

    Segmentation is not perfect: where two letters were written touching, the
    cut between them occasionally lands inside a letter and yields half an 'o'
    or a fragment of a 'Q'. Left in, these do not just add noise — k-means
    happily gives them a cluster of their own, and that cluster then becomes a
    "style", so the user is shown a set of broken letterforms and told it is
    the second way they write.

    Two filters run, because they catch different failures. Width catches bad
    splits directly: half a letter is about half as wide as the rest of its
    kind, however plausible its shape. Shape catches everything else. Both use
    a median-absolute-deviation threshold, which is not itself dragged out by
    the outliers it is meant to find.

    At least `keep_min` of the instances are always kept, so a character whose
    instances are simply varied is never gutted.

    Returns (kept_instances, kept_indices).
    """
    instances, plausible_idx = reject_implausible(instances)

    n = len(instances)
    if n < 4:
        return list(instances), list(plausible_idx)

    def _mad_keep(values, limit_scale, floor_fraction):
        values = np.asarray(values, dtype=float)
        centre = float(np.median(values))
        mad = float(np.median(np.abs(values - centre)))
        # A floor relative to the median matters as much as the MAD itself.
        # When a character is written very consistently the MAD collapses
        # toward zero, and a pure 3-MAD rule then rejects instances that differ
        # by a fraction of a percent — discarding good samples for being
        # ordinary. The floor keeps the filter aimed at genuine breakage
        # (a half-width fragment) rather than at natural variation.
        limit = max(limit_scale * mad, floor_fraction * abs(centre))
        if limit <= 1e-9:
            return np.ones(n, dtype=bool)
        return np.abs(values - centre) <= limit

    width_ok = _mad_keep([i.get("advance_em", 0.0) for i in instances],
                         threshold, 0.30)

    x = features.feature_matrix(instances)
    centre = np.median(x, axis=0)
    distance = np.linalg.norm(x - centre, axis=1)
    shape_ok = _mad_keep(distance, threshold, 0.60)

    keep = np.flatnonzero(width_ok & shape_ok)

    floor = int(np.ceil(keep_min * n))
    if len(keep) < floor:
        keep = np.argsort(distance)[:floor]

    keep = np.sort(keep)
    return [instances[i] for i in keep], [plausible_idx[int(i)] for i in keep]


def cluster_character(instances, seed=0):
    """
    Cluster one character's instances into modes.

    Returns a list of clusters, largest first, each
    ``{"size", "members", "medoid"}`` where members/medoid are indices into
    `instances`.
    """
    n = len(instances)
    if n == 0:
        return []

    kept, kept_idx = reject_outliers(instances)
    if len(kept) < MIN_INSTANCES_FOR_CLUSTERING:
        return [_single_cluster(kept, kept_idx)]

    x = features.feature_matrix(kept)
    n_components = int(min(8, len(kept) - 1, x.shape[1]))
    reduced = PCA(n_components=n_components, random_state=seed).fit_transform(x)

    # A mode has to account for a real share of how the character is written.
    # Without this floor, a couple of unusual instances become a "style".
    min_size = max(MIN_CLUSTER_SIZE, int(round(MIN_CLUSTER_FRACTION * len(kept))))

    best = None
    max_k = int(min(MAX_MODES_PER_CHAR, len(kept) // max(min_size, 1)))
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(reduced)
        labels = km.labels_
        if np.bincount(labels, minlength=k).min() < min_size:
            continue
        score = float(silhouette_score(reduced, labels))
        if score >= MIN_SILHOUETTE and (best is None or score > best[0]):
            best = (score, labels, km.cluster_centers_)

    if best is None:
        return [_single_cluster(kept, kept_idx)]

    _score, labels, centers = best
    clusters = []
    for k in range(centers.shape[0]):
        members = np.flatnonzero(labels == k)
        d = np.linalg.norm(reduced[members] - centers[k], axis=1)
        clusters.append({
            "size": int(len(members)),
            "members": [kept_idx[i] for i in members],
            "medoid": kept_idx[int(members[int(np.argmin(d))])],
        })
    clusters.sort(key=lambda c: -c["size"])
    return clusters


def _single_cluster(instances, indices=None):
    """One mode: the medoid is the instance closest to the median shape."""
    if indices is None:
        indices = list(range(len(instances)))
    x = features.feature_matrix(instances)
    centre = np.median(x, axis=0)
    d = np.linalg.norm(x - centre, axis=1)
    return {
        "size": len(instances),
        "members": list(indices),
        "medoid": indices[int(np.argmin(d))],
    }


def choose_style_count(clusters_by_char):
    """
    How many style sets to build.

    Taken from the data — the typical number of modes found per character —
    rather than fixed, then clamped to a range that stays legible in the UI.
    """
    counts = [len(c) for c in clusters_by_char.values() if c]
    if not counts:
        return MIN_STYLES
    typical = int(round(float(np.median(counts))))
    return int(np.clip(typical, MIN_STYLES, MAX_STYLES))


# --------------------------------------------------------------------------
# Drift modeling
# --------------------------------------------------------------------------

def fit_ou(series_list, phi_floor=PHI_FLOOR):
    """
    Fit an Ornstein-Uhlenbeck process to a set of residual series.

    The discrete form is ``r[t+1] = phi * r[t] + noise``, so a single
    regression through the origin over all pooled consecutive pairs gives
    phi, and the residual spread gives sigma.

    Returns theta (mean-reversion rate per step), sigma (noise per step) and
    std (the stationary standard deviation — the typical size of the wander).

    phi is clamped into [phi_floor, PHI_CEILING]. The ceiling keeps out a walk
    that never corrects, which is not what handwriting does. The floor is a
    deliberate prior rather than a measurement: the residuals are far too few,
    and far too contaminated by segmentation error, to pin down the
    autocorrelation, and an unlucky estimate of phi~0 would make the renderer
    emit per-step jitter — precisely the "random positional noise" the spec
    rules out. Amplitude is measured; smoothness is assumed. The right floor
    depends on what a step is, so callers pass their own.
    """
    prev, nxt = [], []
    for series in series_list:
        s = np.asarray(series, dtype=float)
        if s.size < 2:
            continue
        s = s - s.mean()
        prev.append(s[:-1])
        nxt.append(s[1:])

    if not prev:
        return {"theta": 0.5, "sigma": 0.0, "std": 0.0, "samples": 0}

    p = np.concatenate(prev)
    q = np.concatenate(nxt)
    denom = float(np.dot(p, p))
    phi = float(np.dot(p, q) / denom) if denom > 1e-12 else 0.0
    measured_phi = phi
    phi = float(np.clip(phi, phi_floor, PHI_CEILING))

    # Keep the observed amplitude even after clamping phi, so the wander is
    # as large as the writer's really is and only its smoothness is imposed.
    observed_std = float(np.std(np.concatenate([p, q])))
    sigma = observed_std * float(np.sqrt(max(1.0 - phi ** 2, 1e-6)))
    theta = float(-np.log(max(phi, 1e-6)))

    return {"theta": round(theta, 4), "sigma": round(sigma, 4),
            "std": round(observed_std, 4), "phi": round(phi, 4),
            "measured_phi": round(measured_phi, 4), "samples": int(p.size)}


def _baseline_series(instances, em_px):
    """
    Per-written-line baseline geometry: how each line sat relative to the
    ideal straight baseline.

    Only characters that actually touch the baseline are used. A 'g' or a
    comma sits below it by design, and including them would read as drift
    when it is just letterform.

    Residuals are then centred per character. Round letters like 'o' and 'e'
    are drawn with a slight overshoot below the baseline while 'x' and 'n' sit
    flat on it; without this correction that constant per-letter offset enters
    the series as high-frequency noise and inflates the fitted wander.

    Returns one record per line — its slope, its mean offset, and how much
    scatter is left around its own straight fit — rather than a raw
    per-character sequence. The shape of the output is the model: a hand
    leaves a line *tilted*, not rippled.
    """
    rows = [i for i in instances if i.get("on_baseline", True)]
    if not rows:
        return [], []

    bias = {}
    for inst in rows:
        dy = (inst["actual_xy"][1] - inst["expected_xy"][1]) / em_px
        bias.setdefault(inst["char"], []).append(dy)
    bias = {ch: float(np.median(v)) for ch, v in bias.items()}

    lines = {}
    for inst in rows:
        key = (inst.get("round", 0), inst["band"])
        dy = (inst["actual_xy"][1] - inst["expected_xy"][1]) / em_px
        lines.setdefault(key, []).append((inst["actual_xy"][0], dy - bias[inst["char"]]))

    records, points = [], []
    for key in sorted(lines):
        pts = sorted(lines[key])
        if len(pts) < MIN_LINE_SAMPLES:
            continue
        x0 = pts[0][0]
        xs = np.array([(p[0] - x0) / em_px for p in pts], dtype=float)
        ys = np.array([p[1] for p in pts], dtype=float)

        centred = xs - xs.mean()
        sxx = float((centred ** 2).sum())
        if sxx <= 1e-9:
            continue
        slope = float((centred * ys).sum() / sxx)
        offset = float(ys.mean())
        resid = ys - (offset + slope * centred)
        # Degrees of freedom: two are spent on the slope and the offset.
        scatter = float((resid ** 2).sum() / max(len(ys) - 2, 1))

        records.append({"slope": slope, "offset": offset, "scatter": scatter,
                        "n": len(ys), "sxx": sxx})
        if len(pts) > len(points):
            points = [[round(x, 3), round(y, 4)] for x, y in zip(xs, ys)]

    return records, points


def fit_baseline(records):
    """
    Fit how a hand's baseline actually behaves, with measurement error
    subtracted rather than reproduced.

    Two things are modelled, matching what writing on unruled paper does:

    * **Within a line, a slope.** A hand that starts drifting upward keeps
      going up until the line ends; it does not correct mid-line. So each
      line gets one tilt, not a wander.
    * **Between lines, correction.** Having finished a line high, a writer
      brings the next few lines back down. That is mean reversion on the
      line's offset, one step per *line*.

    Both amplitudes are corrected for measurement noise, which matters more
    than the shape does. Locating a baseline from segmented ink is only
    accurate to a fraction of an em, and that error is uncorrelated between
    characters. Fitted naively, it inflates the drift amplitude — and on a
    sheet with no drift at all it *is* the entire measurement, so the renderer
    would faithfully reproduce wander that the writer never made.

    The scatter left around each line's own straight fit estimates that noise
    directly. Its expected contribution is then subtracted from the observed
    spread of slopes and offsets, and whatever survives is real. When nothing
    survives, the amplitude is zero and the output is dead straight.
    """
    if len(records) < 2:
        return {"slope_std": 0.0, "offset_std": 0.0, "sigma": 0.0, "phi": 0.0,
                "std": 0.0, "noise_std": 0.0, "measured_phi": 0.0, "samples": 0}

    slopes = np.array([r["slope"] for r in records], dtype=float)
    offsets = np.array([r["offset"] for r in records], dtype=float)
    scatter = np.array([r["scatter"] for r in records], dtype=float)
    counts = np.array([r["n"] for r in records], dtype=float)
    sxx = np.array([r["sxx"] for r in records], dtype=float)

    noise_var = float(np.median(scatter))

    # What the observed spread would be if the writer were perfectly straight
    # and only measurement error moved these numbers.
    slope_noise_var = float(np.mean(scatter / np.maximum(sxx, 1e-9)))
    offset_noise_var = float(np.mean(scatter / np.maximum(counts, 1.0)))

    slope_var = max(float(np.var(slopes, ddof=1)) - slope_noise_var, 0.0)
    offset_var = max(float(np.var(offsets, ddof=1)) - offset_noise_var, 0.0)

    # Mean reversion between lines, measured on the noise-corrected variance.
    # Noise is uncorrelated line to line, so it does not bias the lag-1
    # autocovariance, only the variance it is divided by.
    centred = offsets - offsets.mean()
    measured_phi = 0.0
    if len(centred) > 2 and offset_var > 1e-9:
        autocov = float(np.dot(centred[:-1], centred[1:]) / (len(centred) - 1))
        measured_phi = autocov / offset_var
    phi = float(np.clip(measured_phi, PHI_FLOOR_MARGIN, PHI_CEILING))

    offset_std = float(np.sqrt(offset_var))
    sigma = offset_std * float(np.sqrt(max(1.0 - phi ** 2, 0.0)))
    if offset_std <= 0.0:
        phi, sigma = 0.0, 0.0

    return {
        "slope_std": round(float(np.sqrt(slope_var)), 5),
        "offset_std": round(offset_std, 4),
        "sigma": round(sigma, 4),
        "phi": round(phi, 4),
        "std": round(offset_std, 4),
        "noise_std": round(float(np.sqrt(noise_var)), 4),
        "measured_phi": round(measured_phi, 4),
        "samples": int(len(records)),
    }


def _margin_series(round_meta):
    """
    Horizontal residuals down the page: how far right or left of the intended
    left margin each written line actually started, in em.

    Taken from each round's per-band offsets rather than from character
    positions, because a wrapped continuation line starts mid-sentence and its
    first character says nothing about the margin.
    """
    series, points = [], []
    for meta in round_meta or []:
        offsets = meta.get("band_offsets") or []
        if len(offsets) < 4:
            continue
        values = [dx for _band, dx in offsets]
        series.append(values)
        if len(offsets) > len(points):
            points = [[int(band), round(float(dx), 4)] for band, dx in offsets]
    return series, points


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def cluster_and_fit_drift(all_instances, em_px=None, round_meta=None, seed=0):
    """
    Args:
        all_instances: concatenated `instances` lists from all three rounds'
            segment_photo() output. Each should carry a "round" key so that
            drift series are not stitched across sheets.
        em_px: pixels per em, from the capture metadata. Residuals are
            converted into em so the model is independent of photo
            resolution.
        round_meta: the three rounds' `meta` dicts, which carry the per-line
            margin offsets the horizontal drift fit needs.

    Returns:
        dict with:
            "char_clusters": {char: [cluster, ...]} largest mode first.
            "n_styles": how many style sets Stage 4 should build.
            "style_share": [{"id", "share"}] — each style's share of samples.
            "drift": fitted baseline and margin models, each with an OU fit
                and a measured series for display.
            "metrics": per-character advance width and slant statistics,
                measured from the writer's own hand.
    """
    by_char = {}
    for inst in all_instances:
        by_char.setdefault(inst["char"], []).append(inst)

    char_clusters = {ch: cluster_character(items, seed=seed)
                     for ch, items in by_char.items()}

    n_styles = choose_style_count(char_clusters)
    style_share = _style_shares(char_clusters, n_styles)

    if not em_px:
        em_px = 1.0

    baseline_records, baseline_points = _baseline_series(all_instances, em_px)
    margin_series, margin_points = _margin_series(round_meta)

    drift = {
        "baseline": {**fit_baseline(baseline_records),
                     "points": baseline_points,
                     "unit": "em", "axis": "vertical, within a line"},
        "margin": {**fit_ou(margin_series, PHI_FLOOR_MARGIN),
                   "points": margin_points,
                   "unit": "em", "axis": "horizontal, down the page"},
    }

    metrics = {}
    for ch, items in by_char.items():
        advances = [i["advance_em"] for i in items]
        slants = [i["slant"] for i in items]
        metrics[ch] = {
            "advance": round(float(np.median(advances)), 4),
            "slant_mean": round(float(np.mean(slants)), 3),
            "slant_std": round(float(np.std(slants)), 3),
            "count": len(items),
        }

    return {
        "char_clusters": char_clusters,
        "n_styles": n_styles,
        "style_share": style_share,
        "drift": drift,
        "metrics": metrics,
    }


def _style_shares(char_clusters, n_styles):
    """
    What fraction of the writer's samples each style set represents.

    Style i is built from each character's i-th largest mode. A character with
    fewer modes than there are styles reuses its largest, so shares are
    computed by summing the instance counts actually assigned to each style
    and normalizing — which keeps them summing to 1 without double-counting a
    reused mode as if it were new evidence.
    """
    weights = []
    for i in range(n_styles):
        total = 0
        for clusters in char_clusters.values():
            if not clusters:
                continue
            total += clusters[min(i, len(clusters) - 1)]["size"]
        weights.append(total)

    grand = float(sum(weights)) or 1.0
    return [{"id": i + 1, "share": round(w / grand, 4)} for i, w in enumerate(weights)]

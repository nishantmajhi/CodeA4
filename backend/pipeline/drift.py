"""
Drift synthesis for Stage 6.

Stage 3 fits how the writer's baseline and left margin wander and correct.
This module runs that model *forward* to generate the wander for a freshly
rendered page.

Two decisions matter here:

**Resample, don't replay.** The captured drift could simply be looped over the
output, but then every page would repeat the same wobble, and a reader
comparing two pages would see the identical wave twice. Sampling new draws
from the fitted process gives wander with the same amplitude and the same
smoothness, without the same shape.

**Deterministic.** The seed is derived from the session and the page, so the
same document re-rendered produces byte-identical output. Without that, a
preview and the PDF the user then downloads would not match, and no rendering
bug would ever be reproducible.
"""
import hashlib

import numpy as np


def seed_from(*parts):
    """A stable 32-bit seed from arbitrary identifying strings."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8"))
    return int.from_bytes(digest.digest()[:4], "big")


class DriftModel:
    """
    Generates mean-reverting offsets from a fitted OU model.

    `params` is one branch of the Stage 3 drift dict: it needs `phi` (how much
    of the previous offset carries over) and `sigma` (fresh noise per step),
    both in em.
    """

    def __init__(self, params, rng):
        params = params or {}
        self.phi = float(np.clip(params.get("phi", 0.6), 0.0, 0.99))
        self.sigma = max(float(params.get("sigma", 0.0)), 0.0)
        self.std = float(params.get("std", 0.0))
        self._rng = rng

    def walk(self, n, start=None):
        """
        Return `n` successive offsets, in em.

        The walk is started from a draw from the stationary distribution
        rather than from zero, so a page does not always begin perfectly on
        the line and wander off — which would look like a rule the writer was
        obeying at the start of every page.
        """
        if n <= 0:
            return np.zeros(0)
        if self.sigma <= 0 and self.std <= 0:
            return np.zeros(n)

        value = self._rng.normal(0.0, self.std) if start is None else float(start)
        out = np.empty(n)
        for i in range(n):
            value = self.phi * value + self._rng.normal(0.0, self.sigma)
            out[i] = value
        return out

    def last_value(self, values):
        return float(values[-1]) if len(values) else 0.0


class BaselineModel:
    """
    Generates the baseline geometry for successive written lines.

    A line gets a single tilt and a single offset — no within-line wander.
    Writing on unruled paper drifts *monotonically* across a line: a hand that
    starts rising keeps rising until the line ends, because there is nothing
    to correct against until it comes back to the left margin and can see the
    line above. Running a mean-reverting walk along the characters instead
    produces several reversals per line, which reads as a ripple no hand
    makes.

    Correction happens between lines, not within them: `phi` pulls each line's
    offset back toward the page's mean, so a line that ended high is followed
    by lines that come gradually back down.
    """

    def __init__(self, params, rng):
        params = params or {}
        self.phi = float(np.clip(params.get("phi", 0.0), 0.0, 0.99))
        self.sigma = max(float(params.get("sigma", 0.0)), 0.0)
        self.offset_std = max(float(params.get("offset_std",
                                               params.get("std", 0.0))), 0.0)
        self.slope_std = max(float(params.get("slope_std", 0.0)), 0.0)
        self._rng = rng
        self._offset = (self._rng.normal(0.0, self.offset_std)
                        if self.offset_std > 0 else 0.0)

    def next_line(self):
        """Return (offset, slope) in em for the next written line."""
        if self.offset_std <= 0 and self.slope_std <= 0:
            return 0.0, 0.0
        if self.offset_std > 0:
            self._offset = self.phi * self._offset + self._rng.normal(0.0, self.sigma)
        slope = self._rng.normal(0.0, self.slope_std) if self.slope_std > 0 else 0.0
        return float(self._offset), float(slope)


def make_rng(*seed_parts):
    return np.random.default_rng(seed_from(*seed_parts))

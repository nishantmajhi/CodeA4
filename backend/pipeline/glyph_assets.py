"""
Spec Stage 4 — Style asset preparation

Turns Stage 3's clusters into N *internally consistent* glyph sets. "Style 2"
means: for every character, the way this writer forms it in their second-most
common mode. Consistency is the whole point — mixing a mode-1 'a' with a
mode-3 'e' would produce a hand nobody actually has, which is exactly the
"one glyph plus jitter" failure the spec rejects in §2.

Per spec §5 nothing here produces a font file. A glyph is an SVG path plus the
few numbers needed to set it: how far to advance afterwards, how far it
overshoots the baseline, and how much this writer slants it.

Not every character can be observed. The exercise is authored to cover all of
printable ASCII, but a source file may still contain something outside it (a
tab, a non-ASCII identifier). ``resolve_glyph`` handles that with an explicit,
inspectable fallback chain rather than silently dropping the character.
"""

# Characters that can stand in for one another when a glyph is missing,
# preferring something that keeps the code readable over something that merely
# fills space.
_SUBSTITUTES = {
    "\t": " ",
    "‘": "'", "’": "'", "“": "\"", "”": "\"",
    "–": "-", "—": "-", "…": ".",
    "×": "*", "÷": "/", "≤": "<", "≥": ">", "≠": "!",
}

# When a glyph is missing entirely and has no substitute, this is drawn
# instead so the character's absence is visible rather than silent.
FALLBACK_CHAR = "?"


def build_glyph_sets(clustered, instances):
    """
    Args:
        clustered: the dict returned by clustering.cluster_and_fit_drift().
        instances: the same pooled instance list that was clustered — the
            cluster records index into it per character.

    Returns:
        list[dict]: one entry per style —
            {"id", "share", "glyphs": {char: glyph}, "coverage"}
        where glyph is
            {"path", "advance", "slant", "bounds", "instances"}
        `path` is in em units with the baseline at y=0 (see pipeline.vectorize),
        `advance` is how far the pen moves after drawing it, and `instances`
        is how many samples backed the mode this glyph came from.
    """
    by_char = {}
    for inst in instances:
        by_char.setdefault(inst["char"], []).append(inst)

    char_clusters = clustered["char_clusters"]
    metrics = clustered.get("metrics", {})
    n_styles = clustered["n_styles"]
    shares = {s["id"]: s["share"] for s in clustered["style_share"]}

    style_sets = []
    for style_index in range(n_styles):
        glyphs = {}
        for char, clusters in char_clusters.items():
            if not clusters:
                continue
            # A character with fewer modes than there are styles reuses its
            # most common one. That is honest: this writer simply does not
            # have a third way of drawing that letter.
            cluster = clusters[min(style_index, len(clusters) - 1)]
            source = by_char[char][cluster["medoid"]]
            stats = metrics.get(char, {})

            glyphs[char] = {
                "path": source["path"],
                # Advance is the mode's own ink width plus this writer's
                # measured letter gap, not the median across all modes: a
                # wider way of forming a letter genuinely takes more room.
                "advance": round(source["advance_em"], 4),
                "slant": stats.get("slant_mean", source["slant"]),
                "slant_std": stats.get("slant_std", 0.0),
                "bounds": source["bounds"],
                "instances": cluster["size"],
            }

        style_sets.append({
            "id": style_index + 1,
            "share": shares.get(style_index + 1, 0.0),
            "glyphs": glyphs,
            "coverage": len(glyphs),
        })

    return style_sets


def resolve_glyph(glyphs, char):
    """
    Find the best glyph for a character, following an explicit fallback chain:
    the character itself, a typographic substitute, the other case, then a
    visible placeholder.

    Returns (glyph, resolved_char, exact) so callers can report how much of a
    file was rendered from genuinely observed letterforms.
    """
    if char in glyphs:
        return glyphs[char], char, True

    substitute = _SUBSTITUTES.get(char)
    if substitute and substitute in glyphs:
        return glyphs[substitute], substitute, False

    for variant in (char.lower(), char.upper()):
        if variant != char and variant in glyphs:
            return glyphs[variant], variant, False

    if FALLBACK_CHAR in glyphs:
        return glyphs[FALLBACK_CHAR], FALLBACK_CHAR, False

    return None, char, False


def coverage_summary(style_sets, text):
    """
    How much of `text` a style set can render from observed letterforms.
    Used to warn the user when an uploaded file leans on substitutes.
    """
    if not style_sets:
        return {"total": 0, "exact": 0, "missing": []}

    glyphs = style_sets[0]["glyphs"]
    total = exact = 0
    missing = set()
    for char in text:
        if char in (" ", "\t", "\n"):
            continue
        total += 1
        _glyph, _resolved, is_exact = resolve_glyph(glyphs, char)
        if is_exact:
            exact += 1
        else:
            missing.add(char)

    return {"total": total, "exact": exact, "missing": sorted(missing)}

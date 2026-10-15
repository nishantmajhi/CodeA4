"""
Whether the style sets are what the product says they are.

The claim in the README's first paragraph is that output should look
*written* rather than typeset, and the mechanism for that is Stage 3: a hand
has a few natural ways of forming each letter, and clustering finds them. Two
things have to hold for a style set to mean anything.

It has to correspond to a real mode -- style 2 must be a different letterform,
not a different *quality* of extraction. Clustering always returns clusters;
it will separate the well-segmented instances from the badly-segmented ones
just as happily as it separates two ways of writing an 'a', and from the
outside the two results are indistinguishable. The only way to tell them apart
is to ask what the writer actually wrote, which on a synthetic sheet is
recorded.

And every letterform in it has to be a letterform. A cluster made of
mis-segmented fragments becomes a style, and the Style step then shows the
user a broken alphabet and tells them it is 40% of how they write.
"""
import numpy as np
import pytest

from pipeline import clustering, features
from pipeline.glyph_assets import build_glyph_sets


def _by_char(instances):
    out = {}
    for instance in instances:
        out.setdefault(instance["char"], []).append(instance)
    return out


def _cluster_purity(members, modes):
    """What fraction of a cluster came from a single true mode."""
    known = [modes[i] for i in members if modes[i] is not None]
    if len(known) < 3:
        return None
    _values, counts = np.unique(known, return_counts=True)
    return float(counts.max() / counts.sum())


# ------------------------------------------------------- does it find modes?

def test_clustering_recovers_the_modes_the_writer_actually_used(
        labelled_instances):
    """
    The headline claim, tested against what was written rather than against
    itself.

    Each sheet draws every character from one of three faces with fixed
    probabilities, so a cluster that corresponds to a real mode is mostly one
    face, while a cluster of noise is the same mixture as the population. The
    obvious test -- "are the style sets different from each other" -- cannot
    tell those apart, because clusters of noise differ from each other too.

    The floor is a lift over chance rather than an absolute purity, because
    chance is not one in three: the modes are drawn 50/32/18, so a cluster
    assembled at random is already about half one face.
    """
    by_char = _by_char(labelled_instances)

    purities, chance_levels = [], []
    for _char, items in by_char.items():
        modes = [i.get("true_mode") for i in items]
        if sum(m is not None for m in modes) < 20:
            continue
        clusters = clustering.cluster_character(items)
        if len(clusters) < 2:
            continue
        for cluster in clusters:
            purity = _cluster_purity(cluster["members"], modes)
            if purity is not None:
                purities.append(purity)
        known = [m for m in modes if m is not None]
        _values, counts = np.unique(known, return_counts=True)
        chance_levels.append(float(counts.max() / counts.sum()))

    assert len(chance_levels) >= 10, "too few multi-mode characters to judge"
    purity = float(np.mean(purities))
    chance = float(np.mean(chance_levels))
    assert purity - chance >= 0.12, (
        f"clusters are {purity:.3f} pure against {chance:.3f} by chance - "
        "they are separating something other than letterform. If the pipeline "
        "has not changed, check the fixture's three faces in "
        "tools/make_sample_sheet.py: they were once three humanist sans "
        "designs that differ by less than the generator's own per-instance "
        "wobble, which left clustering nothing to recover.")
    assert purity >= 0.65


# -------------------------------------------------- is every letterform real?

def test_no_style_ships_a_letterform_that_cannot_be_that_character(
        labelled_instances):
    """
    Every glyph a style ships is one extracted instance, chosen as its
    cluster's medoid. Nothing used to check that the chosen instance was a
    plausible example of its character, so a cluster of badly-cut instances
    put a half-width 'n' and a 'd' with no bowl into a style set that the
    Style step presents as "41% of your samples".
    """
    by_char = _by_char(labelled_instances)
    clustered = clustering.cluster_and_fit_drift(labelled_instances)
    style_sets = build_glyph_sets(clustered, labelled_instances)

    broken = []
    for style in style_sets:
        for char, glyph in style["glyphs"].items():
            population = np.array([i["advance_em"] for i in by_char[char]])
            seating = np.array([i["bounds"][3] for i in by_char[char]])
            width_ratio = glyph["advance"] / max(float(np.median(population)), 1e-9)
            seat_error = abs(glyph["bounds"][3] - float(np.median(seating)))
            too_odd = not (clustering.MIN_PLAUSIBLE_WIDTH <= width_ratio
                           <= clustering.MAX_PLAUSIBLE_WIDTH)
            if too_odd or seat_error > clustering.MAX_SEATING_DEVIATION:
                broken.append((style["id"], char, round(width_ratio, 2),
                               round(seat_error, 3)))

    assert not broken, f"styles ship letterforms that are not the letter: {broken}"


def test_every_style_covers_the_whole_exercise_alphabet(labelled_instances):
    """Discarding implausible instances must not discard a character: a style
    missing a letter falls back to a substitute glyph on every page."""
    clustered = clustering.cluster_and_fit_drift(labelled_instances)
    style_sets = build_glyph_sets(clustered, labelled_instances)
    written = {i["char"] for i in labelled_instances}
    for style in style_sets:
        assert set(style["glyphs"]) == written, (
            f"style {style['id']} is missing "
            f"{sorted(written - set(style['glyphs']))}")


def test_style_shares_sum_to_one(labelled_instances):
    clustered = clustering.cluster_and_fit_drift(labelled_instances)
    total = sum(s["share"] for s in clustered["style_share"])
    assert total == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------- the filter, on its own

def _instance(advance, seat=0.0, bitmap_width=8):
    bitmap = np.zeros((16, 16), dtype=np.uint8)
    bitmap[4:12, 3:3 + bitmap_width] = 255
    return {"char": "n", "advance_em": advance, "aspect": 0.6, "density": 0.4,
            "bounds": [0.0, -0.5, advance, seat],
            "bitmap": features.encode_bitmap(bitmap)}


def test_a_half_width_instance_is_not_a_letter():
    items = [_instance(0.40 + 0.01 * (i % 3)) for i in range(12)]
    items.append(_instance(0.20))
    kept, indices = clustering.reject_implausible(items)
    assert 12 not in indices and len(kept) == 12


def test_an_instance_that_swallowed_its_neighbour_is_not_a_letter():
    items = [_instance(0.40 + 0.01 * (i % 3)) for i in range(12)]
    items.append(_instance(0.95))
    kept, indices = clustering.reject_implausible(items)
    assert 12 not in indices and len(kept) == 12


def test_an_instance_floating_off_the_baseline_is_not_a_letter():
    items = [_instance(0.40) for _ in range(12)]
    items.append(_instance(0.40, seat=-0.45))
    kept, indices = clustering.reject_implausible(items)
    assert 12 not in indices and len(kept) == 12


def test_the_filter_never_empties_a_character():
    """A character whose instances genuinely vary must survive. A filter that
    can delete a letter is worse than one that lets something through."""
    items = [_instance(0.2 + 0.08 * i) for i in range(10)]
    kept, indices = clustering.reject_implausible(items)
    assert len(kept) >= 5 and len(kept) == len(indices)


def test_the_filter_leaves_a_consistent_character_alone():
    items = [_instance(0.40 + 0.005 * (i % 4)) for i in range(20)]
    kept, _indices = clustering.reject_implausible(items)
    assert len(kept) == 20


def test_indices_still_point_at_the_list_they_came_from():
    """`cluster_character` returns indices into the list it was given, and
    `build_glyph_sets` looks the medoid up in that same list. An off-by-one
    here ships a different character's letterform under this character's
    name -- which is silent, because every index is still valid."""
    items = [_instance(0.40) for _ in range(8)]
    items.insert(3, _instance(0.15))
    kept, indices = clustering.reject_implausible(items)
    assert [items[i] for i in indices] == kept
    assert 3 not in indices

    for cluster in clustering.cluster_character(items):
        assert cluster["medoid"] in cluster["members"]
        for member in cluster["members"]:
            assert 0 <= member < len(items)
            assert items[member]["advance_em"] >= 0.2

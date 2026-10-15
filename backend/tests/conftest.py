"""
Shared fixtures.

Tests run against synthetic sheets from tools/make_sample_sheet.py rather than
real photographs. That is a deliberate trade-off: it means the CV stages are
exercised on input whose ground truth is known exactly, so accuracy can be
asserted rather than eyeballed, at the cost of not covering the messiness of
real paper and real pens. See tests/test_segmentation.py for what that does
and does not prove.

Sheets are generated once per test session because generating and segmenting
one takes a couple of seconds.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("INLINE_JOBS", "1")
os.environ.setdefault("MEMORY_STATE", "1")


@pytest.fixture(scope="session")
def sheets(tmp_path_factory):
    """Three synthetic capture sheets and their ground truth."""
    from tools.make_sample_sheet import make_sheet

    directory = tmp_path_factory.mktemp("sheets")
    out = []
    for seed in (1, 2, 3):
        image = directory / f"sheet{seed}.jpg"
        truth = directory / f"truth{seed}.json"
        make_sheet(str(image), seed=seed, truth_path=str(truth))
        out.append((str(image), str(truth)))
    return out


@pytest.fixture(scope="session")
def segmented(sheets):
    """Stage 1+2 output for all three sheets."""
    from pipeline.segmentation import segment_photo

    results = []
    for index, (image, _truth) in enumerate(sheets, start=1):
        result = segment_photo(image)
        for instance in result["instances"]:
            instance["round"] = index
        results.append(result)
    return results


@pytest.fixture(scope="session")
def labelled_instances(sheets, segmented):
    """
    Every extracted instance, tagged with the mode the fixture actually drew
    it in, where the two can be matched.

    The generator records one truth entry per glyph it draws, including which
    of its three faces it used. Matching those back to extracted instances is
    what lets a test ask whether clustering found the writer's real letterform
    variants rather than only whether it found *something*.

    An instance whose label disagrees with the truth record in the same
    position is left untagged rather than guessed at: a mis-segmented
    character's mode is not knowable.
    """
    import json

    by_round = {}
    for round_n, (_image, truth_path) in enumerate(sheets, start=1):
        with open(truth_path) as f:
            records = json.load(f)
        lines = {}
        for record in records:
            lines.setdefault(record["written_line"], []).append(record)
        for entries in lines.values():
            entries.sort(key=lambda r: r["x"])
        by_round[round_n] = lines

    instances = []
    for round_n, result in enumerate(segmented, start=1):
        bands = {}
        for instance in result["instances"]:
            bands.setdefault(instance["band"], []).append(instance)
        for band, found in bands.items():
            found.sort(key=lambda i: i["actual_xy"][0])
            for instance, record in zip(found, by_round[round_n].get(band, [])):
                instance["true_mode"] = (record["mode"]
                                         if instance["char"] == record["char"]
                                         else None)
            instances.extend(found)
    return instances


@pytest.fixture(scope="session")
def style_bundle(segmented):
    """Stage 3+4 output: clusters, drift and glyph sets."""
    from pipeline.clustering import cluster_and_fit_drift
    from pipeline.glyph_assets import build_glyph_sets

    instances = [i for r in segmented for i in r["instances"]]
    metas = [r["meta"] for r in segmented]
    clustered = cluster_and_fit_drift(instances, em_px=metas[0]["em_px"],
                                      round_meta=metas)
    style_sets = build_glyph_sets(clustered, instances)
    meta = {
        "x_height_mm": metas[0]["x_height_mm"],
        "letter_gap_em": metas[0]["letter_gap_em"],
        "word_gap_em": metas[0]["word_gap_em"],
        "line_spacing_em": metas[0]["line_spacing_em"],
    }
    return {"clustered": clustered, "style_sets": style_sets, "meta": meta,
            "instances": instances}


@pytest.fixture
def client():
    """Flask test client with jobs running inline and state in memory."""
    import models
    from app import create_app

    models.clear_memory()
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client

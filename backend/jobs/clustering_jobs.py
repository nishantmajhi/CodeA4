"""RQ job: cluster styles, fit drift, build glyph assets (Stage 3+4)."""
import json
import logging

import models
import push_service
import storage
from extensions import get_queue
from pipeline.clustering import cluster_and_fit_drift
from pipeline.glyph_assets import build_glyph_sets

log = logging.getLogger(__name__)


def enqueue_clustering(session_id):
    job = models.create_job(session_id, "clustering")
    get_queue().enqueue(clustering_job, job["job_id"], session_id)
    return job


def clustering_job(job_id, session_id):
    models.set_progress(job_id, 15, stage="Pooling samples")

    try:
        all_instances = []
        round_meta = []
        for n in range(1, 4):
            with open(storage.round_chars_path(session_id, n)) as f:
                payload = json.load(f)
            for inst in payload["instances"]:
                # Tag the round so drift series are never stitched across
                # sheets — the pen lifts between them, so consecutive lines on
                # different sheets are not consecutive in time.
                inst["round"] = n
                all_instances.append(inst)
            round_meta.append(payload["meta"])

        models.set_progress(job_id, 40, stage="Finding your letter variants")
        em_px = round_meta[0].get("em_px")
        clustered = cluster_and_fit_drift(all_instances, em_px=em_px,
                                          round_meta=round_meta)

        models.set_progress(job_id, 75, stage="Building glyph sets")
        style_sets = build_glyph_sets(clustered, all_instances)

        # Capture measurements the renderer needs: how large this person
        # writes, and how they space letters, words and lines.
        meta = _merge_meta(round_meta)

        asset = models.save_style_asset(session_id, style_sets,
                                        clustered["drift"], meta)
        with open(storage.style_assets_path(session_id), "w") as f:
            json.dump(asset, f)

        models.mark_style_ready(session_id, True)
        models.update_job(job_id, status="done", progress=100, stage="Done",
                          result={"n_styles": clustered["n_styles"],
                                  "style_share": clustered["style_share"]})

        push_service.notify(session_id, "Your handwriting is ready",
                            "We found your natural letter variants. "
                            "Tap to review them.", url="/#style")

    except Exception:
        log.exception("clustering failed for session %s", session_id)
        message = ("We could not build a handwriting model from these sheets. "
                   "Try recapturing them in even light.")
        models.update_job(job_id, status="failed", error=message, stage=None)
        push_service.notify(session_id, "Handwriting model failed", message)
        raise


def _merge_meta(round_meta):
    """
    Combine the three sheets' measurements into one.

    Medians rather than means: one sheet photographed at an awkward angle
    should not drag the writing size for every page that follows.
    """
    import statistics

    def median_of(key, default):
        values = [m[key] for m in round_meta if m.get(key) is not None]
        return statistics.median(values) if values else default

    return {
        "x_height_mm": median_of("x_height_mm", 3.0),
        "letter_gap_em": median_of("letter_gap_em", 0.06),
        "word_gap_em": median_of("word_gap_em", 0.30),
        "line_spacing_em": median_of("line_spacing_em", 1.4),
    }

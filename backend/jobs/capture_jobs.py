"""RQ job: process one capture round's photo (Stage 1+2)."""
import json
import logging

import models
import push_service
import storage
from jobs.retention import sweep_old_data
from pipeline.imaging import CaptureError
from pipeline.segmentation import segment_photo

log = logging.getLogger(__name__)


def capture_round_job(job_id, session_id, round_n, photo_path):
    models.set_progress(job_id, 10, stage="Reading the sheet")
    models.update_round(session_id, round_n, status="processing", job_id=job_id)

    try:
        result = segment_photo(photo_path)
        models.set_progress(job_id, 80, stage="Extracting characters")

        with open(storage.round_chars_path(session_id, round_n), "w") as f:
            json.dump(result, f)

        # The full instance list is large — hundreds of glyph outlines — and
        # the capture screen only shows a count, so the job result carries the
        # summary and the detail stays on disk for the clustering job.
        summary = {
            "round": round_n,
            "meta": result["meta"],
            "instances": len(result["instances"]),
        }
        models.update_job(job_id, status="done", progress=100,
                          stage="Done", result=summary)
        models.update_round(session_id, round_n, status="done", error=None,
                            instances=len(result["instances"]))

        session = models.get_session(session_id)
        if session and all(r["status"] == "done" for r in session["rounds"]):
            from jobs.clustering_jobs import enqueue_clustering
            enqueue_clustering(session_id)
        else:
            push_service.notify(
                session_id, "Sheet accepted",
                f"Round {round_n} of 3 processed. Ready for the next sheet.",
            )

        sweep_old_data()

    except CaptureError as exc:
        # An expected, user-fixable problem: the message is written for them,
        # so it is surfaced as-is and the round is left retryable.
        message = str(exc)
        models.update_job(job_id, status="failed", error=message, stage=None)
        models.update_round(session_id, round_n, status="failed", error=message)
        push_service.notify(session_id, "Sheet needs retaking", message)

    except Exception:
        log.exception("capture round %s failed for session %s", round_n, session_id)
        message = ("Something went wrong processing this sheet. Please try "
                   "taking the photo again.")
        models.update_job(job_id, status="failed", error=message, stage=None)
        models.update_round(session_id, round_n, status="failed", error=message)
        raise

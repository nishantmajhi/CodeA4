from flask import Blueprint, abort, jsonify

import models

jobs_bp = Blueprint("jobs", __name__)

# Fields that exist for the server's benefit and would either leak filesystem
# layout or bloat every poll response.
_PRIVATE_RESULT_FIELDS = ("pdf_path",)


@jobs_bp.get("/jobs/<job_id>")
def get_job(job_id):
    """
    Generic poll endpoint, reused for capture, clustering and render jobs
    alike — see Backend Plan §3.
    """
    job = models.get_job(job_id)
    if not job:
        abort(404, description="Unknown job")

    result = job.get("result")
    if isinstance(result, dict):
        result = {k: v for k, v in result.items()
                  if k not in _PRIVATE_RESULT_FIELDS}
        # Tell the client whether a PDF exists without telling it where.
        if "pdf_error" in result:
            result["pdf_available"] = job["result"].get("pdf_path") is not None

    return jsonify({
        "job_id": job["job_id"],
        "type": job["type"],
        "status": job["status"],
        "progress": job["progress"],
        "stage": job.get("stage"),
        "error": job["error"],
        "result": result,
    })

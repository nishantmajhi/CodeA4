import os

from flask import Blueprint, Response, abort, jsonify, request, send_file
from werkzeug.utils import secure_filename

import models
import storage
from extensions import get_queue
from jobs.render_jobs import render_job

render_bp = Blueprint("render", __name__)


@render_bp.post("/sessions/<session_id>/render")
def start_render(session_id):
    """UI: the Add Code step's "Render preview" button."""
    session = models.get_session(session_id)
    if not session:
        abort(404, description="Unknown session")
    if not session.get("style_ready"):
        abort(409, description="Your handwriting model isn't ready yet")

    uploaded = request.files.getlist("files")
    if not uploaded:
        abort(400, description="No files uploaded under 'files'")

    job = models.create_job(session_id, "render")

    # Save uploads before enqueueing: the worker is a separate process and
    # cannot read the request's file handles.
    saved = []
    for index, uploaded_file in enumerate(uploaded):
        original = uploaded_file.filename or "untitled.txt"
        # The original name is kept for the handwritten header, but only a
        # sanitized version is ever used as a path — an uploaded name like
        # "../../etc/passwd" must not decide where bytes land on disk.
        # Sanitizing is not the same as making unique, so each upload also
        # gets a directory of its own; see storage.job_upload_path.
        safe = secure_filename(original) or "untitled.txt"
        dest = storage.job_upload_path(job["job_id"], index, safe)
        uploaded_file.save(dest)
        saved.append((original, dest))

    get_queue().enqueue(render_job, job["job_id"], session_id, saved)
    return jsonify({"job_id": job["job_id"]}), 202


@render_bp.get("/render/<job_id>/pages/<int:index>.svg")
def page_svg(job_id, index):
    """
    One rendered page, as SVG.

    Pages are fetched individually rather than embedded in the job result: a
    page of handwriting is a few hundred kilobytes of vector outlines, so a
    twenty-page render would otherwise make every status poll drag megabytes
    of SVG the user may never scroll to.
    """
    job = models.get_job(job_id)
    if not job:
        abort(404, description="Unknown job")

    path = storage.job_page_svg_path(job_id, index)
    if not os.path.exists(path):
        abort(404, description="No such page")

    with open(path) as f:
        return Response(f.read(), mimetype="image/svg+xml")


@render_bp.get("/render/<job_id>/pdf")
def download_pdf(job_id):
    """UI: the Export step's "Download PDF" button."""
    job = models.get_job(job_id)
    if not job:
        abort(404, description="Unknown job")
    if job["status"] != "done":
        abort(409, description=f"Job status is '{job['status']}', not 'done'")

    result = job.get("result") or {}
    pdf_path = result.get("pdf_path")
    if not pdf_path or not os.path.exists(pdf_path):
        abort(409, description=result.get("pdf_error")
              or "This render has no PDF available")

    return send_file(pdf_path, mimetype="application/pdf", as_attachment=True,
                     download_name="codea4-output.pdf")

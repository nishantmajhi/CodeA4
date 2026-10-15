import os

from flask import Blueprint, jsonify, request, abort

import models
import storage
from extensions import get_queue
from jobs.capture_jobs import capture_round_job

sessions_bp = Blueprint("sessions", __name__)

# The extension is the only part of the client's filename that is kept, and it
# decides what ends up on disk. An allowlist keeps that to the formats a phone
# camera or a scanner actually produces; anything else is a sign the upload is
# not the photograph the capture step asked for.
PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".bmp"}
DEFAULT_PHOTO_EXTENSION = ".jpg"


@sessions_bp.post("/sessions")
def create_session():
    """UI: loading the Capture step for the first time."""
    session = models.create_session()
    return jsonify(session), 201


@sessions_bp.get("/sessions/<session_id>")
def get_session(session_id):
    """UI: Capture step's rounds table."""
    session = models.get_session(session_id)
    if not session:
        abort(404, description="Unknown session")
    return jsonify(session)


@sessions_bp.delete("/sessions/<session_id>")
def delete_session(session_id):
    """
    Forget everything about a session, now rather than when it expires.

    The photographs show a person's handwriting and the model built from them
    reproduces it convincingly enough to sign with. There is no account system
    to delete from, so this is the only way to withdraw that data on purpose,
    and it is the reason the API has a DELETE at all.
    """
    existed, job_ids = models.delete_session(session_id)
    removed_files = storage.delete_session_files(session_id)
    for job_id in job_ids:
        # The uploaded source files and the pages rendered from them.
        removed_files = storage.delete_job_files(job_id) or removed_files
    if not existed and not removed_files:
        abort(404, description="Unknown session")
    return jsonify({"deleted": True, "session_id": session_id,
                    "jobs_deleted": len(job_ids)})


@sessions_bp.post("/sessions/<session_id>/rounds/<int:n>/photo")
def upload_round_photo(session_id, n):
    """UI: "Use camera — capture sheet" button."""
    session = models.get_session(session_id)
    if not session:
        abort(404, description="Unknown session")
    if n not in (1, 2, 3):
        abort(400, description="Round must be 1, 2, or 3")

    if "photo" not in request.files:
        abort(400, description="Missing 'photo' file in form data")
    photo = request.files["photo"]
    # A camera capture can arrive with no filename at all, which is not an
    # error — it is a photograph with no name, and JPEG is what the camera
    # gave us.
    ext = os.path.splitext(photo.filename or "")[1].lower()
    if not ext:
        ext = DEFAULT_PHOTO_EXTENSION
    if ext not in PHOTO_EXTENSIONS:
        abort(400, description=(
            "That does not look like a photo. Upload a JPEG, PNG, WebP or "
            "HEIC image of the sheet."))

    # A round holds one photo. Without this, retaking a sheet under a
    # different extension leaves the previous photograph on disk for good.
    storage.clear_round_photos(session_id, n)
    saved_path = storage.round_photo_path(session_id, n, ext)
    photo.save(saved_path)

    job = models.create_job(session_id, "capture_round")
    get_queue().enqueue(capture_round_job, job["job_id"], session_id, n, saved_path)

    return jsonify({"job_id": job["job_id"]}), 202

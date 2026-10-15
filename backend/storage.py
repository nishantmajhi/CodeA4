"""
Filesystem path helpers for the data/ layout described in Backend Plan §5.

    data/
      sessions/<session_id>/rounds/<n>/photo.<ext>
      sessions/<session_id>/rounds/<n>/chars.json
      sessions/<session_id>/style/assets.json
      jobs/<job_id>/render/<file>.pages.json
      jobs/<job_id>/render/output.pdf

Centralizing paths here means switching to object storage (S3-compatible)
later is a change to this file only, not to every route/job that touches
disk.
"""
import os
import shutil
import time

from config import Config


def _ensure(path):
    os.makedirs(path, exist_ok=True)
    return path


def session_dir(session_id):
    return _ensure(os.path.join(Config.DATA_DIR, "sessions", session_id))


def round_dir(session_id, n):
    return _ensure(os.path.join(session_dir(session_id), "rounds", str(n)))


def round_photo_path(session_id, n, ext):
    return os.path.join(round_dir(session_id, n), f"photo{ext}")


def round_chars_path(session_id, n):
    return os.path.join(round_dir(session_id, n), "chars.json")


def style_dir(session_id):
    return _ensure(os.path.join(session_dir(session_id), "style"))


def style_assets_path(session_id):
    return os.path.join(style_dir(session_id), "assets.json")


def job_render_dir(job_id):
    return _ensure(os.path.join(Config.DATA_DIR, "jobs", job_id, "render"))


def render_pdf_path(job_id):
    return os.path.join(job_render_dir(job_id), "output.pdf")


def job_pages_dir(job_id):
    return _ensure(os.path.join(job_render_dir(job_id), "pages"))


def job_page_svg_path(job_id, index):
    return os.path.join(job_pages_dir(job_id), f"{index}.svg")


def job_upload_dir(job_id):
    return _ensure(os.path.join(Config.DATA_DIR, "jobs", job_id, "uploads"))


def job_upload_path(job_id, index, filename):
    """
    Where one uploaded source file is saved.

    Each upload gets a numbered directory of its own, because sanitizing a
    filename makes it safe without making it unique. `secure_filename` strips
    non-ASCII and replaces separators, so two different names routinely reduce
    to the same string -- and two files picked from different folders arrive
    under the same name to begin with. Sharing one directory meant the later
    upload silently replaced the earlier one on disk, and the earlier file's
    pages were then rendered from the later file's contents under the earlier
    file's name.
    """
    directory = _ensure(os.path.join(job_upload_dir(job_id), str(index)))
    return os.path.join(directory, filename)


def clear_round_photos(session_id, n):
    """
    Remove any photo already stored for this capture round.

    A retake under a different extension would otherwise leave the previous
    one behind indefinitely. These are photographs of a person's handwriting,
    and a round holds exactly one: the one they last sent.
    """
    directory = round_dir(session_id, n)
    for name in os.listdir(directory):
        if name.startswith("photo."):
            try:
                os.remove(os.path.join(directory, name))
            except OSError:
                # A file we cannot delete is not a reason to refuse the upload.
                pass


# --------------------------------------------------------------------------
# Retention
# --------------------------------------------------------------------------
#
# Expiring the Redis record makes a session unreachable; it does not touch a
# byte on disk. Without a sweep the photographs, the extracted characters, the
# style model and every rendered page stayed for as long as the disk did,
# belonging to a session id nobody could use any more. These are photographs
# of somebody's hand and a model that reproduces it convincingly enough to
# sign with, so "unreachable" is not the same as "gone".


def _is_safe_segment(name):
    """
    Whether `name` can be used as a directory name under DATA_DIR.

    Session and job identifiers become path segments, so anything that is not
    a plain name — empty, a dot, a separator — has to be refused rather than
    resolved.
    """
    return bool(name) and name not in (".", "..") and         "/" not in name and "\\" not in name and not name.startswith(".")


def _newest_mtime(path):
    """
    The most recent modification time anywhere under `path`.

    Age is measured from the newest thing inside rather than from the
    directory itself: capture is deliberately spread over three sheets with
    breaks in between, so a live session's directory can be old while the
    session is very much in use.
    """
    newest = os.path.getmtime(path)
    for root, dirs, names in os.walk(path):
        for name in list(dirs) + list(names):
            try:
                newest = max(newest, os.path.getmtime(os.path.join(root, name)))
            except OSError:
                continue
    return newest


def _sweep_dir(parent, cutoff):
    removed = 0
    if not os.path.isdir(parent):
        return 0
    for name in os.listdir(parent):
        path = os.path.join(parent, name)
        if not os.path.isdir(path) or not _is_safe_segment(name):
            continue
        try:
            if _newest_mtime(path) > cutoff:
                continue
            shutil.rmtree(path)
        except OSError:
            # One undeletable directory is not a reason to abandon the rest.
            continue
        removed += 1
    return removed


def sweep_expired(max_age_seconds=None, now=None):
    """
    Delete session and job directories nothing has touched in the window.

    Returns how many of each were removed.
    """
    if max_age_seconds is None:
        max_age_seconds = Config.DATA_RETENTION_SECONDS
    cutoff = (now if now is not None else time.time()) - max_age_seconds
    return {
        "sessions": _sweep_dir(os.path.join(Config.DATA_DIR, "sessions"), cutoff),
        "jobs": _sweep_dir(os.path.join(Config.DATA_DIR, "jobs"), cutoff),
    }


def delete_session_files(session_id):
    """Remove everything on disk for one session. True if there was any."""
    return _delete_tree("sessions", session_id)


def delete_job_files(job_id):
    """Remove one job's uploads and rendered pages."""
    return _delete_tree("jobs", job_id)


def _delete_tree(kind, name):
    if not _is_safe_segment(name):
        return False
    path = os.path.join(Config.DATA_DIR, kind, name)
    if not os.path.isdir(path):
        return False
    shutil.rmtree(path, ignore_errors=True)
    return True

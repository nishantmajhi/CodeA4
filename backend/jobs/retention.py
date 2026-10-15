"""
Deleting what the pipeline leaves behind.

Handwriting samples are personal data of an unusual kind: the photographs show
a person's hand, and the model derived from them reproduces it convincingly
enough to sign with. The Redis records already expire, which makes a session
unreachable -- but expiry never touched the files, so the photographs, the
extracted characters, the style model and every rendered page stayed on disk
indefinitely, belonging to a session id nobody had any way to use.

Nothing in this system runs on a timer, so the sweep is attached to the end of
a job instead. That is off the request path, runs in the worker, and happens
often enough that nothing accumulates.
"""
import logging

import storage

log = logging.getLogger(__name__)


def sweep_old_data():
    """
    Delete sessions and jobs past the retention window.

    A sweep that fails must never fail the job that triggered it: the work is
    done and its result is valid whatever the disk is doing.
    """
    try:
        removed = storage.sweep_expired()
    except Exception:
        log.exception("retention sweep failed")
        return {"sessions": 0, "jobs": 0}
    if removed["sessions"] or removed["jobs"]:
        log.info("swept %s expired session(s) and %s job(s)",
                 removed["sessions"], removed["jobs"])
    return removed

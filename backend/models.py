"""
Records for Session / Job / StyleAsset.

**Why this is not a plain dict.** Jobs run in a separate worker process from
the web server (Backend Plan §1). If state lived in process memory, the
worker's progress updates would land in the worker's own dictionary and the
web process polling `GET /api/jobs/<id>` would report "queued" forever. The
bug is invisible in single-process development and total in production, so
state is kept in Redis, which both processes share.

An in-memory backend is still available for tests and single-process dev
(``MEMORY_STATE=1``). It is chosen explicitly rather than by silent fallback,
because a silent fallback would reintroduce exactly the bug above on any
deployment where Redis is briefly unreachable at start-up.

Everything is stored as JSON. No ORM and no schema migrations — the records
are small, short-lived and read by one service.
"""
import json
import secrets
import time

from config import Config
from extensions import get_redis

_MEMORY = {}


def new_id():
    """
    A session or job identifier.

    There is no account system, so a session id is the only thing standing
    between somebody else and a person's handwriting photographs and the model
    built from them. It used to be ``uuid.uuid4().hex[:12]`` — 48 bits, three
    fifths of the entropy thrown away for a shorter string nobody types.
    """
    return secrets.token_urlsafe(24)


# --------------------------------------------------------------------------
# Storage backend
# --------------------------------------------------------------------------

def _key(kind, ident):
    return f"codea4:{kind}:{ident}"


def _read(kind, ident):
    key = _key(kind, ident)
    if Config.MEMORY_STATE:
        raw = _MEMORY.get(key)
    else:
        raw = get_redis().get(key)
    if raw is None:
        return None
    return json.loads(raw)


def _delete(kind, ident):
    key = _key(kind, ident)
    if Config.MEMORY_STATE:
        return _MEMORY.pop(key, None) is not None
    return bool(get_redis().delete(key))


def _write(kind, ident, record):
    key = _key(kind, ident)
    payload = json.dumps(record)
    if Config.MEMORY_STATE:
        _MEMORY[key] = payload
    else:
        get_redis().set(key, payload, ex=Config.STATE_TTL_SECONDS)
    return record


def clear_memory():
    """Wipe the in-memory backend. Used between tests."""
    _MEMORY.clear()


# --------------------------------------------------------------------------
# Session
# --------------------------------------------------------------------------

def create_session():
    sid = new_id()
    return _write("session", sid, {
        "session_id": sid,
        "created_at": time.time(),
        "rounds": [
            {"n": n, "status": "waiting", "job_id": None, "error": None}
            for n in range(1, 4)
        ],
        "style_ready": False,
        # Every job this session starts, so deleting the session can reach
        # what its renders left on disk. Jobs are keyed by job id and carry a
        # session id, but nothing indexes the other way round.
        "job_ids": [],
    })


def get_session(session_id):
    return _read("session", session_id)


def save_session(session):
    return _write("session", session["session_id"], session)


def update_round(session_id, n, **fields):
    """
    Update one capture round.

    Read-modify-write on a shared store is racy in principle. In practice each
    round is written by exactly one job and the user cannot start round n+1
    before round n reports back, so there is no concurrent writer to lose an
    update to. Worth revisiting if rounds are ever processed in parallel.
    """
    session = get_session(session_id)
    if not session:
        return None
    for round_rec in session["rounds"]:
        if round_rec["n"] == n:
            round_rec.update(fields)
            save_session(session)
            return round_rec
    return None


def delete_session(session_id):
    """
    Forget a session: the record, the style model derived from it, the push
    subscription that would otherwise keep notifying a browser about work for
    a session that no longer exists, and every job it started.

    The jobs matter as much as the session does. A render job's directory
    holds the source files the user uploaded and the pages produced from them,
    which are in their handwriting — and those are keyed by job id, so
    deleting only the session left all of it on disk with the record that
    pointed at it gone, which is worse than leaving it reachable.

    Returns (existed, job_ids) so the caller can remove the files too.
    """
    session = _read("session", session_id)
    job_ids = list((session or {}).get("job_ids") or [])
    for job_id in job_ids:
        _delete("job", job_id)
    _delete("session", session_id)
    _delete("style", session_id)
    _delete("push", session_id)
    return session is not None, job_ids


def mark_style_ready(session_id, ready=True):
    session = get_session(session_id)
    if not session:
        return None
    session["style_ready"] = ready
    return save_session(session)


# --------------------------------------------------------------------------
# Job
# --------------------------------------------------------------------------

def create_job(session_id, job_type):
    jid = new_id()
    session = _read("session", session_id)
    if session is not None:
        # A session started before this field existed, or one created by a
        # test with a bare identifier, simply gains it here.
        session.setdefault("job_ids", []).append(jid)
        _write("session", session_id, session)
    return _write("job", jid, {
        "job_id": jid,
        "session_id": session_id,
        "type": job_type,
        "status": "queued",
        "progress": 0,
        "stage": None,
        "error": None,
        "result": None,
        "created_at": time.time(),
    })


def get_job(job_id):
    return _read("job", job_id)


def update_job(job_id, **fields):
    job = get_job(job_id)
    if not job:
        return None
    job.update(fields)
    return _write("job", job_id, job)


def set_progress(job_id, progress, stage=None):
    """Report how far a job has got, for the capture screen's progress bar."""
    fields = {"progress": int(progress), "status": "running"}
    if stage:
        fields["stage"] = stage
    return update_job(job_id, **fields)


# --------------------------------------------------------------------------
# StyleAsset
# --------------------------------------------------------------------------

def save_style_asset(session_id, style_sets, drift, meta=None):
    return _write("style", session_id, {
        "session_id": session_id,
        "style_sets": style_sets,
        "drift": drift,
        "meta": meta or {},
    })


def get_style_asset(session_id):
    return _read("style", session_id)


# --------------------------------------------------------------------------
# Push subscriptions
# --------------------------------------------------------------------------

def save_push_subscription(session_id, subscription):
    return _write("push", session_id, subscription)


def get_push_subscription(session_id):
    return _read("push", session_id)

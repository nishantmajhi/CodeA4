"""
Redis + RQ connection, created lazily.

Kept lazy (rather than connecting at import time) so `app.py` can be imported
— for tests, or `flask routes` — even when no Redis server is reachable. The
connection is only opened the first time it is actually needed.

``Config.INLINE_JOBS`` swaps the queue for one that runs jobs immediately in
the calling process. That exists so the flow can be exercised end to end
without Redis and a second process, which the test suite depends on. It is not
a production mode: the request blocks for the whole pipeline.
"""
import logging

from config import Config

log = logging.getLogger(__name__)

_redis = None
_queue = None


class InlineJob:
    """Minimal stand-in for an RQ job, for the inline queue."""

    def __init__(self, job_id):
        self.id = job_id


class InlineQueue:
    """Runs enqueued callables straight away, in this process."""

    def enqueue(self, func, *args, **kwargs):
        job_id = kwargs.pop("job_id", None)
        try:
            func(*args, **kwargs)
        except Exception:
            # A real worker records the failure against the job record and
            # keeps running; match that here rather than surfacing a 500 for
            # a job that was only ever meant to be fire-and-forget.
            log.exception("inline job failed: %s", getattr(func, "__name__", func))
        return InlineJob(job_id)


def get_redis():
    global _redis
    if _redis is None:
        from redis import Redis
        _redis = Redis.from_url(Config.REDIS_URL)
    return _redis


def get_queue():
    global _queue
    if _queue is None:
        if Config.INLINE_JOBS:
            _queue = InlineQueue()
        else:
            from rq import Queue
            _queue = Queue("codea4", connection=get_redis())
    return _queue


def reset():
    """Drop cached connections. Used by tests that change configuration."""
    global _redis, _queue
    _redis = None
    _queue = None

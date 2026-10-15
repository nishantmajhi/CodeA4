"""
Web push notifications.

The capture and render stages take long enough that a user may switch away
from the tab, so the PWA asks to notify them when a job finishes. Sending
happens from the worker process, right after the job's own state is written —
never before, so a notification can't arrive advertising a result the API
would still report as pending.

Push is optional everywhere: with no VAPID keys configured, or no
subscription for the session, or pywebpush not installed, this is a no-op that
logs. A failed notification must never fail the job that triggered it — the
work is done and the result is on the API regardless.
"""
import json
import logging

import models
from config import Config

log = logging.getLogger(__name__)


def notify(session_id, title, body, url="/"):
    """Send one push message to a session's subscriber, if there is one."""
    if not Config.push_enabled():
        log.debug("push not configured; skipping notification")
        return False

    subscription = models.get_push_subscription(session_id)
    if not subscription:
        return False

    try:
        from pywebpush import WebPushException, webpush
    except ImportError:
        log.warning("pywebpush is not installed; skipping notification")
        return False

    try:
        webpush(
            subscription_info=subscription,
            data=json.dumps({"title": title, "body": body, "url": url}),
            vapid_private_key=Config.VAPID_PRIVATE_KEY,
            vapid_claims={"sub": Config.VAPID_SUBJECT},
        )
        return True
    except WebPushException as exc:
        # 404/410 mean the browser dropped the subscription — clear it so we
        # stop trying, rather than failing on every subsequent job.
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (404, 410):
            models.save_push_subscription(session_id, None)
            log.info("push subscription expired for session %s", session_id)
        else:
            log.warning("push failed for session %s: %s", session_id, exc)
        return False
    except Exception:
        log.exception("unexpected push failure for session %s", session_id)
        return False

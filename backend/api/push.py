from flask import Blueprint, abort, jsonify, request

import models
from config import Config

push_bp = Blueprint("push", __name__)


@push_bp.get("/push/vapid-key")
def vapid_key():
    """
    The public VAPID key the browser needs to create a subscription.

    Served rather than baked into the front-end so that keys can be rotated,
    and so the UI can tell whether push is configured at all and hide the
    option when it is not.
    """
    return jsonify({
        "enabled": Config.push_enabled(),
        "public_key": Config.VAPID_PUBLIC_KEY or None,
    })


@push_bp.post("/push/subscribe")
def subscribe():
    """
    UI: the Capture step's "notify me" checkbox.

    Stores the browser's Push API subscription against a session, so a
    finished round or render can push a real notification instead of the user
    having to keep the tab open (spec §7).
    """
    body = request.get_json(silent=True) or {}
    session_id = body.get("session_id")
    subscription = body.get("subscription")
    if not session_id or not subscription:
        abort(400, description="Requires 'session_id' and 'subscription'")
    if not models.get_session(session_id):
        abort(404, description="Unknown session")

    models.save_push_subscription(session_id, subscription)
    return jsonify({"ok": True})

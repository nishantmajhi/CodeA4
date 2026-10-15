"""
Web push, which is optional everywhere and must stay that way.

Capture and render both take long enough that the user is invited to leave the
tab, so the notification is what brings them back. It is still an enhancement:
with no VAPID keys, no subscription, or no `pywebpush` installed, sending is a
no-op that logs. A failed notification must never fail the job that triggered
it -- the work is done and the result is on the API regardless.

`pywebpush` is not in the environment these tests run in, which is itself one
of the cases worth covering; the rest inject a stand-in.
"""
import sys
import types

import pytest

import models
import push_service
from config import Config


@pytest.fixture(autouse=True)
def clean_store():
    models.clear_memory()
    yield
    models.clear_memory()


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setattr(Config, "VAPID_PUBLIC_KEY", "public")
    monkeypatch.setattr(Config, "VAPID_PRIVATE_KEY", "private")


class _Response:
    def __init__(self, status_code):
        self.status_code = status_code


def _install_pywebpush(monkeypatch, behaviour):
    """Put a stand-in pywebpush in place for the duration of one test."""
    module = types.ModuleType("pywebpush")

    class WebPushException(Exception):
        def __init__(self, message, response=None):
            super().__init__(message)
            self.response = response

    module.WebPushException = WebPushException
    module.webpush = behaviour
    monkeypatch.setitem(sys.modules, "pywebpush", module)
    return module


def test_push_is_disabled_without_keys():
    assert Config.push_enabled() is False
    assert push_service.notify("s", "title", "body") is False


def test_nothing_is_sent_to_a_session_with_no_subscription(configured):
    assert push_service.notify("s", "title", "body") is False


def test_a_notification_carries_the_title_body_and_url(configured, monkeypatch):
    sent = {}

    def webpush(**kwargs):
        sent.update(kwargs)

    _install_pywebpush(monkeypatch, webpush)
    models.save_push_subscription("s", {"endpoint": "https://push.example/1"})

    assert push_service.notify("s", "Ready", "3 pages", url="/#preview") is True

    import json
    payload = json.loads(sent["data"])
    assert payload == {"title": "Ready", "body": "3 pages", "url": "/#preview"}
    assert sent["subscription_info"]["endpoint"] == "https://push.example/1"
    assert sent["vapid_private_key"] == "private"
    assert sent["vapid_claims"]["sub"] == Config.VAPID_SUBJECT


def test_a_missing_library_is_not_an_error(configured, monkeypatch):
    monkeypatch.setitem(sys.modules, "pywebpush", None)
    models.save_push_subscription("s", {"endpoint": "https://push.example/1"})
    assert push_service.notify("s", "title", "body") is False


@pytest.mark.parametrize("status", [404, 410])
def test_a_subscription_the_browser_dropped_is_forgotten(configured, monkeypatch,
                                                         status):
    """
    404 and 410 mean the endpoint is gone. Keeping it would mean every
    subsequent job paying for a request that cannot succeed.
    """
    module = _install_pywebpush(monkeypatch, lambda **_kw: None)

    def webpush(**_kwargs):
        raise module.WebPushException("gone", response=_Response(status))

    module.webpush = webpush
    models.save_push_subscription("s", {"endpoint": "https://push.example/1"})

    assert push_service.notify("s", "title", "body") is False
    assert models.get_push_subscription("s") is None


def test_a_temporary_failure_keeps_the_subscription(configured, monkeypatch):
    """A 503 from the push service is about the push service, not about
    whether this browser still wants notifications."""
    module = _install_pywebpush(monkeypatch, lambda **_kw: None)

    def webpush(**_kwargs):
        raise module.WebPushException("busy", response=_Response(503))

    module.webpush = webpush
    models.save_push_subscription("s", {"endpoint": "https://push.example/1"})

    assert push_service.notify("s", "title", "body") is False
    assert models.get_push_subscription("s") is not None


def test_an_unexpected_failure_is_swallowed(configured, monkeypatch):
    def webpush(**_kwargs):
        raise RuntimeError("something else entirely")

    _install_pywebpush(monkeypatch, webpush)
    models.save_push_subscription("s", {"endpoint": "https://push.example/1"})
    assert push_service.notify("s", "title", "body") is False


def test_a_failed_notification_does_not_fail_the_job(client, monkeypatch,
                                                     tmp_path):
    """The whole point: the pages are rendered and the result is on the API
    whatever the push service is doing."""
    import io

    import fake_style

    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(Config, "VAPID_PUBLIC_KEY", "public")
    monkeypatch.setattr(Config, "VAPID_PRIVATE_KEY", "private")

    def webpush(**_kwargs):
        raise RuntimeError("push service is down")

    _install_pywebpush(monkeypatch, webpush)

    session_id = client.post("/api/sessions").get_json()["session_id"]
    fake_style.install(session_id, models)
    models.save_push_subscription(session_id,
                                  {"endpoint": "https://push.example/1"})
    job_id = client.post(f"/api/sessions/{session_id}/render",
                         data={"files": (io.BytesIO(b"x = 1\n"), "a.py")},
                         content_type="multipart/form-data").get_json()["job_id"]
    assert client.get(f"/api/jobs/{job_id}").get_json()["status"] == "done"


def test_the_public_key_is_served_so_the_ui_can_hide_the_option(client):
    body = client.get("/api/push/vapid-key").get_json()
    assert body["enabled"] is False and body["public_key"] is None


def test_subscribing_needs_both_a_session_and_a_subscription(client):
    import json

    session_id = client.post("/api/sessions").get_json()["session_id"]
    for payload in ({}, {"session_id": session_id}, {"subscription": {"e": 1}}):
        response = client.post("/api/push/subscribe", data=json.dumps(payload),
                               content_type="application/json")
        assert response.status_code == 400


def test_a_subscription_is_stored_against_its_session(client):
    import json

    session_id = client.post("/api/sessions").get_json()["session_id"]
    subscription = {"endpoint": "https://push.example/2", "keys": {"p256dh": "k"}}
    response = client.post("/api/push/subscribe",
                           data=json.dumps({"session_id": session_id,
                                            "subscription": subscription}),
                           content_type="application/json")
    assert response.status_code == 200
    assert models.get_push_subscription(session_id) == subscription

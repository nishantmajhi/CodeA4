"""
The shape of what the API returns, as the front-end depends on it.

`app.js` routes every call through one helper that reads `error` out of a JSON
body. Anything that answers a different shape -- or answers 200 to a request
that failed -- arrives there as an unhandled `undefined` rather than as a
message the user can act on.
"""
import io

import models


def test_health_reports_the_modes_it_is_running_in(client):
    body = client.get("/api/health").get_json()
    assert body["status"] == "ok"
    assert set(body) >= {"status", "inline_jobs", "push_enabled"}


def test_an_unknown_api_path_is_a_json_404(client):
    """
    The catch-all that serves the PWA used to match `/api/...` too, so a
    mistyped or removed endpoint answered `200 text/html` with the app shell
    in it. The front-end checks `res.ok` first, so that is not an error there:
    it is a successful call whose body is a string, and the failure surfaces
    somewhere else entirely as a missing property.
    """
    response = client.get("/api/definitely-not-a-route")
    assert response.status_code == 404
    assert response.mimetype == "application/json"
    assert "error" in response.get_json()


def test_the_wrong_method_on_a_real_route_is_not_a_page(client):
    """`GET /api/sessions` is not a route -- only POST is -- and it used to
    answer with the app shell."""
    response = client.get("/api/sessions")
    assert response.status_code in (404, 405)
    assert response.mimetype == "application/json"


def test_unknown_page_paths_still_fall_back_to_the_app_shell(client):
    """
    Deep links like `/#preview` come from notifications, and client-side
    routing means the server sees paths it has no file for. Those must keep
    returning the shell -- only `/api/` is excluded.
    """
    response = client.get("/some/deep/link")
    assert response.status_code == 200
    assert response.mimetype == "text/html"


def test_errors_carry_a_sentence_for_the_user(client):
    """Every error the capture and render flows can produce is shown to the
    user verbatim, so an empty or missing `error` is a blank error box."""
    session_id = client.post("/api/sessions").get_json()["session_id"]
    cases = [
        client.get("/api/sessions/nope"),
        client.get("/api/jobs/nope"),
        client.post(f"/api/sessions/{session_id}/rounds/9/photo",
                    data={"photo": (io.BytesIO(b"x"), "a.jpg")},
                    content_type="multipart/form-data"),
        client.post(f"/api/sessions/{session_id}/rounds/1/photo", data={},
                    content_type="multipart/form-data"),
    ]
    for response in cases:
        assert response.status_code >= 400
        body = response.get_json()
        assert isinstance(body.get("error"), str) and len(body["error"]) > 10


def test_a_server_error_never_leaks_a_traceback(client, monkeypatch):
    import api.sessions

    def boom(*_args, **_kwargs):
        raise RuntimeError("internal detail /srv/secret/path")

    monkeypatch.setattr(api.sessions.models, "create_session", boom)
    app = client.application
    app.config["PROPAGATE_EXCEPTIONS"] = False
    response = client.post("/api/sessions")
    assert response.status_code == 500
    assert b"secret" not in response.data
    assert b"Traceback" not in response.data


# ----------------------------------------------------------------- identifiers

def test_session_identifiers_are_full_entropy():
    """
    There is no account system, so the session id is the only thing standing
    between somebody else and a person's handwriting photographs and the model
    built from them. It used to be a UUID4 truncated to 12 hex characters --
    48 bits, thrown away for no reason.
    """
    ident = models.new_id()
    assert len(ident) >= 22, f"{len(ident)} characters is not enough entropy"
    assert set(ident) <= set(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")


def test_identifiers_do_not_repeat():
    assert len({models.new_id() for _ in range(2000)}) == 2000


def test_identifiers_are_safe_in_a_url_and_on_disk(client):
    """They are used as path segments and as directory names, so a `/` or a
    `.` in one would be a traversal rather than an id."""
    for _ in range(200):
        ident = models.new_id()
        assert "/" not in ident and "\\" not in ident
        assert not ident.startswith(".")

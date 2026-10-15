"""
End-to-end tests through the HTTP API, with jobs running inline.

These cover the contract the front-end actually depends on: what a poll
returns while work is in flight, what a failure looks like, and what is
deliberately *not* returned to the browser.
"""
import io
import json
import os

import pytest


def _cairo_available():
    """
    Whether this machine can convert SVG to PDF at all.

    cairosvg needs Cairo from the operating system, which the Dockerfile
    installs and a plain Windows machine does not have. Both outcomes are
    documented behaviour -- a PDF, or a preview-only result with a message
    saying so -- and skipping the second would leave the degradation path,
    which is the one users on an under-provisioned server actually meet,
    asserted nowhere.
    """
    try:
        import cairosvg  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


def _upload_sheet(client, session_id, n, image_path):
    with open(image_path, "rb") as f:
        data = {"photo": (io.BytesIO(f.read()), f"sheet{n}.jpg")}
    return client.post(f"/api/sessions/{session_id}/rounds/{n}/photo",
                       data=data, content_type="multipart/form-data")


@pytest.fixture
def captured(client, sheets, tmp_path, monkeypatch):
    """A session with all three sheets processed and a style model built."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from config import Config
    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path))

    session_id = client.post("/api/sessions").get_json()["session_id"]
    for n, (image, _truth) in enumerate(sheets, start=1):
        response = _upload_sheet(client, session_id, n, image)
        assert response.status_code == 202
        job_id = response.get_json()["job_id"]
        job = client.get(f"/api/jobs/{job_id}").get_json()
        assert job["status"] == "done", job.get("error")
    return session_id


def test_health(client):
    assert client.get("/api/health").get_json()["status"] == "ok"


def test_exercise_is_served_from_the_same_source_as_the_segmenter(client):
    from pipeline.exercise import EXERCISE_TEXT

    body = client.get("/api/exercise").get_json()
    assert body["text"] == EXERCISE_TEXT
    assert body["rounds"] == 3


def test_session_starts_with_three_waiting_rounds(client):
    session = client.post("/api/sessions").get_json()
    assert len(session["rounds"]) == 3
    assert all(r["status"] == "waiting" for r in session["rounds"])
    assert session["style_ready"] is False


def test_unknown_session_is_404(client):
    assert client.get("/api/sessions/nope").status_code == 404


def test_upload_without_a_photo_is_rejected(client):
    session_id = client.post("/api/sessions").get_json()["session_id"]
    response = client.post(f"/api/sessions/{session_id}/rounds/1/photo",
                           data={}, content_type="multipart/form-data")
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_render_before_style_is_ready_is_refused(client):
    """The UI gates on this, but the server must not depend on the UI
    behaving — a render without a handwriting model has nothing to draw."""
    session_id = client.post("/api/sessions").get_json()["session_id"]
    data = {"files": (io.BytesIO(b"print(1)\n"), "a.py")}
    response = client.post(f"/api/sessions/{session_id}/render",
                           data=data, content_type="multipart/form-data")
    assert response.status_code == 409


def test_capture_builds_a_style_model(client, captured):
    session = client.get(f"/api/sessions/{captured}").get_json()
    assert session["style_ready"] is True

    style = client.get(f"/api/sessions/{captured}/style").get_json()
    assert style["style_ready"] is True
    assert 2 <= len(style["style_sets"]) <= 4

    for style_set in style["style_sets"]:
        assert style_set["coverage"] > 80          # nearly all of ASCII
        assert 0.0 <= style_set["share"] <= 1.0
        glyph = style_set["glyphs"]["a"]
        assert glyph["path"].startswith("M")
        assert glyph["advance"] > 0

    shares = sum(s["share"] for s in style["style_sets"])
    assert shares == pytest.approx(1.0, abs=0.01)


def test_style_sets_are_internally_distinct(client, captured):
    """
    Style 2 must not be a copy of style 1. If clustering found nothing, the
    sets would be identical and the whole premise — that a hand has several
    natural forms per letter — would be unsupported by the output.
    """
    style = client.get(f"/api/sessions/{captured}/style").get_json()
    sets = style["style_sets"]
    if len(sets) < 2:
        pytest.skip("only one style was found for this sample")

    first, second = sets[0]["glyphs"], sets[1]["glyphs"]
    shared = set(first) & set(second)
    differing = sum(1 for c in shared if first[c]["path"] != second[c]["path"])
    assert differing > 0.25 * len(shared)


def test_drift_model_is_mean_reverting(client, captured):
    style = client.get(f"/api/sessions/{captured}/style").get_json()
    for branch in ("baseline", "margin"):
        model = style["drift"][branch]
        assert 0.0 < model["phi"] < 1.0, "must pull back toward the line"
        assert model["std"] > 0
        assert len(model["points"]) > 3


def test_style_sample_svg(client, captured):
    response = client.get(f"/api/sessions/{captured}/style/sample.svg?style=1")
    assert response.status_code == 200
    assert response.mimetype == "image/svg+xml"
    assert b"<path" in response.data


def test_full_render_flow(client, captured):
    source = b'def greet(name):\n    """Say hello."""\n    print(f"hi {name}")\n'
    data = {"files": (io.BytesIO(source), "greet.py")}
    response = client.post(f"/api/sessions/{captured}/render",
                           data=data, content_type="multipart/form-data")
    assert response.status_code == 202
    job_id = response.get_json()["job_id"]

    job = client.get(f"/api/jobs/{job_id}").get_json()
    assert job["status"] == "done", job.get("error")

    result = job["result"]
    assert result["page_count"] >= 1
    assert result["coverage"]["exact"] == result["coverage"]["total"]
    assert result["pages"][0]["file"] == "greet.py"

    page = client.get(f"/api/render/{job_id}/pages/0.svg")
    assert page.status_code == 200
    assert b"<path" in page.data
    assert b'width="595' in page.data      # A4 in points

    pdf = client.get(f"/api/render/{job_id}/pdf")
    if _cairo_available():
        assert pdf.status_code == 200
        assert pdf.data.startswith(b"%PDF")
    else:
        # The pages rendered; only the conversion is unavailable. The render
        # must still have succeeded, and the refusal must carry the sentence
        # the export screen shows.
        assert pdf.status_code == 409
        assert "preview" in pdf.get_json()["error"].lower()


def test_job_result_hides_server_paths(client, captured):
    """The browser is told whether a PDF exists, never where it lives -- the
    path is a server detail and the flag is what the Export step gates on."""
    data = {"files": (io.BytesIO(b"x = 1\n"), "a.py")}
    job_id = client.post(f"/api/sessions/{captured}/render", data=data,
                         content_type="multipart/form-data").get_json()["job_id"]
    result = client.get(f"/api/jobs/{job_id}").get_json()["result"]
    assert "pdf_path" not in result
    assert result["pdf_available"] is _cairo_available()


def test_uploaded_filename_cannot_escape_its_directory(client, captured, tmp_path):
    """
    A traversal in the upload name must not decide where bytes land, while the
    original name is still what gets written as the page's header.
    """
    data = {"files": (io.BytesIO(b"y = 2\n"), "../../escaped.py")}
    response = client.post(f"/api/sessions/{captured}/render", data=data,
                           content_type="multipart/form-data")
    job_id = response.get_json()["job_id"]
    job = client.get(f"/api/jobs/{job_id}").get_json()
    assert job["status"] == "done", job.get("error")
    assert not os.path.exists(os.path.join(str(tmp_path), "..", "escaped.py"))


def test_missing_page_is_404(client, captured):
    data = {"files": (io.BytesIO(b"z = 3\n"), "a.py")}
    job_id = client.post(f"/api/sessions/{captured}/render", data=data,
                         content_type="multipart/form-data").get_json()["job_id"]
    assert client.get(f"/api/render/{job_id}/pages/99.svg").status_code == 404


def test_push_key_endpoint_reports_when_unconfigured(client):
    body = client.get("/api/push/vapid-key").get_json()
    assert body["enabled"] is False


def test_push_subscribe_requires_a_known_session(client):
    response = client.post("/api/push/subscribe",
                           data=json.dumps({"session_id": "nope",
                                            "subscription": {"endpoint": "x"}}),
                           content_type="application/json")
    assert response.status_code == 404

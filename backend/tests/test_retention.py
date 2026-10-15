"""
What happens to a person's handwriting after they are done with it.

The photographs show somebody's hand, and the model built from them
reproduces it convincingly enough to sign with. The Redis records expire after
a week, which makes a session unreachable -- but the files were never touched
by that. Photographs, extracted character data, the style model and every
rendered page stayed on disk for as long as the disk did, belonging to a
session id nobody had any way to use.
"""
import io
import os
import time

import pytest

import models
import storage
from config import Config
import fake_style


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path))
    return str(tmp_path)


def _age(path, seconds):
    """Backdate everything under `path` by `seconds`."""
    when = time.time() - seconds
    for root, dirs, names in os.walk(path):
        for name in names:
            os.utime(os.path.join(root, name), (when, when))
        for name in dirs:
            os.utime(os.path.join(root, name), (when, when))
    os.utime(path, (when, when))


def _make_session_files(session_id):
    for n in (1, 2, 3):
        with open(storage.round_photo_path(session_id, n, ".jpg"), "wb") as f:
            f.write(b"pretend this is a photograph of a hand")
        with open(storage.round_chars_path(session_id, n), "w") as f:
            f.write("{}")
    with open(storage.style_assets_path(session_id), "w") as f:
        f.write("{}")


# ------------------------------------------------------------------ the sweep

def test_expired_sessions_are_swept(data_dir):
    _make_session_files("old-session")
    _age(os.path.join(data_dir, "sessions", "old-session"), 60 * 60 * 24 * 30)

    removed = storage.sweep_expired(max_age_seconds=60 * 60 * 24 * 7)

    assert not os.path.exists(os.path.join(data_dir, "sessions", "old-session"))
    assert removed["sessions"] == 1


def test_a_session_still_in_its_window_is_left_alone(data_dir):
    _make_session_files("recent")
    _age(os.path.join(data_dir, "sessions", "recent"), 60 * 60)

    storage.sweep_expired(max_age_seconds=60 * 60 * 24 * 7)

    assert os.path.exists(storage.round_photo_path("recent", 1, ".jpg"))


def test_a_session_being_written_to_is_not_swept(data_dir):
    """
    Capture is deliberately spread over three sheets with breaks in between,
    so a session's directory can be old while the session is very much alive.
    Age is measured from the newest thing in it, not from the directory.
    """
    _make_session_files("in-progress")
    _age(os.path.join(data_dir, "sessions", "in-progress"), 60 * 60 * 24 * 30)
    with open(storage.round_photo_path("in-progress", 3, ".jpg"), "wb") as f:
        f.write(b"a sheet captured just now")

    storage.sweep_expired(max_age_seconds=60 * 60 * 24 * 7)

    assert os.path.exists(storage.round_photo_path("in-progress", 1, ".jpg"))


def test_expired_render_jobs_are_swept(data_dir):
    path = storage.job_page_svg_path("old-job", 0)
    with open(path, "w") as f:
        f.write("<svg/>")
    _age(os.path.join(data_dir, "jobs", "old-job"), 60 * 60 * 24 * 30)

    removed = storage.sweep_expired(max_age_seconds=60 * 60 * 24 * 7)

    assert not os.path.exists(os.path.join(data_dir, "jobs", "old-job"))
    assert removed["jobs"] == 1


def test_sweeping_an_empty_data_directory_is_not_an_error(data_dir):
    assert storage.sweep_expired(max_age_seconds=1) == {"sessions": 0, "jobs": 0}


def test_the_sweep_runs_when_work_is_done(client, data_dir, monkeypatch):
    """
    Nothing here runs on a timer, so the sweep has to be attached to something
    that happens. It runs in the worker, when a job finishes, which is both
    off the request path and reliably often.
    """
    calls = []
    monkeypatch.setattr(storage, "sweep_expired",
                        lambda *a, **k: calls.append(True) or
                        {"sessions": 0, "jobs": 0})

    session_id = client.post("/api/sessions").get_json()["session_id"]
    fake_style.install(session_id, models)
    client.post(f"/api/sessions/{session_id}/render",
                data={"files": (io.BytesIO(b"x = 1\n"), "a.py")},
                content_type="multipart/form-data")
    assert calls, "a finished job swept nothing"


def test_a_failing_sweep_never_fails_the_job(client, data_dir, monkeypatch):
    def boom(*_a, **_k):
        raise OSError("disk is busy")

    monkeypatch.setattr(storage, "sweep_expired", boom)
    session_id = client.post("/api/sessions").get_json()["session_id"]
    fake_style.install(session_id, models)
    job_id = client.post(f"/api/sessions/{session_id}/render",
                         data={"files": (io.BytesIO(b"x = 1\n"), "a.py")},
                         content_type="multipart/form-data").get_json()["job_id"]
    assert client.get(f"/api/jobs/{job_id}").get_json()["status"] == "done"


# ------------------------------------------------------------ delete on demand

def test_a_session_can_be_deleted(client, data_dir):
    session_id = client.post("/api/sessions").get_json()["session_id"]
    fake_style.install(session_id, models)
    _make_session_files(session_id)

    response = client.delete(f"/api/sessions/{session_id}")
    assert response.status_code == 200

    assert not os.path.exists(os.path.join(data_dir, "sessions", session_id))
    assert models.get_session(session_id) is None
    assert models.get_style_asset(session_id) is None
    assert client.get(f"/api/sessions/{session_id}").status_code == 404


def test_deleting_a_session_drops_its_push_subscription(client, data_dir):
    session_id = client.post("/api/sessions").get_json()["session_id"]
    models.save_push_subscription(session_id, {"endpoint": "https://x.example"})
    client.delete(f"/api/sessions/{session_id}")
    assert models.get_push_subscription(session_id) is None


def test_deleting_an_unknown_session_is_a_404(client, data_dir):
    assert client.delete("/api/sessions/no-such-session").status_code == 404


def test_a_session_id_cannot_name_a_path_outside_the_data_directory(client,
                                                                    data_dir):
    """Session ids are used as directory names, so deletion has to refuse
    anything that is not one."""
    outside = os.path.join(data_dir, "keepme.txt")
    with open(outside, "w") as f:
        f.write("not part of any session")

    for bad in ("..", "../..", "a/b", ".", ""):
        client.delete(f"/api/sessions/{bad}")

    assert os.path.exists(outside)
    assert os.path.isdir(data_dir)


def test_deleting_a_session_also_deletes_what_it_rendered(client, data_dir):
    """
    A render job's directory holds the source files the user uploaded and the
    pages produced from them, which are in their handwriting. Those are keyed
    by job id, not by session id, so "forget this session" originally left
    every one of them on disk -- with the session record gone, which is worse
    than leaving them reachable.
    """
    session_id = client.post("/api/sessions").get_json()["session_id"]
    fake_style.install(session_id, models)
    job_id = client.post(f"/api/sessions/{session_id}/render",
                         data={"files": (io.BytesIO(b"secret = 1\n"), "a.py")},
                         content_type="multipart/form-data").get_json()["job_id"]
    job_dir = os.path.join(data_dir, "jobs", job_id)
    assert os.path.isdir(job_dir)

    client.delete(f"/api/sessions/{session_id}")

    assert not os.path.exists(job_dir)
    assert models.get_job(job_id) is None
    assert client.get(f"/api/jobs/{job_id}").status_code == 404


def test_deleting_a_session_leaves_another_session_s_renders_alone(client,
                                                                   data_dir):
    keep = client.post("/api/sessions").get_json()["session_id"]
    fake_style.install(keep, models)
    keep_job = client.post(f"/api/sessions/{keep}/render",
                           data={"files": (io.BytesIO(b"x = 1\n"), "a.py")},
                           content_type="multipart/form-data").get_json()["job_id"]

    gone = client.post("/api/sessions").get_json()["session_id"]
    fake_style.install(gone, models)
    client.post(f"/api/sessions/{gone}/render",
                data={"files": (io.BytesIO(b"y = 2\n"), "b.py")},
                content_type="multipart/form-data")

    client.delete(f"/api/sessions/{gone}")

    assert os.path.isdir(os.path.join(data_dir, "jobs", keep_job))
    assert models.get_job(keep_job) is not None


def test_a_sessions_jobs_are_recorded_on_it(client, data_dir):
    """There is no index from session to job, so the session has to keep the
    list -- and it is the only way deletion can find them."""
    session_id = client.post("/api/sessions").get_json()["session_id"]
    fake_style.install(session_id, models)
    ids = [client.post(f"/api/sessions/{session_id}/render",
                       data={"files": (io.BytesIO(b"x = 1\n"), "a.py")},
                       content_type="multipart/form-data").get_json()["job_id"]
           for _ in range(3)]
    recorded = models.get_session(session_id).get("job_ids", [])
    assert set(ids) <= set(recorded)


def test_a_job_for_a_session_that_is_gone_is_not_an_error():
    models.clear_memory()
    job = models.create_job("no-such-session", "render")
    assert models.get_job(job["job_id"]) is not None

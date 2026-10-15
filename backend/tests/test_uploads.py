"""
What happens to the files a user hands the server.

Two different paths save uploads -- the capture photo and the source files --
and both decide a filename from something the client sent. These tests are
about that decision: that a name cannot choose where bytes land, that two
uploads cannot land on top of each other, and that nothing is left behind.
"""
import io
import os

import pytest

import models
import storage
import fake_style


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    from config import Config
    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path))
    return str(tmp_path)


@pytest.fixture
def ready_session(client, data_dir):
    session_id = client.post("/api/sessions").get_json()["session_id"]
    fake_style.install(session_id, models)
    return session_id


def _render(client, session_id, files):
    data = {"files": [(io.BytesIO(body), name) for name, body in files]}
    response = client.post(f"/api/sessions/{session_id}/render", data=data,
                           content_type="multipart/form-data")
    assert response.status_code == 202, response.get_json()
    return response.get_json()["job_id"]


def _saved_bodies(job_id):
    """Every file the render job actually has on disk, sorted."""
    bodies = []
    for root, _dirs, names in os.walk(storage.job_upload_dir(job_id)):
        for name in names:
            bodies.append(open(os.path.join(root, name), "rb").read())
    return sorted(bodies)


# ---------------------------------------------------------------- source files

def test_two_files_with_the_same_name_are_both_rendered(client, ready_session,
                                                        data_dir):
    """
    Selecting `main.py` from two different folders is an ordinary thing to do,
    and the browser sends both under the same name. Sanitizing a name does not
    make it unique, so both used to land on one path: the second overwrote the
    first, and the *first* file's pages were then drawn from the second file's
    contents. Nothing reported it -- the job succeeded and the page count was
    right.
    """
    job_id = _render(client, ready_session, [
        ("main.py", b"first = 1\n"),
        ("main.py", b"second = 2\nsecond_extra = 3\n"),
    ])
    job = client.get(f"/api/jobs/{job_id}").get_json()
    assert job["status"] == "done", job.get("error")

    assert _saved_bodies(job_id) == [b"first = 1\n",
                                     b"second = 2\nsecond_extra = 3\n"]
    # The second file is two lines long and the first one, so a page rendered
    # from the wrong file gives itself away in its own line count.
    assert [p["lines"] for p in job["result"]["pages"]] == [1, 2]


def test_names_that_sanitize_to_the_same_string_stay_separate(client,
                                                              ready_session):
    """
    `secure_filename` strips non-ASCII, so two files with different names in a
    non-Latin script can both reduce to the same string -- in the worst case to
    nothing at all.
    """
    job_id = _render(client, ready_session, [
        ("\u65e5\u672c\u8a9e.py", b"alpha = 1\n"),
        ("\u4e2d\u6587.py", b"beta = 2\nbeta_extra = 3\n"),
    ])
    job = client.get(f"/api/jobs/{job_id}").get_json()
    assert job["status"] == "done", job.get("error")
    assert _saved_bodies(job_id) == [b"alpha = 1\n", b"beta = 2\nbeta_extra = 3\n"]
    assert [p["lines"] for p in job["result"]["pages"]] == [1, 2]


def test_original_names_are_kept_for_the_page_headers(client, ready_session):
    """The sanitized name decides the path; the name the user gave decides
    what is written at the top of the page."""
    job_id = _render(client, ready_session, [("src/main.py", b"x = 1\n")])
    result = client.get(f"/api/jobs/{job_id}").get_json()["result"]
    assert result["pages"][0]["file"] == "src/main.py"


def test_a_traversal_in_an_upload_name_cannot_escape(client, ready_session,
                                                     data_dir):
    job_id = _render(client, ready_session, [("../../escaped.py", b"y = 2\n")])
    job = client.get(f"/api/jobs/{job_id}").get_json()
    assert job["status"] == "done", job.get("error")
    assert not os.path.exists(os.path.join(data_dir, "..", "escaped.py"))
    for root, _dirs, names in os.walk(storage.job_upload_dir(job_id)):
        for name in names:
            resolved = os.path.realpath(os.path.join(root, name))
            assert resolved.startswith(os.path.realpath(data_dir))


def test_a_name_that_sanitizes_to_nothing_still_renders(client, ready_session):
    job_id = _render(client, ready_session, [("..", b"z = 3\n")])
    job = client.get(f"/api/jobs/{job_id}").get_json()
    assert job["status"] == "done", job.get("error")
    assert job["result"]["page_count"] == 1


# --------------------------------------------------------------- capture photo

def test_photo_extension_must_be_an_image(client, data_dir):
    """The saved extension used to be whatever the client's filename ended in,
    so an upload could put a `.php` or `.exe` on disk."""
    session_id = client.post("/api/sessions").get_json()["session_id"]
    for name in ("sheet.php", "sheet.exe", "sheet.html", "sheet.svg"):
        data = {"photo": (io.BytesIO(b"not an image"), name)}
        response = client.post(f"/api/sessions/{session_id}/rounds/1/photo",
                               data=data, content_type="multipart/form-data")
        assert response.status_code == 400, name
        assert "error" in response.get_json()


@pytest.mark.parametrize("name", ["sheet.jpg", "sheet.JPEG", "sheet.png",
                                  "sheet.heic", "sheet.webp"])
def test_ordinary_photo_extensions_are_accepted(client, data_dir, name):
    session_id = client.post("/api/sessions").get_json()["session_id"]
    data = {"photo": (io.BytesIO(b"not an image"), name)}
    response = client.post(f"/api/sessions/{session_id}/rounds/1/photo",
                           data=data, content_type="multipart/form-data")
    assert response.status_code == 202, name


def test_a_photo_with_no_extension_is_accepted_as_jpeg(client, data_dir):
    """A camera capture on some phones arrives with no filename at all."""
    session_id = client.post("/api/sessions").get_json()["session_id"]
    data = {"photo": (io.BytesIO(b"not an image"), "")}
    response = client.post(f"/api/sessions/{session_id}/rounds/1/photo",
                           data=data, content_type="multipart/form-data")
    assert response.status_code == 202


def test_retaking_a_sheet_replaces_the_previous_photo(client, data_dir):
    """
    A retake under a different extension used to leave the first photo on disk
    forever. These are photographs of somebody's handwriting; the round holds
    one, and it is the one they last sent.
    """
    session_id = client.post("/api/sessions").get_json()["session_id"]
    for name in ("sheet.jpg", "sheet.png", "sheet.jpeg"):
        client.post(f"/api/sessions/{session_id}/rounds/1/photo",
                    data={"photo": (io.BytesIO(b"not an image"), name)},
                    content_type="multipart/form-data")
    left = os.listdir(storage.round_dir(session_id, 1))
    assert [n for n in left if n.startswith("photo")] == ["photo.jpeg"]

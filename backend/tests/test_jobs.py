"""
The job layer: what the user is told while work is happening, and what they
are told when it goes wrong.

Every long stage is a job the server owns and the page polls. The job record
is therefore the entire user interface for anything that takes more than a
moment, and its two error paths are not interchangeable -- one is a photo the
user can retake, the other is a fault on our side, and telling them apart is
the difference between a useful sentence and a shrug.
"""
import pytest

import models
import storage
import fake_style
from config import Config
from extensions import InlineQueue, get_queue, reset
from jobs import capture_jobs, clustering_jobs, render_jobs
from pipeline.imaging import CaptureError


@pytest.fixture(autouse=True)
def clean(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path))
    models.clear_memory()
    yield
    models.clear_memory()


@pytest.fixture
def session():
    return models.create_session()


# ---------------------------------------------------------------- the queue

def test_inline_mode_runs_the_work_immediately():
    ran = []
    queue = InlineQueue()
    job = queue.enqueue(lambda value: ran.append(value), 7)
    assert ran == [7]
    assert job.id is None


def test_the_inline_queue_carries_a_job_id_through():
    job = InlineQueue().enqueue(lambda: None, job_id="abc")
    assert job.id == "abc"


def test_a_failing_job_does_not_fail_the_request_that_enqueued_it():
    """
    A real worker records the failure against the job record and keeps
    running. Inline mode has to match that, or a fire-and-forget job turns
    into a 500 on the upload that started it.
    """
    def boom():
        raise RuntimeError("the pipeline fell over")

    assert InlineQueue().enqueue(boom) is not None


def test_the_test_configuration_really_is_running_inline():
    reset()
    try:
        assert isinstance(get_queue(), InlineQueue)
    finally:
        reset()


# --------------------------------------------------------------- capture job

def _fake_segment(instances=530):
    return {
        "instances": [{"char": "a", "advance_em": 0.4, "band": 0,
                       "bounds": [0.0, -0.5, 0.4, 0.0]}
                      for _ in range(instances)],
        "meta": {"em_px": 50.0, "x_height_mm": 3.0, "letter_gap_em": 0.06,
                 "word_gap_em": 0.3, "line_spacing_em": 1.4,
                 "band_offsets": [[i, 0.01 * i] for i in range(16)]},
    }


def test_a_finished_round_reports_a_count_not_every_glyph(session, monkeypatch):
    """The capture screen shows a number. Sending several hundred glyph
    outlines through every status poll to display it would be absurd."""
    monkeypatch.setattr(capture_jobs, "segment_photo",
                        lambda _path: _fake_segment())
    job = models.create_job(session["session_id"], "capture_round")
    capture_jobs.capture_round_job(job["job_id"], session["session_id"], 1,
                                   "unused.jpg")

    stored = models.get_job(job["job_id"])
    assert stored["status"] == "done" and stored["progress"] == 100
    assert stored["result"]["instances"] == 530
    assert isinstance(stored["result"]["instances"], int)
    assert "path" not in str(stored["result"])


def test_the_extracted_characters_are_written_where_clustering_reads_them(
        session, monkeypatch):
    monkeypatch.setattr(capture_jobs, "segment_photo",
                        lambda _path: _fake_segment())
    job = models.create_job(session["session_id"], "capture_round")
    capture_jobs.capture_round_job(job["job_id"], session["session_id"], 2,
                                   "unused.jpg")
    import json
    import os
    path = storage.round_chars_path(session["session_id"], 2)
    assert os.path.exists(path)
    with open(path) as f:
        assert len(json.load(f)["instances"]) == 530


def test_a_photo_the_user_can_retake_gets_their_own_message(session,
                                                            monkeypatch):
    message = ("We could not find any writing on this sheet. Check the photo "
               "is in focus.")

    def refuse(_path):
        raise CaptureError(message)

    monkeypatch.setattr(capture_jobs, "segment_photo", refuse)
    job = models.create_job(session["session_id"], "capture_round")
    capture_jobs.capture_round_job(job["job_id"], session["session_id"], 1,
                                   "unused.jpg")

    stored = models.get_job(job["job_id"])
    assert stored["status"] == "failed"
    assert stored["error"] == message, "the advice was replaced with a shrug"
    round_one = models.get_session(session["session_id"])["rounds"][0]
    assert round_one["status"] == "failed" and round_one["error"] == message


def test_an_internal_failure_does_not_show_the_user_our_problem(session,
                                                                monkeypatch):
    def boom(_path):
        raise ValueError("cv2.error: !ssize.empty() in function 'resize'")

    monkeypatch.setattr(capture_jobs, "segment_photo", boom)
    job = models.create_job(session["session_id"], "capture_round")
    with pytest.raises(ValueError):
        capture_jobs.capture_round_job(job["job_id"],
                                       session["session_id"], 1, "unused.jpg")

    stored = models.get_job(job["job_id"])
    assert stored["status"] == "failed"
    assert "cv2" not in stored["error"] and "resize" not in stored["error"]
    assert "photo" in stored["error"].lower()


def test_a_failed_round_can_be_taken_again(session, monkeypatch):
    calls = []

    def sometimes(_path):
        calls.append(1)
        if len(calls) == 1:
            raise CaptureError("Too dark. Try again in better light.")
        return _fake_segment()

    monkeypatch.setattr(capture_jobs, "segment_photo", sometimes)
    for _ in range(2):
        job = models.create_job(session["session_id"], "capture_round")
        capture_jobs.capture_round_job(job["job_id"], session["session_id"],
                                       1, "unused.jpg")
    round_one = models.get_session(session["session_id"])["rounds"][0]
    assert round_one["status"] == "done" and round_one["error"] is None


def test_clustering_starts_only_once_all_three_sheets_are_in(session,
                                                             monkeypatch):
    """Two sheets is not a handwriting model, and starting on two would give
    the user one built from a third less evidence without saying so."""
    monkeypatch.setattr(capture_jobs, "segment_photo",
                        lambda _path: _fake_segment())
    started = []
    monkeypatch.setattr(clustering_jobs, "enqueue_clustering",
                        lambda sid: started.append(sid))

    for round_n in (1, 2):
        job = models.create_job(session["session_id"], "capture_round")
        capture_jobs.capture_round_job(job["job_id"], session["session_id"],
                                       round_n, "unused.jpg")
        assert started == []

    job = models.create_job(session["session_id"], "capture_round")
    capture_jobs.capture_round_job(job["job_id"], session["session_id"], 3,
                                   "unused.jpg")
    assert started == [session["session_id"]]


# ---------------------------------------------------------------- render job

def test_rendering_without_a_handwriting_model_fails_with_a_message(session):
    job = models.create_job(session["session_id"], "render")
    with pytest.raises(Exception):
        render_jobs.render_job(job["job_id"], session["session_id"], [])
    stored = models.get_job(job["job_id"])
    assert stored["status"] == "failed"
    assert "render" in stored["error"].lower()


def test_a_render_reports_progress_as_it_goes(client, session, tmp_path):
    """The Add Code step has a progress bar, and a job that jumps from 0 to
    100 makes it a spinner with extra steps."""
    seen = []
    original = models.set_progress

    def record(job_id, progress, stage=None):
        seen.append((progress, stage))
        return original(job_id, progress, stage)

    sid = session["session_id"]
    fake_style.install(sid, models)
    models.set_progress = record
    try:
        job = models.create_job(sid, "render")
        path = tmp_path / "a.py"
        path.write_text("x = 1\ny = 2\n")
        render_jobs.render_job(job["job_id"], sid, [("a.py", str(path))])
    finally:
        models.set_progress = original

    progresses = [p for p, _stage in seen]
    assert progresses == sorted(progresses)
    assert progresses[0] < progresses[-1] < 100
    assert all(stage for _p, stage in seen), "a stage with no name"


def test_a_render_result_carries_what_the_preview_needs(session, tmp_path):
    sid = session["session_id"]
    fake_style.install(sid, models)
    job = models.create_job(sid, "render")
    first, second = tmp_path / "a.py", tmp_path / "b.py"
    first.write_text("x = 1\n")
    second.write_text("y = 2\n")
    render_jobs.render_job(job["job_id"], sid,
                           [("a.py", str(first)), ("b.py", str(second))])

    result = models.get_job(job["job_id"])["result"]
    assert result["page_count"] == len(result["pages"]) == 2
    assert [p["file"] for p in result["pages"]] == ["a.py", "b.py"]
    assert [p["index"] for p in result["pages"]] == [0, 1]
    assert result["coverage"]["total"] > 0


def test_each_file_starts_on_its_own_page(session, tmp_path):
    sid = session["session_id"]
    fake_style.install(sid, models)
    job = models.create_job(sid, "render")
    paths = []
    for name in ("a.py", "b.py", "c.py"):
        path = tmp_path / name
        path.write_text("x = 1\n")
        paths.append((name, str(path)))
    render_jobs.render_job(job["job_id"], sid, paths)

    pages = models.get_job(job["job_id"])["result"]["pages"]
    assert [p["page_num"] for p in pages] == [1, 1, 1]


def test_coverage_names_the_characters_that_had_to_be_substituted(session,
                                                                  tmp_path):
    sid = session["session_id"]
    fake_style.install(sid, models, chars=list("xy= 1\n"))
    job = models.create_job(sid, "render")
    path = tmp_path / "a.py"
    path.write_text("x = 1\nz = 2\n")
    render_jobs.render_job(job["job_id"], sid, [("a.py", str(path))])

    coverage = models.get_job(job["job_id"])["result"]["coverage"]
    assert "z" in coverage["missing"] and "2" in coverage["missing"]
    assert coverage["exact"] < coverage["total"]


def test_the_pdf_path_never_leaves_the_server(client, session, tmp_path):
    sid = session["session_id"]
    fake_style.install(sid, models)
    job = models.create_job(sid, "render")
    path = tmp_path / "a.py"
    path.write_text("x = 1\n")
    render_jobs.render_job(job["job_id"], sid, [("a.py", str(path))])

    served = client.get(f"/api/jobs/{job['job_id']}").get_json()["result"]
    assert "pdf_path" not in served
    assert "pdf_available" in served


def test_pages_are_written_one_file_per_page(session, tmp_path):
    """The preview fetches pages one at a time; a page that is not on disk
    under its index is a page the user cannot look at."""
    import os

    sid = session["session_id"]
    fake_style.install(sid, models)
    job = models.create_job(sid, "render")
    path = tmp_path / "a.py"
    path.write_text("\n".join(f"line {i}" for i in range(200)))
    render_jobs.render_job(job["job_id"], sid, [("a.py", str(path))])

    count = models.get_job(job["job_id"])["result"]["page_count"]
    assert count > 1
    for index in range(count):
        assert os.path.exists(storage.job_page_svg_path(job["job_id"], index))

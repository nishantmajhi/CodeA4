"""
The shared record store.

These records are the only thing the web process and the worker process agree
about. Everything the capture screen shows -- which rounds are done, how far a
job has got, whether the style model exists -- is read back out of here by a
process that did not write it, which is why the module exists at all instead
of a dictionary.
"""
import pytest

import models
from config import Config


@pytest.fixture(autouse=True)
def clean_store():
    models.clear_memory()
    yield
    models.clear_memory()


# ------------------------------------------------------------------- session

def test_a_new_session_waits_on_three_rounds():
    session = models.create_session()
    assert [r["n"] for r in session["rounds"]] == [1, 2, 3]
    assert all(r["status"] == "waiting" for r in session["rounds"])
    assert all(r["job_id"] is None and r["error"] is None
               for r in session["rounds"])
    assert session["style_ready"] is False
    assert session["created_at"] > 0


def test_sessions_are_independent():
    first, second = models.create_session(), models.create_session()
    assert first["session_id"] != second["session_id"]
    models.update_round(first["session_id"], 1, status="done")
    reread = models.get_session(second["session_id"])
    assert all(r["status"] == "waiting" for r in reread["rounds"])


def test_an_unknown_session_is_none_not_an_error():
    assert models.get_session("no-such-session") is None
    assert models.update_round("no-such-session", 1, status="done") is None
    assert models.mark_style_ready("no-such-session") is None


def test_updating_one_round_leaves_the_others_alone():
    session = models.create_session()
    models.update_round(session["session_id"], 2, status="done", instances=530)
    rounds = models.get_session(session["session_id"])["rounds"]
    assert rounds[1]["status"] == "done" and rounds[1]["instances"] == 530
    assert rounds[0]["status"] == "waiting" and rounds[2]["status"] == "waiting"


def test_a_round_can_be_retried_after_it_failed():
    session = models.create_session()
    sid = session["session_id"]
    models.update_round(sid, 1, status="failed", error="too dark")
    models.update_round(sid, 1, status="done", error=None)
    round_one = models.get_session(sid)["rounds"][0]
    assert round_one["status"] == "done" and round_one["error"] is None


def test_an_out_of_range_round_changes_nothing():
    session = models.create_session()
    assert models.update_round(session["session_id"], 9, status="done") is None
    assert len(models.get_session(session["session_id"])["rounds"]) == 3


def test_style_ready_is_a_flag_that_can_be_taken_back():
    session = models.create_session()
    sid = session["session_id"]
    models.mark_style_ready(sid, True)
    assert models.get_session(sid)["style_ready"] is True
    models.mark_style_ready(sid, False)
    assert models.get_session(sid)["style_ready"] is False


# ----------------------------------------------------------------------- job

def test_a_new_job_is_queued_at_zero():
    session = models.create_session()
    job = models.create_job(session["session_id"], "render")
    assert job["status"] == "queued"
    assert job["progress"] == 0
    assert job["stage"] is None and job["error"] is None
    assert job["result"] is None
    assert job["session_id"] == session["session_id"]


def test_progress_moves_a_job_into_running():
    job = models.create_job("s", "capture_round")
    models.set_progress(job["job_id"], 40, stage="Extracting characters")
    stored = models.get_job(job["job_id"])
    assert stored["status"] == "running"
    assert stored["progress"] == 40
    assert stored["stage"] == "Extracting characters"


def test_progress_without_a_stage_keeps_the_previous_one():
    """The capture screen shows the stage text; blanking it mid-job would
    make the panel flicker back to nothing."""
    job = models.create_job("s", "render")
    models.set_progress(job["job_id"], 10, stage="Reading your files")
    models.set_progress(job["job_id"], 20)
    assert models.get_job(job["job_id"])["stage"] == "Reading your files"


def test_a_finished_job_keeps_its_result():
    job = models.create_job("s", "render")
    models.update_job(job["job_id"], status="done", progress=100,
                      result={"page_count": 3})
    stored = models.get_job(job["job_id"])
    assert stored["status"] == "done" and stored["result"]["page_count"] == 3


def test_an_unknown_job_is_none_not_an_error():
    assert models.get_job("nope") is None
    assert models.update_job("nope", status="done") is None
    assert models.set_progress("nope", 50) is None


# ---------------------------------------------------------------- style asset

def test_a_style_asset_round_trips():
    sets = [{"id": 1, "share": 1.0, "glyphs": {"a": {"path": "M0,0Z"}},
             "coverage": 1}]
    drift = {"baseline": {"phi": 0.5}, "margin": {"phi": 0.4}}
    models.save_style_asset("s", sets, drift, {"x_height_mm": 3.0})
    stored = models.get_style_asset("s")
    assert stored["style_sets"] == sets
    assert stored["drift"] == drift
    assert stored["meta"]["x_height_mm"] == 3.0


def test_a_style_asset_without_meta_is_still_readable():
    models.save_style_asset("s", [], {})
    assert models.get_style_asset("s")["meta"] == {}


# ----------------------------------------------------------------- deletion

def test_deleting_a_session_forgets_everything_derived_from_it():
    session = models.create_session()
    sid = session["session_id"]
    models.save_style_asset(sid, [], {})
    models.save_push_subscription(sid, {"endpoint": "https://x.example"})

    existed, _job_ids = models.delete_session(sid)
    assert existed is True
    assert models.get_session(sid) is None
    assert models.get_style_asset(sid) is None
    assert models.get_push_subscription(sid) is None


def test_deleting_a_session_that_is_not_there_says_so():
    assert models.delete_session("never-existed") == (False, [])


def test_deleting_a_session_reports_the_jobs_it_started():
    """Job directories are keyed by job id, so the caller needs the list to
    find the files -- there is no index the other way round."""
    session = models.create_session()
    sid = session["session_id"]
    ids = [models.create_job(sid, "render")["job_id"] for _ in range(3)]
    existed, job_ids = models.delete_session(sid)
    assert existed is True and sorted(job_ids) == sorted(ids)
    assert all(models.get_job(job_id) is None for job_id in ids)


def test_deleting_one_session_leaves_another_alone():
    keep = models.create_session()
    models.save_style_asset(keep["session_id"], [{"id": 1}], {})
    gone = models.create_session()
    models.delete_session(gone["session_id"])
    assert models.get_session(keep["session_id"]) is not None
    assert models.get_style_asset(keep["session_id"]) is not None


# --------------------------------------------------------------- the backend

def test_records_survive_being_read_by_another_reader():
    """
    The reason this is not a dictionary: jobs run in a separate process from
    the web server, so a progress update has to be visible to a reader that
    did not write it. Serializing through the store is what makes that true,
    and storing an object by reference would pass this test in-process and
    fail in production.
    """
    job = models.create_job("s", "render")
    live = models.get_job(job["job_id"])
    live["status"] = "done"
    assert models.get_job(job["job_id"])["status"] == "queued"


def test_everything_stored_is_json(monkeypatch):
    import json

    written = {}
    monkeypatch.setattr(models, "_MEMORY", written)
    session = models.create_session()
    models.create_job(session["session_id"], "render")
    models.save_style_asset(session["session_id"], [], {})
    for key, payload in written.items():
        assert isinstance(payload, str), key
        json.loads(payload)


def test_redis_records_are_written_with_an_expiry(monkeypatch):
    """
    Handwriting samples are personal data and there is no account system, so
    nothing is meant to live indefinitely. A record written without a TTL is
    a record that outlives the person's interest in it.
    """
    calls = []

    class FakeRedis:
        def set(self, key, value, ex=None):
            calls.append((key, ex))

        def get(self, _key):
            return None

        def delete(self, _key):
            return 1

    monkeypatch.setattr(Config, "MEMORY_STATE", False)
    monkeypatch.setattr(models, "get_redis", lambda: FakeRedis())
    models.create_session()
    assert calls and all(ex == Config.STATE_TTL_SECONDS for _key, ex in calls)


def test_keys_are_namespaced_so_two_kinds_cannot_collide():
    """A session and a job could be handed the same identifier; they must not
    be able to overwrite one another."""
    models.save_style_asset("same-id", [{"id": 1}], {})
    models.save_push_subscription("same-id", {"endpoint": "https://x.example"})
    assert models.get_style_asset("same-id")["style_sets"] == [{"id": 1}]
    assert models.get_push_subscription("same-id")["endpoint"] == \
        "https://x.example"

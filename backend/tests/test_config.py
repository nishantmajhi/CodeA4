"""
Configuration, which is where this project's security posture lives.

There is no authentication anywhere in the API. That is a defensible choice
for a tool someone runs on their own machine, and it makes the defaults load
bearing: each one has to be the safe answer for a server that is reachable,
because the Dockerfile and the compose file mean some of them will be.

The values are read from the environment once, at import, so these tests call
the parsing functions directly rather than reloading the module -- reloading
it rebinds `config.Config` while every module that has already imported it
keeps the old class, which makes later tests fail for reasons that have
nothing to do with what they assert.
"""
import os
import re

import pytest

from config import Config, parse_flag, parse_origins, parse_seconds


def test_cross_origin_access_is_not_open_by_default():
    """
    `CORS_ORIGINS` defaulted to `*`. With no authentication anywhere, that
    means any page the user visits can call this API from their browser and
    read a session, its style model and its rendered pages, given only the
    session id. The app serves its own front-end from the same origin, so the
    ordinary deployment needs no cross-origin access at all.
    """
    default = parse_origins(None)
    assert default != "*" and "*" not in default
    assert all("localhost" in o or "127.0.0.1" in o for o in default)
    assert Config.CORS_ORIGINS == default or os.environ.get("CORS_ORIGINS")


def test_cors_origins_can_be_configured():
    assert parse_origins("https://a.example, https://b.example") == \
        ["https://a.example", "https://b.example"]
    assert parse_origins("  ") == parse_origins(None)


def test_a_deployment_can_still_open_it_deliberately():
    """Refusing `*` outright would make the setting a lie. It has to be
    reachable on purpose -- just not by default."""
    assert parse_origins("*") == "*"


def test_an_unknown_origin_gets_no_cors_header(monkeypatch):
    """The end-to-end version of the above, through a real app."""
    import app as app_module

    class Defaults(Config):
        CORS_ORIGINS = parse_origins(None)

    monkeypatch.setattr(app_module, "Config", Defaults)
    test_client = app_module.create_app().test_client()
    response = test_client.get("/api/health",
                               headers={"Origin": "https://evil.example"})
    allowed = response.headers.get("Access-Control-Allow-Origin")
    assert allowed not in ("*", "https://evil.example")


def test_the_debugger_is_off_unless_asked_for():
    """
    Werkzeug's debugger offers a Python console on any traceback, which is
    remote code execution for anyone who can reach it.
    """
    assert parse_flag(None) is False
    assert parse_flag("1") is True
    assert parse_flag("true") is True
    assert parse_flag("TRUE") is True
    assert parse_flag("0") is False
    assert parse_flag("") is False


def test_the_entry_point_does_not_hard_code_the_debugger():
    """
    `app.py` ended with `app.run(debug=True)`, so `python app.py` -- the most
    obvious way to start it, and the one the module docstring shows -- served
    the interactive debugger. A test on the flag above cannot see that, so
    this one reads the file.
    """
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source = open(os.path.join(here, "app.py"), encoding="utf-8").read()
    assert not re.search(r"debug\s*=\s*True", source)
    assert "app.run(" in source


def test_retention_window_is_finite_and_configurable():
    assert 0 < Config.STATE_TTL_SECONDS <= 60 * 60 * 24 * 30
    assert Config.DATA_RETENTION_SECONDS >= Config.STATE_TTL_SECONDS
    assert parse_seconds("60", 99) == 60
    assert parse_seconds(None, 99) == 99
    assert parse_seconds("not a number", 99) == 99
    assert parse_seconds("-5", 99) == 99


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " on "])
def test_flags_read_the_usual_spellings(value):
    assert parse_flag(value) is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "maybe"])
def test_anything_else_is_off(value):
    assert parse_flag(value) is False

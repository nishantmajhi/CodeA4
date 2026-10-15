"""
Env-driven configuration. Keep this the only place that reads os.environ.

The values are parsed by the small functions below rather than inline, so the
rules can be tested without reimporting this module -- a reload rebinds
``Config`` while every module that already imported it keeps the old class,
which is a hard failure to read.

There is no authentication anywhere in this API, which makes these defaults
part of the security posture rather than a convenience: each one has to be the
safe answer for a reachable server, because the Dockerfile and the compose
file mean some deployments will be reachable.
"""
import os

_HERE = os.path.dirname(__file__)

# Where the front-end is normally opened from during development. The app
# serves the PWA itself, from its own origin, so the ordinary deployment needs
# no cross-origin access at all; these cover running a separate dev server.
DEFAULT_CORS_ORIGINS = [
    "http://localhost:5000", "http://127.0.0.1:5000",
    "http://localhost:5173", "http://127.0.0.1:5173",
]


def parse_flag(raw, default=False):
    """Read a boolean environment value. Anything unrecognised is off."""
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def parse_origins(raw):
    """
    Read CORS_ORIGINS: a comma-separated list, or the literal ``*``.

    ``*`` is still available, because a setting that cannot express "open"
    would be a lie — but it has to be asked for. It used to be the default,
    which with unauthenticated endpoints meant any page the user visited could
    read their session, their handwriting model and their rendered pages given
    only the session id.
    """
    if raw is None or not raw.strip():
        return list(DEFAULT_CORS_ORIGINS)
    if raw.strip() == "*":
        return "*"
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_seconds(raw, default):
    """Read a positive duration, falling back rather than failing to start."""
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


class Config:
    REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    DATA_DIR = os.environ.get("DATA_DIR", os.path.join(_HERE, "data"))
    MAX_CONTENT_LENGTH = 25 * 1024 * 1024  # 25MB — a phone photo or a few source files
    CORS_ORIGINS = parse_origins(os.environ.get("CORS_ORIGINS"))

    # Werkzeug's debugger offers a Python console on any traceback, which is
    # remote code execution for anyone who can reach it. Off unless asked for.
    DEBUG = parse_flag(os.environ.get("FLASK_DEBUG"))

    # Where the PWA lives. Served by this app so the service worker's scope
    # covers the API too, which avoids a cross-origin setup for the common
    # single-server deployment.
    FRONTEND_DIR = os.environ.get(
        "FRONTEND_DIR", os.path.abspath(os.path.join(_HERE, "..", "frontend"))
    )

    # Run jobs inline in the web process instead of handing them to a worker.
    # Intended for development and for the test suite, where standing up Redis
    # and a second process to check a route's behaviour is not worth it. In
    # this mode a request blocks for as long as the pipeline takes, which is
    # exactly what the queue exists to avoid in production.
    INLINE_JOBS = parse_flag(os.environ.get("INLINE_JOBS"))

    # Keep session state in process memory rather than Redis. Only safe for a
    # single-process dev server; see models.py.
    MEMORY_STATE = parse_flag(os.environ.get("MEMORY_STATE"))

    # How long session state lives in Redis. Handwriting samples are personal
    # data and there is no account system, so state is deliberately temporary.
    STATE_TTL_SECONDS = parse_seconds(os.environ.get("STATE_TTL_SECONDS"),
                                      60 * 60 * 24 * 7)

    # ...and how long the *files* live. Expiring the Redis record only makes a
    # session unreachable; the photographs, the extracted characters, the
    # style model and the rendered pages stayed on disk indefinitely. This is
    # the window after which they are swept, and it is deliberately not
    # shorter than the state TTL, so nothing is deleted out from under a
    # session that is still usable.
    DATA_RETENTION_SECONDS = max(
        parse_seconds(os.environ.get("DATA_RETENTION_SECONDS"),
                      60 * 60 * 24 * 7),
        STATE_TTL_SECONDS,
    )

    # Web push (VAPID). Generate a key pair with:
    #   python -c "from py_vapid import Vapid01; v=Vapid01(); v.generate_keys(); ..."
    # Push is simply disabled when these are absent, rather than erroring —
    # it is an enhancement, not a requirement for the flow to work.
    VAPID_PUBLIC_KEY = os.environ.get("VAPID_PUBLIC_KEY", "")
    VAPID_PRIVATE_KEY = os.environ.get("VAPID_PRIVATE_KEY", "")
    VAPID_SUBJECT = os.environ.get("VAPID_SUBJECT", "mailto:admin@example.com")

    @classmethod
    def push_enabled(cls):
        return bool(cls.VAPID_PUBLIC_KEY and cls.VAPID_PRIVATE_KEY)

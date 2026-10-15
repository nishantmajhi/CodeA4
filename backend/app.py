"""
Flask app factory.

This process serves the API and the PWA's static files. It never runs Stage
1-8 processing inline (Backend Plan §1) — pipeline work is handed to the queue
and executed by `worker.py`.

Local dev, the full setup:
    redis-server &
    python worker.py &
    flask --app app run --debug

Local dev, no Redis (jobs run inline, requests block):
    INLINE_JOBS=1 MEMORY_STATE=1 flask --app app run --debug

`python app.py` also works and is the shortest thing to type, which is why
the debugger it starts is gated on FLASK_DEBUG rather than switched on here —
Werkzeug's debugger hands a Python console to anyone who can reach a
traceback, and that is not something to enable by being the default.
"""
import logging
import os

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

from api import register_blueprints
from config import Config


def create_app():
    app = Flask(__name__, static_folder=None)
    app.config.from_object(Config)
    CORS(app, origins=Config.CORS_ORIGINS)
    logging.basicConfig(level=logging.INFO)

    register_blueprints(app)

    @app.get("/api/health")
    def health():
        return jsonify({
            "status": "ok",
            "inline_jobs": Config.INLINE_JOBS,
            "push_enabled": Config.push_enabled(),
        })

    _register_frontend(app)

    @app.errorhandler(400)
    @app.errorhandler(404)
    @app.errorhandler(409)
    @app.errorhandler(413)
    def handle_error(e):
        return jsonify({"error": getattr(e, "description", str(e))}), e.code

    @app.errorhandler(500)
    def handle_server_error(e):
        # Never surface a traceback or internal path to the browser.
        app.logger.exception("unhandled error")
        return jsonify({"error": "Something went wrong on our end."}), 500

    return app


def _json_404(description):
    return jsonify({"error": description}), 404


def _register_frontend(app):
    """
    Serve the PWA from the same origin as the API.

    This matters for more than convenience: a service worker can only control
    pages within its own scope, so serving `sw.js` from the app's root is what
    lets it cache the shell and receive push events. Hosting the front-end
    elsewhere would need its own service worker and a CORS configuration for
    every API call.
    """
    root = Config.FRONTEND_DIR
    if not os.path.isdir(root):
        app.logger.warning("frontend directory not found: %s", root)
        return

    @app.get("/")
    def index():
        return send_from_directory(root, "index.html")

    @app.get("/<path:filename>")
    def static_files(filename):
        if filename == "api" or filename.startswith("api/"):
            # The shell is the right answer for an unknown *page*, and the
            # wrong one for an unknown API path: the front-end checks the
            # status before the body, so 200 with a page in it is not an
            # error there — it is a successful call whose body is a string,
            # and the failure resurfaces somewhere unrelated as a missing
            # property. Answer in the shape every other endpoint answers in.
            return _json_404("No such endpoint")

        full = os.path.join(root, filename)
        if not os.path.isfile(full):
            # Unknown paths fall back to the shell so client-side routing and
            # deep links (e.g. /#preview from a notification) still load.
            return send_from_directory(root, "index.html")
        response = send_from_directory(root, filename)
        if filename == "sw.js":
            # A stale service worker would keep serving an old app forever.
            response.headers["Cache-Control"] = "no-cache"
        return response


app = create_app()

if __name__ == "__main__":
    app.run(debug=Config.DEBUG)

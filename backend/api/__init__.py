from api.exercise import exercise_bp
from api.jobs import jobs_bp
from api.push import push_bp
from api.render import render_bp
from api.sessions import sessions_bp
from api.style import style_bp


def register_blueprints(app):
    for blueprint in (sessions_bp, style_bp, render_bp, jobs_bp, push_bp,
                      exercise_bp):
        app.register_blueprint(blueprint, url_prefix="/api")

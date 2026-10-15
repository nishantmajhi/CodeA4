from flask import Blueprint, abort, jsonify

import models

style_bp = Blueprint("style", __name__)


@style_bp.get("/sessions/<session_id>/style")
def get_style(session_id):
    """UI: Your Style step — style cards and the drift chart."""
    session = models.get_session(session_id)
    if not session:
        abort(404, description="Unknown session")
    if not session.get("style_ready"):
        return jsonify({"style_ready": False}), 202

    asset = models.get_style_asset(session_id)
    if not asset:
        return jsonify({"style_ready": False}), 202

    return jsonify({"style_ready": True, **asset})


@style_bp.get("/sessions/<session_id>/style/sample.svg")
def style_sample(session_id):
    """
    A rendered sample of one style set, for the style cards.

    Returned as SVG rather than as raw glyph paths so the UI does not have to
    reimplement advance-width layout — which is engine logic, and would drift
    out of step with the renderer the moment either side changed.
    """
    from flask import Response, request

    from pipeline.render_svg import sample_svg

    asset = models.get_style_asset(session_id)
    if not asset:
        abort(404, description="No style assets for this session")

    try:
        style_id = int(request.args.get("style", 1))
    except ValueError:
        abort(400, description="style must be a number")

    text = request.args.get("text", "handwriting 123")
    svg = sample_svg(asset["style_sets"], asset.get("meta", {}), style_id, text)
    if svg is None:
        abort(404, description="Unknown style")
    return Response(svg, mimetype="image/svg+xml")

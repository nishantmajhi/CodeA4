"""The capture exercise text, served so the UI and the segmenter agree."""
from flask import Blueprint, jsonify

from pipeline.exercise import (
    EXERCISE_LINES, INK_LINES, LONG_LINE_INDEX, EXERCISE_TEXT,
)

exercise_bp = Blueprint("exercise", __name__)


@exercise_bp.get("/exercise")
def get_exercise():
    """
    UI: the block of text shown on the Capture step.

    Served from the same constant the segmenter aligns against. If the UI
    hard-coded its own copy, an edit to one would silently break the other:
    the segmenter would be matching ink against text the user never wrote.
    """
    return jsonify({
        "lines": EXERCISE_LINES,
        "text": EXERCISE_TEXT,
        "ink_lines": len(INK_LINES),
        "long_line_index": LONG_LINE_INDEX,
        "rounds": 3,
    })

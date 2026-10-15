"""The exercise text is ground truth for everything downstream."""
from pipeline.exercise import (
    EXERCISE_LINES, INK_LINES, LONG_LINE_INDEX, coverage_report,
)


def test_every_renderable_character_is_covered():
    """
    A character the exercise never asks for cannot be rendered from the user's
    own hand, only substituted. This test is what stops an innocuous edit to
    the exercise text from silently degrading output for, say, every file
    containing a backtick.
    """
    missing, _thin = coverage_report()
    assert missing == [], f"exercise never asks the user to write: {missing}"


def test_no_character_appears_only_once():
    """
    A single instance per sheet is three across all rounds — enough to trace a
    glyph from, but too few to tell a genuine second way of writing it from a
    smudge. Clustering needs more than that.
    """
    _missing, thin = coverage_report()
    assert thin == [], f"written only once per sheet: {thin}"


def test_long_line_is_actually_the_longest():
    """
    The wrap behaviour can only be observed if the marked line really is too
    long to fit, so the index must keep pointing at the longest line.
    """
    longest = max(range(len(INK_LINES)), key=lambda i: len(INK_LINES[i]))
    assert LONG_LINE_INDEX == longest


def test_ink_lines_excludes_blanks():
    assert all(line.strip() for line in INK_LINES)
    assert len(INK_LINES) < len(EXERCISE_LINES)

"""
Stages 6 and 7: where text becomes a page.

Everything here is measured in em and only becomes points when the SVG is
emitted, so these tests mostly ask geometric questions -- does every character
survive wrapping, does every line reach a page, does anything land outside the
margins -- rather than comparing rendered output to a stored blob. A stored
blob would fail on every legitimate change to the drift model and prove
nothing about layout.
"""
import re

import numpy as np
import pytest

import fake_style
from pipeline import drift
from pipeline.glyph_assets import coverage_summary, resolve_glyph
from pipeline.render_svg import (A4_HEIGHT_PT, A4_WIDTH_PT, MARGIN_BOTTOM_MM,
                                 MARGIN_LEFT_MM, MARGIN_RIGHT_MM, MARGIN_TOP_MM,
                                 MM_PT, Metrics, _paginate, assign_block_styles,
                                 compose_page_svg, expand_tabs,
                                 find_block_boundaries, layout_file,
                                 render_file_to_pages, sample_svg, wrap_line)

TRANSFORM = re.compile(r"translate\((-?[\d.]+),(-?[\d.]+)\)")


@pytest.fixture
def style_sets():
    return fake_style.style_sets()


@pytest.fixture
def metrics(style_sets):
    return Metrics(style_sets, fake_style.META)


class FixedMetrics(Metrics):
    """Every character the same width, so wrapping maths is exact."""

    def __init__(self, advance=0.5):
        self.letter_gap = 0.0
        self.word_gap = 0.5
        self.line_spacing = 1.5
        self.x_height_mm = 3.0
        self.styles = {}
        self._advance = advance

    def advance(self, _char, _style_id):
        return self._advance


# ---------------------------------------------------------------- whitespace

@pytest.mark.parametrize("line,expected", [
    ("", ""),
    ("\t", "    "),
    ("a\tb", "a   b"),
    ("abc\td", "abc d"),
    ("abcd\te", "abcd    e"),
    ("\t\t", "        "),
    ("  \tx", "    x"),
])
def test_a_tab_reaches_the_next_tab_stop(line, expected):
    assert expand_tabs(line) == expected


def test_tabs_and_spaces_line_up_with_each_other():
    """A tab-indented block and a four-space-indented block have to end up at
    the same place, which is what the writer would have done by hand."""
    assert expand_tabs("\tif (x) {") == expand_tabs("    if (x) {")


# ------------------------------------------------------------------ wrapping

def test_wrapping_never_loses_a_character():
    """Property check over random lines and widths. Only spaces may vanish,
    and only where the break was taken at one."""
    rng = np.random.default_rng(11)
    alphabet = list("abcdefgh ij  klmnop")
    metrics = FixedMetrics()
    for _ in range(3000):
        line = "".join(rng.choice(alphabet, rng.integers(0, 60)))
        width = float(rng.uniform(0.6, 8.0))
        rows = wrap_line(line, 1, metrics, content_width_em=width)
        assert "".join(rows).replace(" ", "") == line.replace(" ", "")


def test_wrapping_prefers_a_late_word_break():
    rows = wrap_line("aaa bbb ccc ddd", 1, FixedMetrics(), content_width_em=5.0)
    assert rows[0] == "aaa bbb"
    assert not any(row.endswith("-") for row in rows), "no continuation marker"


def test_a_long_token_breaks_mid_word():
    rows = wrap_line("abcdefghijklmnop", 1, FixedMetrics(), content_width_em=3.0)
    assert len(rows) > 1
    assert "".join(rows) == "abcdefghijklmnop"


def test_indentation_survives_wrapping():
    line = "        deeply_indented_call(argument_one, argument_two)"
    rows = wrap_line(line, 1, FixedMetrics(), content_width_em=8.0)
    assert rows[0].startswith("        ")


def test_a_character_wider_than_the_line_still_gets_a_line():
    """A hand that runs out of paper mid-letter has nowhere to go; the layout
    must terminate rather than loop."""
    rows = wrap_line("abcd", 1, FixedMetrics(advance=5.0), content_width_em=1.0)
    assert rows == ["a", "b", "c", "d"]


@pytest.mark.parametrize("line", ["", " ", "    "])
def test_an_empty_or_blank_line_stays_one_line(line):
    assert wrap_line(line, 1, FixedMetrics(), content_width_em=3.0) == [line]


# ------------------------------------------------------------ style blocks

def test_block_boundaries_are_language_agnostic():
    """The product accepts any source file, so structure is read from layout
    rather than from `def` -- keying off Python would fail on everything."""
    boundaries = find_block_boundaries([
        "function first() {", "    return 1;", "}", "",
        "function second() {", "    return 2;", "}",
    ])
    assert 0 in boundaries
    assert 4 in boundaries       # blank line, then a new top-level unit
    assert 1 not in boundaries   # inside a block


def test_a_return_to_the_left_margin_starts_a_block():
    boundaries = find_block_boundaries(["def a():", "    x = 1", "def b():"])
    assert 2 in boundaries


def test_block_detection_reads_expanded_tabs_not_raw_ones():
    """Indentation is counted in spaces, so the expansion has to happen
    first. layout_file does it in that order; this pins the order down."""
    raw = ["def a():", "\treturn 1", "def b():"]
    assert 2 not in find_block_boundaries(raw)
    assert 2 in find_block_boundaries([expand_tabs(line) for line in raw])


def test_style_is_constant_within_a_block():
    """Switching style per line would be the "one glyph plus jitter" the spec
    rejects; it may only change at a natural break."""
    lines = ["def a():", "    x = 1", "    y = 2", "    return x", "",
             "def b():", "    return 2"]
    rng = drift.make_rng("session", "file", "styles")
    assigned = assign_block_styles(lines, [1, 2, 3], [0.5, 0.3, 0.2], rng)
    assert len(set(assigned[0:4])) == 1
    assert len(set(assigned[5:7])) == 1


def test_every_assigned_style_exists():
    rng = drift.make_rng("session", "file", "styles")
    assigned = assign_block_styles(["a", "", "b", "", "c"] * 10, [1, 2],
                                   [0.6, 0.4], rng)
    assert set(assigned) <= {1, 2}


def test_style_assignment_follows_the_measured_shares():
    """Styles are drawn in proportion to how much of the writer's samples
    each represents, so the usual hand dominates."""
    rng = drift.make_rng("session", "file", "styles")
    lines = [line for _ in range(400) for line in ("x = 1", "")]
    assigned = assign_block_styles(lines, [1, 2], [0.8, 0.2], rng)
    share = assigned.count(1) / len(assigned)
    assert 0.7 < share < 0.9


# ---------------------------------------------------------------- pagination

def test_no_line_is_dropped_or_duplicated_by_pagination(metrics):
    for count in (0, 1, 5, 40, 200, 1000):
        written = [{"text": f"line {i}", "style_id": 1} for i in range(count)]
        pages = _paginate("f.py", written, metrics)
        assert [row for page in pages for row in page["lines"]] == written


def test_only_the_first_page_of_a_file_gets_the_header(metrics):
    written = [{"text": "x", "style_id": 1} for _ in range(200)]
    pages = _paginate("f.py", written, metrics)
    assert len(pages) > 1
    assert pages[0]["filename_header"] == "f.py"
    assert all(page["filename_header"] is None for page in pages[1:])


def test_page_numbers_run_from_one(metrics):
    written = [{"text": "x", "style_id": 1} for _ in range(200)]
    pages = _paginate("f.py", written, metrics)
    assert [page["page_num"] for page in pages] == list(range(1, len(pages) + 1))


def test_an_empty_file_still_produces_a_page(metrics):
    pages = _paginate("empty.py", [], metrics)
    assert len(pages) == 1
    assert pages[0]["lines"] == []
    assert pages[0]["filename_header"] == "empty.py"


def test_the_first_page_holds_fewer_lines_than_the_rest(metrics):
    """The header and its rule take room that later pages do not spend."""
    written = [{"text": "x", "style_id": 1} for _ in range(400)]
    pages = _paginate("f.py", written, metrics)
    assert len(pages[0]["lines"]) < len(pages[1]["lines"])


def test_a_larger_hand_fits_fewer_lines_on_a_page(style_sets):
    """How large a person writes is measured, and it decides the page."""
    written = [{"text": "x", "style_id": 1} for _ in range(400)]
    small = _paginate("f.py", written, Metrics(style_sets,
                                               {**fake_style.META,
                                                "x_height_mm": 2.0}))
    large = _paginate("f.py", written, Metrics(style_sets,
                                               {**fake_style.META,
                                                "x_height_mm": 5.0}))
    assert len(small) < len(large)


# ----------------------------------------------------------- page composition

def _marks(svg):
    return [(float(x), float(y)) for x, y in TRANSFORM.findall(svg)]


def test_every_non_space_character_becomes_one_mark(style_sets, metrics):
    page = {"filename_header": None, "page_num": 1,
            "lines": [{"text": "abc def", "style_id": 1},
                      {"text": "gh", "style_id": 1}]}
    models = {"baseline": drift.BaselineModel(fake_style.DRIFT["baseline"],
                                              drift.make_rng("s", "f")),
              "margin": drift.DriftModel(fake_style.DRIFT["margin"],
                                         drift.make_rng("s", "f"))}
    svg = compose_page_svg(page, metrics, models, ("s", "f"))
    assert svg.count("<path") == len("abcdefgh")


def test_nothing_is_drawn_outside_the_printable_area(style_sets):
    """A page is meant to be printed. Ink past the margin is ink the printer
    may simply not put down."""
    lines = ["    some_function(argument_one, argument_two, argument_three)"
             for _ in range(60)]
    pages = render_file_to_pages("long.py", lines, style_sets,
                                 fake_style.DRIFT, fake_style.META,
                                 session_id="s")
    for page in pages:
        for x, y in _marks(page["svg"]):
            assert -2 <= x <= A4_WIDTH_PT - MARGIN_RIGHT_MM * MM_PT
            assert MARGIN_TOP_MM * MM_PT - 2 <= y <= \
                A4_HEIGHT_PT - MARGIN_BOTTOM_MM * MM_PT


def test_the_page_is_a4_in_points(style_sets):
    pages = render_file_to_pages("a.py", ["x = 1"], style_sets,
                                 fake_style.DRIFT, fake_style.META,
                                 session_id="s")
    assert f'width="{A4_WIDTH_PT:.2f}"' in pages[0]["svg"]
    assert f'height="{A4_HEIGHT_PT:.2f}"' in pages[0]["svg"]


def test_the_header_is_written_above_the_first_line(style_sets):
    pages = render_file_to_pages("header.py", ["x = 1"], style_sets,
                                 fake_style.DRIFT, fake_style.META,
                                 session_id="s")
    marks = _marks(pages[0]["svg"])
    assert "<line" in pages[0]["svg"], "the rule under the header is missing"
    assert min(y for _x, y in marks) < MARGIN_TOP_MM * MM_PT + 30


def test_a_blank_line_leaves_a_blank_line(style_sets, metrics):
    page = {"filename_header": None, "page_num": 1,
            "lines": [{"text": "a", "style_id": 1},
                      {"text": "", "style_id": 1},
                      {"text": "b", "style_id": 1}]}
    models = {"baseline": drift.BaselineModel(fake_style.DRIFT["baseline"],
                                              drift.make_rng("s", "f")),
              "margin": drift.DriftModel(fake_style.DRIFT["margin"],
                                         drift.make_rng("s", "f"))}
    svg = compose_page_svg(page, metrics, models, ("s", "f"))
    ys = sorted({y for _x, y in _marks(svg)})
    assert len(ys) == 2
    gap = ys[1] - ys[0]
    assert gap == pytest.approx(2 * metrics.line_spacing * metrics.em_pt(),
                                rel=0.2)


# -------------------------------------------------------------- determinism

def test_the_same_render_twice_is_byte_identical(style_sets):
    """The preview and the PDF downloaded after it are two separate renders;
    if they differed, the user would be printing something they never saw."""
    lines = ["def a():", "    return 1"]
    first = render_file_to_pages("a.py", lines, style_sets, fake_style.DRIFT,
                                 fake_style.META, session_id="s")
    second = render_file_to_pages("a.py", lines, style_sets, fake_style.DRIFT,
                                  fake_style.META, session_id="s")
    assert [p["svg"] for p in first] == [p["svg"] for p in second]


def test_two_sessions_do_not_share_a_wobble(style_sets):
    lines = ["def a():", "    return 1"]
    one = render_file_to_pages("a.py", lines, style_sets, fake_style.DRIFT,
                               fake_style.META, session_id="one")
    two = render_file_to_pages("a.py", lines, style_sets, fake_style.DRIFT,
                               fake_style.META, session_id="two")
    assert one[0]["svg"] != two[0]["svg"]


def test_two_files_with_the_same_name_do_not_share_a_wobble(style_sets):
    """
    The seed was session id plus filename, so two files uploaded under the
    same name -- `main.py` from two different folders, which is an ordinary
    thing to do -- rendered with byte-identical drift. That is the failure
    drift.py's docstring names: "every page would repeat the same wobble, and
    a reader comparing two pages would see the identical wave twice."

    Until uploads stopped overwriting each other this could not be observed,
    because only one of the two files reached the renderer at all.
    """
    lines = ["def a():", "    return 1"]
    first = render_file_to_pages("main.py", lines, style_sets,
                                 fake_style.DRIFT, fake_style.META,
                                 session_id="s", file_index=0)
    second = render_file_to_pages("main.py", lines, style_sets,
                                  fake_style.DRIFT, fake_style.META,
                                  session_id="s", file_index=1)
    assert first[0]["svg"] != second[0]["svg"]


def test_successive_pages_of_one_file_differ(style_sets):
    lines = ["line {}".format(i) for i in range(120)]
    pages = render_file_to_pages("a.py", lines, style_sets, fake_style.DRIFT,
                                 fake_style.META, session_id="s")
    assert len(pages) > 1
    assert pages[0]["svg"] != pages[1]["svg"]


# --------------------------------------------------------------- style cards

def test_a_sample_renders_the_requested_style(style_sets):
    svg = sample_svg(style_sets, fake_style.META, 1, "handwriting 123")
    assert svg.startswith("<svg")
    assert svg.count("<path") == len("handwriting123")


def test_an_unknown_style_has_no_sample(style_sets):
    assert sample_svg(style_sets, fake_style.META, 99, "abc") is None


def test_a_sample_of_nothing_is_still_valid_svg(style_sets):
    svg = sample_svg(style_sets, fake_style.META, 1, "   ")
    assert svg.startswith("<svg") and svg.endswith("</svg>")


def test_a_sample_is_tall_enough_for_its_letters(style_sets):
    svg = sample_svg(style_sets, fake_style.META, 1, "hy")
    height = float(re.search(r'height="([\d.]+)"', svg).group(1))
    assert height > 0.5 * Metrics(style_sets, fake_style.META).em_pt()


# ------------------------------------------------------------ glyph fallback

def test_the_fallback_chain_is_followed_in_order():
    glyphs = {"a": {"m": "a"}, "'": {"m": "'"}, "-": {"m": "-"},
              "?": {"m": "?"}}
    assert resolve_glyph(glyphs, "a")[1:] == ("a", True)
    assert resolve_glyph(glyphs, "A")[1:] == ("a", False)      # other case
    assert resolve_glyph(glyphs, "’")[1:] == ("'", False)  # substitute
    assert resolve_glyph(glyphs, "—")[1:] == ("-", False)
    assert resolve_glyph(glyphs, "é")[1:] == ("?", False)  # placeholder


def test_a_missing_glyph_with_no_placeholder_is_reported_not_guessed():
    glyph, resolved, exact = resolve_glyph({"a": {}}, "z")
    assert glyph is None and resolved == "z" and exact is False


def test_coverage_counts_only_what_was_actually_observed():
    summary = coverage_summary([{"id": 1, "glyphs": {"a": {}, "b": {}}}],
                               "a b\tc\n")
    assert summary == {"total": 3, "exact": 2, "missing": ["c"]}


def test_coverage_of_a_style_set_that_does_not_exist():
    assert coverage_summary([], "abc") == {"total": 0, "exact": 0, "missing": []}


def test_whitespace_is_never_counted_as_a_missing_glyph():
    summary = coverage_summary([{"id": 1, "glyphs": {"a": {}}}], "a \t\na")
    assert summary["total"] == 2 and summary["missing"] == []


# ------------------------------------------------------------------- metrics

def test_a_space_advances_by_the_measured_word_gap(metrics):
    assert metrics.advance(" ", 1) == pytest.approx(fake_style.META["word_gap_em"])


def test_measuring_a_string_is_the_sum_of_its_advances(metrics):
    assert metrics.measure("abc", 1) == pytest.approx(
        sum(metrics.advance(c, 1) for c in "abc"))


def test_line_spacing_has_a_floor(style_sets):
    """A cramped exercise sheet must not produce output whose ascenders and
    descenders collide."""
    cramped = Metrics(style_sets, {**fake_style.META, "line_spacing_em": 0.4})
    assert cramped.line_spacing >= 1.3


def test_the_em_follows_how_large_the_person_writes(style_sets):
    small = Metrics(style_sets, {**fake_style.META, "x_height_mm": 2.0})
    large = Metrics(style_sets, {**fake_style.META, "x_height_mm": 4.0})
    assert large.em_pt() == pytest.approx(2 * small.em_pt())


def test_layout_uses_the_whole_content_width(style_sets, metrics):
    """A wrap that fires early wastes the page; one that fires late runs off
    it. Neither is visible from a line count alone."""
    lines = ["x" * 400]
    pages = layout_file("a.py", lines, metrics, style_sets, ("s", "a.py"))
    rows = [row["text"] for page in pages for row in page["lines"]]
    content_width = (A4_WIDTH_PT - (MARGIN_LEFT_MM + MARGIN_RIGHT_MM) * MM_PT) \
        / metrics.em_pt()
    for row in rows[:-1]:
        measured = metrics.measure(row, 1)
        assert measured <= content_width + 1e-6
        assert measured > 0.75 * content_width

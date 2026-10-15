"""
Stage 5, which is one function, and is where a file's bytes become the text
that will be written out by hand.

The spec's requirement is that nothing is reformatted: indentation, tabs and
trailing spaces are the author's and are reproduced as they are. The only
decision this stage makes is how to turn bytes into characters, and getting
that wrong is visible on paper as a wrong character rather than as an error.
"""
from pipeline.code_parser import parse_source


def _write(tmp_path, name, body):
    path = tmp_path / name
    path.write_bytes(body)
    return str(path)


def test_lines_come_back_exactly_as_written(tmp_path):
    body = b"def f():\n\tif x:\n        return 1   \n"
    lines = parse_source(_write(tmp_path, "a.py", body))
    assert lines == ["def f():", "\tif x:", "        return 1   "]


def test_windows_line_endings_are_not_left_on_the_line(tmp_path):
    lines = parse_source(_write(tmp_path, "a.py", b"a = 1\r\nb = 2\r\n"))
    assert lines == ["a = 1", "b = 2"]


def test_a_file_with_no_trailing_newline_keeps_its_last_line(tmp_path):
    assert parse_source(_write(tmp_path, "a.py", b"x = 1")) == ["x = 1"]


def test_an_empty_file_has_no_lines(tmp_path):
    assert parse_source(_write(tmp_path, "a.py", b"")) == []


def test_blank_lines_are_kept(tmp_path):
    lines = parse_source(_write(tmp_path, "a.py", b"a\n\n\nb\n"))
    assert lines == ["a", "", "", "b"]


def test_a_byte_order_mark_is_not_a_character(tmp_path):
    """
    Notepad, and Visual Studio, and PowerShell's `>` all write UTF-8 with a
    byte-order mark. Read as plain UTF-8 the mark survives as U+FEFF, which
    has no glyph -- so the first character of the file came out as the
    placeholder '?', and the preview warned the user that a character they
    cannot see was not in their exercise sheets.
    """
    path = _write(tmp_path, "a.py", "\ufeffprint(1)\n".encode("utf-8"))
    assert parse_source(path) == ["print(1)"]


def test_a_utf16_file_is_read_as_text_not_as_noise(tmp_path):
    """
    PowerShell wrote UTF-16 by default for years, so a redirected file is a
    plausible upload. Decoded as UTF-8 it came back as a replacement
    character followed by a NUL between every letter -- a page of
    placeholders where the source should be.
    """
    for encoding in ("utf-16", "utf-16-le", "utf-16-be"):
        body = "x = 1\ny = 2\n".encode(encoding)
        if encoding != "utf-16":
            body = ("\ufeff" + "x = 1\ny = 2\n").encode(encoding)
        assert parse_source(_write(tmp_path, f"{encoding}.py", body)) == \
            ["x = 1", "y = 2"], encoding


def test_undecodable_bytes_become_a_visible_placeholder(tmp_path):
    """
    A file in an encoding we cannot identify is still rendered, with the
    bytes we could not read shown rather than silently dropped. Refusing the
    file outright would be worse: most of it is readable.
    """
    lines = parse_source(_write(tmp_path, "a.py",
                                "caf\u00e9 = 1\n".encode("latin-1")))
    assert len(lines) == 1
    assert lines[0].startswith("caf") and lines[0].endswith(" = 1")


def test_unicode_in_a_utf8_file_survives(tmp_path):
    path = _write(tmp_path, "a.py", "s = \"\u65e5\u672c\u8a9e\"\n".encode("utf-8"))
    assert parse_source(path) == ['s = "\u65e5\u672c\u8a9e"']


def test_a_very_long_line_is_not_truncated(tmp_path):
    body = ("x" * 5000 + "\n").encode("utf-8")
    assert len(parse_source(_write(tmp_path, "a.py", body))[0]) == 5000

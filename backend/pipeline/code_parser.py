"""
Spec Stage 5 — Code parsing

Read the uploaded file as literal text, preserving whitespace, tabs and
indentation exactly (no reformatting, per spec §3/§5).

The only decision this stage makes is how to turn bytes into characters, and
it is worth making carefully, because getting it wrong is not an error the
user sees: it is a wrong character written out by hand on a printed page.
"""
import codecs

# Encodings a file can announce with a byte-order mark, longest mark first so
# UTF-32's is not mistaken for UTF-16's.
_BOMS = [
    (codecs.BOM_UTF32_LE, "utf-32-le"),
    (codecs.BOM_UTF32_BE, "utf-32-be"),
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF16_LE, "utf-16-le"),
    (codecs.BOM_UTF16_BE, "utf-16-be"),
]


def detect_encoding(head):
    """
    Choose an encoding from a file's first few bytes.

    UTF-8 is the answer for almost everything, but two cases produce visible
    nonsense rather than a slightly wrong character:

    * **A UTF-8 byte-order mark.** Notepad, Visual Studio and PowerShell's
      `>` all write one. Decoded as plain UTF-8 it survives as U+FEFF, which
      has no glyph, so the first character of the file was written out as the
      placeholder '?' and the preview warned about a character the user
      cannot see.
    * **UTF-16.** PowerShell redirected output to UTF-16 for years, so it is
      a plausible upload. Decoded as UTF-8 it comes back as a replacement
      character followed by a NUL between every letter — a page of
      placeholders where the source should be.

    Both announce themselves in the first bytes, so neither needs guessing.
    """
    for bom, encoding in _BOMS:
        if head.startswith(bom):
            return encoding
    return "utf-8"


def parse_source(file_path):
    """
    Args:
        file_path: path to an uploaded source file.

    Returns:
        list[str]: the file's lines, exactly as written (line endings
        stripped, everything else — indentation, tabs, trailing spaces —
        left untouched).

    Bytes that cannot be decoded are replaced rather than refused. A file in
    an encoding we cannot identify is mostly readable, and rendering it with
    the unreadable parts visible is more use than rendering nothing.
    """
    with open(file_path, "rb") as f:
        head = f.read(4)

    encoding = detect_encoding(head)
    with open(file_path, "r", encoding=encoding, errors="replace") as f:
        lines = [line.rstrip("\n").rstrip("\r") for line in f.readlines()]

    # A UTF-16 or UTF-32 mark is not stripped by the codec the way
    # `utf-8-sig` strips its own.
    if lines and lines[0].startswith("\ufeff"):
        lines[0] = lines[0][1:]
    return lines

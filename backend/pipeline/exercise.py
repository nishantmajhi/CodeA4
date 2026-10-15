"""
The guided capture exercise (spec Stage 1).

This is the single source of truth for what the user is asked to copy onto
each of the three A4 sheets. The API exposes it at ``GET /api/exercise`` so
the front-end renders exactly the text the segmenter will later align
against — if the two ever drifted apart, Stage 2 would be aligning against
the wrong ground truth and every downstream stage would silently degrade.

Two properties are deliberate (spec §4 Stage 1):

1. **Coverage** — nearly every character a source file can contain appears
   here at least once, and most appear several times. ``coverage_report()``
   checks this and the test suite asserts it, so editing the text can't
   quietly drop a character.
2. **One over-long line** — ``LONG_LINE_INDEX`` marks a line that cannot fit
   on an A4 line at natural handwriting size. The writer is forced to wrap it
   organically, which is the only way to observe how *they* break a line.
   Stage 2's line alignment knows this line may occupy two written lines.
"""

# Lines exactly as the user should copy them. Blank entries are blank lines.
EXERCISE_LINES = [
    "Q1. Write a hello world program in C, and say what printf does.",
    "",
    "Ans:",
    "#include <stdio.h>",
    "#define TAG \"v1.0\"      /* build tag */",
    "",
    "int main(void) {",
    "    char *msg = \"Hello, world!\";",
    "    printf(\"%s [%d] & done;\\n\", msg, 42);   // this long line is meant to run past the right margin, so keep writing and let it wrap by itself",
    "    int q = (7 + 3) * 2 - 6 / 3;",
    "    if (q >= 8 && q <= 9 || q != 0) q = q % 5;",
    "    return 0;",
    "}",
    "",
    "printf sends text to standard output. The quick brown fox jumps over a lazy dog.",
    "THE QUICK BROWN FOX JUMPS OVER A LAZY DOG. 0123456789",
    "PACK MY BOX WITH FIVE DOZEN LIQUOR JUGS. 9876543210",
    "Odd keys: ~ ` ^ _ | \\ ? ' $ @ + - = : ; < > { } [ ] and 6 more.",
    "Again: ~ ` ^ _ | \\ ? ' $ @ + - = : ; < > { } [ ] jazz & quiz.",
]

# Index into the *non-blank* lines (INK_LINES) of the line that is deliberately
# too long to fit — the one the writer must wrap by hand.
LONG_LINE_INDEX = 6

EXERCISE_TEXT = "\n".join(EXERCISE_LINES)

# Non-blank lines, which is what the segmenter actually works with — a blank
# line leaves no ink and so cannot be detected on the sheet.
INK_LINES = [ln for ln in EXERCISE_LINES if ln.strip()]

# The set of characters the capture is expected to teach us, ignoring spaces
# (a space is an advance, not a glyph).
EXPECTED_CHARS = sorted({c for c in EXERCISE_TEXT if c.strip()})

# Every character we would like to be able to render. Anything here but not in
# EXPECTED_CHARS has to fall back to a substitute glyph at render time
# (see pipeline.glyph_assets.resolve_glyph), which is a visible quality loss —
# hence the test that keeps this set empty.
RENDERABLE_CHARS = sorted(
    {chr(c) for c in range(33, 127)}
)


def char_counts():
    """How many times each character is written across one sheet."""
    counts = {}
    for ch in EXERCISE_TEXT:
        if ch.strip():
            counts[ch] = counts.get(ch, 0) + 1
    return counts


def coverage_report():
    """
    Returns (missing, thin) where `missing` are renderable characters the
    exercise never asks for, and `thin` are ones written only once per sheet
    (so only three instances across all three rounds — enough to build a
    glyph from, but too few to cluster into distinct modes).
    """
    counts = char_counts()
    missing = [c for c in RENDERABLE_CHARS if c not in counts]
    thin = sorted(c for c, n in counts.items() if n == 1)
    return missing, thin

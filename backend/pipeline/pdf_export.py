"""
Spec Stage 8 — PDF export

Deliberately thin. Every layout decision was already made in Stage 6/7, which
emitted finished A4 SVG pages; all that remains is to convert each page to PDF
and bind them into one document.

The conversion keeps the glyphs as **vectors**. Handwriting rendered as
outlines stays crisp at any zoom and prints at the printer's own resolution,
which is the whole reason Stage 4 traces contours instead of keeping the
photographed bitmaps.
"""
import io
import os

from pypdf import PdfWriter


class ExportError(Exception):
    """Raised when pages cannot be converted, with a user-facing message."""


def _svg_to_pdf_bytes(svg):
    """
    Convert one SVG page to a single-page PDF.

    cairosvg is imported lazily so that importing this module — which the API
    does at start-up — does not require the native Cairo libraries to be
    present. Only an actual export needs them.
    """
    try:
        import cairosvg
    except (ImportError, OSError) as exc:  # OSError: Cairo missing at runtime
        raise ExportError(
            "PDF export is unavailable on this server because its graphics "
            "libraries are missing. The pages can still be viewed and printed "
            "from the preview."
        ) from exc

    return cairosvg.svg2pdf(bytestring=svg.encode("utf-8"))


def export_pages_to_pdf(pages, out_path, convert=None):
    """
    Args:
        pages: the page dicts from render_svg.render_file_to_pages(), each
            carrying its composed SVG under "svg". Pages from several files
            may be concatenated; they are written in the order given, so each
            file starts on its own fresh page (spec §4 Stage 7).
        out_path: where to write the combined PDF.
        convert: SVG-to-PDF-bytes function, for tests. Cairo is a native
            library that is simply absent on some machines — including every
            Windows one without a GTK runtime — so the binding and ordering
            done here has to be checkable without it. The default is the real
            converter and nothing in the product passes anything else.

    Returns:
        The path written.
    """
    if not pages:
        raise ExportError("There are no rendered pages to export.")

    convert = convert or _svg_to_pdf_bytes
    writer = PdfWriter()
    for page in pages:
        svg = page.get("svg")
        if not svg:
            continue
        writer.append(io.BytesIO(convert(svg)))

    if len(writer.pages) == 0:
        raise ExportError("None of the rendered pages could be converted.")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        writer.write(f)
    writer.close()
    return out_path

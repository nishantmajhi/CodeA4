"""
Stage 8, which is thin by design and is where the one native dependency is.

cairosvg needs Cairo from the operating system. It is present in the
Dockerfile and absent from a plain Windows machine, so the export path has two
documented behaviours and both matter: it produces a PDF where Cairo exists,
and where it does not it degrades to a preview-only result rather than failing
the render that produced the pages.

The binding, the ordering and the error handling are the project's own code
and are checked here with the conversion injected; whether Cairo draws a
correct page is Cairo's business.
"""
import io
import os

import pytest
from pypdf import PdfReader, PdfWriter

from pipeline.pdf_export import ExportError, export_pages_to_pdf


def _one_page_pdf(width=595, height=842):
    writer = PdfWriter()
    writer.add_blank_page(width=width, height=height)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def _fake_convert(_svg):
    return _one_page_pdf()


def _pages(count, prefix="<svg/>"):
    return [{"svg": f"{prefix}{i}", "file": "a.py", "page_num": i + 1,
             "lines": []} for i in range(count)]


def test_every_page_reaches_the_pdf_in_order(tmp_path):
    seen = []

    def convert(svg):
        seen.append(svg)
        return _one_page_pdf()

    out = str(tmp_path / "out.pdf")
    export_pages_to_pdf(_pages(5), out, convert=convert)

    assert seen == [f"<svg/>{i}" for i in range(5)]
    assert len(PdfReader(out).pages) == 5


def test_the_written_file_is_a_pdf(tmp_path):
    out = str(tmp_path / "out.pdf")
    export_pages_to_pdf(_pages(2), out, convert=_fake_convert)
    with open(out, "rb") as f:
        assert f.read(5) == b"%PDF-"


def test_the_output_directory_is_created(tmp_path):
    out = str(tmp_path / "deep" / "deeper" / "out.pdf")
    export_pages_to_pdf(_pages(1), out, convert=_fake_convert)
    assert os.path.exists(out)


def test_exporting_nothing_is_an_error_with_a_sentence_in_it():
    with pytest.raises(ExportError) as excinfo:
        export_pages_to_pdf([], "unused.pdf", convert=_fake_convert)
    assert len(str(excinfo.value)) > 20


def test_pages_with_no_svg_are_skipped_not_written_blank(tmp_path):
    pages = _pages(3)
    pages[1]["svg"] = ""
    out = str(tmp_path / "out.pdf")
    export_pages_to_pdf(pages, out, convert=_fake_convert)
    assert len(PdfReader(out).pages) == 2


def test_a_render_with_nothing_convertible_is_an_error(tmp_path):
    pages = [{"svg": "", "file": "a.py", "page_num": 1, "lines": []}]
    with pytest.raises(ExportError):
        export_pages_to_pdf(pages, str(tmp_path / "out.pdf"),
                            convert=_fake_convert)


def test_a_missing_cairo_is_reported_as_advice_not_a_stack_trace(monkeypatch):
    """
    The message reaches the user through the job result, so it has to say what
    they can still do -- the pages rendered fine and the preview works. This
    is the one failure in the pipeline that is the server's fault and not
    theirs.
    """
    import builtins

    real_import = builtins.__import__

    def no_cairo(name, *args, **kwargs):
        if name == "cairosvg":
            raise ImportError("no library called cairo")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_cairo)
    with pytest.raises(ExportError) as excinfo:
        export_pages_to_pdf(_pages(1), "unused.pdf")
    message = str(excinfo.value)
    assert "cairo" not in message.lower() or "graphics libraries" in message
    assert "preview" in message.lower()


def test_page_size_is_a4(tmp_path):
    """Stage 6 emits A4 in points and Stage 8 must not resize it -- this is
    meant to be printed and written on."""
    out = str(tmp_path / "out.pdf")
    export_pages_to_pdf(_pages(1), out, convert=_fake_convert)
    box = PdfReader(out).pages[0].mediabox
    assert float(box.width) == pytest.approx(595, abs=2)
    assert float(box.height) == pytest.approx(842, abs=2)

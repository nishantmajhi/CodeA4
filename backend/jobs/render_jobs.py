"""RQ job: parse code, compose pages, export PDF (Stage 5-8)."""
import logging
import os

import models
import push_service
import storage
from jobs.retention import sweep_old_data
from pipeline.code_parser import parse_source
from pipeline.glyph_assets import coverage_summary
from pipeline.pdf_export import ExportError, export_pages_to_pdf
from pipeline.render_svg import render_file_to_pages

log = logging.getLogger(__name__)


def render_job(job_id, session_id, file_paths):
    """
    Args:
        file_paths: list of (original_filename, saved_path) tuples for the
            uploaded source files.
    """
    models.set_progress(job_id, 10, stage="Reading your files")

    try:
        asset = models.get_style_asset(session_id)
        if not asset:
            raise ValueError("no style assets for session")

        style_sets = asset["style_sets"]
        drift = asset["drift"]
        meta = asset.get("meta", {})

        all_pages = []
        missing = set()
        total = exact = 0

        for index, (original_filename, saved_path) in enumerate(file_paths):
            lines = parse_source(saved_path)

            cover = coverage_summary(style_sets, "\n".join(lines))
            total += cover["total"]
            exact += cover["exact"]
            missing.update(cover["missing"])

            pages = render_file_to_pages(
                original_filename, lines, style_sets, drift, meta,
                session_id=session_id, file_index=index,
            )
            all_pages.extend(pages)
            models.set_progress(
                job_id, 10 + int(55 * (index + 1) / len(file_paths)),
                stage=f"Writing out {original_filename}",
            )

        # Page SVGs are large and the client fetches them one at a time as it
        # scrolls the preview, so they are written to disk and the job result
        # carries only what the thumbnail rail needs.
        _write_pages(job_id, all_pages)

        models.set_progress(job_id, 75, stage="Making the PDF")
        pdf_path = storage.render_pdf_path(job_id)
        pdf_error = None
        try:
            export_pages_to_pdf(all_pages, pdf_path)
        except ExportError as exc:
            # The pages rendered fine; only the PDF conversion failed. The
            # preview is still worth showing, so this is reported alongside a
            # successful result rather than failing the whole job.
            pdf_error = str(exc)
            pdf_path = None
            log.warning("pdf export unavailable: %s", exc)

        result = {
            "pdf_path": pdf_path,
            "pdf_error": pdf_error,
            "page_count": len(all_pages),
            "pages": [
                {"file": p["file"], "page_num": p["page_num"],
                 "index": i, "lines": len(p["lines"])}
                for i, p in enumerate(all_pages)
            ],
            "coverage": {
                "total": total, "exact": exact,
                "missing": sorted(missing),
            },
        }
        models.update_job(job_id, status="done", progress=100, stage="Done",
                          result=result)

        push_service.notify(session_id, "Your pages are ready",
                            f"{len(all_pages)} page(s) rendered in your "
                            "handwriting.", url="/#preview")
        sweep_old_data()

    except Exception:
        log.exception("render failed for session %s", session_id)
        message = "We could not render these files. Please try again."
        models.update_job(job_id, status="failed", error=message, stage=None)
        push_service.notify(session_id, "Render failed", message)
        raise



def _write_pages(job_id, pages):
    """
    Write each page where `GET /api/render/<job>/pages/<index>.svg` will find
    it. The index the client scrolls by is in the job result, which it is
    already polling, so there is no manifest here for anything to read.
    """
    directory = storage.job_pages_dir(job_id)
    for index, page in enumerate(pages):
        with open(os.path.join(directory, f"{index}.svg"), "w") as f:
            f.write(page["svg"])

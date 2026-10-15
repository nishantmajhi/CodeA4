# CodeA4 backend

Flask API plus an RQ/Redis job queue, following `../docs/CodeA4_Backend_Plan.md`.
Every pipeline stage lives in `pipeline/` behind a stable function signature.
All eight stages are implemented for real — nothing here returns fabricated
data.

## Run locally

**With Docker (simplest):**
```bash
docker compose up --build
```
The app is at `http://localhost:5000` — it serves both the API (`/api/...`)
and the PWA.

**Without Docker**, the full three-process setup:
```bash
pip install -r requirements.txt
redis-server &                          # in one terminal
python worker.py &                      # in another
flask --app app run --debug             # in another
```

**Without Redis at all**, for quick development — jobs run inline in the web
process and state is kept in memory. Requests block for as long as the
pipeline takes, so this is never a production mode:
```bash
INLINE_JOBS=1 MEMORY_STATE=1 flask --app app run --debug
```

`python app.py` also works. The debugger it starts is gated on `FLASK_DEBUG`
rather than switched on in the file, because Werkzeug's debugger hands a
Python console to anyone who can reach a traceback.

## Tests

```bash
python -m pytest -q
```

287 tests, roughly 75 seconds. They need no Redis — `tests/conftest.py` sets
inline/in-memory mode.

| File | What it covers |
| ---- | -------------- |
| `test_exercise.py` | The capture text: full coverage of printable ASCII, and no character asked for only once |
| `test_imaging.py` | Page geometry, binarization, line bands, skew — on synthetic masks |
| `test_segmentation_units.py` | The alignment passes on component boxes, with no photograph involved |
| `test_segmentation.py` | Stages 1+2 end to end on generated sheets, against ground truth |
| `test_style_quality.py` | Whether the style sets are real letterform variants, and whether every glyph they ship is a plausible instance of its character |
| `test_pipeline_units.py` | Shape descriptors, vectorization, drift fitting |
| `test_render_layout.py` | Stages 6+7: wrapping, pagination, style blocks, determinism |
| `test_code_parser.py` | Stage 5: bytes to text, including byte-order marks |
| `test_pdf_export.py` | Stage 8, with the conversion injected so it runs without Cairo |
| `test_models.py` | The record store and what it guarantees across processes |
| `test_jobs.py` | The queue, progress reporting, and the two error paths |
| `test_retention.py` | The sweep and the delete endpoint |
| `test_uploads.py` | What happens to the files a user hands the server |
| `test_http_contract.py` | Status codes and error shapes the front-end depends on |
| `test_config.py` | The defaults, which are the security posture |
| `test_api.py` | The whole flow through the HTTP API |

The CV stages are tested against synthetic sheets from
`tools/make_sample_sheet.py`, which produce photograph-like input with known
ground truth. Read the docstring in `tests/test_segmentation.py` for what that
does and does not prove.

**The generator's three faces must be three genuinely different designs.**
They were once three humanist sans faces that differ by less than the
generator's own per-instance wobble, which left the clustering stage nothing
real to recover — so it separated the well-segmented instances from the badly
segmented ones instead, and style 2 came out a broken alphabet.
`test_style_quality.py` measures cluster purity against the modes the
generator recorded, which is the check that catches this.

To look at a generated sheet yourself:
```bash
python tools/make_sample_sheet.py /tmp/sheet.jpg --seed 1 --truth /tmp/truth.json
```

Tests that are not about the capture pipeline install a synthetic style model
from `tests/fake_style.py` rather than building a real one, which costs about
a minute a time.

## Smoke test

```bash
curl -X POST http://localhost:5000/api/sessions
# -> {"session_id": "...", "rounds": [...], "style_ready": false}

curl http://localhost:5000/api/exercise
# -> the exact text the user must copy onto each sheet

curl -X POST http://localhost:5000/api/sessions/<session_id>/rounds/1/photo \
  -F "photo=@/path/to/sheet1.jpg"
# -> {"job_id": "..."}

curl http://localhost:5000/api/jobs/<job_id>
# -> poll until "status": "done"; "stage" says what it is doing
```

Repeat for rounds 2 and 3. Once round 3 finishes, a clustering job is
enqueued automatically. Poll `GET /api/sessions/<session_id>/style` until
`style_ready` is true, then:

```bash
curl -X POST http://localhost:5000/api/sessions/<session_id>/render \
  -F "files=@/path/to/some/code.py"
curl http://localhost:5000/api/jobs/<job_id>          # poll until done
curl http://localhost:5000/api/render/<job_id>/pages/0.svg   # one page
curl -OJ http://localhost:5000/api/render/<job_id>/pdf       # the whole thing

curl -X DELETE http://localhost:5000/api/sessions/<session_id>
# -> forgets the session, the style model and every file belonging to it
```

If a capture fails, the job's `error` is a sentence written for the user
("...take the photo again in brighter, even light...") — the capture screen
shows it verbatim, so keep it that way when adding new failure paths.
Everything else must reach the browser as a sentence too: an unexpected
exception becomes "Something went wrong on our end", never a traceback.

## Layout

```
app.py            Flask factory; serves the API and the PWA
worker.py         RQ worker entrypoint (a separate process)
config.py         the only place that reads os.environ
models.py         Session/Job/StyleAsset records, shared via Redis
storage.py        filesystem paths, and the retention sweep
extensions.py     Redis/RQ wiring, plus the inline queue for dev
push_service.py   web push, a no-op when unconfigured
api/              one blueprint per resource
jobs/             the three queued jobs, plus the retention sweep they trigger
pipeline/         all eight stages — see below
tools/            development tools, not part of the product
tests/
```

### pipeline/

| Module            | Stage | What it does                                                   |
| ----------------- | ----- | -------------------------------------------------------------- |
| `exercise.py`     | 1     | The capture text — ground truth for everything downstream       |
| `imaging.py`      | 1     | Page detection, deskew, binarization, line bands                |
| `segmentation.py` | 1+2   | Aligns ink to known characters; emits per-instance samples      |
| `features.py`     | 2/3   | Shape descriptors                                               |
| `vectorize.py`    | 4     | Ink contours → smooth SVG paths in em units                     |
| `clustering.py`   | 3     | Letterform modes + drift fitting                                |
| `glyph_assets.py` | 4     | N internally consistent glyph sets                              |
| `code_parser.py`  | 5     | Reads source files as literal text                              |
| `drift.py`        | 6     | Resamples the fitted drift model, deterministically             |
| `render_svg.py`   | 6+7   | Style blocks, layout, wrapping, pagination, SVG composition     |
| `pdf_export.py`   | 8     | Vector SVG → PDF                                                |

## Configuration

See `.env.example`, and the table in `../README.md`. Everything is optional
for local development. The ones worth knowing:

* `INLINE_JOBS` / `MEMORY_STATE` — the no-Redis mode above.
* `CORS_ORIGINS` — defaults to localhost origins, not `*`. None of this API is
  authenticated, so `*` means any page the user visits can read their session
  given only its id. The app serves its own front-end from its own origin, so
  the ordinary deployment needs nothing here.
* `DATA_RETENTION_SECONDS` — how long files survive. `STATE_TTL_SECONDS`
  expires the Redis records, which makes a session unreachable; it does not
  touch a byte on disk, which is what this is for.
* The VAPID keys, without which push is disabled and the UI hides the option.

## A note on state

Session and job records live in **Redis**, not in process memory. This is not
incidental: jobs run in a separate worker process, so an in-memory dict would
mean the worker's progress updates never reach the web process and every job
would poll as "queued" forever. `MEMORY_STATE=1` restores the in-memory
backend for single-process use, and must be set deliberately — it is never a
silent fallback, because a silent fallback would reintroduce that bug on any
deployment where Redis blipped at start-up.

## A note on the data

The photographs are of somebody's handwriting and the model derived from them
reproduces it convincingly. Two things follow for anyone adding to this.

Files are swept by `storage.sweep_expired`, which runs at the end of a capture
or render job because nothing here runs on a timer. If you add a new place
that writes under `DATA_DIR`, put it under `sessions/<id>/` or
`jobs/<id>/` so the sweep and `DELETE /api/sessions/<id>` both reach it.

Identifiers become directory names. `storage._is_safe_segment` is what keeps
one from naming a path of its own, and every path helper that takes one from
outside should go through it.

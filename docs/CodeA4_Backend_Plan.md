# CodeA4 — Backend Plan

> **Historical.** This was written before the backend existed, and it says so
> throughout: it describes a scaffold whose stages return "structurally
> correct fake data" so the queue and the front-end could be built ahead of
> the CV work. All eight stages are real now, and no module returns fabricated
> data. The architecture it lays out — Flask never running a stage inline, one
> generic job-status endpoint, shared state in Redis, the `data/` layout — is
> still exactly what is built, and it is still the best explanation of *why*.
> Where it differs from the code, the code is right; the current API surface is
> tabulated in `../README.md` and the current layout in `../backend/README.md`.
>
> Two things have been added since: `DELETE /api/sessions/<id>` and the
> retention sweep that deletes files past their window, neither of which this
> plan anticipated. See `review.md` for why.

Companion to `CodeA4_Spec.md` (product spec) and `tracker.md` (project
status). This plan turns spec §8 (High-level architecture) and §9
(Suggested build order) into a concrete API surface and folder layout,
before any code is written. The scaffold in `backend/` follows this
plan exactly.

---

## 1. Design goals

- Flask is API-only. It never runs Stage 1–8 processing inline — every
  slow step (segmentation, clustering, rendering, PDF export) is an RQ job.
- The API surface should map 1:1 onto the 5 UI steps already validated in
  `codea4_ui_prototype_v2.html`, so the front-end can eventually be pointed
  at real endpoints with minimal rework.
- Every pipeline stage (Stage 1–8 in the spec) is isolated into its own
  module with a stable function signature, even before it's really
  implemented. Early on, these modules return **structurally correct fake
  data** (same shape the real algorithm will produce) rather than
  `NotImplementedError`, so the rest of the system — job queue, polling,
  storage, PDF assembly — can be built and demoed end-to-end before the
  hard CV/ML work is done. This mirrors how the front-end mock already
  fakes its data.
- A single generic job-status endpoint is reused for every long-running
  step (capture-round processing, clustering, rendering), rather than a
  bespoke status endpoint per stage — matches the spec's "frontend polls a
  job-status endpoint" note in §8.

---

## 2. Data model (in-memory for now, DB-shaped on purpose)

No database yet — `models.py` holds everything in a process-memory dict.
Every record already looks like a DB row so swapping in SQLite/Postgres
later is a storage-layer change only, not an API change.

**Session** (one per user's three-round capture flow)

```
session_id, created_at,
rounds: [ {n: 1..3, status: waiting|processing|done|failed, job_id, error} ],
style_ready: bool
```

**Job** (one per background unit of work — capture round, clustering, or render)

```
job_id, session_id, type: capture_round|clustering|render,
status: queued|processing|done|failed,
progress: 0-100, error: str|null, result: dict|null
```

**StyleAsset** (produced once, after round 3 clusters — Stage 3/4 output)

```
session_id,
style_sets: [ {id, share, glyphs: {char: svg_path, ...}} ],
drift: {baseline: [[x,y], ...], margin: [[x,y], ...]}
```

**RenderJob result** (Stage 5–8 output, per code upload)

```
pages: [ {file, page_num, filename_header_svg, lines: [{text, style_id}] } ],
pdf_path
```

---

## 3. API surface

All routes prefixed `/api`. All uploads are `multipart/form-data`. All
long-running actions return a `job_id` immediately; the client polls
`GET /jobs/<job_id>`.

| Method & path                           | Maps to UI step                                                      | Purpose                                                                                                                                                                                                     |
| --------------------------------------- | -------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `POST /sessions`                        | Capture (on load)                                                    | Create a new capture session. Returns session with 3 empty rounds.                                                                                                                                          |
| `POST /sessions/<sid>/rounds/<n>/photo` | Capture → "Use camera"                                               | Upload one round's photo. Enqueues a `capture_round` job (Stage 1+2: segmentation vs. known exercise text). Returns `{job_id}`.                                                                             |
| `GET /jobs/<job_id>`                    | Capture progress bar, Style loading, Preview loading, Export loading | Generic poll endpoint. Returns `{status, progress, error, result}`. When the 3rd round's job finishes, the session auto-enqueues a `clustering` job (Stage 3+4) and sets `style_ready` once that completes. |
| `GET /sessions/<sid>`                   | Capture rounds table                                                 | Session + round statuses + `style_ready` flag.                                                                                                                                                              |
| `GET /sessions/<sid>/style`             | Your Style step                                                      | Once `style_ready`, returns the `StyleAsset` (style sets + drift curves) for direct rendering into the style cards / drift chart.                                                                           |
| `POST /sessions/<sid>/render`           | Add Code → "Render preview"                                          | Upload one or more source files. Enqueues a `render` job (Stage 5–8). Returns `{job_id}`.                                                                                                                   |
| `GET /render/<job_id>/pdf`              | Export → "Download PDF"                                              | Streams the finished PDF once the render job's status is `done`.                                                                                                                                            |
| `POST /push/subscribe`                  | Capture → notification checkbox                                      | Stores a Push API subscription against a session, so a finished round/job can notify the user instead of them polling in-app (spec §7).                                                                     |

Notes:

- No auth/user model yet — spec doesn't call for one in v1; sessions are
  the only identity boundary. Worth flagging as an open item before any
  real deployment (uploaded handwriting is sensitive).
- `GET /jobs/<job_id>` is intentionally generic across all three job types
  so the front-end only needs one polling function.

---

## 4. Job pipeline (RQ)

Three job types, each a thin wrapper around the pipeline modules:

1. **`capture_round_job(session_id, round_n, photo_path)`**
   Stage 1 (compare to known exercise text) + Stage 2 (segment every
   character instance). Stores extracted instances to disk; updates the
   round's status.

2. **`clustering_job(session_id)`**
   Runs once, after round 3's capture job completes. Stage 3 (shape
   clustering + drift-curve fitting) + Stage 4 (build N vector glyph sets).
   Writes the `StyleAsset` record and flips `style_ready`.

3. **`render_job(session_id, files)`**
   Stage 5 (parse source as literal text) → Stage 6 (flow-consistent style
   assignment + SVG composition) → Stage 7 (pagination/line wrap) →
   Stage 8 (SVG → PDF via `cairosvg`, combine into one PDF). Writes the
   `RenderJob` result and the final PDF to disk.

Each job updates its own `Job.progress`/`status` as it runs so the generic
poll endpoint has something real to report.

---

## 5. Storage layout (filesystem, for now)

```
data/
  sessions/<session_id>/
    rounds/<n>/photo.<ext>          # raw uploaded photo
    rounds/<n>/chars.json           # Stage 2 output: segmented instances
    style/assets.json               # Stage 3+4 output: style sets + drift
  jobs/<job_id>/
    render/<file>.pages.json        # Stage 6+7 output per file
    render/output.pdf               # Stage 8 final PDF
```

Swappable for object storage (S3-compatible) later without touching the
API layer, since access is already funneled through `storage.py`.

---

## 6. Folder structure

```
backend/
  app.py              # Flask app factory, registers blueprints
  config.py           # env-driven config (Redis URL, data dir, etc.)
  extensions.py       # Redis connection + RQ queue, created lazily
  worker.py           # RQ worker entrypoint (separate process from web)
  models.py           # in-memory Session/Job/StyleAsset records
  storage.py           # filesystem path helpers (data/ layout above)
  api/
    __init__.py       # blueprint registration
    sessions.py       # /sessions, /sessions/<id>, /sessions/<id>/rounds/<n>/photo
    style.py          # /sessions/<id>/style
    render.py         # /sessions/<id>/render, /render/<job_id>/pdf
    jobs.py           # /jobs/<job_id>
    push.py           # /push/subscribe
  jobs/
    __init__.py
    capture_jobs.py   # capture_round_job
    clustering_jobs.py# clustering_job
    render_jobs.py    # render_job
  pipeline/            # Stage 1-8 implementations — stubbed, real seams
    __init__.py
    segmentation.py    # Stage 1+2
    clustering.py      # Stage 3
    glyph_assets.py     # Stage 4
    code_parser.py      # Stage 5
    render_svg.py        # Stage 6+7
    pdf_export.py         # Stage 8
  requirements.txt
  docker-compose.yml   # redis + web + worker, for local dev
  .env.example
  README.md
```

---

## 7. Implementation status

All eight stages are implemented for real. Nothing in `pipeline/` returns
fabricated data any more, and the `# STUB — Stage N` markers are gone.

Where the built system departs from this plan, and why:

- **§2 said state was in-memory, "DB-shaped on purpose".** It is now in Redis.
  The in-memory version was not merely temporary but *incorrect*: jobs run in a
  separate worker process, so a module-level dict meant the worker's updates
  never reached the web process and every job polled as "queued" forever. The
  get/save seam this plan called for is what made the swap a single-file
  change, as intended. An in-memory backend remains for tests and
  single-process development, behind an explicit `MEMORY_STATE=1`.
- **§3's job result for renders** no longer embeds page SVGs. A page of
  handwriting is a few hundred kilobytes of vector outlines, so a twenty-page
  render would drag megabytes through every status poll. Pages are written to
  disk and fetched individually from `GET /render/<job_id>/pages/<n>.svg`.
- **§3 gained three endpoints**: `GET /exercise` (so the UI and the segmenter
  share one copy of the capture text — two copies would drift apart and the
  segmenter would silently align against text the user never wrote),
  `GET /sessions/<id>/style/sample.svg` (so the style cards use the real layout
  engine rather than a second approximate one), and `GET /push/vapid-key`.
- **§3's drift shape changed.** It is no longer a bare list of points but a
  fitted Ornstein-Uhlenbeck model (`phi`, `sigma`, `std`) *plus* the measured
  points for display. The renderer needs the model, because it resamples fresh
  wander rather than replaying the captured wobble on every page.
- **§5's storage layout** gained `jobs/<job_id>/pages/` and
  `jobs/<job_id>/uploads/`, following from the change above.
- **The web app also serves the PWA.** A service worker only controls pages
  within its own scope, so serving the front-end from the app's root is what
  lets it cache the shell and receive push events without a second origin and
  a CORS configuration on every call.

The design goals in §1 held up unchanged: Flask still never runs a pipeline
stage inline, every stage still sits behind a stable signature in `pipeline/`,
and the queue is still the only path to doing work.

---

## 8. Next steps

The build order in §8 of the original plan is complete. What remains is
validation rather than construction:

1. **Run real photographed sheets through the pipeline.** Everything to date is
   measured against synthetic fixtures with known ground truth, which makes
   accuracy assertable but proves nothing about ink bleed, ruled paper or
   joined writing. See `tracker.md` §5.
2. Tune `pipeline/imaging.py` against whatever real photos reveal —
   binarization block size and the tall-band splitting tolerance are the most
   exposed parameters.
3. Consider persisting sessions in SQLite if they ever need to outlive the
   Redis TTL; `models.py` is still the only file that would change.

# CodeA4

Renders source code as print-ready A4 pages in your own handwriting.

You copy a short exercise onto three sheets of paper and photograph them.
CodeA4 measures how you actually form each character — including the two or
three different ways you naturally draw the same letter — and how your
baseline and left margin wander and correct themselves down a page. It then
writes your source files out in that hand, preserving indentation and
whitespace exactly, and exports a PDF.

The point is that the output should look *written*, not typeset in a
handwriting font. That means no single glyph per character repeated down the
page, and no random jitter pretending to be variation — the variation comes
from your own letterforms, and the drift from a model fitted to your own
wander.

> **Status: early.** The pipeline is complete and tested end to end, but it
> has been validated against synthetic sheets and one screenshot rather than
> a sample of real handwriting. See [`docs/tracker.md`](docs/tracker.md) for
> what is and is not proven, and `review.md` for what was fixed and what was
> deliberately left alone.

---

## Contents

- [What this is for](#what-this-is-for)
- [Quick start](#quick-start)
- [Your handwriting data](#your-handwriting-data)
- [Running without Docker](#running-without-docker)
- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Configuration](#configuration)
- [Running the tests](#running-the-tests)
- [Troubleshooting](#troubleshooting)

---

## What this is for

This produces handwritten-looking pages of source code, and "copy this program
out by hand" is a common assignment, so it is worth saying what the tool is
for rather than leaving readers to guess.

It was built as a typography and signal-processing exercise: the interesting
problem is recovering a person's several natural letterforms from a photograph
and reproducing a hand that drifts the way theirs does. It is also useful as
an accessibility aid, and for producing legible handwritten material by
someone who finds sustained writing difficult or painful.

Whether any particular use is legitimate is a question about that use, not
about the tool, and this README is not the place it gets settled. What is
settled here: the output is a convincing imitation of one person's hand, and
that is worth knowing before you point it at anything.

---

## Quick start

Requires Docker with the Compose plugin.

```bash
git clone <your-fork-url> codea4
cd codea4
docker compose -f backend/docker-compose.yml up --build
```

Then open **http://localhost:5000** and follow the five steps:
Capture → Your Style → Add Code → Preview → Export.

This starts three containers: `web` (Flask, serving both the API and the
frontend), `worker` (runs the image pipeline), and `redis` (job queue and
shared session state).

To stop, press `Ctrl+C`, then:

```bash
docker compose -f backend/docker-compose.yml down
```

`down` on its own **keeps your data**. To also delete it, see the next
section.

### There is no login, and no access control

Anyone who can reach the server can create a session, and anyone who knows a
session id can read that session's photographs, style model and rendered
pages. That is a reasonable arrangement for something you run on your own
machine, which is what the quick start above sets up — it binds to your own
Docker host and is not published anywhere.

If you put this on a reachable host, put something in front of it. The
defaults assume you have not: `CORS_ORIGINS` allows only localhost origins, so
a page you visit cannot call the API from your browser, and the Werkzeug
debugger is off unless `FLASK_DEBUG` is set.

---

## Your handwriting data

The photographs show your hand, and the model derived from them reproduces it
convincingly enough to pass as yours. This section is about where that lives
and how to get rid of it.

### What is kept, and where

| Where | What it holds | Cleared by |
| ----- | ------------- | ---------- |
| `DATA_DIR` (Docker volume `backend_data`) | The three photographs, the extracted character instances, the fitted style model, the rendered pages and PDFs | `down -v`, `DELETE /api/sessions/<id>`, or the retention sweep |
| Redis | The session record, job status, and the style model | Restarting the Redis container; expires after `STATE_TTL_SECONDS` |
| Browser `localStorage` | Your session id, under the key `codea4.session` | Clearing site data |

Nothing is sent anywhere else. There is no analytics, and the only outbound
request the server makes is a web push notification, and only if you asked for
one and configured VAPID keys.

### How long it is kept

Both the Redis records and the files expire after seven days by default —
`STATE_TTL_SECONDS` and `DATA_RETENTION_SECONDS` respectively. The file sweep
runs when a capture or render job finishes, and measures a session's age from
the newest file in it, so a session you are part way through is never swept
out from under you.

### Deleting it now

```bash
curl -X DELETE http://localhost:5000/api/sessions/<session_id>
```

That removes the session record, the style model, the push subscription and
every file belonging to it, including the pages it rendered. Your session id is
in browser local storage under
`codea4.session`; clear that too, or the page will keep asking about a session
that is gone.

### Start over completely

```bash
docker compose -f backend/docker-compose.yml down -v
```

The `-v` is the part that matters — it removes the named volume along with the
containers. **Deleting or rebuilding the Docker image resets nothing**, which
catches people out: the image holds only the code. Then clear
`codea4.session`:

- DevTools → Application → Local Storage → `http://localhost:5000`
- Delete the key `codea4.session`, or run
  `localStorage.removeItem('codea4.session')` in the console, then reload.

A hard reload alone is not enough, and neither is a private window if you then
return to your normal one.

### Recapture your handwriting, keep everything else

Clear `codea4.session` and reload — the app starts a new session.

### Clear queue and fitted models, keep uploaded files

```bash
docker compose -f backend/docker-compose.yml restart redis
```

Redis holds no on-disk state here, so restarting it drops all sessions, jobs
and style models while leaving the `backend_data` volume untouched.

### Non-Docker installs

```bash
rm -rf backend/data
redis-cli FLUSHDB
```

---

## Running without Docker

Requires Python 3.11 or later and a local Redis. PDF export additionally needs
the Cairo system library that `cairosvg` binds to; without it everything else
works and the Export step says the download is unavailable.

```bash
# System dependency (Debian/Ubuntu). macOS: brew install cairo
sudo apt-get install libcairo2

cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then run the three pieces in separate terminals:

```bash
redis-server                                    # terminal 1
cd backend && python worker.py                  # terminal 2
cd backend && flask --app app run --port 5000   # terminal 3
```

The web process never runs pipeline stages itself. If the worker is not
running, captures and renders queue forever without erroring — this is the
most common cause of "it just sits at 0%".

On Windows, Cairo is not straightforward to install and there is no need to:
the pipeline runs, the preview works and the pages can be printed from it. The
PDF download is the one thing that will not work.

---

## How it works

1. **Capture** — three sheets of a fixed exercise. Because the text is known,
   segmentation is an alignment problem rather than handwriting recognition:
   we never guess what a mark is, only which marks belong to which known
   character. Three sheets rather than one so natural variation can be told
   apart from one-off accidents.
2. **Learn** — each character's instances are clustered into the small number
   of ways you actually form it, giving two to four internally consistent
   style sets. Instances that cannot be that character — a cut that landed
   inside a letter, a crop that swallowed its neighbour — are discarded first,
   because a cluster of those would otherwise become a "style" and you would
   be shown a broken alphabet and told it is how you write. Baseline behaviour
   is fitted as one tilt per line plus mean-reverting correction *between*
   lines, with measurement error subtracted rather than reproduced — a sheet
   written dead straight is modelled as dead straight.
3. **Render** — your files are laid out using your own measured letter, word
   and line spacing, wrapped the way a hand wraps at the edge of a page, with
   one internally consistent style set per block of code.

Whitespace and indentation are preserved exactly. There is no syntax colouring
and no reformatting.

Rendering is deterministic: the same files in the same session always produce
the same pages, so the preview and the PDF you download after it are the same
document.

---

## Repository layout

| Path | What it is |
| ---- | ---------- |
| `docs/CodeA4_Spec.md` | Product source of truth — scope, the eight pipeline stages |
| `docs/CodeA4_Backend_Plan.md` | Backend architecture: API, jobs, data model, storage |
| `docs/tracker.md` | Current status, how the hard parts work, and honest limitations |
| `review.md` | What a cleanup pass fixed, and what it deliberately did not |
| `backend/` | Flask API, RQ worker, the full pipeline implementation |
| `backend/pipeline/` | The eight stages |
| `frontend/` | The PWA (served by Flask, so the service worker scope covers both) |
| `prototypes/` | The original design mocks, superseded but kept for reference |

**Read [`docs/tracker.md`](docs/tracker.md) before changing anything.** It
records why several non-obvious choices are the way they are, and what the
system has not yet been proven to handle.

### API surface

| Method | Path | What it is for |
| ------ | ---- | -------------- |
| `GET` | `/api/health` | Liveness, and which modes the server is in |
| `GET` | `/api/exercise` | The text to copy, served from the same constant the segmenter aligns against |
| `POST` | `/api/sessions` | Start a capture |
| `GET` | `/api/sessions/<session_id>` | The three rounds and whether the style model is ready |
| `DELETE` | `/api/sessions/<session_id>` | Forget a session and everything derived from it |
| `POST` | `/api/sessions/<session_id>/rounds/<n>/photo` | Upload one sheet; returns a job to poll |
| `GET` | `/api/sessions/<session_id>/style` | The fitted style sets and drift model |
| `GET` | `/api/sessions/<session_id>/style/sample.svg` | A rendered sample of one style |
| `POST` | `/api/sessions/<session_id>/render` | Upload source files; returns a job to poll |
| `GET` | `/api/jobs/<job_id>` | Status, progress and result for any job |
| `GET` | `/api/render/<job_id>/pages/<index>.svg` | One rendered page |
| `GET` | `/api/render/<job_id>/pdf` | The whole render as a PDF |
| `GET` | `/api/push/vapid-key` | Whether push is configured, and the public key |
| `POST` | `/api/push/subscribe` | Store a browser's push subscription |

---

## Configuration

All settings are environment variables, read in `backend/config.py` and
nowhere else. See `backend/.env.example`.

| Variable | Default | What it does |
| -------- | ------- | ------------ |
| `REDIS_URL` | `redis://localhost:6379/0` | Queue and shared state |
| `DATA_DIR` | `backend/data` | Photos, extracted characters, rendered pages, PDFs |
| `FRONTEND_DIR` | `../frontend` | Where the PWA is served from |
| `CORS_ORIGINS` | localhost dev origins | Comma-separated, or `*`. The app serves its own front-end from its own origin, so the ordinary deployment needs nothing here |
| `FLASK_DEBUG` | off | Werkzeug's interactive debugger. It offers a Python console on any traceback; never set it on a reachable host |
| `STATE_TTL_SECONDS` | `604800` (7 days) | How long the Redis records survive |
| `DATA_RETENTION_SECONDS` | `604800` (7 days) | How long the files survive. Never shorter than the above |
| `INLINE_JOBS` | off | Run jobs in the web process. Development and tests only — requests block for the whole pipeline |
| `MEMORY_STATE` | off | Keep records in process memory. Single-process only; see `backend/README.md` |
| `VAPID_PUBLIC_KEY` | unset | Web push; notifications are off when unset |
| `VAPID_PRIVATE_KEY` | unset | As above |
| `VAPID_SUBJECT` | `mailto:admin@example.com` | Contact address sent with push |

Push notifications are entirely optional — with no VAPID keys configured the
app runs normally, does not attempt to send any, and the UI hides the option.

---

## Running the tests

```bash
cd backend && python -m pytest -q
```

287 tests, roughly 75 seconds, and no Redis needed —
`backend/tests/conftest.py` puts everything in inline/in-memory mode. Run it
before and after any change.

Most of them are about things that are invisible by eye: that drift is not
invented from measurement noise, that the style sets correspond to real
letterform variants rather than to how well each instance happened to be
extracted, that wrapping never loses a character, and that two files uploaded
under the same name do not overwrite each other. The two tests that need the
Cairo library assert the documented degradation when it is absent rather than
skipping.

---

## Troubleshooting

**Capture sticks at 0% and never finishes.**
The worker process is not running, or cannot reach Redis. Check
`docker compose -f backend/docker-compose.yml logs worker`.

**"We could not find any writing on this sheet."**
The page edges were not detected, or the photo is too dark. Photograph the
sheet flat, filling the frame, in even light without a shadow across it.

**The rendered handwriting contains the wrong letters.**
Check `alignment_confidence` in the capture metadata. Below about 0.95 means
characters were matched to the wrong ink and the sheet should be recaptured. A
matched-character count alone will *not* reveal this.

**Old handwriting keeps coming back after a rebuild.**
See [Your handwriting data](#your-handwriting-data) — you almost certainly
need `down -v` *plus* clearing `codea4.session` from browser local storage.

**The Export step says the PDF is unavailable.**
The Cairo system library is missing. `apt-get install libcairo2`, or
`brew install cairo` on macOS. The Docker image installs it already. The
preview is still complete and printable.

**Port 5000 is already in use.**
On macOS this is usually AirPlay Receiver. Either turn it off in System
Settings → General → AirDrop & Handoff, or change the published port in
`backend/docker-compose.yml`.

## Licence

Free for personal, non-commercial use. You may use, study and modify this
software for yourself, but you may not use it commercially, and you may not
share or publish it, in its original form or modified. See [LICENSE](LICENSE)
for the full terms.

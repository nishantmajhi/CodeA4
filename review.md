# Code Review — CodeA4

First reviewed 2026-09-04. **Remediated 2026-09-20** on branch
`cleanup/2026-09-20`, not pushed. This file is now a record rather than a
to-do list: what was wrong, what was fixed, what was deliberately left, and
what the next person should know.

Verify with:

```sh
cd backend && python -m pytest -q      # 287 tests, ~75s, no Redis needed
```

---

## Corrections to the first review

**Finding 5 was half right, and the half it got wrong was the worse half.** It
said of the source-file upload path: *"Note that `api/render.py:38` already
does the right thing with `secure_filename`; make the two paths consistent."*

`secure_filename` makes a name safe to use as a path. It does not make it
unique, and the render path saved every upload into one directory keyed on it.
Two files picked from different folders arrive under the same name to begin
with — `main.py` from two projects is an ordinary thing to do — and
`secure_filename` strips non-ASCII, so `日本語.py` and `中文.py` both reduce to
`py`. **The second upload overwrote the first, and both files' pages were then
rendered from the survivor's contents, under both files' names.** The job
succeeded, reported the right page count, and listed both original filenames in
its result. Nothing anywhere could notice.

So the finding pointed at the safe path as the model for the unsafe one. It is
the clearest thing in this pass: a name being *sanitized* and a name being
*usable as a key* are different properties, and only the first was checked.

Everything else the review said was accurate.

---

## What the review found, and what happened to it

| # | Finding | Verdict | Status |
| - | ------- | ------- | ------ |
| 1 | `app.run(debug=True)` | Real | Fixed |
| 2 | `CORS_ORIGINS` defaults to `*` | Real | Fixed |
| 3 | 48-bit session identifiers | Real | Fixed |
| 4 | No retention, storage or disclosure for handwriting data | Real | Fixed |
| 5 | Photo extension comes from the client | Real, and understated | Fixed |
| 6 | No authentication at all | Real | Author's decision; documented |
| 7 | Backend plan is historical; no LICENSE | Real | Plan marked; licence left to the author |

**1. The debugger.** `python app.py` is the shortest way to start this and the
module docstring shows it. Werkzeug's debugger offers a Python console on any
traceback. Now `app.run(debug=Config.DEBUG)`, off unless `FLASK_DEBUG` says
otherwise. A test reads `app.py` for a hard-coded `debug=True`, because a test
of the flag cannot see the entry point.

**2. CORS.** Defaulted to `*`, which with unauthenticated endpoints means any
page the user visits can read their session, model and pages from their
browser given only a session id. The default is now the usual localhost dev
origins. `*` is still settable — a setting that cannot express "open" would be
a lie — but it has to be asked for.

**3. Identifiers.** `uuid.uuid4().hex[:12]` was 48 bits where 122 were
available, in the one value standing between somebody else and a person's
handwriting photographs. Now `secrets.token_urlsafe(24)`. The front-end already
falls back to creating a new session when an old id is not found, so no
migration was needed.

**4. The handwriting data.** This was the finding with a consequence for
somebody other than the author, and all three of its parts were right.

*Retention.* `STATE_TTL_SECONDS` expires the Redis records, which makes a
session unreachable; it never touched a byte on disk. The photographs, the
extracted character instances, the style model and every rendered page stayed
for as long as the disk did, owned by a session id that had stopped working —
so not even the person who made them could reach them again to ask for them
back. `storage.sweep_expired` now deletes session and job directories nothing
has touched inside `DATA_RETENTION_SECONDS`, measured from the newest file
inside rather than from the directory, because capture is spread over three
sheets with breaks in between and a live session's directory is routinely a
day old. Nothing here runs on a timer, so the sweep is attached to the end of a
capture or render job: off the request path, in the worker, often enough.

*Deletion.* `DELETE /api/sessions/<id>` forgets the record, the style model,
the push subscription, the files, and every job the session started. There is
no account system, so without it there was no way to withdraw the data on
purpose at all.

The jobs were an oversight in the first version of this fix, caught by running
the real server rather than the test client: a render job's directory holds the
source files the user uploaded and the pages produced from them, and those are
keyed by job id, not by session id. Deleting the session left all of it on
disk *with the record that pointed at it gone*, which is worse than leaving it
reachable. Sessions now record the jobs they start, because nothing indexes
that way round.

*Storage.* Checked, and it was already right: `DATA_DIR` defaults to
`backend/data` while the static route serves only `FRONTEND_DIR`, which is
`../frontend`. The upload directory is not reachable over HTTP.

*Disclosure.* The README now has a section saying what is stored, in which of
the three places, for how long, how to delete it, and that nothing leaves the
machine.

**5. The photo extension.** Taken verbatim from the client's filename, so
`sheet.php` or `sheet.exe` put exactly that on disk. The review was right that
this is not a traversal — `splitext` will not let a separator into an
extension — and right that it should be an allowlist anyway. It also missed
that a retake under a different extension left the previous photograph behind
indefinitely, so a single round could accumulate `photo.jpg`, `photo.png` and
`photo.jpeg` side by side. A round holds one photo: the one last sent.

**6. Authentication.** Left as it is, because the answer depends on a decision
only the author can make. What changed is that the defaults now assume the
server is *not* reachable, and the README states the posture plainly instead of
leaving it to be discovered: anyone who can reach the server can create a
session, and anyone with a session id can read it.

**7. Cleanup.** `docs/tracker.md` stays — the review was right that it is a
live document doing its job, and it has been brought current.
`docs/CodeA4_Backend_Plan.md` is marked historical at the top, with a note on
what in it is still true, which is most of the architecture and all of the
reasoning. **No LICENSE has been added**: choosing terms is the author's, not a
cleanup pass's.

---

## What the review did not find

Seven more, in order of how much they matter.

### 1. Two uploads could overwrite each other, and the wrong file was rendered

Described above under corrections. The worst defect in this pass: silent, and
producing output that is confidently wrong rather than absent.

### 2. The style sets did not correspond to anything

This is the product's central claim — that a hand has a few natural ways of
forming each letter and this system finds them — and it had never been tested.

The synthetic sheets draw each character from one of three faces with fixed
probabilities so that clustering has real modes to recover. The three faces
were three humanist sans designs, and the difference between them in the 16×16
normalized bitmap the clustering works on is **smaller than the per-instance
wobble the generator applies on top** — rotation, shear, scale and an elastic
warp. Measured on crops straight out of the generator, with no photograph and
no segmentation in the way, the true face labels score a silhouette of -0.002
to +0.111. There was nothing there to find.

So clustering found something else. Its clusters were each a proportional
mixture of all three faces (adjusted Rand index against the true modes: 0.03),
separated instead by *segmentation quality*: cluster 1 was the cleanly cut
instances and cluster 2 the badly cut ones. Style 2 was therefore not a second
way of writing. It was the same hand extracted worse, and the Style step
presented it to the user as "41% of your samples". Rendered at four times
thumbnail size, style 2's alphabet was visibly broken — 'n' and 'm' collapsed
to fragments, 'o' broken open.

The one test in the area asserted that the two style sets *differ*. Clusters of
noise satisfy that completely.

The faces are now a sans, a serif and a monospace, and
`tests/test_style_quality.py` measures cluster purity against the modes the
generator recorded — 0.71 against 0.51 by chance, where the old faces gave
0.60 against 0.53. That test fails on the old faces by the margin it is meant
to.

### 3. Mis-segmented instances were becoming styles

`reject_outliers` exists to prevent exactly this and its docstring says so:
*"k-means happily gives them a cluster of their own, and that cluster then
becomes a style, so the user is shown a set of broken letterforms and told it
is the second way they write."* It filters on median-absolute-deviation, which
measures *spread* — and when a third of a character's instances are cut badly
in the same way, the spread widens to accommodate them and they survive.

`reject_implausible` now runs first with flat bounds instead: no hand writes
one 'n' at two thirds the width of its other 'n's, one 1.6× too wide has
swallowed its neighbour, and one whose ink sits 0.2 em off the baseline its own
kind sit on has lost part of itself. At least half of every character always
survives, because a filter that can empty a letter is worse than one that lets
something through.

Without it, style 2 shipped `}` at half the width of every other `}` on the
sheet, plus five more between 0.5× and 0.65× of their character's median. With
it, none — and the letterforms that were being crowded out come back.

### 4. The width calibration had never once run

`calibrate_widths` fits `actual = alpha + beta * prior` so the alignment DP,
which scores purely on width, is not comparing this writer's ink against a
generic table that may be out by a factor of two. `docs/tracker.md` records it
as the third of five fixes in the round that cured glyph mis-assignment.

It returned the identity fallback on every sheet this project has. Its evidence
was characters from lines whose component count already equals their character
count, and it wanted thirty such samples. One or two of the sixteen lines on a
sheet ever qualify, because a single merge or split anywhere disqualifies the
whole line and on real ink there is always one. Measured across three sheets: 1,
1 and 2 clean lines, giving 1, 1 and 27 samples.

The condition is also backwards. A line matches 1:1 under the *uncalibrated*
prior only when that prior already fits this hand, so the mechanism could fire
only when it was not needed, and was guaranteed silent on the hand it exists
for. It presumably did fire on the monospaced screenshot the tracker says these
fixes were validated against — the one input where nearly every line is
trivially 1:1 — which is why nobody saw it.

It now fits over every unwrapped line, which needs no per-character matching at
all: we know which characters are on a line, and its components' total inked
width is measurable whether or not the letters within were cut correctly.

**No output changed.** Across six synthetic sheets spanning four width regimes
— proportional, condensed, monospaced and wide — character labelling is
3154/3166 both before and after, and median crop placement error is 0.0684 em
against 0.0677 em. This is repaired because a safety net that has never
deployed is not a safety net, not because anything measurable improved. Said
plainly here rather than dressed up.

### 5. A byte-order mark was written out as a character

Notepad, Visual Studio and PowerShell's `>` all write UTF-8 with a BOM. Read as
plain UTF-8 it survives as U+FEFF, which has no glyph — so the first character
of every such file was rendered as the placeholder `?`, and the preview told
the user that a character they cannot see was missing from their exercise
sheets. A UTF-16 file, PowerShell's default for years, came back as a
replacement character followed by a NUL between every letter: a whole page of
placeholders. Both announce themselves in the first four bytes.

### 6. Unknown `/api/` paths returned the app shell with a 200

The catch-all serving the PWA also matched `/api/...`, so a mistyped or removed
endpoint answered `200 text/html`. `app.js` checks the status before the body,
so that is not an error there — it is a successful call whose body is a string,
and the failure reappears somewhere unrelated as an undefined property.
`/api/` paths now get the same JSON 404 as everything else, while unknown
*page* paths still fall back to the shell so deep links keep working.

### 7. Two files with the same name rendered with identical drift

The render seed was the session id plus the filename. `drift.py`'s docstring
says why that matters: *"every page would repeat the same wobble, and a reader
comparing two pages would see the identical wave twice."* Two uploads under one
name produced byte-identical pages. This was unobservable until finding 1 was
fixed, because only one of the two files ever reached the renderer.

---

## The test suite

44 tests became 287, in sixteen files. `backend/README.md` tabulates what each
covers. Three things about it are worth knowing before changing it.

**The fixtures only worked on Linux.** `FONT_CANDIDATES` listed three Debian
paths, and when none existed `_load_fonts` silently fell back to
`ImageFont.load_default()` — a bitmap face fixed at about 11 px, identical for
all three modes. A sheet was still produced and still looked superficially like
writing; the failure surfaced four stages later as *"Only 12 lines of writing
were found, but the exercise has 16"*. Each mode now lists a Windows face too,
and a machine with none of a mode's faces raises `MissingFonts` naming the
paths it looked for.

**Two tests need Cairo and assert the degradation when it is absent**, rather
than skipping. `cairosvg` binds to a native library the Dockerfile installs and
a plain Windows machine does not have. Both outcomes are documented behaviour —
a PDF, or a preview-only result with a sentence explaining why — and the second
is the one a user on an under-provisioned server actually meets.
`export_pages_to_pdf` takes its converter as an argument, defaulting to the
real one, so the binding, ordering, page size and error paths are all checkable
without Cairo.

**Several are properties rather than examples**, because that is where these
bugs were: wrapping is checked over three thousand random lines and widths for
losing a character, pagination over six line counts for dropping or duplicating
one, fragment absorption for losing ink at either end of a line, character
alignment for producing exactly one entry per non-space character, and band
splitting for ever returning a band whose top is past its bottom.

---

## Deliberately not done

**No LICENSE.** Choosing terms is the author's.

**No authentication.** See finding 6.

**The PDF export still depends on Cairo.** Replacing it would remove the
project's only native dependency and make export work on Windows, and the SVG
this project emits is a small enough subset that a direct converter is
plausible. It is also a rewrite of a working stage to fix a deployment problem
the Dockerfile already solves, so it was not attempted. Noted because it is the
obvious next thing somebody will consider.

**One glyph in the style sets is still visibly wrong** — style 2's 'd' has a
neighbouring stem attached. It is the right width and sits on the baseline, so
no plausibility bound can see it. It is the ~3% mis-segmentation
`docs/tracker.md` §5 already records honestly, and chasing it means improving
segmentation rather than filtering its output.

**The UI was not touched**, per the standing constraint. `frontend/` and
`prototypes/` have no changes in this branch. Two server-side changes are
visible to a user and are worth naming: the Export step's message when Cairo is
missing was already written and is now reachable in more cases, and a session
can now be deleted, which the UI offers no button for — it is a `curl` in the
README.

---

## Notes for a future session

**Look at the output.** Findings 2 and 3 were found by rendering the style
cards to an image and reading them, not by reading code. At thumbnail size
style 2 looks plausible. Headless Chrome will rasterise an SVG with
`--headless=new --disable-gpu --screenshot` and needs no driver, no protocol
client and no dependency of any kind; Chrome is at
`C:/Program Files/Google/Chrome/Application/chrome.exe` on this machine. It
took about a minute to set up and found the two worst things here.

**Read a mechanism's inputs before believing it runs.** Findings 2 and 4 are
both mechanisms that exist, are documented at length, are called on every
sheet, and never do anything. In both cases the giveaway was in the data rather
than the code: `width_fit` was `{alpha: 0.0, beta: 1.0}` — the literal fallback
value — in the metadata of every sheet the project has ever produced.

**The fixture is the weakest part of the whole arrangement.** Everything known
about this pipeline's accuracy comes from `tools/make_sample_sheet.py`, and
finding 2 is what happens when a fixture quietly stops providing the property
the tests assume. If you change its fonts, its wobble, or its exercise text,
`tests/test_style_quality.py` is the test that tells you whether the fixture
still supports the claim it exists to support.

**Real handwriting remains item 1.** `docs/tracker.md` §7 has said so since
August and this pass does not change it. Everything above was measured against
synthetic sheets; the one real capture this project has ever seen was a
screenshot of its own exercise, and five of the eight defects in the tracker's
fix round 1 came out of that single input.

## Licence, 2026-10-02

The author chose one licence for all of their projects: free for personal,
non-commercial use, modifiable for the user's own use, and not to be
distributed, original or modified. No standard licence says that, so `LICENSE`
is a short custom text, the same in every project apart from the year and what
it excludes. The README's licence section says the same. Anything above that
calls the licence an open question, or calls the project MIT, is superseded.

## CI, 2026-10-02

`.github/workflows/ci.yml` runs `pytest` in `backend/` on Python 3.12, on
`windows-latest`, with the full `requirements.txt`. Nothing has been pushed,
so it has not run on GitHub yet. Every job was run before committing, from a
fresh clone of this branch, the way a runner would run it: Windows jobs here,
in PowerShell 7 or the shell the step names, with only the tools the job sets
up on `PATH`. 287.

**Why Windows, and an open question for the author:** the fixture sheets are
drawn in the first face found from each list in `tools/make_sample_sheet.py`,
and the lists put DejaVu first, with a comment saying the same three
distinguishable faces are found "on a Linux CI box and on a Windows desktop".
The suite had only ever run on Windows, where it draws in Verdana, Times New
Roman and Consolas and passes. On Ubuntu with DejaVu, four tests fail:
character labelling falls to 79.8% against a 90% bar, a line is mapped to the
wrong exercise line, and two styles ship a `!` that is not one. With
Liberation instead, one fails (the clustering does not recover the writer's
modes). Whether the pipeline should be held to those thresholds for every
face, or the fixtures should name their faces, is a design decision, so CI
runs where the suite passes and the code is unchanged.

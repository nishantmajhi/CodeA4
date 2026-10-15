# CodeA4 — Project Tracker

Last updated: 2026-09-20

Purpose of this file: a running status doc so any new session (or the project
owner) can pick up context quickly without re-reading the whole conversation
history. Update it whenever a stage moves from planned → in progress → done,
or a design decision is made.

---

## 1. Where the project actually is

**Phase: complete vertical slice.** Every stage of the spec's Stage 1–8
pipeline is implemented for real and runs end to end: photograph a sheet →
segment characters → cluster letterforms and fit drift → build glyph sets →
render source files as A4 pages in that handwriting → export a PDF. The PWA is
wired to the real API. No stage returns fabricated data any more.

| Area                                           | Status                                                                                                       |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Product spec                                   | ✅ Done — `CodeA4_Spec.md`                                                                                   |
| UI wireframes v1 / v2                          | ✅ Done — kept for reference in `prototypes/`                                                                |
| Backend plan                                   | ✅ Done — `CodeA4_Backend_Plan.md` (§7 updated to match the built system)                                     |
| Backend (Flask + RQ/Redis)                     | ✅ Real — shared state in Redis, inline mode for dev/tests                                                    |
| Stage 1 — Guided capture                       | ✅ Real — camera capture in the PWA, uploaded and processed per round                                        |
| Stage 2 — Character segmentation               | ✅ Real — `pipeline/segmentation.py` + `imaging.py`, OpenCV and two alignment DPs                             |
| Stage 3 — Shape clustering + drift modeling    | ✅ Real — `pipeline/clustering.py`, scikit-learn + Ornstein-Uhlenbeck fits                                    |
| Stage 4 — Style asset (SVG glyph) prep         | ✅ Real — `pipeline/glyph_assets.py` + `vectorize.py`, traced contours per character per style                |
| Stage 5 — Code parsing                         | ✅ Real — `pipeline/code_parser.py`                                                                          |
| Stage 6 — Flow-consistent SVG rendering        | ✅ Real — `pipeline/render_svg.py`, per-glyph placement with synthesized drift                                |
| Stage 7 — Pagination / line wrapping           | ✅ Real — same module, measured advance widths                                                                |
| Stage 8 — SVG → PDF export                     | ✅ Real — `pipeline/pdf_export.py`, vector output                                                             |
| PWA shell (service worker, push notifications) | ✅ Real — `frontend/sw.js`, manifest, icons, VAPID push sent from the worker                                  |
| Tests                                          | ✅ 287 passing — see `../backend/README.md` for what each file covers                                         |
| Data retention                                 | ✅ Files swept past their window; `DELETE /api/sessions/<id>` forgets a session on demand                     |

### Measured quality (synthetic sheets, see §5)

| Metric                                        | Result        |
| --------------------------------------------- | ------------- |
| Characters extracted per sheet                | 528 / 528     |
| Written lines mapped to the right exercise line | 59 / 59       |
| Character label accuracy vs. ground truth     | **99.4%**     |
| Cluster purity against the modes actually drawn | **0.71** (0.51 by chance) |

The last row is new, and it is the one that matters most: it is the first
measurement of the claim in §2 — that a hand has a few natural variants per
letter and this system finds them. Before the 2026-09-20 pass it was 0.60
against 0.53 by chance, which is to say not measurably better than assigning
instances at random. See "Fix round 3" below.

Label accuracy moved because the fixture's three faces changed, not because
segmentation did.

---

## 2. Files in this project

| File / directory              | What it is                                                                                                                                          |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CodeA4_Spec.md`              | Source of truth for product scope, pipeline (Stages 1–8), architecture, build order. Authoritative — nothing else may contradict it.                 |
| `CodeA4_Backend_Plan.md`      | Backend architecture — API surface, job/queue design, data model, storage layout. Read before touching `backend/`.                                   |
| `README.md`                   | How to run the whole thing.                                                                                                                          |
| `backend/`                    | Flask API, RQ worker, and the full `pipeline/` implementation. See `../backend/README.md`.                                                               |
| `frontend/`                   | The real PWA: `index.html`, `app.js`, `sw.js`, `manifest.webmanifest`, `icons/`. Served by Flask so the service worker's scope covers the API.        |
| `prototypes/`                 | The v1 and v2 design mocks, kept for reference. **Superseded** — they run on an internal JS mock and are no longer the app. `frontend/` reuses v2's stylesheet verbatim. |
| `tracker.md`                  | This file.                                                                                                                                           |

---

## 3. How the hard parts actually work

Worth reading before changing any of it — several of these choices exist
because the obvious approach was tried and failed.

**Known text is the whole trick (Stage 1+2).** Because the capture exercise is
fixed, segmentation is never recognition — we never guess what a blob is, only
which blobs belong to which known characters. Two dynamic programs do it:
expected lines → written bands, then components → characters (allowing an 'i'
and its dot to merge, and two touching letters to be split apart).

**A line only wraps because it hit the margin.** Matching lines by how much ink
they contain is too weak a cue: a six-character wrapped tail looks like a short
line, and reading it as the next exercise line throws every following line off
by one. The decisive evidence is that every band a line claims except its last
must itself reach the right margin. This took line assignment from 2 errors to 0.

**Widths are measured, not assumed.** A first pass aligns using a generic width
prior, then the cleanly matched characters are measured to build *this writer's*
width table, and alignment and cuts are redone with it. Without this the cut
between touching letters lands inside a letter — `c` came out as `d` and `t` as
`l`, and those wrong glyphs propagated all the way into rendered pages.

**Mis-segmented fragments must be rejected before clustering.** k-means will
happily give half-letters a cluster of their own, and that cluster then becomes
a "style" — so the user gets shown a broken alphabet and told it is how they
write. Outlier rejection filters on width (which catches bad splits directly)
and shape, both with a floor relative to the median so a very consistent hand
isn't over-pruned.

**Drift is resampled, not replayed.** Stage 3 fits an Ornstein-Uhlenbeck
process — how strongly the writer self-corrects, and how far they wander. The
renderer draws *new* wander from that model, seeded deterministically per
session and page, so pages don't repeat an identical wobble and a preview
matches the PDF downloaded after it byte for byte.

**Style blocks are language-agnostic.** Block boundaries come from blank lines
and returns to zero indentation, not from `def`/`class` — the product accepts
any source file, so keying off Python keywords would fail on everything else.

---

## 4. Bugs found and fixed while building

Recorded because each was invisible until measured against ground truth:

1. **Job state was per-process.** `models.py` kept sessions and jobs in a
   module-level dict, but RQ workers run in a *separate process* — so a
   worker's progress updates landed in its own memory and the web process would
   have reported "queued" forever. Invisible in single-process dev, total in
   production. State now lives in Redis; in-memory mode must be opted into.
2. Sparse lines (a lone `}`) were dropped by a page-relative band threshold.
3. Touching lines merged into one band — added a tall-band splitter.
4. The over-long line wraps onto **three** written lines at a large hand; the
   alignment DP capped at two.
5. Baseline drift was measured against a line fitted through the very points
   being measured — circular, and it smoothed away the signal.
6. Margin drift was contaminated by code indentation and by page-edge shadow
   read as ink.
7. Expected positions were laid out per *source* line, so every character after
   a wrap sat hundreds of pixels from its real position.
8. MAD-based outlier rejection over-pruned very consistent distributions.

---

## 5. Honest limitations

- **Tested on synthetic sheets, not real handwriting.** `backend/tools/
  make_sample_sheet.py` generates photograph-like sheets with known ground
  truth — perspective, uneven lighting, blur, noise, drifting baselines,
  organic wrapping, touching letters, and several genuine modes per character.
  The modes only became genuine on 2026-09-20; before that the three faces
  differed by less than the generator's own per-instance wobble.
  That makes accuracy measurable rather than eyeballed. It does **not** cover
  ink bleed, ruled paper, crossings-out, or joined cursive. The accuracy
  figures in §1 are floors against regression, not field claims.
- **The drift smoothness parameter is a prior, not a measurement.** Margin
  drift is estimated from ~16 written lines per sheet, far too few to pin down
  autocorrelation. Amplitude is measured; smoothness is assumed, with the
  fitted value recorded alongside as `measured_phi` so the assumption is
  visible rather than hidden.
- **~3% of characters land on the wrong ink**, almost entirely where letters
  were written touching. Outlier rejection keeps most of these out of the final
  glyph sets, but a character with few instances (a rarely used symbol) is more
  exposed.
- **The exercise is long** — 16 lines, copied three times — because covering
  all of printable ASCII with at least two instances each is what stops the
  renderer falling back on substitute glyphs. That is a real cost to the user.

---

## 6. Open items (from spec §6, still unresolved)

- Support for cursive/joined handwriting — out of scope for v1, and the
  segmenter's component-splitting approach would need rethinking for it.
- Paper texture / scan-effect realism layer — out of scope for v1.
- User control over the number of style variants (N), or manual review of
  clusters. N is currently chosen from the data (median modes per character,
  clamped to 2–4). The Style step shows the result but offers no way to say
  "exclude this instance" or "merge these two styles".
- Adjustable block granularity for style switching — fixed at flow-break
  granularity per spec §4 Stage 6.

## 7. Suggested next steps

1. **Validate on real handwriting.** This is the single highest-value next
   step and the one thing the current tests cannot tell you. Capture three real
   sheets, run them through, and check the accuracy numbers hold.
2. Tune `imaging.py` thresholds against whatever real photos reveal —
   binarization block size and the band-splitting tolerance are the likely
   first casualties of real paper.
3. Consider a lighter exercise once real-world coverage data exists: if
   substitutes turn out to be rare in practice for uncommon symbols, the sheet
   could be shortened considerably.
4. Add a cluster-review affordance to the Style step (§6), which would also
   give users a way to correct the ~3% mis-segmentation without a recapture.

---

## How to use this file

1. Read `CodeA4_Spec.md` first — it's the product source of truth.
2. Read this file for current status, how the hard parts work, and limitations.
3. Read `CodeA4_Backend_Plan.md`, then `../backend/README.md`, before touching
   `backend/` — the plan explains *why* it's structured this way.
4. Run `cd backend && python -m pytest tests/ -q` to confirm the system still
   works before and after any change.
5. Update the tables/notes above after any meaningful change.

---

## Fix round — glyph mis-assignment and per-letter jitter

Diagnosed from a capture where the user photographed the *preview* of the
exercise rather than a handwritten sheet. Clean monospaced input is the
easiest case the pipeline can be given, so the fact that it still produced
badly wrong output isolated the fault to our code rather than the handwriting.

Symptom: the rendered PDF read `StudentLame` for `StudentName`, `Dntity` for
`Entity`, `OO` for `DB`, `VEiewOag` for `ViewBag`. Every character was matched
(`528/528`) and the capture reported clean — the *counts* were right while the
*ink behind each label* was wrong, which nothing checked for.

Five defects, in causal order:

1. **`imaging.binarize` broke thin strokes.** A stem at grey ~150 against a
   local mean of 154 failed the `C=12` margin and was read as paper, leaving
   only the darker serifs. One letter became two or three components, which
   pushed every later label onto the wrong ink. Replaced with hysteresis
   thresholding (strict pass seeds, lenient pass grows, isolated faint specks
   dropped). `WEAK_PAPER_FRAC = 0.70` chosen by sweep, not by guess: 0.90
   merged adjacent letters badly enough to collapse the width calibration.

2. **`_WIDTH_PRIOR` covered almost nothing.** 23 uppercase, 18 lowercase and
   8 digits all fell through to `_DEFAULT_WIDTH`, so the DP's only scoring
   signal was near-constant. Table now covers the alphabet.

3. **The prior was never fitted to the hand.** Scale *and* shape can both be
   wrong, and rescaling cannot fix a wrong ratio. `calibrate_widths` fits
   `actual = alpha + beta * prior` over lines that are already unambiguously
   1:1 — `beta` carries scale, `alpha` carries how uniform the hand is.

   **Correction (2026-09-20): as written, this never ran.** Requiring whole
   lines to be 1:1, and thirty samples of them, meant one or two of sixteen
   lines ever qualified, so the fit returned the identity on every sheet.
   Worse, the condition is backwards: a line matches 1:1 under the
   *uncalibrated* prior only when that prior already fits, so the mechanism
   could fire only when it was not needed. It presumably did fire on the
   monospaced screenshot these fixes were validated against, which is the one
   input where nearly every line is trivially 1:1 — and that is why nobody
   saw it. It is now fitted over every unwrapped line, which needs no
   per-character matching at all.

4. **The DP could split a component across a word space.** A space is a pen
   lift, so this is physically impossible, yet it was consistently the
   cheapest wrong answer available. Now forbidden outright; the space cue
   applies at every step boundary rather than only 1:1 ones; the threshold is
   derived per line from the known space count. Biggest single win.
   Also added `_absorb_fragments` (slivers told from punctuation by contact,
   not size) and a recut move for ink that is both touching and fragmented.

5. **`clustering.PHI_FLOOR` was shared between two different series.** A
   margin step is a written line, a baseline step is a *character* — 0.35 gave
   every letter an independent vertical offset, which reads as bouncing rather
   than wander. Split into `PHI_FLOOR_BASELINE = 0.92` / `PHI_FLOOR_MARGIN =
   0.35`. Measured baseline phi is 0.104, so the floor is doing the work; the
   residuals are still mostly segmentation noise.

Also: `vectorize._smooth_path` rounded every vertex to its edge midpoints,
which turns a thin rectangle into an ellipse — the cause of the bulbous
letterforms. Now a bounded fillet, capped at half of each adjoining edge, with
tolerances in em rather than pixels so a glyph looks the same at any capture
resolution.

New diagnostic: `segment_photo` reports `alignment_confidence` and
`suspect_chars` in `meta`. The corrupted run scored 0.9924 *after* the fixes
and would have flagged low before them — a matched-character count alone
cannot detect this class of failure.

Frontend: the Add Code file table's thick top/bottom rules removed via
`#fileTable` only. The global `table` rule is deliberately untouched, so the
Capture rounds table keeps its rules.

**Caveat: every fix above was validated against a monospaced screenshot, not
real handwriting.** The affine width fit should come back with a large `beta`
and near-zero `alpha` on a real sheet — the opposite regime to the one tested
here. Item 1 of "Suggested next steps" is now more important, not less.

---

## Fix round 2 — drift was being invented, not measured

Reported against the same screenshot capture. The exercise preview on screen
has *no* baseline drift at all, yet the rendered output wandered — and
wandered several times within a single line, which no hand does. Both halves
of that were real bugs.

### The rendered shape was wrong

`render_svg._render_run` drew a fresh mean-reverting walk **per line** and
stepped it **per character**. Two consequences:

* Within a line, an OU walk reverses direction every few characters, giving a
  ripple. Real writing on unruled paper drifts monotonically across a line —
  a hand that starts rising keeps rising until the line ends, because it has
  nothing to correct against until it returns to the left margin.
* Between lines, restarting the walk meant no correction at all. A hand that
  finished a line high brings the next few lines back down.

Replaced with `drift.BaselineModel`: one offset and one tilt per line,
mean reversion applied *between* lines. `DriftModel` is retained for the
margin series, where a step genuinely is one line.

### The measured amplitude was almost entirely error

Root cause was in `segmentation._page_grid`, which fitted line pitch by
regressing baseline position against **band index**. Band indices count only
bands containing ink, so a blank line in the exercise puts two bands one
index apart but two rows apart. Mixing single and double gaps over-estimated
the pitch by about 15% (122.5 px fitted against a true 106 px), and the error
accumulated down the page as a linear ramp in every character's distance from
where it was "expected" — which the drift fit then dutifully modelled as the
writer's wander.

Pitch is now recovered from the gaps themselves, which are near-integer
multiples of one true spacing, and rows are counted off in units of it.
Effect on the screenshot capture: per-line offsets fell from ±1.5 em to
±0.04 em, and `line_spacing_em` corrected from 1.78 to 1.66.

`fit_baseline` additionally subtracts measurement noise before reporting an
amplitude. The scatter left around each line's own straight fit estimates
that noise directly; its expected contribution is removed from the observed
spread of slopes and offsets, and only what survives is rendered. On a
perfectly straight sheet nothing survives and the output is dead straight.

This replaced the `PHI_FLOOR_BASELINE` prior added in fix round 1. That was
treating the symptom: forcing phi high while keeping the full noise amplitude
converts white noise into *smooth, believable* invented drift, which is worse
than obvious jitter because it looks correct.

Three regression tests added: flat input yields no drift, genuinely drifting
input still does, and a rendered line never reverses direction mid-line.

### Docs

Root `README.md` rewritten. The previous one never explained where state
lives, so a user trying to reset their learned handwriting deleted the Docker
image — which holds only code. Style models live in Redis, uploads in the
`backend_data` volume, and the session id in browser `localStorage` under
`codea4.session`; all three now documented with the commands to clear each.
Spec, backend plan and this tracker moved to `docs/`.


---

## Fix round 3 — the cleanup pass (2026-09-20)

A full pass over the backend: bugs fixed, the test suite taken from 44 checks
to 287, both READMEs rewritten. `../review.md` is the record. What follows is
the part a future session needs in *this* file rather than in that one.

### The fixture could not demonstrate the product's central claim

The generator draws each character from one of three faces so that clustering
has real modes to recover. The three faces were three humanist sans designs,
and the difference between them in the 16x16 normalized bitmap is **smaller
than the per-instance wobble the generator applies on top**. Measured on crops
straight out of the generator — no photograph, no segmentation in the way —
the true face labels score a silhouette of -0.002 to +0.111. There was nothing
there to find.

So clustering found something else. Its clusters were each a proportional
mixture of all three faces (adjusted Rand index against the true modes: 0.03),
separated instead by *segmentation quality*: cluster 1 the cleanly cut
instances, cluster 2 the badly cut ones. Style 2 was therefore not a second
way of writing. It was the same hand extracted worse, presented in the Style
step as "41% of your samples", and rendered large its alphabet was visibly
broken — 'n' and 'm' collapsed to fragments, 'o' broken open.

The only test in the area asserted that the two style sets *differ*, which
clusters of noise satisfy completely.

The faces are now a sans, a serif and a monospace, and
`backend/tests/test_style_quality.py` measures cluster purity against the modes
the generator recorded. **If you change those faces, that test is the one that
tells you whether the fixture still supports the claim it exists to support.**

### Mis-segmented instances were becoming styles

`reject_outliers` exists to stop exactly this, and its docstring says so. It
filters on median-absolute-deviation, which measures *spread* — and when a
third of a character's instances are cut badly in the same way, the spread
widens to accommodate them and they survive. `reject_implausible` now runs
first with flat bounds instead: no hand writes one 'n' at two thirds the width
of its other 'n's. Without it, style 2 shipped `}` at half the width of every
other `}` on the sheet, plus five more between 0.5x and 0.65x of their
character's median.

### Two lessons worth carrying

**Look at the output.** Both of the above were found by rendering the style
cards to an image and reading them, not by reading code. At thumbnail size
style 2 looks plausible; at four times that it is obviously broken. Headless
Chrome rasterises an SVG with `--headless=new --screenshot` and no driver or
dependency of any kind, which is how this was done.

**A passing test is evidence about what it covers, not about the project.**
Three separate findings in this pass were mechanisms that had never run, or
tests that asserted something equally true of correct and broken output.
`calibrate_widths` is the sharpest: documented at length in this file, called
on every sheet, and returning its fallback every single time.

### Still open

- **No authentication, and the Docker setup implies deployment.** The
  defaults now assume the server is not reachable — localhost-only CORS, the
  debugger off unless asked for — but deciding the posture is the author's.
- **No LICENSE.** Deliberately left to the author.
- **Real handwriting.** Unchanged, and still item 1 of §7. Everything in this
  pass was measured against synthetic sheets.

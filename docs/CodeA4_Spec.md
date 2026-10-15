# CodeA4 — Project Specification v1

## 1. One-line summary

A web app that scans a user's handwriting, learns its natural variations, and renders their source code files onto printable A4 pages that look genuinely handwritten by them.

## 2. Core Principle

Humans don't write the same letter identically every time, but they also don't write with random noise on every character. Real handwriting has:

- **A small number of natural variants per letter** (e.g., you might write 'a' in 2-4 distinct ways), not infinite randomness.
- **Consistency within a flow** — style stays stable across a stretch of writing and only shifts at natural breaks (new paragraph, new page, pen change), not letter-by-letter.

The system is built around extracting these variants and flow-consistent blocks, rather than treating handwriting as a single glyph shape with jitter applied.

This same principle extends to position, not just shape: on unlined paper, humans drift off both axes as they write (baseline wandering vertically within a line, left margin wandering horizontally down the page) and self-correct _gradually_, not abruptly. This drift-and-correction behavior must also be learned from the samples, and applied as smooth curves during rendering rather than random positional jitter.

## 3. Inputs & Output

**Inputs:**

1. Handwriting sample photos, captured via a guided exercise (see Stage 1) rather than an open template.
2. Source code file(s) — one or more files.

**Output:**

- One or more A4-size, print-ready PDF pages, plain white background, black ink only, no syntax color-coding.
- One new page per uploaded source file, with a handwritten-style filename header.
- Code rendered in the user's own extracted handwriting style, exactly preserving the original file's whitespace/tabs/indentation — no reformatting.
- Long lines wrap naturally to the next line, as a human would on paper — no continuation marker.

## 4. Pipeline

### Stage 1 — Guided handwriting capture

- Present the user a fixed Q&A copying exercise: a coding question at the top and its answer/source code below (e.g., "Write a hello world program in C" / `Ans: #include<stdio.h>...`).
- The question/code is deliberately authored so that: (a) nearly all target characters (letters, digits, code punctuation) appear naturally, and (b) at least one line is too long to fit a real A4 line at natural handwriting size, forcing genuine organic line-wrapping.
- The user copies this same exercise onto a blank A4 sheet and photographs it, three times in total (three separate sessions/sheets), with a paced interval between rounds to keep sessions independent.
- Because the expected text is known exactly, each character's actual position can be directly compared to its expected position on a straight line — giving ground-truth data for both character segmentation and axis drift, with no fuzzy detection needed.

### Stage 2 — Character instance extraction

- Using the known expected text as ground truth, segment every character instance across all three captured photos.
- Every occurrence is extracted as its own sample — no averaging or discarding at this stage.

### Stage 3 — Style clustering & drift modeling

- **Shape clustering:** for each character, cluster its extracted instances by shape similarity, resulting in a small set of natural "modes" per character (not one idealized shape, not unlimited random variation).
- **Drift modeling:** using each character's actual position vs. its expected position on a straight line, fit a smooth mean-reverting drift curve for (a) vertical baseline wander-and-correct within a line, and (b) horizontal margin wander-and-correct down the page.

### Stage 4 — Style asset preparation

- Build N internally-consistent vector glyph sets ("Style 1", "Style 2", ... "Style N"), each assembled entirely from one cluster/mode per character.
- Each character instance is stored as a vector outline (SVG path), grouped by style — no font file (TTF/OTF) is generated; glyphs are composed directly at render time instead.

### Stage 5 — Code parsing

- Parse uploaded source file(s) as literal text, preserving whitespace/tabs/indentation exactly.

### Stage 6 — Flow-consistent rendering

- Divide the code into natural blocks (e.g., per page, or per logical unit such as a function/class).
- Assign one style set per block, consistent throughout; style may shift at the next natural break point — mimicking natural drift, not random per-letter switching.
- Compose each A4 page directly as SVG: for every character, place its vector glyph (from the assigned style set) at a computed (x, y, rotation), offset by the learned drift curves (Stage 3), so lines wander and self-correct the way the user's own writing does.

### Stage 7 — Pagination & line wrapping

- Start each file on a fresh A4 page with a handwritten-style filename header.
- Wrap lines exceeding page width to the next line.
- Add further pages as needed if a file doesn't fit on one.

### Stage 8 — Export

- Convert each composed SVG page to PDF (e.g., via `cairosvg`), then combine into a single print-ready multi-page PDF (A4 size, standard margins).

## 5. Explicit non-goals (v1)

- No neural handwriting synthesis model — style comes from real extracted/clustered glyphs, not generated approximations.
- No font file (TTF/OTF) generation — glyphs are composed directly as vector shapes at render time, avoiding font-tooling dependencies.
- No cursive/joined-letter connection logic (scope assumes print-style handwriting).
- No line numbers.
- No ruled/lined or grid paper background.

## 6. Open items for future iteration

- Support for cursive/joined handwriting styles.
- Paper texture / scan-effect realism layer.
- User control over number of style variants (N) or manual review/editing of clusters.
- Adjustable block granularity for style switching (e.g., per-line vs per-function vs per-page).

## 7. Capture flow UX note

The three capture rounds (Stage 1) are separated by a paced interval (e.g., ~5 minutes), which naturally encourages independent, non-identical writing sessions across the three sheets and gives real backend processing time to run. Rather than a fake countdown timer, use the PWA's Push API to notify the user when the next sheet is ready — same pacing effect, but transparent rather than a manufactured wait.

## 8. High-level architecture

**Frontend — Progressive Web App (PWA)**

- Installable web app; uses `<input type="file" capture="environment">` for direct camera capture of handwriting sheets.
- Service worker caches the app shell for offline-capable UI (processing itself still requires connectivity).
- Push API notifies the user when each capture round's processing is done and the next sheet is ready (replaces a fake countdown timer — see Stage 7 UX note).
- Handles: upload flow (handwriting sheets, code files), live preview of rendered pages, PDF download.

**Backend — Flask**

- Flask serves the API only (upload endpoints, job status, PDF download) — it does not run the processing pipeline directly in the request/response cycle, since segmentation/clustering/rendering are too slow for that.
- **RQ + Redis** runs the actual pipeline (Stages 1–8) as background jobs; the frontend polls a job-status endpoint or receives a push notification when a stage completes.
- Processing libraries: OpenCV (image preprocessing, segmentation), numpy/scikit-learn (shape clustering, drift-curve fitting), `cairosvg` (SVG → PDF export).
- Style assets (vector glyph sets per cluster) and fitted drift curves are cached per user after the three capture rounds complete, so rendering new code files later reuses them without redoing capture/clustering.

## 9. Suggested build order

1. Flask API skeleton + RQ + Redis job queue wiring (upload → background job → status polling).
2. Character extraction + clustering pipeline (photo → grouped variant vector glyphs).
3. Drift-curve fitting from captured samples (baseline and margin wander-and-correct models).
4. Code parser + whitespace-preserving layout engine with line wrapping.
5. Flow-consistent block-based style assignment + SVG page composition (glyphs + drift curves).
6. SVG → PDF export.
7. PWA frontend: capture flow (camera input, push notifications), code upload, live preview, download.
8. End-to-end polish and testing with real handwriting samples and real code files.

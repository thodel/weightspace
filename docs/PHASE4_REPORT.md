# Phase 4 report — the collapsed lines are a data question (asteraix, 06.11.2026)

Branch `phase4` from `main` (cb94b44, PRs #1, #2, #4, #5 merged). The task, from the phase-1 report §5 item 3: find what the collapsed held-out lines share, separate "the crop is bad" from "the model fails on a readable line", and propose a session only if a readable-line subset exists. CPU only; no budgeted session; nothing written to the share, Hugging Face, the registry or the trainer. (The brief said the Hertwig dataset fix stays unapplied; it had already been applied by hand the same evening, commit `dda578ea`, see docs/HERTWIG_REVISIONS.md. Nothing here touched it.)

**In one sentence:** of the 428 lines `xix-blb-v1` "collapses" on in `blb-eval`, none is a readable handwritten line the model gets wrong; they are broken boxes, rotated lines, OCR-junk references and a detector that flags correct text, and together they carry 40 % of the held-out errors — with them re-cut the held-out CER is 9.6 %, with them left out 8.1 %, against 12.3 % as reported. **No weight-space session is proposed.**

## 1. What "collapsed" counted

`wsp.metrics.is_collapsed` flags a line whose hypothesis is under half or over twice the reference length, **or** in which one character trigram occurs six times or more. The second test fires on correct long lines: normalised punctuation (` . `, ` , `) and endings (`en `) repeat that often in ordinary text. 142 of the 428 flagged lines are of this kind, read almost perfectly (for example `batis assignat , receptis ab ipso 4 . lib . et dimidia denar . Constant . Act . 1276 . ind . IV . sexto non . Maji .` read with one substitution). `wsp.collapse.degenerate_repeat` is a stricter test (the trigram must also occur at least twice as often as in the reference, plus two); under it 286 lines remain collapsed. `is_collapsed` itself was **not** changed: every reported "collapsed" count so far, and the programmer's collapsed check in the sessions, used it (§6).

## 2. What the 428 lines are (`docs/collapse/blb-eval__xix-blb-v1.json`)

`python -m wsp.collapse` puts each flagged line in one class, in this order; the parent `xix-v1` is the sibling that decides "model-specific":

| class | lines | share of all held-out edits | what it is |
|---|---:|---:|---|
| degenerate box | 111 | **25.9 %** | the line's box is 1–7 px high while its reference is a whole line; all in K 2913,17 |
| vertical | 100 | 8.5 % | envelope addresses and marginal notes written vertically; the crop is taller than wide (K 2911,103 46, K 2915,3 27, K 2913,17 12, …) |
| detector false positive | 142 | 2.3 % | correct text, flagged by the trigram test (§1) |
| short, every model | 36 | 1.3 % | OCR junk as reference (`— 2 73`, `88 8`, `eee e .` for an ornament), headings whose reference is garbled (`STIFTES KREUZLINGEN` = `rks AAIAinen`), a box over two lines, six upside-down lines (K 2915,3 p. 10), one typewritten page |
| box holds more than its text | 9 | 1.4 % | reference `1 5`, crop a full printed line |
| reference of 1–3 characters | 30 | 0.5 % | page numbers, signs |
| model-specific (the parent reads it) | **0** | 0 | — |

The 428 lines are 7.6 % of the lines and 9 % of the reference characters, and carry 39.8 % of the edits (16,586 of 41,655). Contact sheets of random samples of each kind were inspected by eye; every line in the "short, every model" class was looked at.

**K 2913,17 is not a manuscript.** It is the printed *Regesten des Stiftes Kreuzlingen* (1853) with a few of Laßberg's marks, filed under `manuskripte`. Its 2,825 lines are **half of `blb-eval`** (5,628), and its references are the BLB's own ALTO OCR of the print, not a hand correction: hence the junk lines and garbled headings. All 111 degenerate boxes are in it. The handwriting model reads the print well where the box is right (median CER of its normal lines is low; it is the boxes that fail).

## 3. Re-cut: what the model reads when the crop is right (`scripts/recut_blb_eval.py`, `docs/collapse/recut-blb-eval.json`)

Crops cut as plain rectangles from the page image; scored with `xix-blb-v1`.

| lines | as cut | plain box | re-cut | CER |
|---|---:|---:|---|---:|
| 115 degenerate boxes | 0.999 | 1.000 | widened to the page's median line height, anchored at the box's top / **middle** / bottom | 0.491 / **0.354** / 0.378 |
| 100 vertical lines | 0.999 | 0.999 | rotated 90° counter-clockwise / 270° / best of the two per line | **0.342** / 0.785 / 0.277 |
| 6 upside-down lines | 0.914 | 0.940 | rotated 180° | **0.225** |

The plain box gives the same ~100 % as the held-out crop, so `extract_polygons` is not the cause; the boxes are. The degenerate boxes sit roughly in the middle of their line (the ALTO→PAGE conversion seems to lose the line height, not the position); the remaining 35 % there is the price of a guessed height, not of the model. Most vertical lines are written bottom-to-top.

## 4. How much of the held-out CER the collapsed lines carry

| `xix-blb-v1` on `blb-eval` | lines | CER |
|---|---:|---:|
| as cut (phase 1, docs/baselines/heldout-blb-eval.json) | 5,628 | 12.34 % |
| the 221 degenerate, vertical and upside-down lines re-cut (centred, 90°, 180°) | 5,628 | **9.55 %** |
| the 286 data-defect lines left out (all classes but false positives) | 5,342 | **8.07 %** |
| parent `xix-v1`: as cut / defects left out | | 31.96 % / 29.38 % |

The trainer's recorded held-out CER for this model is 8.7 % (docs/MODELS.md): the gap between that and phase 1's 12.3 % is these lines. The gap to the parent does not depend on them.

## 5. Family 1: `combined-v1` on the Bern sets (`docs/collapse/bern-*__combined-v1.json`)

Transkribus polygons, not ALTO rectangles, so no degenerate boxes; siblings are `hertwig-v2` and `schlussredaktion-v1`.

| set | flagged | short, every model | model-specific | vertical | 1–3 chars | CER as is | without data defects |
|---|---:|---:|---:|---:|---:|---:|---:|
| `bern-eval` (held-out) | 34 | 21 (14.9 % of edits) | **7 (2.2 %)** | 1 | 5 | 19.33 % | 16.97 % |
| `bern-train-300` (practice) | 29 | 15 (29.4 %) | 3 (6.0 %) | 4 | 7 | 14.56 % | 10.57 % |

The "short, every model" lines are mostly **crop/text mismatches in the Transkribus ground truth**: the line polygon holds a fragment of a line (often in table regions, `tr_*_tl_*`) while the text holds the whole line, or the next line's text. The model reads the crop correctly (`stanlich / unnd wol gegrundt` read exactly, against a reference `sich erlütteret unnd approbieret bisß uff den vierden artickel/`). A few are struck-through words. The **7 model-specific held-out lines** (`putation / mitt`, `Uff das Herr Berchtold`, `aber`, `vnnd`, `Im v capit`, …) are lines `hertwig-v2` reads and `combined-v1` drops — the Hertwig damage of target 1, measured in phase 3 (R3), where no edit removed it.

## 6. Conclusion and recommendations

**No session is preregistered.** On `blb-eval` there is no readable-line subset the subject fails on and its parent reads; on `bern-eval` there are 7 lines (2.2 % of the errors), too few to carry a paired interval, and the damage they show was the subject of the phase-3 session. A session on collapsed lines would be a session on data defects.

What would change the numbers, all outside this repo's remit:

1. **BLB ground truth (`lassberg-blb-gt`, built by `agentic_historian scripts/blb_gt_dataset.py`):** (a) the ALTO→PAGE conversion produces boxes 1–7 px high for 111 lines of K 2913,17 — check the ALTO `HEIGHT` values and the scaling; (b) vertical and upside-down lines need an orientation (rotate the crop, or mark them so evaluation can skip them); (c) **K 2913,17 is a printed book with OCR text as ground truth** and is half of the handwriting held-out set: it should leave the handwriting `_eval` split, or be a stratum of its own. These are the dataset owner's decisions.
2. **Bern ground truth (Transkribus):** about 21 held-out lines whose polygon and text disagree (§5); a correction pass in Transkribus on those pages, or excluding them, is the owner's call.
3. **Metric (this repo):** replace the trigram test of `is_collapsed` by `degenerate_repeat` (or add it), with a PROTOCOL note: collapsed counts reported so far include correct long lines. Not done here because it changes every reported count and the programmer's collapsed check; it is a protocol change to agree first.
4. **For the paper's held-out number:** report `blb-eval` as cut *and* with the data-defect classes left out, with the class table, so the 12 % and the 8 % are both traceable.

## 7. Reproduce

```bash
cd ~/Repo/weightspace && git checkout phase4
export TMPDIR=~/atr-cache/tmp CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12
.venv/bin/python -m pytest -q tests                                                           # 39 passed, 2 skipped
R=docs/baselines/readings
.venv/bin/python -m wsp.collapse --set data/heldout/blb-eval.jsonl --readings $R/blb-eval/kraken-german-xix-blb-v1.jsonl \
    --siblings $R/blb-eval/kraken-german-xix-v1.jsonl --out docs/collapse/blb-eval__xix-blb-v1.json
.venv/bin/python -m wsp.collapse --set data/heldout/bern-eval.jsonl --readings $R/bern-eval/kraken-bern-combined-v1.jsonl \
    --siblings $R/bern-eval/kraken-bern-hertwig-v2.jsonl $R/bern-eval/kraken-bern-schlussredaktion-v1.jsonl --out docs/collapse/bern-eval__combined-v1.json
.venv/bin/python scripts/recut_blb_eval.py                                                    # ~12 min
```

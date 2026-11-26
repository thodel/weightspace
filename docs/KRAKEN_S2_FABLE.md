# Kraken session 2 with Claude Fable 5.1 as programmer

Run `runs/20261111-bern-combined-v1-f1-b10-1`, replicating `runs/20261106-bern-combined-v1-b10-1`
(Claude Opus 5.5). Times CET unless labelled; logs are UTC.

## 1. Setup and what the operator knew (written 11.11.2026, before the first pass)

The operator is a Claude Opus 5.5 session. Before the first pass it had read session 2's
pre-registration, PROTOCOL.md and protocol/docs/amendments.md. From those it knows that
session 2 kept an output-layer transplant from `hertwig-v2` that lost on held-out (amendment A3).
It had not read session 2's log, notes, program or summary.

**Attempt 0, discarded.** The first version of the run's `preregistration.md` §7 stated that
knowledge. The programmer is allowed to read that file, and the first Fable subagent did read it
while writing its hypotheses, and said so. Its round-0 files (hypotheses and `program_01.py`)
were set aside before any pass. Section 7 was rewritten to point here; this file is outside
what the programmer may read. A fresh Fable subagent then started from the same brief.

## 2. The session (11.11.2026, 09:56–10:18 CET)

Same subject, practice set (`bern-train-300`, training side), held-out set (`bern-eval`), budget
(10) and keep rule as session 2. The programmer was Claude Fable 5.1, a Claude Code subagent
with `model: fable`. It read only `PROGRAMMER_BRIEF.md` and the files it lists: a pack of
phase-0 readings, describe/diff of subject, parent and sibling, C0/C1, the practice list and the
harness source. It wrote §5 of the pre-registration and three programs in rounds. In each round
it reported that nothing it had read named an edit or result of session 2. The operator ran each program in the sandbox.
One difference from session 2: there, one `program.py` was written in full in advance; here
later programs were written after seeing earlier results.

| pass | hypothesis | edit | practice CER | Δ vs kept, line 95 % | decision |
|---:|---|---|---:|---|---|
| 1 | C0 | — | 0.1456 | | baseline |
| 2 | H1 | blank bias −0.20 | 0.1444 | −0.12 [−0.23, −0.02] | **kept** |
| 3 | H2 | all tensors α 0.5 toward the sibling | 0.1429 | −0.16 [−0.76, +0.33] | refused |
| 4 | H1b | blank bias −0.20 more | 0.1410 | −0.34 [−0.55, −0.16] | **kept** |
| 5 | H3 | space bias +0.30 | 0.1426 | +0.16 [−0.03, +0.38] | refused |
| 6 | H6 | `"` output row from the parent | 0.1410 | 0.00 (identical output) | refused |
| 7 | H4 | all tensors α 0.25 toward the parent | 0.1357 | −0.53 [−1.16, +0.02] | refused |
| 8 | H1c | blank bias −0.20 more | 0.1397 | −0.13 [−0.33, +0.05] | refused |
| 9 | H4b | all tensors α 0.125 toward the parent | 0.1357 | −0.54 [−1.04, −0.13] | **kept** |

Nine of ten passes were used, with 8 edits and 3 kept. The frozen model is the subject with the
blank bias lowered by 0.40 (+0.998 → +0.598), then every trained tensor moved 12.5 % of the way
toward `hertwig-v2`. Output rows were aligned by character; the rows of `½ ƒ ̊` and the blank
bias were held. H4b is the pre-registered follow-up of H4, which narrowly failed the rule, so
two step sizes were tried on the parent axis.

## 3. Held-out, scored once (11.11.2026, 10:18 CET)

Intervals are paired, 2,000 resamples. "page" is the two-stage page-clustered interval (A4), in
CER points. C0 is the unedited subject.

| | practice | held-out all | held-out Hertwig | held-out Schlussred. |
|---|---:|---:|---:|---:|
| C0 | 0.1456 | 0.1933 | 0.2669 | 0.1348 |
| **Fable (this run)** | 0.1357 | **0.1875** | 0.2536 | 0.1350 |
| Δ, line | −0.99 [−1.65, −0.47] | −0.57 [−0.88, −0.29] | −1.32 [−1.91, −0.80] | +0.02 [−0.28, +0.31] |
| Δ, **page** | −0.99 [−1.80, −0.36] | **−0.57 [−1.15, −0.05]** | −1.32 [−2.36, −0.60] | +0.02 [−0.41, +0.52] |
| Opus (session 2) | 0.1414 | 0.1948 | 0.2639 | 0.1399 |
| Δ, page | −0.42 [−0.93, +0.04] | +0.15 [−0.29, +0.60] | −0.30 [−0.82, +0.21] | +0.51 [−0.04, +1.08] |
| Fable − Opus, page | −0.57 [−1.17, −0.05] | −0.73 [−1.19, −0.35] | | |

Collapsed held-out lines: C0 34, Fable 23, Opus 23 (current detector).

The Fable edit is a held-out gain whose page interval excludes 0. It lies wholly on the
Hertwig pages: −1.32 points, with Schlussredaktion unchanged. That is the size and place of the
C4 retrain on corrected data (R11 arm A: −0.54 to −0.56 all, −1.30 Hertwig).

*Note of 24.11.2026:* three later reruns of the same session (Fable 5.2, `docs/KRAKEN_S2_FABLE52.md`;
Gemini 4 Argon and GPT-6.2 Astra, `docs/XVENDOR_REPORT.md`) took the same direction and gained
−0.57 to −0.65 points, but their page intervals reach or include 0. This run's interval is the only
one of five that lies below 0, so read it as one draw of a shared effect, not as a result of its own.

## 4. Controls

`runs/…-f1-b10-1/controls.json`, `scripts/sham_combined_f1.py`, CPU. Each control is scored
against C0.
- **C5 sham:** 5 seeds. Every edited tensor gets Gaussian noise with the Frobenius norm of the
  kept diff of that tensor, 10.68 in all.
  - Practice: +0.03 to +0.14 points. The keep rule kept 0 of 5.
  - Held-out: −0.12 to +0.02. Every page interval contains 0.
  - So the direction of the edit matters, not its size.
- **Split of the kept diff:**
  - Blank −0.40 alone: practice −0.46 [page −0.81, −0.17], held-out **+0.04** [−0.18, +0.25].
  - Interpolation alone: practice −0.74 [−1.45, −0.20], held-out **−0.53** [−1.14, −0.01].
  - The held-out gain comes from the step toward the parent. The two blank steps, both kept on
    practice, do not transfer. This agrees with R8(c): the deletions are decided upstream of
    the output layer.
- **C0, C1:** `docs/baselines/` (unchanged since phase 0).
- **C6:** the phase-0 readings.
- **C4:** R11.

## 5. The two programmers compared

- **Same direction.** Both kept edits that move the subject toward the parent, `hertwig-v2`:
  - Opus transplanted the parent's whole output layer.
  - Fable moved every tensor 12.5 % of the way there.
  - Both tried the clean sibling, and in both sessions the keep rule refused it: Opus block by
    block, Fable as a half-way soup.
  - Both lowered the blank bias. Opus's single step to the parent's value was refused. Fable
    took two steps of −0.20, both kept, and they did nothing on held-out.
- **Different outcome.**
  - The full output-layer transplant traded Schlussredaktion for Hertwig: +0.51 and −0.30 on
    held-out, net +0.15, not a gain.
  - The small whole-network step kept Schlussredaktion and recovered 1.3 Hertwig points: net
    −0.57, page interval excluding 0. It beats Opus's frozen model by −0.73 [−1.19, −0.35].
- **Why the training-side practice set still misleads.**
  - Fable's practice gain on Hertwig lines (−2.73) is twice its held-out gain there (−1.32).
  - The parent was trained on those 103 lines, so a parent-ward edit is over-rewarded on
    practice, as in session 2 (amendment A3). Fable stated this confound in its
    pre-registration and notes, and chose a small step rather than a block transplant.
  - What survives to held-out is the part of the parent direction that undoes the label noise:
    the same Hertwig-only gain that retraining on corrected images buys (R11).
- **Limits.**
  - One replicate per programmer.
  - Fable worked in rounds and Opus in one program.
  - Fable tried two step sizes on the parent axis.
  - The page interval's upper end (−0.05) is close to 0.
  - On validation-side practice (R7), five attempts by other programmers moved toward the
    sibling and none gained by the page interval. No attempt there moved toward the parent.
    The parent was trained on every Hertwig validation line, so under the donor rule a
    parent-ward edit could be judged only on Schlussredaktion lines.

## 6. Registration (11.11.2026, 15:08 CET, at the user's request)

After the session the user asked for the frozen model to be registered. It is registered on the
research share as a **disabled** candidate, in the same way as `latin_incunabula` v1 and v3 (R4, R12):

- `python -m wsp.run export runs/20261111-bern-combined-v1-f1-b10-1 --as kraken-bern-combined-wsp-v1`
  wrote the weights, `metadata.json` and a registry record to `runs/…/export/kraken-bern-combined-wsp-v1/`
  (weights not committed).
- The weights and `metadata.json` were copied to
  `training_folder/trained/kraken-bern-combined-wsp-v1/` on the share. File sha256 `bc08d2f9…`
  and tensor digest `4c6d1798…`, equal to `final.mlmodel` and to `summary.json`.
- Registered with `atr_training.registration` (training-atr-models `81539a6`) as
  `registry/trained/kraken-bern-combined-wsp-v1.yaml`, `enabled: false`, engine kraken, task htr,
  level page. The description names it a candidate, not for production.

### Serving path (11.11.2026, 22:14–22:39 CET, at the user's request)

As R4 and R12, with the user's OK for two writes to the share:

- The 14 `bern-eval` pages (6 Hertwig, 8 Schlussredaktion) were written to
  `tei/datasets/bern-disputation-wsp/bern_eval/`: images (PNG) and PAGE XML unchanged from the
  parquet shards the line set was cut from, `imageFilename` set to the new names
  `hertwig_pNNNN` / `schlussredaktion_pNNNN`, with a README and a manifest of sha256s.
- The candidate was enabled for the evaluation only (22:14 CET). The gateway listed it about
  20 s later.
- `start_batch`, source `datasets:bern-disputation-wsp/bern_eval`, both models in one run: smoke
  run `wsp-bern-heldout-f1-smoke` (3 pages), then `wsp-bern-heldout-f1` (14 pages).
- `score_ground_truth` by page name (report `score-gt-20261111T213917`, all 14 pages taken by
  name, none missing). Then the candidate was **disabled again** (22:39 CET).
- Paired over pages with `wsp.serving_path` (2,000 resamples, seed 20261106):
  `runs/…/serving_path.json`; the per-page numbers are in `runs/…/score_report_serving.json`.

| measurement, Δ in CER points | all 14 pages | Hertwig (6) | Schlussredaktion (8) |
|---|---:|---:|---:|
| harness, lines, page-clustered (§3) | −0.57 [−1.15, −0.05] | −1.32 [−2.36, −0.60] | +0.02 [−0.41, +0.52] |
| serving path, median page CER (unedited → candidate) | 23.83 → 20.91 % | 37.06 → 36.99 % | 18.54 → 15.57 % |
| serving path, median of page deltas | −2.42 [−3.61, −0.07] | +0.17 [−0.88, +0.87] | −3.61 [−5.25, −2.48] |
| serving path, character-weighted | **−2.16 [−3.32, −1.05]** | −0.04 [−0.68, +0.56] | **−3.84 [−4.89, −2.91]** |
| pages better / worse | 11 / 3 | 3 / 3 | 8 / 0 |

**The candidate is not worse through the serving path; it is better, and by more than in the
harness. But the gain sits on the other hand.** In the harness the gain is on Hertwig and
nothing moves on Schlussredaktion. Through the gateway Hertwig does not move and every
Schlussredaktion page improves. The two measurements differ in segmentation: the harness reads the
ground-truth line crops, the gateway segments each page itself, and its CER is about twice the
harness's. Not tested here, so only a guess: the edit changes how the model reads the gateway's
line crops of the Schlussredaktion pages (for example the lowered blank bias, which did nothing
on the ground-truth crops). The split control of §4 was not repeated through the gateway.
Fourteen pages from two documents; the interval is over pages, not documents.

### Split and Opus through the serving path (12.11.2026, 06:58–07:28 CET, at the user's request)

The open question above was whether the two blank steps, which did nothing on the ground-truth
crops, carry the gateway gain. Three controls were registered on the share as disabled models,
enabled for this read only, and disabled again:

- `kraken-bern-combined-wsp-v1-blank`: only the blank part of the kept diff (−0.40);
- `kraken-bern-combined-wsp-v1-interp`: only the interpolation part (every other element);
- `kraken-bern-combined-wsp-opus-s2`: session 2's frozen model (Opus), tensor digest
  `90765a26…`, equal to that run's summary.

Both parts are built by `build()` in `scripts/sham_combined_f1.py`, as in §4. They were read
into the same gateway run `wsp-bern-heldout-f1`, so the unedited model's reading is the same one
used above. Scoring is in report `score-gt-20261112T062756`; the pairing is in
`runs/…/serving_path_split.json`, and the line-level per-hand split in
`runs/…/split_by_hand_lines.json`.

Δ CER against the unedited model, in points. The serving-path rows are character-weighted with
an interval over pages. The line rows use the page-clustered interval (A4).

| model | measure | all | Hertwig | Schlussredaktion |
|---|---|---:|---:|---:|
| Fable final | lines | −0.57 [−1.15, −0.05] | −1.32 [−2.36, −0.60] | +0.02 [−0.41, +0.52] |
| | serving path | **−2.16 [−3.32, −1.05]** | −0.04 [−0.68, +0.56] | −3.84 [−4.89, −2.91] |
| blank part only | lines | +0.04 [−0.18, +0.25] | +0.01 [−0.44, +0.42] | +0.05 [−0.14, +0.26] |
| | serving path | −0.41 [−0.78, −0.06] | +0.19 [−0.10, +0.50] | −0.89 [−1.27, −0.55] |
| interpolation part only | lines | −0.53 [−1.14, −0.01] | −1.41 [−2.42, −0.67] | +0.16 [−0.17, +0.55] |
| | serving path | **−1.86 [−2.82, −1.03]** | −0.46 [−0.82, −0.06] | −2.97 [−4.05, −2.01] |
| Opus (session 2) | lines | +0.15 [−0.29, +0.60] | −0.30 [−0.82, +0.21] | +0.51 [−0.04, +1.08] |
| | serving path | −0.74 [−1.38, −0.17] | +0.39 [+0.11, +0.69] | −1.64 [−2.13, −1.15] |
| Fable − Opus | lines | −0.73 [−1.19, −0.35] | | |
| | serving path | −1.41 [−2.07, −0.79] | −0.43 [−0.87, +0.04] | −2.20 [−2.85, −1.54] |

What this settles:

- **The interpolation carries the gain on both measures.** Through the gateway it gives −1.86 of
  the −2.16 and is better on 12 of 14 pages. The blank part adds about −0.4 through the gateway,
  all of it on Schlussredaktion. On the ground-truth crops it adds nothing. The two parts add up
  roughly (−0.41 − 1.86 = −2.27 against −2.16).
- **The hand reversal belongs to the measurement, not to this edit.** Every edited model,
  Opus's included, gains on Schlussredaktion through the gateway (−0.89 to −3.84, 8 of 8 pages
  each), whether or not it gains there on the ground-truth crops. On Hertwig the gateway shows
  less than the crops for every model. Edits from the subject toward the parent help on the
  gateway's own line crops of the Schlussredaktion pages. Why is not tested: segmentation
  details, or how much of a line the gateway's crops include.
- **Fable's model beats Opus's on both measures**: −0.73 on lines and −1.41 through the gateway.
  Through the gateway Opus's model is not worse than the unedited one (−0.74), unlike on lines
  (+0.15). So "session 2 lost on held-out" holds for the line-level measure only.
- The reading in §5, that what transfers is the label-noise part of the parent direction (as
  in C4), rests on the line-level Hertwig gain. Through the gateway that gain is small (−0.46
  for the interpolation alone, −0.04 for the final model), and the larger gain is on the hand
  without label noise. Both readings hold only under their own measurement.

Fourteen pages from two documents; intervals are over pages. Four candidate and control
registrations are on the share, all disabled.

The candidate stays registered and **disabled**. Offering it for use is the user's decision.

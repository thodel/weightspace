# Phase 1 report — the first budgeted session, asteraix, 06.11.2026

Branch `phase1` on top of `phase0` (PR thodel/weightspace#1). Subject `kraken-german-xix-blb-v1`, practice `blb-train-300`, budget 10, held-out `blb-eval`. Everything on CPU (`CUDA_VISIBLE_DEVICES=""`, 12 threads): the session's 8 passes took 173 s, each held-out scoring of 5,628 lines about 35 minutes. Writes only under `~/Repo/weightspace/` and `~/atr-cache/wsp/`. No export, no registry write, no upload, no change to any service. Phase-0 report: `docs/PHASE0_REPORT.md`.

**The result in one sentence:** the readings' hypotheses for this model were tested and are wrong at the scale they predicted; the session kept a blank-bias edit of −1.0 that lowers practice CER by 0.02 points and held-out CER by 0.01 points, both with intervals that contain 0, and that is reported as no gain.

## 1. What was verified

| item | result |
|---|---|
| held-out set | `data/heldout/blb-eval.jsonl`: every line of the four handwriting `_eval` projects, 5,628 of 5,644 lines (16 skipped: baseline shorter than 5 px, listed in the meta file), 33 objects, list sha256 `0729b93b…`. `wsp.audit heldout`: every object an `export.json` eval object, 0 shared with any practice list (`data/heldout/overlap.json`). One crop (6 × 1655 px) is unscalable and reads as empty text for every model (commit b9dd4b1) |
| audit hardening | the phase-0 audit compared shelfmarks as raw strings, and the share spells them `K2911,14` where `export.json` spells `K2911-14`; under one spelling (`wsp.audit.norm_id`) the practice sets are still clean, so no phase-0 number changes (commit aeaf441, tests in `tests/test_audit_heldout.py`) |
| preregistration | `runs/20261106-german-xix-blb-v1-b10-1/preregistration.md`, complete before pass 1: digests of subject and donor, practice and held-out identities with sha256, 8 hypotheses with predicted direction and size, the controls; `program.py` written in full before the sandbox run |
| sandbox | `env -i … unshare -Un .venv/bin/python -I program.py` (the line is in `summary.json`); the programmer saw the practice set and the parent's weights, nothing else |
| C0 / C1 on held-out, before the session | `docs/baselines/heldout-blb-eval.json`: subject CER 0.1234, WER 0.286, 428 collapsed, blank share 0.877; parent `xix-v1` CER 0.3196, WER 0.779, 455 collapsed |
| serving-path cross-check | the gateway's `blb-eval-*` runs scored by page name against the same `_eval` projects (`score_ground_truth`, reports `score-gt-20261106T1511{34,38,42,44}` on tei): subject weighted CER 7.1 / 25.6 / 11.0 / 12.1 % (lassberg / korrespondenten / unbestimmt / manuskripte), parent 15.9 / 28.1 / 22.0 / 21.9 %. Line-level here: 5.3 / 24.4 / 14.7 / 11.7 % — same ordering, same gap to the parent; the two paths cut lines differently (the gateway segments, the set uses the GT rectangles), so the numbers are not expected to be equal |

The held-out CER of 12.3 % here against the trainer's recorded 8.7 % (docs/MODELS.md) is the same kind of difference: the trainer scores its own compile of the eval lines; this set is cut from the PAGE rectangles and keeps 428 lines the model collapses on.

## 2. The session (`runs/20261106-german-xix-blb-v1-b10-1`)

Decision rule (PROTOCOL.md §3): keep only if practice CER falls and the collapsed count does not rise; ties revert. This is the rule as it stood before 06.11.2026 evening; the minimum-effect rule now in PROTOCOL.md §3 (interval must exclude 0) was written after this session and would have kept nothing here. Practice: 300 lines, 12,867 reference characters, C0 831 edits.

| pass | hypothesis | edit | practice CER | WER | collapsed | decision |
|---:|---|---|---:|---:|---:|---|
| 1 | C0 | — | 0.06458 | 0.1711 | 10 | baseline |
| 2 | H1 blank bias back to the parent (−0.113) | `add` `O_18.lin.bias[0]` | 0.06443 | 0.1703 | 10 | kept (−0.016 pt, CI [−0.06, +0.03]) |
| 3 | H2 blank −0.5 total | `add` | 0.06505 | 0.1747 | 9 | reverted |
| 4 | H3 space row bias +1.0 | `add` `bias[space]` | 0.06474 | 0.1707 | 10 | reverted |
| 5 | H4 space +0.5 | `add` | 0.06489 | 0.1719 | 10 | reverted |
| 6 | H5 12 most-moved rows from the parent | `set_rows` | 0.06458 | 0.1711 | 10 | reverted (tie: +0.016 pt vs kept state, 0 vs C0) |
| 7 | H6 frequent rows `e r s i t n a` × 1.1 | `set_rows` | 0.06505 | 0.1731 | 9 | reverted |
| 8 | H7 blank −1.0 total (1.232 → 0.232) | `add` | 0.06435 | 0.1755 | 9 | kept (−0.008 pt vs kept state, CI [−0.34, +0.30]) |
| 9–10 | — | unspent | | | | every lever measured at two sizes; a third size would fit the practice set |

Frozen: 7 edits, 2 kept, one tensor changed (`O_18.lin.bias[0]`: 1.2323 → 0.2323; `diff.json` is that one element). Practice CER 0.06458 → 0.06435 (−0.023 points, paired bootstrap interval [−0.34, +0.30]). Digests: start `f35e075a…`, final `78d6b7d5…`, `final.mlmodel` sha256 `8cdd9231…`.

What each lever did, from the confusion tables (`log.jsonl`): every output-layer lever trades deletions for insertions and substitutions roughly one for one. Blank −0.5 took ` `→∅ from 103 to 90 and `e`→∅ from 40 to 35, and CER rose. Space +1.0 took ` `→∅ to 90 and `∅→ ` insertions ate the gain. Scaling the frequent rows reduced `e`→∅ and `n`→∅ as predicted and raised substitutions among those letters more. The 12 typographic rows are inert, as the readings said. H7's "gain" is 1 edit in 12,867 characters with WER up half a point and `∅→ ` entering the top three confusions: kept by the letter of the rule, not a gain.

## 3. Held-out, once, after `freeze()`

`wsp.run heldout … --set blb-eval`, scored against `docs/baselines/heldout-blb-eval.json` (the same lines, read by the unedited subject before the session; paired per-line bootstrap, 2,000 resamples, seed 20261106).

| | lines | CER before | CER after | parent | Δ (points) | 95 % interval |
|---|---:|---:|---:|---:|---:|---|
| **blb-eval** | 5,628 | **12.335 %** | **12.325 %** | 31.96 % | **−0.010** | [−0.048, +0.026] |
| lassberg_eval | 446 | 5.27 | 5.32 | 19.76 | +0.050 | [−0.037, +0.139] |
| korrespondenten_eval | 452 | 24.43 | 24.42 | 33.74 | −0.011 | [−0.157, +0.131] |
| unbestimmt_eval | 193 | 14.72 | 14.64 | 42.82 | −0.083 | [−0.336, +0.162] |
| manuskripte_eval | 4,537 | 11.75 | 11.73 | 32.52 | −0.013 | [−0.051, +0.027] |

WER 0.2858 → 0.2866, collapsed 428 → 403, length ratio 0.946 → 0.955; 1,510 of 5,628 lines read differently. Within the top-10 confusions, deletions fell from 12,451 to 10,670 and insertions rose from 0 to 775 — the same trade as on practice, at scale. **The interval contains 0 on the whole set and on every project; the edit is not a gain.** No C5 sham is owed because no gain is reported; the kept edit is nonetheless reproducible from `diff.json`.

## 4. What the readings got right and wrong

- Right: the deletion-dominated confusion tables, the inert typographic rows (H5 is a measured null), the direction of every lever (each reduced the deletions it targeted).
- Wrong: the size. docs/READINGS.md predicted −0.5 to −1 point for the blank-bias edit; the measured effect is −0.02 points on practice and −0.01 on held-out. The blank's +0.113 after fine-tuning is cosmetic; the model's deletions are decided in the recurrent stack frame by frame, and shifting any output row only moves which of two errors the frame makes. A bias reading from the weights cannot see that the softmax margins at the deleted frames are large.
- The ratio kept to attempted: 2 of 7, both noise-sized. Under a rule with a minimum effect (say the interval must exclude 0) the session would have kept nothing, which is the honest description.

## 5. Open questions for phase 2

1. **A minimum effect in the keep rule.** `Session.measure` already computes the paired interval against the baseline; keep should require the interval to exclude 0 (or a preregistered minimum), otherwise a budget of 10 keeps noise. A protocol change, to be written into PROTOCOL.md §3 before the next session.
2. **Levers below the output layer.** On this subject the next hypotheses are in `L_16` (the last BiLSTM) and `C_3.co.bias` (the tensor that moved most in the fine-tune, rel. L2 1.09, cosine 0.68): an α-sweep `parent + α(child − parent)` per block (C3 of the protocol) would locate where the fine-tune's gain sits before any edit is tried. Cheap: 7 blocks × 2 α values = 14 passes.
3. **The collapsed lines.** 403 held-out lines (7 %) collapse; they carry a large share of the CER (the subject's length ratio is 0.955). Whether they share a geometry (tall or narrow crops, the GT rectangles' synthesised baselines) is a data question to answer before a weight-space one; `readings/heldout_blb-eval.jsonl` has them.
4. **Scoring path.** Line-level CER from GT rectangles (12.3 %) vs the trainer's compile (8.7 %) vs the gateway's segmentation (weighted 7–26 % per project): one of these should be declared the paper's number. The protocol says held-out through the serving path when a candidate is exported; for sessions the line-level set is the stable one.
5. Family 1 and 3 remain blocked on the phase-0 questions (dataset revision; shard00 arrows).

## 6. Reproduce

```bash
cd ~/Repo/weightspace && git checkout phase1
export TMPDIR=~/atr-cache/tmp CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12
.venv/bin/python -m pytest -q tests                                                    # 22 passed, 2 skipped
D=/mnt/wbkolleg_dh_1/tei/datasets/lassberg-blb-gt
.venv/bin/python -m wsp.data build --heldout --set blb-eval --n 0 --source "pagexml:$D/lassberg_eval,$D/korrespondenten_eval,$D/unbestimmt_eval,$D/manuskripte_eval"
.venv/bin/python -m wsp.audit heldout --heldout data/heldout/blb-eval.jsonl --practice data/practice/*-300.jsonl
.venv/bin/python -m wsp.baselines --practice data/heldout/blb-eval.jsonl --out docs/baselines/heldout-blb-eval.json \
    --model models/kraken/kraken-german-xix-blb-v1.mlmodel models/kraken/kraken-german-xix-v1.mlmodel
R=runs/20261106-german-xix-blb-v1-b10-1        # the session: program.py as committed; the sandbox line is in summary.json
.venv/bin/python -m wsp.run heldout $R --set blb-eval
```

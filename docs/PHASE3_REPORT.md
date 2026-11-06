# Phase 3 report — family 1 unblocked, target 1 measured, family 3 dropped (asteraix, 06.11.2026)

Branch `phase3` on top of `phase2` (PR thodel/weightspace#4). In PLAN.md's numbering this is phase 2 ("Patch") continued: the decisions of 06.11.2026 on the Hertwig data and on family 3, and the first session on target 1. CPU only. No write to Hugging Face, the share, the registry or the trainer. Phase-1 report: `docs/PHASE1_REPORT.md`.

**In one sentence:** with the Hertwig pages correctly paired, `combined-v1` turns out to be the best family-1 model overall, its label-noise damage is three points on the Hertwig hand, and a budget-10 session found no edit that removes it on held-out pages; the one edit the keep rule accepted was an artefact of a donor trained on the practice lines.

## 1. Decisions taken (06.11.2026) and what was done

| decision | done |
|---|---|
| Fix the dataset on Hugging Face, run by hand from a terminal | `scripts/fix_hertwig_hf.py`: replaces the two 25.10 Hertwig shards on `main` with the byte-identical 22.10 files in one commit and verifies; dry run by default. Dry-run checked: XML 50/50 and 6/6 identical, images 6/50 and 1/6. **Applied by hand the same evening: commit `dda578ea`**, verified |
| Keep target 1, document it | `docs/HERTWIG_REVISIONS.md`, PLAN.md target 1, docs/MODELS.md: "can a weight-space edit undo training on label noise", C4 = retrain on the corrected data (a trainer job, needs an OK; not run) |
| Add the 8 Schlussredaktion test pages to the held-out set | `data/heldout/bern-eval.jsonl`: 6 Hertwig test pages (22.10 images) + 8 Schlussredaktion test pages, 545 lines, audited at page level, none among the 168 training pages |
| Drop the family-3 checkpoints | docs/MODELS.md, PLAN.md target 4: their training and validation arrows were not archived; the Cr255 → Cr1 rewrite (R1) stays |

Harness changes on the way: a parquet source may name several shards; `wsp.audit heldout` checks Transkribus test projects at (document, page) level; the practice audit checked eval page numbers against *all* rows, so Schlussredaktion page p0008 looked like Hertwig eval page 8 (fixed: per document; Schlussredaktion eval pages added); `Session.edit(per_tensor=…)` for block transplants; `RunSession.keep` now returns the rule's decision and persists only kept states. 36 tests pass.

## 2. The premise of target 1 does not hold on correct pages

C0/C1 before the session (`docs/baselines/bern-train-300.json`, `docs/baselines/heldout-bern-eval.json`):

| CER | practice Hertwig (103) | practice Schlussred. (197) | held-out Hertwig (287) | held-out Schlussred. (258) | **held-out all (545)** |
|---|---:|---:|---:|---:|---:|
| `combined-v1` | 0.205 | 0.118 | 0.267 | 0.135 | **0.193** |
| `hertwig-v2` (parent) | 0.114 | 0.299 | 0.236 | 0.306 | 0.275 |
| `schlussredaktion-v1` (clean sibling) | 0.236 | 0.113 | 0.295 | 0.131 | 0.204 |
| `cursive_inzigkofen` (grandparent) | | | | | 0.695 |

"combined-v1 (28.8 %) is worse than its parent and sibling (16.7 %)" (docs/MODELS.md) came from numbers scored on mismatched pages or on different sets. On correctly paired pages combined-v1 beats both overall; the noise cost it about 3 points on Hertwig against its parent, and it still reads Hertwig better than the sibling that never saw the noisy pages. The practice set samples both training projects (`bern-train-300`, 103 + 197 lines), because a Hertwig-only set would reward turning the subject back into the parent.

## 3. The session (`runs/20261106-bern-combined-v1-b10-1`)

Keep rule of PROTOCOL §3 (06.11.2026): the paired 95 % interval against the last kept state must lie below 0, enforced by the harness; the programmer also reverts any edit that raises the collapsed count.

| pass | hypothesis | practice all | Hertwig | Schlussred. | collapsed | decision | prediction |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | C0 | 0.1456 | 0.205 | 0.118 | 29 | — | |
| 2 | H1 blank bias → parent | 0.1451 | 0.204 | 0.118 | 29 | refused (interval contains 0) | refused ✓ |
| 3 | H2 output layer ← sibling | 0.1458 | 0.204 | 0.119 | 29 | refused | refused ✓ |
| 4 | H3 L_16 ← sibling | 0.1505 | 0.213 | 0.122 | 27 | refused (CER rose) | refused ✓ |
| 5 | H4 L_14 ← sibling | 0.1514 | 0.212 | 0.123 | 31 | reverted (collapsed up) | refused ✓ |
| 6 | H5 L_12 ← sibling | 0.1449 | 0.197 | 0.121 | 25 | refused (interval contains 0) | refused ✓ |
| 7 | H6 conv stack ← sibling | 0.1508 | 0.212 | 0.123 | 32 | reverted (collapsed up) | refused ✓ |
| 8 | H7 output layer ← parent | **0.1414** | 0.190 | 0.119 | 20 | **kept**, [−0.85, −0.04] pt | refused ✗ |
| 9 | H8 L_16 half-way → parent | 0.1427 | 0.172 | 0.129 | 18 | refused (interval contains 0) | refused ✓ |
| 10 | — | | | | | unspent | |

Final: one kept edit, the output layer (266 shared character rows and the blank) from hertwig-v2, the rows of `½ ƒ ̊` kept.

**Held-out, once after freeze** (paired per-line bootstrap against the subject's C0 reading, 2,000 resamples):

| bern-eval | lines | before | after | Δ (points) | 95 % interval |
|---|---:|---:|---:|---:|---|
| **all** | 545 | 19.33 % | 19.48 % | **+0.15** | [−0.10, +0.42] |
| Hertwig | 287 | 26.69 % | 26.39 % | −0.30 | [−0.72, +0.13] |
| Schlussredaktion | 258 | 13.48 % | 13.99 % | **+0.51** | [+0.21, +0.82] |

**Not a gain.** The practice gain (−0.42 points) came from the 103 Hertwig practice lines, which are the parent's own training lines: the parent's output layer fits them, so a transplant from it looks good on practice and does not transfer. On held-out the edit trades a non-significant Hertwig gain for a significant Schlussredaktion loss.

## 4. What this says

1. **The noise damage is not localised.** No single block of the clean sibling improves the subject; the sibling is worse on Hertwig in every block. The subject's Hertwig skill and its Schlussredaktion skill share the same weights. That is the joint-gain profile the xix-blb α-sweep found, now for a fine-tune on partly wrong labels.
2. **A donor trained on the practice lines biases the keep rule.** A transplant from such a donor is measured on lines the donor has memorised. This is a protocol gap, not an accident of this session: PROTOCOL §2 should require that a donor has not been trained on the practice lines, or that the practice set is scored per donor-seen and donor-unseen lines and the rule applied to the unseen part.
3. **The readings' hypotheses for this pair are refuted on correct data**: blank bias (H1) is noise-sized again; the output layer from the parent (readings (b)) helps practice and not held-out; the noise "profile" (blank up, rare rows pushed down) did not point at an edit that repairs anything.
4. The honest comparator is still C4, the same fine-tune on the corrected images. Given that combined-v1 already beats both relatives overall, the expected gain of a retrain is the 3 Hertwig points.

## 5. Open

1. ~~Apply the dataset fix~~: applied by hand on 06.11.2026, commit `dda578ea` on `main`, verified (docs/HERTWIG_REVISIONS.md). Still open: the trainer should pin that revision for family-1 jobs.
2. **C4 retrain** of combined-v1 on the corrected 22.10 Hertwig pages + Schlussredaktion, through the trainer on asteraix (needs an OK; a GPU job).
3. **Protocol:** a donor must not have been trained on the practice lines (or the rule is applied to donor-unseen lines only). Write into PROTOCOL §2 before the next session with a donor.
4. **Where the trainer's 28.8 % and 16.7 % came from** (which pages, which export): a check on the trainer side.
5. Target 2 (`hertwig-v2` alone, "a blank-dominant direction") is unblocked by bern-eval and the 22.10 practice set; given two sessions in which output-layer levers were noise-sized, the visualisation epic (thodel/weightspace#3) should come first.

## 6. Reproduce

```bash
cd ~/Repo/weightspace && git checkout phase3
export TMPDIR=~/atr-cache/tmp CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12
.venv/bin/python -m pytest -q tests                                                   # 36 passed, 2 skipped
.venv/bin/python scripts/fix_hertwig_hf.py                                            # dry run only
H=~/atr-cache/wsp/hf     # the shards are fetched by fix_hertwig_hf.py / hf_hub_download at the pinned revisions
.venv/bin/python -m wsp.data build --heldout --set bern-eval --n 0 --source "parquet:$H/bern-disputation-1528-rev20261022/data/train/A_V_1443_Hertwig_test/20261022_114935-0000.parquet,$H/bern-disputation-1528-rev20261025/data/train/A_V_1447_Schlussredaktion_test/20261025_170131-0000.parquet"
.venv/bin/python -m wsp.data build --set bern-train-300 --source "parquet:$H/bern-disputation-1528-rev20261022/data/train/A_V_1443_Hertwig_train/20261022_114935-0000.parquet,$H/bern-disputation-1528-rev20261025/data/train/A_V_1447_Schlussredaktion_train/20261025_170131-0000.parquet"
R=runs/20261106-bern-combined-v1-b10-1    # program.py as committed; the sandbox line is in summary.json
.venv/bin/python -m wsp.run heldout $R --set bern-eval
```

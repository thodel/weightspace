# Pre-registration — 20261111-planted-k1-zero-row-e-b-f1-b0-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (b), **Fable 5.1 rerun** of `runs/20261107-planted-k1-zero-row-e-b-b0-1` (Sonnet 5.5). Written before the first pass; the program is `runs/planted-f1/s06/b/program.py`, copied unchanged.

1. **Subject:** `k1-zero-row-e.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: output row of `e` (weights and bias) set to 0; `runs/planted/plant.json`) — tensor digest `f15cb502574f8562a2c7f8fcd819163ba2d17150ceac5f78d7e2e793fe47a7a8`, file sha256 `560e51f8525f14df97da88e09a174a29187993a5adc74bfdce368345bc13ad3c`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: parent `kraken-german-xix-v1`, a lineage donor; it was never trained on BLB data, so the donor rule holds.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 0 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. Budget 0: edits are accepted unmeasured (`keep_unmeasured`).
5. **Hypotheses (predicted direction and magnitude):**

# s06, task (b): blind programmer (budget 0)

Model: claude-fable-5-1 (Claude Fable 5.1). Settings: default reasoning effort (agent run under Claude Code); tools used: Read/Bash
(cat, ls, grep, wc) on the three pack files, the brief and `wsp/ops.py`, `wsp/session.py`, `wsp/run.py`, `wsp/kraken_subject.py`,
`wsp/align.py`; `python -m py_compile` on the program. No pass, no model loaded, no other folder read.

## Reading of the pack

- `describe.md`: output row `e` has norm **0.00** (every other common letter sits at 7.8-8.3: `r` 7.80, `s` 8.05, `t` 8.08, `i` 8.17,
  `n` 8.33, `a` 8.33). `diff_parent_to_subject.md`: `e` moved most, rel L2 1.000, cosine 0.000, norm parent -> child 7.64 -> 0.00,
  bias Δ only -0.142. That is a zeroed weight row with the bias left (nearly) alone.
- `c0_practice.md`: CER 0.1623, 2088 edits; `e`->∅ 910, `e`->a 131, `e`->i 90, `e`->o 52, `e`->r 36, `e`->c 35, `e`->u 32: about
  1290 of the 2088 edits are missed `e`s. With a zero weight row the `e` logit is a constant (its bias), so the blank (bias +1.23)
  or a neighbouring vowel wins at every `e`. Length ratio 0.900 and 9 collapsed lines are consistent with ~7 % of the characters
  never being emitted.
- Everything else looks like ordinary fine-tuning drift: all three BiLSTM blocks moved uniformly (mean rel L2 0.137-0.141, cosines
  0.95-0.98), conv blocks 0.13-0.31. Two things I noticed but am **not** acting on: `C_3.co.bias` rel L2 1.09 / cosine 0.68, but the
  tensor is tiny in both models (child L2 0.10, implied parent ~0.07) so a large relative change there says little; and
  `L_16.layer.weight_ih_l0` has abs max 4.70 where its siblings top out at ~2, but its rel L2 (0.296) is in line with the other
  input matrices, so if anything is off it is one or a few weights I cannot locate from these readings. Neither shows up in the
  confusions, and a blind edit on either would be a guess.

## Hypotheses

| # | hypothesis | edit | predicted effect | applied? |
|---|---|---|---|---|
| H1 | The `O_18.lin.weight` row of `e` was zeroed; the model can never emit `e`. | `set_rows` on `O_18.lin.weight`, row `cm['e']`, with the parent's `e` row (`pm['e']`, aligned by character; norm 7.64, in line with the child's other common letters). Child bias kept (it is within 0.14 of the parent's). | CER down by roughly 0.06-0.09 (0.162 -> ~0.07-0.10): most of the 910 deletions and a good part of the ~380 `e`->vowel substitutions recovered. The parent row sits in a feature space that drifted ~0.3 rel L2, so it is an imperfect detector; some `e`s stay missed and a few `a`/`i`/`o` may flip to `e`, but a zero row is strictly worse than any reasonable `e` detector. Collapsed lines and length ratio should move towards 0 and 1.0. | yes |
| H2 | `C_3.co.bias` was tampered with. | none | not confident; both tensors are near zero, no confusion pattern points at the convs. | no |
| H3 | A spike in `L_16.layer.weight_ih_l0` (abs max 4.70). | none | not confident; cannot locate it from the pack and its rel L2 is normal. | no |

Guard inside the program: the edit is only applied if the child's `e` row really has norm < 1e-3 and the parent's `e` row has a
sane norm (3-15). If either check fails the program notes it and freezes without editing.


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Claude Fable 5.1 (claude-fable-5-1) as a fresh Claude Code subagent (`model: fable`), one per run, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack (runs/planted-f1/s06/pack/, an opaque folder name) and runs/planted-f1/PROGRAMMER_BRIEF.md (the brief of the Sonnet arms with the model id and the read-restriction to its own folder changed); not told which defect was planted; never shown the Sonnet runs, their programs or results. Budget 0. Operator: a Claude Opus 5.5 (claude-opus-5-5) session; it knows the defects and the Sonnet results, set up the runs and does the held-out scoring and the write-up; it wrote no hypothesis and no edit and passed nothing of this to the programmers.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).

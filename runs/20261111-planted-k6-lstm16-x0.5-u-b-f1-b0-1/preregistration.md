# Pre-registration — 20261111-planted-k6-lstm16-x0.5-u-b-f1-b0-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (b), **Fable 5.1 rerun** of `runs/20261107-planted-k6-lstm16-x0.5-u-b-b0-1` (Sonnet 5.5). Written before the first pass; the program is `runs/planted-f1/s07/b/program.py`, copied unchanged.

1. **Subject:** `k6-lstm16-x0.5.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: last BiLSTM block (L_16, all weights and biases) x0.5; `runs/planted/plant.json`) — tensor digest `a339a718a392342d8f0c47c62505ed6b3f9c3f008b0ba5480471f534d1de55f1`, file sha256 `3c7da1f6f6ae42d70ccd49f5cea8b35dc781168bc70b46f557f7af9c3320fae5`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: `kraken-bern-hertwig-v2`, **an unrelated donor** (other lineage, trained on Bern 1528 data, never on BLB); the parent is not available to this arm.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 0 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. Budget 0: edits are accepted unmeasured (`keep_unmeasured`).
5. **Hypotheses (predicted direction and magnitude):**

# Hypotheses — planted-f1 s07, task (b), budget 0 (blind programmer)

Model: claude-fable-5-1 (Claude Fable 5.1). Settings: reasoning effort low-to-medium (the default of this
session), tools used: Read/Bash (`cat`, `ls`, `py_compile`) on the pack files and the allowed `wsp/` modules only.
No model loaded, no pass run. Donor not used.

## What the readings show

- `c0_practice.md`: CER 0.199 on a 300-line *training-side* practice set, WER 0.49, 24 collapsed lines,
  length ratio 0.828, and every one of the top-10 confusions is a deletion (` `, `e`, `n`, `i`, `r`, `c`, `s`,
  `u`, `t`, `m` → nothing). The model under-emits: blank wins too often. That is the symptom of a
  weakened signal reaching the output layer (small, indecisive logits lose to the blank's positive bias),
  not of a swapped or noisy output row (that would give substitutions of particular characters).
- `describe.md`: the three BiLSTMs are built identically (200 units each). L_12 and L_14 agree closely
  (weight_ih rms 0.22, weight_hh rms 0.30, bias_hh rms 0.92–0.93 in both directions). L_16 is at roughly
  half of that in **every** trainable tensor, both directions: weight_ih 0.121/0.113 (×0.56/×0.52 of L_14),
  weight_hh 0.154/0.157 (×0.50/×0.53), bias_hh 0.424/0.438 (×0.46/×0.47); bias abs max 1.20 vs 2.52 (×0.48).
  Weights and biases of one block shrinking by the same factor is not what training does; it is what a
  `scale` does.
- `diff_donor_to_subject.md` (unrelated donor, cosines ≈ 0 for the weights, so rel L2² ≈ 1 + k² with
  k = ‖subject‖/‖donor‖): the subject's L_12 and L_14 weights are 1.44–1.66× the donor's, its convs 1.25×,
  but its L_16 weights are 0.74–0.77× the donor's. In the donor, L_16 has the same norm as L_14
  (≈85–93 for weight_ih, ≈75–80 for weight_hh); in the subject L_16 is half of L_14. For the biases
  (cos ≈ 0.89 with the donor, so the lineages share the bias pattern), the rel L2 of 0.62 for L_16 against
  0.47 for L_12/L_14 is solved by k ≈ 0.46 for L_16 with the donor's three bias norms all ≈ 26 — again
  "L_16 halved".
- Output layer: blank bias +1.23 (donor ≈ +0.88), blank row norm 4.6, frequent letters the weakest rows,
  rare glyphs the strongest with strongly negative biases — the usual shape of a trained CTC output layer.
  Nothing there is far enough from the donor to call a defect. Convs: uniform rms ≈ 0.19, nothing odd.

## H1 (applied): the L_16 BiLSTM block was scaled by ≈ 0.5

- **Edit:** `scale`, factor 2.0, over the six non-zero tensors of L_16 (`weight_ih_l0`, `weight_hh_l0`,
  `bias_hh_l0`, and their `_reverse` twins; `bias_ih` is all zero and unaffected). One edit.
  Guard inside the program (free, no pass): the edit is applied only if the measured rms ratio
  L_16/L_14 over those six tensors lies in [0.40, 0.60]; otherwise nothing is applied.
- **Predicted effect:** large CER drop, from 0.199 to roughly 0.03–0.08 (a training-side set); length ratio
  back near 1.0; collapsed lines near 0; the deletion confusions mostly gone. If the true factor was not
  0.5 but e.g. 0.6, the model ends at 1.2× its original L_16, which an LSTM tolerates much better than 0.5×.
- **If wrong (L_16 is naturally small in this lineage):** doubling it would saturate its gates and push CER
  up, possibly well above 0.2. I judge this unlikely: the co-scaling of biases and weights, the within-model
  comparison with two identical sibling layers, and the donor's L_16 ≈ L_14 all point the same way, and the
  deletion-dominated readings are what a damped last layer produces.

## Not applied

- Blank bias down (`add` on `O_18.lin.bias[0]`): the deletions would also fit a raised blank bias, but
  +1.23 is within the normal range (donor +0.88) and, if H1 is the defect, lowering the blank after the
  repair would cause insertions. Not fairly sure → not applied.
- Anything from the donor (transplant/interpolate): the donor is of a different lineage (cos ≈ 0 on every
  weight); nothing in it is a healthy copy of the subject.


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Claude Fable 5.1 (claude-fable-5-1) as a fresh Claude Code subagent (`model: fable`), one per run, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack (runs/planted-f1/s07/pack/, an opaque folder name) and runs/planted-f1/PROGRAMMER_BRIEF_UNRELATED.md (the brief of the Sonnet arms with the model id and the read-restriction to its own folder changed); not told which defect was planted; never shown the Sonnet runs, their programs or results. Budget 0. Operator: a Claude Opus 5.5 (claude-opus-5-5) session; it knows the defects and the Sonnet results, set up the runs and does the held-out scoring and the write-up; it wrote no hypothesis and no edit and passed nothing of this to the programmers.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).

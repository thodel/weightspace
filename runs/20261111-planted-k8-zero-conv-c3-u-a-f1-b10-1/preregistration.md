# Pre-registration — 20261111-planted-k8-zero-conv-c3-u-a-f1-b10-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (a), **Fable 5.1 rerun** of `runs/20261107-planted-k8-zero-conv-c3-u-a-b10-1` (Sonnet 5.5). Written before the first pass; the program is `runs/planted-f1/s01/a/program.py`, copied unchanged.

1. **Subject:** `k8-zero-conv-c3.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: conv block C_3 (weights and bias) set to 0; `runs/planted/plant.json`) — tensor digest `df5577f17d54eec6a967ac38de6e09b18aefd467b1b80041c119ece5631304db`, file sha256 `80df9c7b020b643618164105826e3d7ee088addaad390b2e24232a5e6aa4ebd4`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: `kraken-bern-hertwig-v2`, **an unrelated donor** (other lineage, trained on Bern 1528 data, never on BLB); the parent is not available to this arm.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 10 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. 
5. **Hypotheses (predicted direction and magnitude):**

# Pre-registered hypotheses — planted-f1 / s01 / task (a), budget 10

- **Programmer:** claude-fable-5-1 (Claude Fable 5.1), run as a Claude Code subagent on 11.11.2026.
- **Settings:** reasoning effort as set by the harness (not visible to me; believed default); tools used: Read, Bash
  (`cat`/`ls` on the pack and the five allowed `wsp/` files, `py_compile` of `program.py`), Write. No model loaded, no pass
  run, no pack file other than the three in `pack/` read. Unrelated donor only: `kraken-bern-hertwig-v2.mlmodel`
  (no parent available). Inside the program I use only counted `s.measure` passes: no `activations()`/`logits()` calls,
  which the harness does not count but which would be uncounted forward passes over the practice images.

## Reading of the pack

1. `describe.md`: `C_3.co.weight` [32,32,3,13] and `C_3.co.bias` [32] have L2 **0.00**, rms 0, abs max 0: the second
   convolution is identically zero. `diff_donor_to_subject.md` agrees (rel L2 exactly 1.0000, cosine nan, 39936/39936
   and 32/32 elements "changed"). A zero `Cr` layer outputs ReLU(0) = 0 everywhere, so `C_6` sees a constant zero map,
   emits its (tiny, rms 0.03) biases, and everything downstream is a constant per frame → the CTC blank (bias +1.23,
   the only positive bias of note) wins every frame.
2. `c0_practice.md` confirms the symptom exactly: CER 1.0000, WER 1.0000, 300/300 lines collapsed, length ratio 0.000,
   12,867 edits = all deletions (`x → ∅` for every top confusion). The model emits nothing.
3. Everything else looks like a trained network: conv blocks `C_0/C_6/C_9` rms 0.18–0.19; LSTM `weight_ih/hh` rms
   0.22/0.30, `bias_ih` zero by construction, `bias_hh` rms ~0.9 with cosine 0.88 to the donor (forget-gate-style biases
   are shared across lineages, which is expected); output layer blank row weak (4.64) with positive bias, frequent
   letters (`e r s t i n a`) with the smallest norms and rare accented letters with the largest, which is the usual
   CTC pattern. The most negative bias (−7.85) belongs to the *new* row ` ` (NBSP, in the "added" codec list), not to
   the ordinary space. One mild oddity: `L_16.layer.weight_ih_l0` abs max 4.70 vs 1.3–2.8 for the other LSTM matrices,
   but its L2 (136.85 vs 128.07 reverse) is only 7 % higher, so this is not a block-scale plant and a single outlier
   would not move CER; the program counts such elements and notes them, it does not spend a pass on them.

So: **one defect, `C_3` zeroed**, with no parent to copy it back from. The donor's `C_3` was trained on *its own*
`C_0` channels for a different hand; the subject's `C_6` was trained on the subject's (lost) `C_3` channels. Any
replacement therefore feeds `C_6` features in the wrong channel slots. I expect partial repair at best.

## Hypotheses (edit class — predicted CER change on the 300 practice lines — conclusion if it fails)

Baseline pass 1 (`C0`): CER ≈ 1.00, collapsed ≈ 300 (checks the pack; if `C_3` is not all-zero at run time, nothing is
edited and the session is frozen with a note).

- **H1 — donor `C_3` transplant** (edit class: *transplant*, `C_3.co.weight` + `C_3.co.bias` from the donor, aligned
  by position as the shapes are identical). A trained 32→32 conv restores an image-dependent signal through `C_6`.
  Prediction: CER falls from 1.00 to **~0.85** (plausible range 0.60–1.10), collapsed lines from 300 to **< 50**;
  character identities mostly wrong, spacing/segmentation partly right. If refused (CER not significantly below 1.00,
  or > 1 through hallucinated insertions): the subject's `C_6`→LSTM cannot use mid-level filters from another lineage;
  the defect is located but not repairable at this layer from this donor.
- **H2 — identity pass-through `C_3`** (edit class: *set/transplant of a constructed tensor*: weight[i,i,1,6] = 1, bias 0,
  so `C_6` receives the pooled `C_0` features unchanged, channel i → channel i). Prediction: CER **~0.85** (0.60–1.10),
  collapsed < 50. Tournament against H1: kept only if significantly better than whichever state is current. If neither
  H1 nor H2 is kept, the rescue below is run; if that fails too, the model stays all-blank and I conclude that the gap
  between the unrelated donor's feature space and the subject's is too large for a single-layer graft.
- **H3 — amplitude of the grafted `C_3`** (edit class: *scale*, `C_3.co.weight` and `C_3.co.bias` together by ×2 or
  ×0.5; the stack is nearly homogeneous so this rescales the LSTM input). Direction chosen from the diagnostics of the
  best state: many collapsed lines / length ratio < 0.8 → ×2; length ratio > 1.2 → ×0.5; otherwise ×0.5 first. Up to two
  probes (same direction again if the first is kept, opposite if it is refused). Prediction: **−0.02 to −0.10** CER
  relative to the kept state. If both refused: signal amplitude is not the limiting factor; the mismatch is in feature
  identity.
- **Rescue (if H1 and H2 both refused)**: the same two grafts at ×4 (if the measured outputs were still collapsed) or
  ×0.25 (if they hallucinated, CER ≥ 1 with few collapsed lines), lower-CER candidate first. Prediction: as H1/H2.
- **H4 — whole donor conv stack** (edit class: *block transplant* of `C_0, C_3, C_6, C_9` weights and biases), tried
  only if the best CER is still > 0.60: the LSTM then receives features that are at least coherent among themselves.
  Prediction: CER **0.50–1.10**, probably *not* better than H1/H2 because the subject LSTM is as foreign to donor `C_9`
  features as `C_6` is to donor `C_3` features. If refused: a repair must keep the subject's own `C_6/C_9`.
- **H5 — blank-bias recalibration** (edit class: *bias shift*, `O_18.lin.bias[0]` by ∓0.75), only after a graft is kept,
  direction from the deletion/insertion balance (over-blank → −0.75; over-emitting → +0.75; balanced → skipped). Up to
  two steps. Prediction: **−0.01 to −0.04** CER. If refused: the residual error is not a blank-calibration problem.

Considered and rejected: replacing *all* shared tensors with the donor's (that would be the donor model with the
subject's 124 extra rows, not a repair of the subject); editing output rows from the donor (its rows live in a
different 400-d feature space, LSTM weights have cosine ≈ 0 to the subject's, so they carry no information here);
clipping the 4.70 outlier in `L_16.weight_ih_l0` (effect indistinguishable from zero, the keep rule would refuse).

## Pass plan (≤ 10)

1 baseline · 2 H1 · 3 H2 · 4–5 H3 (or rescue) · 6 H4 if CER > 0.60 · 7–8 H5 · spare. Every edit is decided by
`s.keep()`; the program branches only on `m.cer`, `m.collapsed`, the length ratio and the deletion/insertion counts
from `m.confusions`. It always ends with `s.freeze()` (reverting an undecided edit first if something raised).


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** Claude Fable 5.1 (claude-fable-5-1) as a fresh Claude Code subagent (`model: fable`), one per run, default reasoning effort, tools Read/Write/Bash (py_compile only); read only its pack (runs/planted-f1/s01/pack/, an opaque folder name) and runs/planted-f1/PROGRAMMER_BRIEF_UNRELATED.md (the brief of the Sonnet arms with the model id and the read-restriction to its own folder changed); not told which defect was planted; never shown the Sonnet runs, their programs or results. Wrote its program in advance, did not run it. Operator: a Claude Opus 5.5 (claude-opus-5-5) session; it knows the defects and the Sonnet results, set up the runs and does the held-out scoring and the write-up; it wrote no hypothesis and no edit and passed nothing of this to the programmers.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).

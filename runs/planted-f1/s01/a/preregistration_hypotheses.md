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

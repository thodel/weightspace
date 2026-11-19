Model id: claude-fable-5-2. Settings: reasoning effort low, task (a), budget 10, WITH the donor. Tools used: Read and Bash (grep/python
over the three pack files and the four API files of the brief; py_compile). No model loaded, no pass run, no other pack or run folder read.

# Readings the hypotheses rest on
- `describe.md`: `enc.06.attn.o.weight` (768x768) and `enc.06.attn.o.bias` have L2 norm 0 and share of zeros 1.0000. No other tensor has
  zeros or non-finite values; the per-layer norms are otherwise smooth.
- `diff_parent_to_subject.md`: the same two tensors have relative distance 2.0 and norm ratio 0 against the parent. Every other tensor is
  inside or below the healthy range (encoder matrices 0.03 to 0.09, layer norms and biases tiny, tied embedding 0.527 against "about 0.5");
  no token row with cosine below 0.5 or small norm, no outlier attention head. `dec.embed.pos` has an identical share of 0.47, which I read
  as untrained positions beyond the line lengths seen in fine-tuning, not as a defect.
- `c0_practice.md`: CER 0.3316, 12 collapsed lines, length ratio 0.905, confusions dominated by deletions (space 431, e, n, r, s ...): the
  decoder skips text, consistent with a degraded encoder representation, not with a token-specific fault.

# Hypotheses (one line each)
- H1 (edit class: transplant of a block from the donor). The whole attention output projection of encoder layer 6 was zeroed, so the layer
  adds no attention. Edit: transplant `enc.06.attn.o.weight` and `enc.06.attn.o.bias` from the parent. The neighbouring layers moved only
  0.06 to 0.09 (matrix) and 0.002 to 0.003 (bias) from the parent, so the parent's block is close to what the healthy child had.
  Prediction: practice CER falls from 0.33 by at least 0.15 absolute, to about 0.05 to 0.15; collapsed lines fall from 12 to at most 3; the
  deletion confusions shrink by more than half; the keep rule is met. If it fails (not kept, or the fall is under 0.05): the zeroed block
  is not the main cause, or the parent's block does not fit the child's layer; I then try H1b.
- H1b (edit class: transplant of a whole layer; only run if H1 is not kept). Transplant all tensors of `enc.06` from the parent, so the
  block meets the q/k/v and norms it was trained with. Prediction: CER falls by at least 0.10. If this fails as well: I conclude the damage
  cannot be repaired from these readings, apply nothing, and freeze.
- H2 (no edit). There is no second defect visible in the readings. Nothing else is edited. If after a kept H1 the CER is still above 0.20, I
  note that a defect invisible to these readings (or an intrinsically hard hand) remains; I do not edit blind.

Passes planned: 2 (C0, H1), 3 if H1b is needed. The remaining budget stays unspent.

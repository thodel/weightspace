# Pre-registration, 20261119-trocr-planted-p05-zero-enc6-attn-f-b10-1

TrOCR planted-defect control (docs/TROCR_STAGE3_PREREG.md, committed before this run), programmer arm `f`: Claude Fable 5.2 (`claude-fable-5-2`), arm (a2), with the parent donor.
The programmer was a fresh subagent with the tools Read, Write and Bash (`py_compile` only), default reasoning effort; it read only its pack
(`runs/trocr-planted/s01/pack/`) and `runs/trocr-planted/PROGRAMMER_BRIEF.md`; it was not told which defect was planted and was shown no other session.
Program: `runs/trocr-planted/s01/f/program.py`, copied unchanged (sha256 `a9a98603f46b8a8cb268118b4da500bd3f9560963dde95039ad26d0a58f87ae9`); hypotheses below as the programmer wrote them, before any pass.
Subject: planted `p05-zero-enc6-attn` = encoder layer 6 attention output projection (weight and bias) set to 0 (tensor digest `875ba809c9935636ca3433f3f7fece10d0701cbade4efae6a1500ddcc327bed2`; original `3027fea2a05614552c3f0c3a7d8999c9a34a25b1d24315eb7fcfd6c2eaac0151`).
Practice: `data/practice/trocr-blb-train-300-seen.jsonl`, sha256 `e102f7dd7e8fa492ff8c63191e7adf9584d264ac615398630d7708625c91bef4`. Held-out: `data/heldout/blb-eval-hand.jsonl`, 2272 lines, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`.
Budget 10, min effect 0.005, page-clustered interval, generation settings {"num_beams": 4, "max_new_tokens": 256, "no_repeat_ngram_size": 3, "length_penalty": 1.0, "early_stopping": true, "do_sample": false, "use_cache": true}.

## Hypotheses (the programmer's file `preregistration_hypotheses.md`)

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


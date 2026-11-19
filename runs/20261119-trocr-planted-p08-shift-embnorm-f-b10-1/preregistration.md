# Pre-registration, 20261119-trocr-planted-p08-shift-embnorm-f-b10-1

TrOCR planted-defect control (docs/TROCR_STAGE3_PREREG.md, committed before this run), programmer arm `f`: Claude Fable 5.2 (`claude-fable-5-2`), arm (a2), with the parent donor.
The programmer was a fresh subagent with the tools Read, Write and Bash (`py_compile` only), default reasoning effort; it read only its pack
(`runs/trocr-planted/s02/pack/`) and `runs/trocr-planted/PROGRAMMER_BRIEF.md`; it was not told which defect was planted and was shown no other session.
Program: `runs/trocr-planted/s02/f/program.py`, copied unchanged (sha256 `6972fb1b5901533280858395749f5baf94a5ae0c87da81a2686a3c7a804f8b30`); hypotheses below as the programmer wrote them, before any pass.
Subject: planted `p08-shift-embnorm` = bias shift: a fixed random direction added to the decoder embedding layer-norm bias, a fraction of the norm of its weight (tensor digest `783d096bc6a72d33921dd7f21311e01168673f467232ee708d29df0764ab7cbc`; original `3027fea2a05614552c3f0c3a7d8999c9a34a25b1d24315eb7fcfd6c2eaac0151`).
Practice: `data/practice/trocr-blb-train-300-seen.jsonl`, sha256 `e102f7dd7e8fa492ff8c63191e7adf9584d264ac615398630d7708625c91bef4`. Held-out: `data/heldout/blb-eval-hand.jsonl`, 2272 lines, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`.
Budget 10, min effect 0.005, page-clustered interval, generation settings {"num_beams": 4, "max_new_tokens": 256, "no_repeat_ngram_size": 3, "length_penalty": 1.0, "early_stopping": true, "do_sample": false, "use_cache": true}.

## Hypotheses (the programmer's file `preregistration_hypotheses.md`)

Model: claude-fable-5-2. Settings: low reasoning effort, one sitting, no passes run before writing. Tools used: file read and shell (cat/grep/awk) on the pack, the brief, wsp/ops.py and wsp/session.py; py_compile. Task (a), budget 10, WITH the donor; MASK = s.unseen_by("trocr-kurrent-XVI-XVII").

# Readings that carry the hypotheses
- C0 practice: CER 0.298, 33 collapsed lines, length ratio 0.915, confusions are almost all deletions (space 492, e, n, r, i, s): the decoder drops text, it does not confuse letters.
- `dec.embed.norm.bias`: relative distance 1.71 to the parent, norm ratio 6.31 (29.81 against about 4.73), cosine 0.12. A healthy child moves layer-norm vectors by 0.007 to 0.13. Its std (0.93) equals its rms, so the added part has mean near 0 (not one constant added to all entries) and is nearly orthogonal to the parent's vector.
- Nothing else is out of the healthy range: `dec.embed.norm.weight` 0.004, tied embedding 0.527 (normal about 0.5, no row with cosine below 0.5, none zero), no outlier head, no tensor with zeros or non-finite values, all weight matrices 0.03 to 0.24. The 0.47 identical share of `dec.embed.pos` is read as untrained positions beyond the line lengths, not a defect. The late encoder layers (0.03 to 0.05) moved a little less than the reference child; not treated as a defect.

# Hypotheses (one line each)
- H1 (edit class: donor repair of one layer-norm vector; `add` of minus the estimated planted part, only if the difference to the parent has a near-constant magnitude, relative spread of |d| below 0.15): the defect is a planted offset of fixed size on `dec.embed.norm.bias`; removing sign(d)*median|d| restores the child's own bias. Prediction: CER falls from about 0.30 to 0.05 to 0.12 (a drop of 18 to 25 points), collapsed lines from 33 to at most 5, deletions of space no longer the top confusion by a wide margin. If it fails (not kept): the planted part is not of fixed size; go to H2.
- H2 (edit class: donor transplant of one layer-norm vector, `transplant` of `dec.embed.norm.bias` from the parent; run when H1's condition is not met or H1 was not kept): the bias carries added noise; the parent's vector is within the healthy 0.13 of the child's true one. Prediction: CER falls to 0.05 to 0.12 (drop of 18 to 25 points), collapsed lines to at most 5. If it fails: the large bias is not the cause of the deletions (or the defect is elsewhere and invisible to these readings); I apply nothing more and conclude the pack does not localise the defect.
- H3 (edit class: donor interpolation, `interpolate` alpha 0.5 of `dec.embed.norm.weight` toward the parent; only if a repair of the bias was kept and CER is still above 0.15 or more than 8 lines collapse): a second, small defect sits in the same layer norm. Prediction: weak, change within ±0.5 points, most likely not kept (the weight is at 0.004 from the parent). If it fails: no further defect visible; stop.
- H0 (no edit): no other tensor is touched. The tied embedding, `dec.embed.pos`, and the encoder show no defect, so editing them would only pull the child toward a parent that reads this hand worse.

Expected passes: 2 to 4 of 10. Unspent passes stay unspent.


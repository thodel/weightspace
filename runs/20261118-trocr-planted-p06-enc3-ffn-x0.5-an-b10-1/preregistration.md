# Pre-registration, 20261118-trocr-planted-p06-enc3-ffn-x0.5-an-b10-1

TrOCR planted-defect control (docs/TROCR_STAGE3_PREREG.md, committed before this run), programmer arm `an`: Claude Sonnet 5.5 (`claude-sonnet-5-5`), arm (a), **without** the donor.
The programmer was a fresh subagent with the tools Read, Write and Bash (`py_compile` only), default reasoning effort; it read only its pack
(`runs/trocr-planted/s05/pack_nodonor/`) and `runs/trocr-planted/PROGRAMMER_BRIEF.md`; it was not told which defect was planted and was shown no other session.
Program: `runs/trocr-planted/s05/an/program.py`, copied unchanged (sha256 `325f581d1d3c1d5e148cff47a652a7a0e2bd6d10ae6da8204e571db43914ffb1`); hypotheses below as the programmer wrote them, before any pass.
Subject: planted `p06-enc3-ffn-x0.5` = encoder layer 3 FFN output projection (weight and bias) x0.5 (tensor digest `cfaca90a9d4f64012196994ac5d55049ccff0fff34eae08799c0158c6e74d6c2`; original `3027fea2a05614552c3f0c3a7d8999c9a34a25b1d24315eb7fcfd6c2eaac0151`).
Practice: `data/practice/trocr-blb-train-300-seen.jsonl`, sha256 `e102f7dd7e8fa492ff8c63191e7adf9584d264ac615398630d7708625c91bef4`. Held-out: `data/heldout/blb-eval-hand.jsonl`, 2272 lines, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`.
Budget 10, min effect 0.005, page-clustered interval, generation settings {"num_beams": 4, "max_new_tokens": 256, "no_repeat_ngram_size": 3, "length_penalty": 1.0, "early_stopping": true, "do_sample": false, "use_cache": true}.

## Hypotheses (the programmer's file `preregistration_hypotheses.md`)

model id: claude-sonnet-5-5 | settings: reasoning effort low, no donor (pack_nodonor only), MASK = s.unseen_by() | tools: Read, Bash (ls/cat/grep/py_compile), Write; no model loaded, no pass run

Readings: C0 CER 0.2346, 7 collapsed lines, top confusion = space deleted (377), then dropped letters. describe.md shows one sharp anomaly:
enc.03.ffn.out.weight L2 31.0 / bias 5.4 against neighbours 61-67 / 10.7-11.4 (about half). Other candidates are weak (dec.11.norm_ffn.weight 67 vs ~30.6 elsewhere).

H1 scale enc.03.ffn.out.{weight,bias} x2.0 (undo a halving): CER down ~0.03 to 0.10 (encoder features are damaged; likely the largest effect). If it fails: the low norm is natural for this layer; drop it.
H2 scale dec.11.norm_ffn.{weight,bias} x0.6 (norm of 67 vs 30 in all other decoder norms): CER change -0.005 to +0.01, low confidence. If it fails: norm is natural for the last layer; the edit is undone.
H3 scale all other enc.NN.ffn.out.{weight,bias} x1.3 (a global damping of encoder FFNs, not visible in single-tensor norms): CER down ~0.01 to 0.03 if present; if fail try x1.15; if both fail, no global encoder FFN damping.
H4 scale all dec.NN.cross.o.{weight,bias} x1.3 (global damping of cross attention output; would show as dropped characters/spaces): CER down ~0.01 to 0.03 if present; if fail try x1.15; if both fail, no cross-o damping.
Not attempted (no visible signature in norms/zeros/non-finite): zeroed rows/heads, swapped rows, noise, embedding shifts. Zero share is 0 and nothing is non-finite everywhere, so zeroing is unlikely.
Conclusion rule: an edit is believed only if s.keep accepts it (interval below -0.005); otherwise treated as no defect there. Max 7 passes used of 10.


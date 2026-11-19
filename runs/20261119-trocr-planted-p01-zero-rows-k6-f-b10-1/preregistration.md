# Pre-registration, 20261119-trocr-planted-p01-zero-rows-k6-f-b10-1

TrOCR planted-defect control (docs/TROCR_STAGE3_PREREG.md, committed before this run), programmer arm `f`: Claude Fable 5.2 (`claude-fable-5-2`), arm (a2), with the parent donor.
The programmer was a fresh subagent with the tools Read, Write and Bash (`py_compile` only), default reasoning effort; it read only its pack
(`runs/trocr-planted/s03/pack/`) and `runs/trocr-planted/PROGRAMMER_BRIEF.md`; it was not told which defect was planted and was shown no other session.
Program: `runs/trocr-planted/s03/f/program.py`, copied unchanged (sha256 `4a04265e77a6858acd5c5796bc18eb40a2a135368daccf48c1406d04474d6330`); hypotheses below as the programmer wrote them, before any pass.
Subject: planted `p01-zero-rows-k6` = the rows of the k most frequent token ids of the practice text set to 0 in the tied embedding (input and readout) (tensor digest `22b6a4c8bdabbc6fb8829ec7f60c743999ccb893f669d5a30b4724f9a488b0ca`; original `3027fea2a05614552c3f0c3a7d8999c9a34a25b1d24315eb7fcfd6c2eaac0151`).
Practice: `data/practice/trocr-blb-train-300-seen.jsonl`, sha256 `e102f7dd7e8fa492ff8c63191e7adf9584d264ac615398630d7708625c91bef4`. Held-out: `data/heldout/blb-eval-hand.jsonl`, 2272 lines, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`.
Budget 10, min effect 0.005, page-clustered interval, generation settings {"num_beams": 4, "max_new_tokens": 256, "no_repeat_ngram_size": 3, "length_penalty": 1.0, "early_stopping": true, "do_sample": false, "use_cache": true}.

## Hypotheses (the programmer's file `preregistration_hypotheses.md`)

Model id: claude-fable-5-2. Settings: reasoning effort low, single agent, no subagents. Tools used: Read and Bash (cat/grep/awk on the pack, the brief and wsp/ops.py, wsp/session.py, wsp/metrics.py, wsp/trocr_subject.py; py_compile of program.py). No model loaded, no pass run, no other pack or run folder read.
Task (a): budget 10, WITH the donor. MASK = s.unseen_by("trocr-kurrent-XVI-XVII").

# Readings that the hypotheses rest on
- `describe.md`: every tensor finite; the only tensor with a non-zero share of zeros is `dec.embed.tokens` (0.0001 = about 6 rows of 50265). Per-layer norms are smooth.
- `diff_parent_to_subject.md`: all weight matrices at 0.08 to 0.24 relative distance, layer norms at 0.003 to 0.13, tied embedding at 0.53: all inside the 'healthy child' range. No head over 3 times its tensor's median. Exactly six rows of the tied embedding are exactly zero: ids 6 `,`, 2432 `Ġund`, 364 `Ġe`, 225 `en`, 90 `t`, 4 `.`; the next lowest cosine is 0.74 (ordinary).
- `c0_practice.md`: CER 0.2508, 8 collapsed lines; top confusions are deletions of space (319), `e`, `.`, `n`, `,`, and `,`->`;`: the model cannot write (zero logit, no bias) or read back (zero input embedding) these six very frequent tokens and substitutes or drops them.
- Judged not defects: `dec.embed.pos` identical share 0.47 (positions never reached in training keep the parent's values; distance 0.044 is normal); `enc.01.attn.o.bias` identical share 0.0013 (one element of 768).

# Hypotheses (CER on the 300 practice lines; C0 expected 0.251)
- H1 (edit class: set_rows, donor rows plus common drift). The single defect is the six zeroed rows of `dec.embed.tokens`. Setting them to the parent's rows plus the mean drift of the intact rows (mean over intact rows of subject minus parent; a shift common to all rows leaves the softmax unchanged, so the restored rows must carry it too) repairs the model. Prediction: CER falls by 12 to 21 points, to 0.04 to 0.13; collapsed lines fall from 8 to at most 2; deletions of space, `.`, `,` leave the top of the confusion list or shrink by more than half. Kept by the rule. If it fails (not kept, or CER stays above 0.18): the common drift is harmful or the rows are not the whole defect; go to H1b.
- H1b (edit class: set_rows, donor rows verbatim; only run if H1 is not kept). Prediction: CER falls by 10 to 20 points. If this also fails: the zero rows are not the cause of the CER (I would then conclude a defect invisible in norm/distance statistics, and stop without further edits, because nothing else in the pack points anywhere).
- H2 (edit class: set_rows, refinement; only if H1 or H1b is kept). Replacing the common drift by the local drift (mean drift of the 50 intact rows nearest in the parent's cosine space) gives the six rows the child's adaptation more exactly. Prediction: change between -1.0 and +0.3 points; most likely smaller than the keep threshold of 0.5 points, so most likely NOT kept. If not kept: the simple restoration is as good as can be told on 300 lines.
- H3 (edit class: scale of the six rows via set_rows; only if a repair is kept; direction chosen from the confusions of the kept state). If after the repair the six tokens' characters (`,` `.` `t` `e` `n` space) are still more often deleted than inserted (by more than 30 edits in the top-10), the restored rows are too short for the tied readout: scale them by 1.06; if more inserted than deleted by the same margin, scale by 0.94; otherwise skip. Prediction: change of -0.2 to -0.8 points if run; probably not kept by the rule. If not kept: row length from the parent is adequate.
- H0 (no edit). Nothing else is wrong: no other tensor is edited. After the repair I expect the residual confusions to be ordinary look-alike letters (`n`/`r`/`m`, `e`), not a pattern pointing at a second defect. If CER after the kept repair is still above 0.15, I record that a second defect is likely but leave it, since the pack gives no location.

Passes planned: 1 (C0) + 1 (H1) + at most 1 (H1b) + 1 (H2) + at most 1 (H3) = at most 5 of 10. Unspent passes stay unspent.


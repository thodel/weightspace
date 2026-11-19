# Pre-registration, 20261118-trocr-planted-p06-enc3-ffn-x0.5-ad-b10-1

TrOCR planted-defect control (docs/TROCR_STAGE3_PREREG.md, committed before this run), programmer arm `ad`: Claude Sonnet 5.5 (`claude-sonnet-5-5`), arm (a), **with** the parent donor.
The programmer was a fresh subagent with the tools Read, Write and Bash (`py_compile` only), default reasoning effort; it read only its pack
(`runs/trocr-planted/s05/pack/`) and `runs/trocr-planted/PROGRAMMER_BRIEF.md`; it was not told which defect was planted and was shown no other session.
Program: `runs/trocr-planted/s05/ad/program.py`, copied unchanged (sha256 `96864639f56868ed92b26276e27b93b808cbbabcd648251c34e63b6e9b60d6fe`); hypotheses below as the programmer wrote them, before any pass.
Subject: planted `p06-enc3-ffn-x0.5` = encoder layer 3 FFN output projection (weight and bias) x0.5 (tensor digest `cfaca90a9d4f64012196994ac5d55049ccff0fff34eae08799c0158c6e74d6c2`; original `3027fea2a05614552c3f0c3a7d8999c9a34a25b1d24315eb7fcfd6c2eaac0151`).
Practice: `data/practice/trocr-blb-train-300-seen.jsonl`, sha256 `e102f7dd7e8fa492ff8c63191e7adf9584d264ac615398630d7708625c91bef4`. Held-out: `data/heldout/blb-eval-hand.jsonl`, 2272 lines, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`.
Budget 10, min effect 0.005, page-clustered interval, generation settings {"num_beams": 4, "max_new_tokens": 256, "no_repeat_ngram_size": 3, "length_penalty": 1.0, "early_stopping": true, "do_sample": false, "use_cache": true}.

## Hypotheses (the programmer's file `preregistration_hypotheses.md`)

# Preregistration, task (a), budget 10, with donor (pack s05)

Model id: claude-sonnet-5-5 (Sonnet 5.5). Settings: default/low reasoning effort; tools used: file read, shell (grep/sort over my pack files, py_compile). No model was loaded or run.

Evidence: C0 CER 0.2346, 7 collapsed lines; top confusion is deleted spaces (377) and deleted letters. Against the donor, nearly everything sits in the healthy range (0.07-0.24; embedding 0.53 as expected). Two clear outliers: `enc.03.ffn.out.weight` (rel. distance 0.667, norm ratio 0.5027, cosine 0.996) and `enc.03.ffn.out.bias` (0.667, ratio 0.5000, cosine 1.0000). A pure halving of the encoder-layer-3 FFN output (weight and bias). No zero rows, no non-finite values, no head outliers, norm ratios elsewhere 0.98-1.06.

H1 (scale, main). enc.03 FFN output was multiplied by about 0.5. Edit: scale `enc.03.ffn.out.{weight,bias}` by 2.0. Predicted CER change: about -0.04 to -0.10 (C0 0.2346 to roughly 0.13-0.19), collapsed lines down. If it fails (not kept, change within +-0.005): the halving is not what hurts; try H2 and then conclude the defect is elsewhere or the residual stream tolerates it.

H2 (fallback if H1 is kept but a smaller factor might fit better, or if H1 is not kept). Scale by 1.8 and by 1.5 instead of 2.0 (revert H1 first if not kept). Predicted: 2.0 best, because the norm ratio is 0.5000 to four digits and the cosine is about 1. If 1.8 beats 2.0 beyond noise, conclude the child's own step also changed that tensor by ~0.1 and keep 1.8.

H3 (interpolate, only if H1 kept). Interpolate the repaired `enc.03.ffn.out.{weight,bias}` 0.25 toward the donor. Predicted change about 0 (within +-0.003); the donor reads this hand worse. Expect revert. Conclusion if kept: the child's drift in that tensor was harmful.

H4 (null on the rest). Because everything else is in the healthy range, I predict no other edit helps. One cheap probe: interpolate `dec.embed.tokens` 0.15 toward the donor (embedding moved 0.53, plausible but normal). Predicted +0.00 to +0.01 (worse or neutral); expect revert. If it helps by more than 0.5 CER points, keep it and conclude the embedding drift overfit the child.

Budget plan: C0, H1, then at most H2 (1-2 passes), H3, H4. At most 6 measures; the rest unspent.


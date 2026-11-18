# Pre-registration, 20261118-trocr-planted-p05-zero-enc6-attn-ad-b10-1

TrOCR planted-defect control (docs/TROCR_STAGE3_PREREG.md, committed before this run), programmer arm `ad`: Claude Sonnet 5.5 (`claude-sonnet-5-5`), arm (a), **with** the parent donor.
The programmer was a fresh subagent with the tools Read, Write and Bash (`py_compile` only), default reasoning effort; it read only its pack
(`runs/trocr-planted/s01/pack/`) and `runs/trocr-planted/PROGRAMMER_BRIEF.md`; it was not told which defect was planted and was shown no other session.
Program: `runs/trocr-planted/s01/ad/program.py`, copied unchanged (sha256 `1f3b127336d9ca20bbe096e346f8ad5785e09968322fdef9493033e9db649af2`); hypotheses below as the programmer wrote them, before any pass.
Subject: planted `p05-zero-enc6-attn` = encoder layer 6 attention output projection (weight and bias) set to 0 (tensor digest `875ba809c9935636ca3433f3f7fece10d0701cbade4efae6a1500ddcc327bed2`; original `3027fea2a05614552c3f0c3a7d8999c9a34a25b1d24315eb7fcfd6c2eaac0151`).
Practice: `data/practice/trocr-blb-train-300-seen.jsonl`, sha256 `e102f7dd7e8fa492ff8c63191e7adf9584d264ac615398630d7708625c91bef4`. Held-out: `data/heldout/blb-eval-hand.jsonl`, 2272 lines, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`.
Budget 10, min effect 0.005, page-clustered interval, generation settings {"num_beams": 4, "max_new_tokens": 256, "no_repeat_ngram_size": 3, "length_penalty": 1.0, "early_stopping": true, "do_sample": false, "use_cache": true}.

## Hypotheses (the programmer's file `preregistration_hypotheses.md`)

# Preregistration, pack s01, task (a) budget 10, with the parent donor

Model id: claude-sonnet-5-5 (Sonnet 5.5). Settings: reasoning effort low; tools used: file read, shell (text parsing of the pack only, no model loaded, no program run), file write.

Readings: C0 CER 0.3316, 12 collapsed lines, length ratio 0.905, dominant confusion is deletion (space 431, e 152, n 102, ...). In the pack, the only anomaly:
`enc.06.attn.o.weight` and `enc.06.attn.o.bias` are exactly zero (share of zeros 1.0, norm 0; distance to parent 2.0, norm ratio 0). Every other tensor is within the healthy-child range (matrices 0.03 to 0.23, norms 0.003 to 0.13, tied embedding 0.53); no zero rows, no outlier heads, no non-finite values.

| # | edit class | hypothesis | predicted effect | if it fails |
|---|---|---|---|---|
| H1 | transplant (parent values) of `enc.06.attn.o.weight`, `enc.06.attn.o.bias` | encoder layer 6 attention output was zeroed, so layer 6 has no attention contribution and encoder features are degraded, producing deletions | CER falls from 0.33 by 8 to 18 points (to about 0.15 to 0.25), collapsed lines fall, length ratio rises toward 1 | If CER falls by less than 0.5 points or rises: the zeroing is not the cause of the deletions; revert and test H3 (the child's own o was never the parent's, so the donor o may mismatch the child's q,k,v); conclude that readings do not localise the fault in the encoder |
| H2 | scale of the repaired `enc.06.attn.o.{weight,bias}` by 1.15 (only if H1 kept) | donor o is a stand-in for the child's own, which may carry a different gain; the child's neighbours show norm ratios 1.00, so no real gain change expected | CER change within +-0.5 points; kept only if more than 0.5 points better | Revert; conclude that the donor o is as good as it gets with one tensor-wide factor |
| H3 | interpolate (alpha 0.5 toward parent) of `enc.06.attn.o.{weight,bias}` (only if H1 failed or made things worse) | the donor o is partly compatible with the child's q,k,v; half of it is a gentler repair | CER change of -1 to -6 points relative to the C0 state | Revert; conclude that nothing in the encoder attention is repairable from the donor |
| H4 | interpolate of `dec.embed.tokens` alpha 0.2 toward parent (only if CER after H1 is still above 0.20) | residual damage comes from the tied embedding drifting from the parent (rel. distance 0.53) | likely no gain or a loss, +-1 point; the drift is in the healthy range, so I expect a revert | Revert; conclude that the embedding is not defective |

Nothing else is planned. Total passes at most 5 of 10.


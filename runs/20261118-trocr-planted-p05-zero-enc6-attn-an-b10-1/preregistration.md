# Pre-registration, 20261118-trocr-planted-p05-zero-enc6-attn-an-b10-1

TrOCR planted-defect control (docs/TROCR_STAGE3_PREREG.md, committed before this run), programmer arm `an`: Claude Sonnet 5.5 (`claude-sonnet-5-5`), arm (a), **without** the donor.
The programmer was a fresh subagent with the tools Read, Write and Bash (`py_compile` only), default reasoning effort; it read only its pack
(`runs/trocr-planted/s01/pack_nodonor/`) and `runs/trocr-planted/PROGRAMMER_BRIEF.md`; it was not told which defect was planted and was shown no other session.
Program: `runs/trocr-planted/s01/an/program.py`, copied unchanged (sha256 `a3e76b93218984b82dfbc1b76da62cae2e9f7635022e3a38869b4d0567f6e635`); hypotheses below as the programmer wrote them, before any pass.
Subject: planted `p05-zero-enc6-attn` = encoder layer 6 attention output projection (weight and bias) set to 0 (tensor digest `875ba809c9935636ca3433f3f7fece10d0701cbade4efae6a1500ddcc327bed2`; original `3027fea2a05614552c3f0c3a7d8999c9a34a25b1d24315eb7fcfd6c2eaac0151`).
Practice: `data/practice/trocr-blb-train-300-seen.jsonl`, sha256 `e102f7dd7e8fa492ff8c63191e7adf9584d264ac615398630d7708625c91bef4`. Held-out: `data/heldout/blb-eval-hand.jsonl`, 2272 lines, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`.
Budget 10, min effect 0.005, page-clustered interval, generation settings {"num_beams": 4, "max_new_tokens": 256, "no_repeat_ngram_size": 3, "length_penalty": 1.0, "early_stopping": true, "do_sample": false, "use_cache": true}.

## Hypotheses (the programmer's file `preregistration_hypotheses.md`)

model id: claude-sonnet-5-5
settings: default reasoning effort (not set explicitly); no donor (pack_nodonor only)
tools used: Read, Bash (ls/cat/grep/wc, mkdir, py_compile), Write. Read only the brief, pack_nodonor/describe.md and c0_practice.md, and the API files wsp/ops.py, wsp/session.py. No pass run.

Readings: C0 CER 0.3316 on practice set, 12 collapsed lines, length ratio 0.905; top confusions are all deletions (space 431, e, n, r, s, i, u, o, a): the model drops material, no systematic substitution. describe.md: enc.06.attn.o.weight and enc.06.attn.o.bias are exactly zero (share of zeros 1.0, norm 0), all else finite and nonzero. Other oddities: dec.11.norm_ffn.weight norm 67.5 (neighbours ~31) and bias 3.5; enc.10.norm_ffn weight 52.9 / bias 19.6 (neighbours ~20 / ~7).

H1 (strong): enc.06 self-attention output projection was zeroed (dead layer, encoder attention missing at layer 6), which explains the deletions. No donor, so the weight cannot be restored. Edit class: add (to the zero o.weight) a small attention-readout built from pinv(enc.06.attn.v) times 0.3, i.e. make attention return a damped copy of the mean of its inputs. Prediction: CER change -0.00 to -0.03; honest prior that it does NOT help (p keep ~0.2) because the correct o is not recoverable from weights alone. If it fails: conclude enc.06 attention is unrecoverable without the parent; leave zeros. If kept: try 0.3 more (total 0.6) once.
H2 (weak): dec.11.norm_ffn weight/bias inflated (~2.2x) as a planted scale of the last decoder FFN-norm. Edit class: scale x0.7 of dec.11.norm_ffn.{weight,bias}. Prediction if true: CER -0.01 to -0.05; if natural last-layer outlier: CER +0.01 to +0.05. If fails: conclude the large norm is natural; revert.
H3 (weak): enc.10.norm_ffn weight/bias inflated (~2.5x vs neighbours). Edit class: scale x0.7 of enc.10.norm_ffn.{weight,bias}. Prediction: same shape as H2, probability of kept ~0.1. If fails: natural massive-activation outlier; revert.
H4 (weak): length ratio 0.905 and deletions suggest EOS emitted too eagerly. Edit class: scale of the EOS row (id 2) of dec.embed.tokens by 0.95 (tied: also changes its input embedding). Prediction: CER -0.005 to -0.01 if early EOS is a cause, else ~0 or worse. If fails: deletions are inside the line, not at the end; revert.

Budget plan: 1 baseline + H1 + H2 + H3 + H4 (+1 branch on H1) = at most 6 of 10 passes. Every edit is judged by the harness keep rule (interval below -0.005) and a collapsed-line check (not more than C0+3).


# Pre-registration, 20261118-trocr-planted-p07-noise-dec8-60-an-b10-1

TrOCR planted-defect control (docs/TROCR_STAGE3_PREREG.md, committed before this run), programmer arm `an`: Claude Sonnet 5.5 (`claude-sonnet-5-5`), arm (a), **without** the donor.
The programmer was a fresh subagent with the tools Read, Write and Bash (`py_compile` only), default reasoning effort; it read only its pack
(`runs/trocr-planted/s08/pack_nodonor/`) and `runs/trocr-planted/PROGRAMMER_BRIEF.md`; it was not told which defect was planted and was shown no other session.
Program: `runs/trocr-planted/s08/an/program.py`, copied unchanged (sha256 `cb23cbfd4d9b809162f8b0ca22f813af2289c7a1c115a1f4ec0ee02c244dd075`); hypotheses below as the programmer wrote them, before any pass.
Subject: planted `p07-noise-dec8-60` = Gaussian noise on the weights of decoder layer 8 (self, cross, FFN), total norm a fraction of the layer's (tensor digest `1547142fbf5cd16953eec8cec5b0de50535bd5fd1d294fefe00309925161d6a1`; original `3027fea2a05614552c3f0c3a7d8999c9a34a25b1d24315eb7fcfd6c2eaac0151`).
Practice: `data/practice/trocr-blb-train-300-seen.jsonl`, sha256 `e102f7dd7e8fa492ff8c63191e7adf9584d264ac615398630d7708625c91bef4`. Held-out: `data/heldout/blb-eval-hand.jsonl`, 2272 lines, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`.
Budget 10, min effect 0.005, page-clustered interval, generation settings {"num_beams": 4, "max_new_tokens": 256, "no_repeat_ngram_size": 3, "length_penalty": 1.0, "early_stopping": true, "do_sample": false, "use_cache": true}.

## Hypotheses (the programmer's file `preregistration_hypotheses.md`)

model id: claude-sonnet-5-5
settings: default reasoning effort (low), no extended-thinking budget set; tools used: Bash (read pack + wsp/ops.py/session.py only, awk/python over describe.md text, py_compile), Read, Write. Pack: pack_nodonor (no donor). MASK = s.unseen_by(). No model loaded, no pass run.

Reading: C0 CER 0.2321, 381 of 2986 edits are a deleted space, then deleted e/n/r/s, n->r, e->a (a weak, noisy reader; not collapsed). No zeros, no non-finite, no dead tensor. Encoder and decoder layer norms grow smoothly with depth EXCEPT dec.08: every weight matrix is inflated against its neighbours dec.07/dec.09 (ffn.out 60.3 vs 44.6/46.3, ffn.in 70.0 vs 56.4/58.7, cross.o 40.3 vs 35.9/33.8, cross.v 36.7 vs 32.9/31.6, self q/k 55.8/54.3) while biases/norm vectors are not. Excess norm about 39 on ffn (std ~0.019) and about 19 on 1024x1024 matrices (std ~0.019): the same std in every matrix = additive i.i.d. Gaussian noise on dec.08 matrices (planted noise defect). Nothing else stands out (enc.10 norm_ffn and dec.11 norm_ffn.weight are large but look like normal final-layer outliers).

H1 (edit class: scale, per-tensor Wiener shrinkage). Noise on dec.08 matrices; clean norm estimated as the mean of dec.07/dec.09 norms, scale each matrix by clean^2/observed^2 (about 0.55-0.8). Prediction: CER falls by 0.5 to 3 points (to about 0.205-0.227); half of a point is the keep threshold. If it fails: shrinkage hurts the signal as much as noise; conclude plain scaling is the wrong tool and go to H2 from the original state.
H2 (edit class: add of an SVD soft-threshold delta; low-rank denoise). Same dec.08 matrices; singular values soft-thresholded at the Marchenko-Pastur noise edge sigma*(sqrt(m)+sqrt(n)) with sigma from the excess norm. The noise has a flat spectrum, the signal a decaying one, so this removes more noise per unit signal than H1. Prediction: CER falls 1 to 4 points, better than H1 (applied on the kept H1 state if H1 was kept, else on original). If it fails: the signal is not low-rank enough or sigma is mis-estimated; conclude dec.08 noise is not the main cause or is not removable without data.
H3 (edit class: add, milder SVD threshold at 0.6x the edge, only if H2 was kept). Prediction: within +-0.5 points of H2; kept only if clearly better, else revert (tests whether H2 over-shrinks).
H4 (edit class: scale, only if H2 and H3 both failed, i.e. the noise story is wrong). Try scaling dec.08 cross.o.weight+bias by 1.0/0.85 inverse test is NOT planned; instead stop and freeze. Prediction: no change. Unspent passes are fine.
Null: if C0 is healthy-ish and no edit passes the keep rule, freeze unchanged (the 0.23 CER may be the base level of this child on this hand).


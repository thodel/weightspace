# Pre-registration, 20261120-trocr-stage4-A-p3s-b10-1

TrOCR stage 4 (docs/TROCR_STAGE4_PREREG.md), subject A `from-kurrent-v2`, programmer Claude Sonnet 5.5, arm P3, donor absent. The programmer was a fresh subagent with the tools Read, Write
and Bash (`py_compile` only); it read only its pack (`runs/trocr-stage4/A/pack_nodonor/`) and `runs/trocr-stage4/PROGRAMMER_BRIEF.md`.
Program `runs/trocr-stage4/A/p3s/program.py`, copied unchanged (sha256 `91cdf30c0b890af445489572020fd8e5a600e49124c3d1f5ac1e48724608de44`); hypotheses below as written, before any pass.
Subject digest `3027fea2a05614552c3f0c3a7d8999c9a34a25b1d24315eb7fcfd6c2eaac0151`. Practice `data/practice/trocr-blb-train-1200-seen.jsonl`, sha256 `b6b33a8d0c19b733538655f58f07a40518737174af467e2193b6b96b9deceb73`. Budget 10, min effect 0.005,
page-clustered interval, generation settings {"num_beams": 4, "max_new_tokens": 256, "no_repeat_ngram_size": 3, "length_penalty": 1.0, "early_stopping": true, "do_sample": false, "use_cache": true}. Landlock allow-list: `landlock.json`; donor files absent.

## Hypotheses (`hypotheses.md`)

Model: claude-sonnet-5-5 (Sonnet 5.5). Settings: reasoning effort low; tools used: Read, Write, Bash (py_compile only). Task (a), budget 10, no donor, MASK = s.unseen_by().

Reading of the pack: healthy fine-tune. C0 CER 0.2307, length ratio 0.952, deletions dominate (space -> nothing 1540, e/r/n/s/i -> nothing, only 130 insertions of e). The model under-writes: it drops spaces and letters. No tensor shows zeros or non-finite values. Outliers: dec.11.norm_ffn.weight (norm 67 vs ~31 elsewhere), enc.00/enc.10 norm_ffn weights. No token list or tokenizer is available to me, so I make no token-specific edits.

Edits are uniform scalings (no direction guessed), applied one at a time; each is decided by the keep rule. Expected outcome for all: nothing to gain, |delta CER| < 0.3 points, rejected.

H1 (scale, all dec.NN.cross.o weights and biases x1.05): more weight on the image evidence relative to the language prior should reduce omissions. Predicted CER change -0.2 to -0.5 points if the under-reading comes from a prior-dominated decoder; else +0.1 to +0.5. If it fails with CER up: try x0.95 (branch). If CER down but not significant: try x1.10.
H2 (scale, dec.embed.tokens x1.04): the tied readout is sharper; the ranking of beams (shorter vs longer) shifts slightly. Predicted -0.1 to +0.3. The input side is almost undone by the embedding LayerNorm. If CER rises: try x0.96.
H3 (scale, enc.norm_final.weight and bias x1.03): larger encoder output increases cross-attention key/value magnitude, sharpening attention to the image. Predicted -0.1 to +0.3. If it fails: nothing further on the encoder.
H4 (scale, dec.11.ffn.out weight and bias x1.05): late-layer FFN contribution up; the odd norm_ffn of layer 11 hints at a large late contribution that the fine-tune tuned, so I expect a rise. Predicted +0.2 to +1.0 (so only tested if budget remains; revert expected).

Conclusion if all fail: the model sits at a local optimum for scalar rescalings; report no gain, keep the unedited weights.


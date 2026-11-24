# Pre-registration, 20261120-trocr-stage4-A-p3f-b10-2

TrOCR stage 4 (docs/TROCR_STAGE4_PREREG.md), subject A `from-kurrent-v2`, programmer Claude Fable 5.2, arm P3, donor absent. The programmer was a fresh subagent with the tools Read, Write
and Bash (`py_compile` only); it read only its pack (`runs/trocr-stage4/A/pack_nodonor/`) and `runs/trocr-stage4/PROGRAMMER_BRIEF.md`.
Program `runs/trocr-stage4/A/p3f/program.py`, copied unchanged (sha256 `5e80c195dba431e8d22d1788c82c326692248bb9fb1488c3d5e97713c553dce6`); hypotheses below as written, before any pass.
Subject digest `3027fea2a05614552c3f0c3a7d8999c9a34a25b1d24315eb7fcfd6c2eaac0151`. Practice `data/practice/trocr-blb-train-1200-seen.jsonl`, sha256 `b6b33a8d0c19b733538655f58f07a40518737174af467e2193b6b96b9deceb73`. Budget 10, min effect 0.005,
page-clustered interval, generation settings {"num_beams": 4, "max_new_tokens": 256, "no_repeat_ngram_size": 3, "length_penalty": 1.0, "early_stopping": true, "do_sample": false, "use_cache": true}. Landlock allow-list: `landlock.json`; donor files absent.

## Hypotheses (`hypotheses.md`)

# Hypotheses: TrOCR stage 4, subject A, task (a), budget 10, WITHOUT the donor (p3f)

Programmer: model id `claude-fable-5-2` (Claude Code subagent). Settings: reasoning effort as set by the harness (not visible to me, not changed by me);
tools used: Read (brief, `pack_nodonor/describe.md`, `pack_nodonor/c0_practice.md`, `wsp/ops.py`, `wsp/session.py`, `wsp/run.py`, `wsp/trocr_subject.py`),
Write (this file and `program.py`), Bash (`python -m py_compile program.py` only). No model loaded, no pass run, no donor read, named or looked for.
`MASK = s.unseen_by()` (all lines).

## What the pack shows
- C0 on practice: CER 0.2307, 11,809 character edits, length ratio 0.952, 29 collapsed lines.
- One confusion dominates: **space → nothing, 1,540 edits = 13 % of all edits = 3.0 CER points.** The next (e → nothing) has 247. Spaces are about one
  character in seven of German text, so if the deletions came from dropped words or truncated lines the space share would be near 15 % of deletions; it
  is far above that. The model merges words: at a word boundary it writes a continuation BPE token where a word-initial (`U+0120`) token is due.
- `describe.md`: no tensor is degenerate (no zeros, no non-finite, norms smooth across layers). The only outlier is `dec.11.norm_ffn.weight`
  (norm 67.5 against about 30.7 in the other decoder layers) with `dec.11.norm_ffn.bias` 3.53: this is the last operation before the tied readout.
- Mechanism used: the decoder is post-norm and there is no final norm and no output bias, so logits = `dec.embed.tokens` · (g·norm(x) + b) with
  b = `dec.11.norm_ffn.bias`. Adding a vector δ to b shifts the logit of token i by exactly `tokens[i]·δ` at every position: b is a
  1024-dimensional stand-in for the missing output bias. It does not touch the input side of the tied embedding.

## Hypotheses (ΔCER on the practice set, in CER points = 0.01)
- **H1 (class: add, direction built from the model's own embedding and token strings; `dec.11.norm_ffn.bias`).** δ is the ridge least-squares direction
  that raises the logit of every word-initial token by c nats against every other token (EOS pinned to the other class). Doses c = 1.0, then 2.0 (and
  4.0) if it helps, or 0.5 if it does not; the best dose is kept through the harness rule.
  Prediction: space deletions fall by a third to a half at the best dose (1,540 → 800 to 1,000), space insertions rise by fewer than 200;
  **ΔCER −0.7 points (range −0.3 to −1.5)**, length ratio up toward 0.97. Probability that the keep rule (interval below −0.5 points) is met: about 40 %.
  If it fails (no dose lowers CER): the missing spaces are not a readout prior but come from the visual side or from the reference convention
  (the model does not see the gap), and a constant logit shift only trades deletions for insertions; nothing is kept.
  If the fit note shows that word-initial tokens are not linearly separable in the embedding (class separation below 2 within-class std), a failure
  says nothing about the prior, only that this tensor cannot express it.
- **H2 (class: add, same tensor; EOS logit −1.0 nat, other tokens pinned).** Length ratio 0.952 and letter deletions (e, r, n, s, i: 786 in the top
  ten) suggest some early stops. Prediction: **ΔCER −0.1 points (range −0.4 to +0.2)**, collapsed lines not up by more than 5; the keep rule will most
  likely refuse it. If it fails: the deletions are inside words (minims of the Kurrent hand), not line ends.
- **H3 (class: scale; `dec.11.norm_ffn.weight` and `.bias` × 1.25, i.e. logit temperature 0.8; × 0.8 if × 1.25 hurts and passes remain).** Only beam
  search can react (argmax is unchanged). Prediction: **|ΔCER| < 0.2 points**, not kept. If it moves more than that, the beam's length handling matters
  and H2 should be read with it.
- **H5 (sham for H1, only if H1 was kept; always reverted).** A random vector with the norm of the kept δ on the same tensor.
  Prediction: **|ΔCER| < 0.15 points**: the H1 effect is the direction, not the size of the move.
- **H4 (class: scale; all `dec.NN.cross.o.{weight,bias}` × 1.1; last, if a pass remains).** More weight on the image against the decoder's own context.
  Prediction: **ΔCER between −0.1 and +0.3 points**, not kept; a global scale of this kind is what a random comparator also gives, so even a small
  gain would not be a finding.

## What I expect overall
The likely outcome is either one kept edit (H1, about −0.7 points on practice, and it should transfer to unseen lines of the same hand because it is a
property of the readout, not of the lines) or nothing kept. H2 to H4 are expected to be reverted. Passes: C0 1, H1 2 to 4, H2 1, H3 1 to 2, H5 1, H4 1;
never more than 10 (guarded in the program).


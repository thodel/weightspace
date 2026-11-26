# Pre-registration, 20261120-trocr-stage4-A-p2-b10-1

TrOCR stage 4 (docs/TROCR_STAGE4_PREREG.md), subject A `from-kurrent-v2`, programmer Claude Fable 5.2 (`claude-fable-5-2`), arm P2, with the donor. The programmer was a fresh subagent with the tools Read, Write
and Bash (`py_compile` only); it read only its pack (`runs/trocr-stage4/A/pack/`) and `runs/trocr-stage4/PROGRAMMER_BRIEF.md`.
Program `runs/trocr-stage4/A/p2/program.py`, copied unchanged (sha256 `f88e43ede1023ea1f2c306de7e604f22871bdaeb4404c4ce8fed7c1d39a2ea8a`); hypotheses below as written, before any pass.
Subject digest `3027fea2a05614552c3f0c3a7d8999c9a34a25b1d24315eb7fcfd6c2eaac0151`. Practice `data/practice/trocr-blb-train-1200-seen.jsonl`, sha256 `b6b33a8d0c19b733538655f58f07a40518737174af467e2193b6b96b9deceb73`. Budget 10, min effect 0.005,
page-clustered interval, generation settings {"num_beams": 4, "max_new_tokens": 256, "no_repeat_ngram_size": 3, "length_penalty": 1.0, "early_stopping": true, "do_sample": false, "use_cache": true}. Landlock allow-list: `landlock.json`; donor files visible.

## Hypotheses (`hypotheses.md`)

Model id: claude-fable-5-2 (Claude Code subagent). Settings: reasoning effort as set by the harness (not visible to me, not changed by me); tools used: Read (brief, the three pack files, wsp/ops.py, wsp/session.py, wsp/run.py, wsp/trocr_subject.py), Write (this file and program.py), Bash (only `python -m py_compile program.py`). No pass run, no model loaded, nothing else read.
Task (a), budget 10, WITH the donor. Subject A.

# Readings the hypotheses rest on

- `c0_practice.md`: CER 0.2307, 11,809 character edits, of which 1,540 (13 %, 3.0 CER points) are a reference space with nothing written; no space insertion is in the top ten (so fewer than 91). Length ratio 0.952, 29 collapsed lines. The letter deletions (`e` 247, `r` 151, `n` 146) are far too few for the missing spaces to come from dropped stretches of line: the model merges words. In this BPE a space is not a token of its own, it is the leading `U+0120` of a word-initial token, so "merging" means the model writes the continuation variant of a token where the word-initial variant was due.
- Brief: the output projection is the tied embedding, there is no output bias, and the decoder is post-norm, so `dec.11.norm_ffn.bias` (1024) is added directly in front of the readout: adding a vector d to it adds exactly `E @ d` to the logits of every step, where E is `dec.embed.tokens`. This is the one place where a per-token logit offset can be written without touching how a token is read in.
- `diff_magnitudes.md`: the model sits 0.04 to 0.24 from its donor in every weight matrix, norm ratios 1.000 to 1.03 (a small step). The model reads its own training pages at 23 % CER, which is an underfitted step, not an overfitted one. Magnitudes only; I read no direction from the pack.

# Hypotheses (CER changes are on the practice set, against the last kept state)

H1 (edit class: logit offset through the final layer-norm bias, `add` on `dec.11.norm_ffn.bias`). The model under-writes word-initial tokens. A vector d fitted by weighted ridge regression so that `E @ d` is +b on tokens with a leading `U+0120` and 0 on the others (end token held at 0), with b = 1.0 nat, restores part of the 1,540 missing spaces at a small cost in inserted spaces. Predicted: CER down by 0.5 to 1.5 points (0.005 to 0.015) at the best b; space deletions down by a third or more, space insertions up by fewer than the deletions saved. Ladder (at most three passes): b = 1.0; if kept, one further +1.0 on top; if lower but refused by the rule, b = 2.0 (then 3.0 if still falling); if not lower, b = 0.5. If it fails (no b lowers CER by more than noise): the missing spaces are not a prior that a constant offset can move (they are decided by the image, or they come from the decoding clean-up of spaces before punctuation, which no weight reaches), and I conclude the dominant confusion is not editable at the readout.

H2 (edit class: same, the end token alone). Length ratio 0.952 and 29 collapsed lines suggest early stops. `E @ d` = -1.0 on the end token, 0 on the tokens in use. Predicted: small, CER down 0.1 to 0.3 points, collapsed lines down by a few; not expected to pass the keep rule on its own. If it fails or CER rises: the short lines are merged words and dropped letters, not early stops; nothing to gain at the end token.

H3 (edit class: extrapolation away from the donor, `interpolate` with alpha = -0.3 on every tensor except the tied embedding). The fine-tuning step is small and the model still reads its training hand at 23 %: the step toward this hand is too short, and lengthening it (w + 0.3 (w - donor)) lowers CER on this hand. The embedding is left out because 86 % of the distance lies there, mostly in rows of tokens this hand never uses. Predicted: CER down 0.2 to 0.8 points; my confidence is low (about one in three that it is lower at all). Ladder: if kept, a further 0.3; if lower but refused, 0.6; if not lower, 0.15. This is the opposite sign of the comparator (interpolation toward the donor), not a copy of it. If it fails: the step is not underfit along its own direction (the residual error is not more of the same move), and the donor holds no usable direction for this hand in either sign.

H4 (edit class: partial interpolation toward the donor, encoder tensors only, alpha = +0.3; a contrast for H3, not a candidate). Predicted: CER unchanged or up by 0 to 0.5 points; reverted unless the rule keeps it. If instead it lowers CER while H3 raises it, the encoder was moved too far and the finding is that, to be set against the global-interpolation comparator.

H5 (edit class: one `add` over all tensors, the best refused H1 offset and the best refused H3 extrapolation together; run only if both lowered CER by more than 0.2 points and neither was kept). Predicted: the two gains add (they act on different places), total 0.5 to 1.5 points, enough for the rule where each alone was not. If it fails: the two are not additive and nothing is kept.

# Expected outcome overall

H1 is the only hypothesis I expect to pass the keep rule (about an even chance). The others are more likely than not to be reverted. If nothing is kept the program freezes the unchanged model, which the brief names as a legitimate result.

Caveat written before any pass: the tokens "in use" for the fit of H1 and H2 are taken from the reference texts of the practice lines (token types and counts only, to weight the regression and to set its scale); no image, no reading and no line-level error enters the edit. The offset is one vector of 1,024 numbers with one free scalar b.


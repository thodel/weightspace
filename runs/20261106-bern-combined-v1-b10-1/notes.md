# Notes — 20261106-bern-combined-v1-b10-1

Programmer: Claude Opus 5.5, Claude Code on asteraix. `program.py` was written in full before the sandbox run; these notes are written after it from `log.jsonl` and the per-pass readings.

- Before the run, the C0/C1 baselines changed the premise: on correctly paired pages the subject is the best of the three family-1 models overall (held-out 19.3 % against 27.5 % for the parent and 20.4 % for the sibling); its noise damage is about 3 points on Hertwig held-out lines. H8 was rewritten from "the refused sibling blocks half-way" to "L_16 half-way toward the parent" before pass 1; the preregistration states the predictions after that change.
- Pass 1 C0: practice 0.1456 (Hertwig lines 0.205, Schlussredaktion 0.118), 29 collapsed.
- H1 blank bias to the parent's: 0.1451, interval contains 0, refused. As predicted.
- H2–H6, the clean sibling one block at a time: output layer 0.1458 (refused), L_16 0.1505 (CER rose), L_14 0.1514 with 31 collapsed (reverted), L_12 0.1449 (interval contains 0, refused; Hertwig lines 0.197, Schlussredaktion 0.121), conv stack 0.1508 with 32 collapsed (reverted). As predicted, no block of the sibling helps; the noise damage is not concentrated in one block that the sibling has clean.
- H7 output layer from the parent: 0.1414, 20 collapsed, interval [−0.85, −0.04] points: **kept**. Predicted refused. Per project: Hertwig lines 0.205 → 0.190, Schlussredaktion 0.118 → 0.119. The program printed "REFUSED" here: `RunSession.keep` returned None (see deviations in `summary.json`; fixed in `wsp/run.py`).
- H8 L_16 half-way toward the parent, against the H7 state: 0.1427, interval contains 0, refused. Hertwig lines 0.172, Schlussredaktion 0.129: the trade the prediction named, larger on both sides.
- Pass 10 unspent. Frozen: 8 edits, 1 kept (`O_18.lin.weight` and `O_18.lin.bias` from hertwig-v2 for the 266 shared rows and the blank; the rows of ½ ƒ ̊ kept).

Held-out, once after freeze (bern-eval, paired against the subject's C0 reading): all 19.33 % → 19.48 % (+0.15 points, [−0.10, +0.42]); Hertwig 26.69 → 26.39 (−0.30, [−0.72, +0.13]); Schlussredaktion 13.48 → 13.99 (**+0.51, [+0.21, +0.82]**). Not a gain. The practice gain came from the 103 Hertwig practice lines, which are the parent's own training lines: a donor trained on the practice lines makes any transplant from it look good on practice.

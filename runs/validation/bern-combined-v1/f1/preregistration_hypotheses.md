# Pre-registered hypotheses — bern-combined-v1, attempt f1 — programmer model: claude-fable-5-1 (Claude Fable 5.1)

Settings: subject `kraken-bern-combined-v1` (4.1 M parameters, output `O_18` 270 rows, blank = row 0); practice set bern-val, 513 lines
(212 Hertwig, 301 Schlussredaktion), C0 CER 0.1386, 27 collapsed lines, length ratio 0.949, 2408 character edits.
Budget 10 passes (pass 1 = baseline); keep rule of the harness (paired 95 % interval of CER(after) − CER(last kept) below 0, min_effect 0,
2000 resamples, default seed). One donor: `kraken-bern-schlussredaktion-v1` (sibling fine-tune of the same parent hertwig-v2; CER 0.1820 on
the 297 practice lines it was not trained on). Every edit that uses it is judged with `rule_mask=s.unseen_by('kraken-bern-schlussredaktion-v1')`.
`kraken-bern-hertwig-v2` is not used as a donor: it trained on all 212 Hertwig practice lines and its CER on the rest is 0.2766, worse than the subject.
Written before any pass; the program is fixed in advance and branches only on the keep decisions it gets.

Readings the hypotheses rest on: (a) the error profile is dominated by deletions (` `→∅ 195, `/`→∅ 98, `e`/`n`/`t`/`r`/`i`/`s`→∅ 373; insertions
∅→` ` 99, ∅→`/` 57), length ratio 0.949, blank row norm 2.78 with bias +1.00 (the highest bias of the layer): the blank wins too many frames.
(b) Subject and schlussredaktion-v1 both sit ~0.2 relative L2 from hertwig-v2 with per-tensor cosines 0.96–0.99 to it, so they are two short,
partly independent displacements from one parent; averaging such siblings usually lands in the same basin.

| # | hypothesis | edit class | predicted direction and magnitude of the CER change | if it fails |
|---|---|---|---|---|
| H1 | Global interpolation towards schlussredaktion-v1 at α = 0.5 (all 34 tensors; output rows paired by character, blank↔blank; the subject-only row `ƒ` untouched) lowers CER on the 297 rule lines. | interpolate (donor) | down, −0.010 to −0.020 absolute on the rule lines (−0.005 to −0.015 on all lines) | the two fine-tunes are not linearly connected enough for a half-way point to read both hands; sibling averaging is not a free gain for this family (R-series α-controls stand) |
| H1b | The α-response is monotone near the best point: if 0.5 is kept, α = 0.75 (then 0.9) improves further; if 0.5 is refused, α = 0.25 still helps a little. | interpolate (donor) | 0.75: −0.002 to −0.005 more; 0.9: −0.001 to −0.003 more; 0.25 after a refused 0.5: −0.003 to −0.008 | a refused 0.75 means the optimum is near 0.5 (stay); a refused 0.25 after a refused 0.5 means no α helps on the unseen lines: the interpolation line is closed for this pair |
| H2 | Lowering the blank bias (`O_18.lin.bias[0]` −0.4) cuts the deletions faster than it adds insertions. | add (output bias, blank row) | down, −0.003 to −0.007 absolute on all lines (≈ 50–120 edits of 2408) | the deletions are not marginal blank-vs-character decisions but come from collapsed or misaligned lines; the blank is left alone |
| H2b | A second step of the same size (cumulative −0.8) still helps but less; if H2 instead *raised* CER, the opposite sign (+0.3) is tried once. | add (output bias, blank row) | second step: −0.001 to −0.003; opposite sign: −0.001 to −0.003 | the blank bias is at or past its optimum: the H2 state (or the unedited one) stays |
| H3 | Raising the biases of the two most-deleted symbols, ` ` and `/`, by +0.25 each converts net deletions (96 and 41) into correct emissions. | add (output bias, 2 rows) | down, −0.001 to −0.003 absolute; may sit below the keep rule's resolution | those errors are placement/segmentation errors of the line, not thresholds; per-character bias nudges are not worth a pass on this model |
| H4 | Scaling `O_18.lin.weight` by 1.15 (bias unchanged) sharpens feature-driven character logits against the bias-driven blank (blank row norm 2.78 vs 13.7 for the others), reducing deletions where features are informative. | scale (output weight) | down, −0.002 to −0.005 absolute | the output layer is already calibrated for greedy CTC; no temperature edit is useful here |

Order and budget: pass 1 baseline; passes 2–4 H1/H1b (2–3 passes); passes 5–6 H2/H2b; pass 7 H3; pass 8 H4; up to two passes unused.
Overall expectation: H1 is the main bet (≈ 60 % that α = 0.5 is kept); H2 ≈ 50 %; H3 and H4 ≈ 25 % each. If nothing is kept the frozen model is the
unedited subject and the conclusion is that this model's errors are not reachable by output-layer calibration or sibling averaging.

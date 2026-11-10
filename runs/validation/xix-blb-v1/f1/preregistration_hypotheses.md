Model: claude-fable-5-1 (Fable 5.1), programmer attempt f1 for kraken-german-xix-blb-v1

## Settings
- Subject `kraken-german-xix-blb-v1`; donor `kraken-german-xix-v1` (parent, CER 0.3266 on the practice lines vs 0.1383 for the subject; `seen_by` empty for every line, so the donor-rule mask covers all 300 lines).
- Read: the three pack files, `PROGRAMMER_BRIEF.md`, `wsp/ops.py`, `wsp/session.py`, `wsp/run.py`, `wsp/kraken_subject.py`, `wsp/align.py`. Nothing else; no other attempt's folder; the program was not run and no model was loaded (only `py_compile`).
- Budget 10 passes (pass 1 = C0 baseline), 9 measured edits, each decided by `s.keep` (95 % paired interval below 0). Written in advance; branches depend only on `keep` outcomes.
- Agent launched with the default model and effort of the Agent tool call, no override that I know of.

## Reading of the pack that drives the hypotheses
C0: CER 0.1383, length ratio 0.970, 2 collapsed lines. Seven of the ten top confusions are deletions (space 78, e 44, i 29, n 26, r 23, t 21, s 21); the one insertion is space (30). The reference has roughly 1701/0.1383 ≈ 12 300 characters, so the hypothesis is about 370 characters short net: the model under-emits. The blank row is the weakest output row (norm 4.64, bias +1.23) and the deleted letters are exactly the weakest letter rows (e 7.57, r 7.80, s 8.05, t 8.08, i 8.17, n 8.33, a 8.33); the space bias (−7.85) is the most negative of all rows. Parent is 2.4× worse on these lines, so the donor is weak; interpolation toward it is a regularisation test, not a transfer of knowledge.

## Hypotheses (one line each; direction and size are of CER on the practice set, baseline 0.1383)
- H1 (pass 2) `add` −0.5 to `O_18.lin.bias[0]` (blank bias shift). Predicted −0.005 (−0.002 … −0.010): deletions fall more than insertions rise. If refused: the deletions are not marginal blank wins; blank calibration is not the lever, the loss is in the features.
- H1b (pass 3) dose step on the blank bias, adaptive: −0.5 more if H1 was kept (predicted a further −0.002, −0.004 … +0.002), else −0.25 instead (predicted −0.002; if that is refused too, the blank-bias lever is dead at every size). Doubled letters (nn, ss, tt, ll) are the expected cost of going too far.
- H2 (pass 4) `add` +0.6 to the space row of `O_18.lin.bias`. Predicted −0.002 (−0.001 … −0.005): 78 deleted vs 30 inserted spaces. If refused: the missing spaces are narrow gaps that produce no space frame at all, and a bias cannot create one.
- H2b (pass 5) adaptive dose step on the space bias: +0.6 more if H2 kept (predicted −0.001), else +0.3 (predicted −0.001). If both refused, word-boundary deletions are feature-level.
- H3 (pass 6) `add` +8 % of themselves to the weight rows of e, r, s, t, i, n, a in `O_18.lin.weight` (row scaling, evidence-conditioned, unlike a bias). Predicted −0.002 (−0.001 … −0.003). If refused: the small norm of these rows is not what loses them; the deletions are decided upstream of the output layer.
- H4 (pass 7) `interpolate` every conv and LSTM tensor toward the parent at α = 0.1, judged with `rule_mask=s.unseen_by("kraken-german-xix-v1")` (WiSE-FT-style regularisation with a weak donor). Predicted +0.002 (0.000 … +0.005), i.e. I expect a refusal; kept only if the fine-tuning overshot into a basin where the pre-fine-tuning direction is benign.
- H4b (pass 8) if H4 kept: a second α = 0.1 step (predicted −0.001); if refused: conv stack only (`C_0 … C_9`) at α = 0.2, the block that moved most from the parent (C_3 bias cosine 0.68), predicted ±0.003 (refusal expected). If both refused: the parent carries nothing this subject can use at any granularity; the donor is simply worse.
- H5 (pass 9) `add` +0.25 to the r row of `O_18.lin.bias`: r is deleted 23×, read as n 23× and as e 19×, and it is the second-weakest letter row. Predicted −0.001 (−0.003 … +0.001). If refused: r's confusions are feature-level and the bias trades n→r and e→r for the gains.
- H6 (pass 10) if any blank step was kept: one more blank step in the kept direction (−0.5 after two kept steps, −0.25 after one kept −0.5, −0.125 after a kept −0.25), predicted −0.001; if no blank step was kept: `scale` the whole `O_18.lin.weight` by 1.05 (evidence against biases, a different mechanism than the bias shifts), predicted −0.001 (−0.003 … +0.002).

## Overall prediction
Final practice CER 0.128 … 0.134 (−0.004 … −0.010 against 0.1383), most of it from H1/H1b; 3 to 5 of 9 edits kept; the donor edits refused. If the final CER is ≥ 0.136, the under-emission of this model is not a calibration problem of the output layer and nothing in weight space without a gradient reaches it with 10 passes.

# TrOCR stage 3: non-LLM baselines on the planted models (practice level)

Run 18.11.2026 on card 0, 10 passes, practice set (300 lines), page-clustered keep rule, minimum effect 0.5 points. Arm c: hand-written rule list with the parent donor. Arm d: scale coordinate search, no donor. CER of the kept state (taken from the session logs; the 'done' lines in the chain log print the last pass, not the kept state). Parent-level CER of the unplanted subject: 0.2145. Repaired % = (start - final) / (start - 0.2145). Held-out scoring of the finals is not done yet; classes below are practice-level only.

| defect | start CER | c: final (kept/passes) | repaired% c | d: final (kept/passes) | repaired% d |
| p01-zero-rows-k6 | 0.2508 | 0.2163 (1/2) | 95% | 0.2508 (0/10) | 0% |
| p02-swap-rows | 0.2343 | 0.2161 (1/2) | 92% | 0.2343 (0/10) | 0% |
| p03-cross-o5-zero | 0.2320 | 0.2145 (1/3) | 100% | 0.2320 (0/10) | 0% |
| p04-zero-heads-8 | 0.2468 | 0.2177 (1/2) | 90% | 0.2243 (1/10) | 70% |
| p05-zero-enc6-attn | 0.3316 | 0.2207 (1/3) | 95% | 0.3065 (1/10) | 21% |
| p06-enc3-ffn-x0.5 | 0.2346 | 0.2346 (0/2) | 0% | 0.2346 (0/10) | 0% |
| p07-noise-dec8-60 | 0.2321 | 0.2321 (0/6) | 0% | 0.2321 (0/10) | 0% |
| p08-shift-embnorm | 0.2984 | 0.2147 (1/2) | 100% | 0.2331 (2/10) | 78% |
| p09-ln11-zero | 10.4328 | 0.2156 (1/2) | 100% | 10.4328 (0/10) | 0% |
| p10-noise-all-20 | 0.2426 | 0.2426 (0/1) | 0% | 0.2426 (0/10) | 0% |
| p11-noise-embed-50 | 0.2335 | 0.2335 (0/1) | 0% | 0.2335 (0/10) | 0% |
| p12-enc-ffn-all-x0.7 | 0.2736 | 0.2736 (0/4) | 0% | 0.2585 (1/10) | 26% |
| p13-cross-o-all-x0.7 | 0.2921 | 0.2921 (0/1) | 0% | 0.2197 (2/10) | 93% |
| p14-null | 0.2145 | 0.2145 (0/1) | - | 0.2145 (0/10) | - |

Readings (practice level, not held-out):
- Arm c repairs 7 of the 8 donor-reachable defects at 90 % or more (p01, p02, p03, p04, p05, p08, p09), with one kept edit each and no harmful keep. It repairs none of the diffuse ones (p06, p07, p10, p11, p12, p13), as its rules are built for localised damage. On p14 (null) it keeps nothing.
- Arm d (no donor) keeps nothing on the null, and repairs p13 (93 %), p08 (78 %), p04 (70 %); partly p12 (26 %) and p05 (21 %). It cannot touch token rows, a zeroed layer-norm or the cross-attention zeroing (p01, p02, p03, p09).
- No arm kept a harmful edit on any model (all kept edits lowered practice CER). Practice-level selection can still overfit; held-out is needed.
- Arm d is limited by its fixed factors (0.8, 1.25) and by the 10-pass budget: on p12 the damaged group (encoder FFN) was reached only in the last pass, with the wrong direction.
- p10 (diffuse noise, +2.8 points) and p11, p07 are not repaired by either arm; these are the cases where a programmer must do better than both baselines.

Run folders under `runs/*-trocr-planted-*` are committed without `diff.json` (up to 1.9 GB each), `readings/`, and the model copies (`final.trocr`, `current.trocr`); these stay local.

## Held-out classes (added 18.11.2026, 11:43 CET; blb-eval-hand, 2,272 lines, each model read once)

Unplanted subject: held-out CER 19.74 %. Every one of the 12 defects scored counts (damage at least 1.5 points); p07 is the smallest (+1.54). Definitions and the script: `scripts/trocr_stage3_classify.py`; data: `docs/trocr/heldout_baselines.json`, `docs/trocr/stage3_classes.json`. Intervals are page-clustered paired bootstraps of final minus corrupted CER (points).

| defect | damage (pts) | counts | arm | passes/kept | final CER | recovery | interval of final − corrupted (pages) | class |
|---|---:|---|---|---|---:|---:|---|---|
| p01-zero-rows-k6 | 3.45 | yes | c | 2/1 | 19.76 | 99 % | [-3.80, -3.00] | repaired |
| p01-zero-rows-k6 | 3.45 | yes | d | 10/0 | 23.19 | 0 % | [+0.00, +0.00] | not repaired |
| p02-swap-rows | 1.71 | yes | c | 2/1 | 19.74 | 100 % | [-1.91, -1.49] | repaired |
| p02-swap-rows | 1.71 | yes | d | 10/0 | 21.45 | 0 % | [+0.00, +0.00] | not repaired |
| p03-cross-o5-zero | 2.07 | yes | c | 3/1 | 19.76 | 99 % | [-2.59, -1.52] | repaired |
| p03-cross-o5-zero | 2.07 | yes | d | 10/0 | 21.81 | 0 % | [+0.00, +0.00] | not repaired |
| p04-zero-heads-8 | 2.49 | yes | c | 2/1 | 19.71 | 101 % | [-3.17, -1.93] | repaired |
| p04-zero-heads-8 | 2.49 | yes | d | 10/1 | 20.96 | 51 % | [-1.81, -0.78] | partly |
| p05-zero-enc6-attn | 13.42 | yes | c | 3/1 | 20.19 | 97 % | [-14.17, -11.80] | repaired |
| p05-zero-enc6-attn | 13.42 | yes | d | 10/1 | 29.95 | 24 % | [-3.81, -2.62] | partly |
| p06-enc3-ffn-x0.5 | 2.07 | yes | c | 2/0 | 21.80 | 0 % | [+0.00, +0.00] | not repaired |
| p06-enc3-ffn-x0.5 | 2.07 | yes | d | 10/0 | 21.80 | 0 % | [+0.00, +0.00] | not repaired |
| p07-noise-dec8-60 | 1.54 | yes | c | 6/0 | 21.27 | 0 % | [+0.00, +0.00] | not repaired |
| p07-noise-dec8-60 | 1.54 | yes | d | 10/0 | 21.27 | 0 % | [+0.00, +0.00] | not repaired |
| p08-shift-embnorm | 6.87 | yes | c | 2/1 | 19.74 | 100 % | [-8.04, -5.80] | repaired |
| p08-shift-embnorm | 6.87 | yes | d | 10/2 | 21.82 | 70 % | [-5.96, -3.69] | partly |
| p09-ln11-zero | 863.38 | yes | c | 2/1 | 19.74 | 100 % | [-972.58, -775.61] | repaired |
| p09-ln11-zero | 863.38 | yes | d | 10/0 | 883.11 | 0 % | [+0.00, +0.00] | not repaired |
| p10-noise-all-20 | 2.87 | yes | c | 1/0 | 22.61 | 0 % | [+0.00, +0.00] | not repaired |
| p10-noise-all-20 | 2.87 | yes | d | 10/0 | 22.61 | 0 % | [+0.00, +0.00] | not repaired |
| p12-enc-ffn-all-x0.7 | 6.33 | yes | c | 4/0 | 26.07 | 0 % | [+0.00, +0.00] | not repaired |
| p12-enc-ffn-all-x0.7 | 6.33 | yes | d | 10/1 | 25.06 | 16 % | [-1.50, -0.54] | not repaired |
| p13-cross-o-all-x0.7 | 9.06 | yes | c | 1/0 | 28.80 | 0 % | [+0.00, +0.00] | not repaired |
| p13-cross-o-all-x0.7 | 9.06 | yes | d | 10/2 | 19.96 | 98 % | [-10.54, -7.01] | repaired |

Readings: held-out agrees with practice level. Arm c (rule list, donor) repairs 7 defects (p01, p02, p03, p04, p05, p08, p09: 97–101 % recovered, interval below 0); arm d (no donor) repairs p13 only (98 %) and partly p04 (51 %), p05 (24 %), p08 (70 %); p12 stays not repaired (16 %). **Neither baseline repairs p06, p07, p10** (diffuse or without signature; damage 2.07, 1.54, 2.87 points), and neither makes a harmful keep. Where a programmer can add something is therefore p06, p07, p10, p12 (and p12/p13-type scale defects without the donor, where arm d reaches only p13).

## Pilot: arm (a) with donor, Sonnet 5.5, 8 core models (20:30 CET, 18.11.2026)
Held-out classes (full table: `stage3_classes.md`): repaired p01 (99 %), p03 (97 %), p05 (101 %), p06, p07, p08, p12 (100 %); not repaired p10 (0 %, no edit kept, held-out 22.61 %). No harmful keep. Arm c repaired 3 of these 8 and arm d none. Only arm (a) with donor ran; arms (a) without donor, (b) and (a2) are not run.
Note: the chain stalled 45 min after p12 (its wait matched its own launcher); no effect on any session.

## Arm (a) without donor, Sonnet 5.5, 8 core models (22:07 CET, 18.11.2026)
Held-out classes (`stage3_classes.md`, rows `an`): repaired p06 (100 %, 7 passes, 1 kept); not repaired p03, p05, p07, p08, p10, p12 (0 %, no edit kept, model unchanged) and p01 (crashed). No harmful keep.
**p01 crash (dated note 18.11.2026, 21:54 CET):** the programmer's `try_edit` called `s.revert` after `s.keep` had already undone the rejected edit (`RuntimeError: no edit to decide on`, program.py line 36) after 3 passes. No frozen model, no `summary.json`, no held-out score. Per the planning session's decision the program was not patched and not rerun; the session counts as not repaired (final = corrupted model, nothing kept; marked `an*` in the classes table). The control flow of the p03, p05 and p08 programs was read and cannot raise the same error.
**Gate 3 (prereg section 4):** with donor, 7 of 8 counted defects repaired by arm (a); without donor, 1 (p06); harmful keeps overall: 0. Gate 3 is met on these counts. Arms (b) and (a2) were not run. Limit: the no-donor rule is enforced by the brief only.

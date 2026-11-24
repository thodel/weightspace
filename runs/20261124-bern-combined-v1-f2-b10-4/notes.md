# Notes — 20261124-bern-combined-v1-f2-b10-4


## 2026-11-24T16:59Z — round 0 (before any pass; heading first written as 17:05Z, corrected in round 1 to the clock time)

- **Programmer:** Claude Fable 5.2, model id `claude-fable-5-2` (as stated in my system prompt).
  Headless Claude Code session, default reasoning effort, no subagents, no web.
- **Read:** `PROGRAMMER_BRIEF.md`; `preregistration.md`, `notes.md`, `session.json` of this
  folder (a directory listing also showed `env.txt`, which I did not open); the whole pack
  (`c0_c1.md`, `readings_budget0.md`, `describe_bern-combined-v1.md`, both `diff_*.md`,
  `per_line_edits.json`; the other `describe_*` and the `.json` twins not opened);
  `data/practice/bern-train-300.jsonl` and the head of its `.meta.json`; `wsp/ops.py`,
  `wsp/session.py`, `wsp/kraken_subject.py`, `wsp/subject.py`, `wsp/align.py`, `wsp/metrics.py`,
  and of `wsp/run.py` the function list and `RunSession`/`open_session`. No line image opened.
- **Ran (CPU, read-only, nothing written, no forward):** two inline reading scripts.
  1. Per-hand CER and overlap from `per_line_edits.json`: subject 0.2050 / 0.1184
     (Hertwig / Schlussredaktion), parent 0.1137 / 0.2992, sibling 0.2355 / 0.1129. Per-line
     oracle subject-or-parent 0.1116, all three 0.1014. The 30 worst lines hold 51 % of the
     subject's edits and all three models fail on most of them; 40 lines have line CER > 0.5.
     Without those: subject Hertwig 0.133 (parent 0.048), Schlussredaktion 0.065 (sibling 0.058).
  2. Deltas from the parent, subject against sibling, per tensor: similar norms; cosine 0.85 /
     0.96 in `O_18` weight / bias, 0.54–0.80 in `L_16`, 0.42–0.70 in `L_14`, 0.24–0.41 in
     `L_12`, 0.36–0.48 in the conv weights. Blank bias +0.998 / +0.879 / +0.936 (subject /
     parent / sibling). Codec rows: subject and parent share indices for shared characters; the
     sibling does not (it lacks `ƒ`), so its rows must be paired by character. The practice
     texts contain U+00A0 22 times, which is not in the codec (scoring flattens whitespace, so
     this costs nothing).
- **Wrote:** section 5 of `preregistration.md`; `program_01.py` (syntax checked with
  `compile()`, not executed): pass 1 C0 (stops if CER is not 0.1456), pass 2 H1 (all 28 trained
  tensors 0.25 toward the parent), pass 3 H2 (blank bias −0.25), pass 4 H3 (0.30 toward the
  sibling). Each edit is measured, reverted if collapsed rises, otherwise left to `s.keep()`.
- **Earlier sessions:** `preregistration.md` (sections 1 and 3, by the operator) says this run
  replicates "kraken session 2" and that earlier sessions and frozen models on this subject
  exist, and a docstring in `wsp/run.py` names a session id `20261106-bern-combined-v1-b10-1`
  in connection with a harness bug (keep() returning None). Neither names an edit, a kept
  decision or a result of those sessions. `readings_budget0.md` lists budget-0 hypotheses
  (blank bias back to the parent's value; output layer from the parent; per-layer α), which
  are phase-0 proposals, not session results.

## 2026-11-24T17:04Z — round 1 (after passes 1–4, before pass 5)

- **Programmer:** `claude-fable-5-2`, same session, same settings.
- **Results of `program_01.py`** (stdout from the operator; decisions checked in `log.jsonl`):

  | pass | edit | CER | Hertwig | Schlussred. | collapsed | Δ vs kept (95 % interval) | decision |
  |---|---|---:|---:|---:|---:|---|---|
  | 1 | C0 | 0.1456 | 0.2050 | 0.1184 | 29 | | baseline, as predicted |
  | 2 | H1: 0.25 toward the parent, all tensors | 0.1381 | 0.1671 | 0.1247 | 22 | −0.0076 (−0.0147, −0.0015) | kept |
  | 3 | H2: blank bias −0.25 | 0.1364 | 0.1648 | 0.1234 | 21 | −0.0017 (−0.0032, −0.0002) | kept |
  | 4 | H3: 0.30 toward the sibling, all tensors | 0.1349 | 0.1686 | 0.1195 | 20 | −0.0015 (−0.0045, +0.0017) | refused by the rule, undone |

- **Against the predictions.** H1: total predicted −0.006, got −0.0076; both hands inside their
  ranges. H2: predicted −0.002, got −0.0017; I had said the interval would more likely contain
  0, and it did not (upper end −0.0002, so only just). H3: predicted −0.002 with
  Schlussredaktion down and Hertwig up; got −0.0015 with exactly that trade (−0.004 / +0.004),
  not separable from 0.
- **Read:** `log.jsonl` (decisions) and `readings/pass_001…004.jsonl`, through one inline
  script (CPU, no forward, nothing written) that counts per-line changes by hand:
  - H1: Hertwig 57 lines better, 7 worse, −129 edits (22 of them in one line);
    Schlussredaktion 25 better, 72 worse, +47 edits. The trade is broad on both hands.
  - H2: Hertwig −8 edits, Schlussredaktion −10 edits, about 35 lines touched in all.
  - H3: Hertwig +13 edits (17 better, 28 worse), Schlussredaktion −29 (47 better, 26 worse).
- **Kept state:** H1 + H2, CER 0.1364, collapsed 21, blank bias about +0.718. Six passes left.
- **Wrote:** the round-1 paragraph of section 5 and `program_02.py` (syntax checked with
  `compile()`, not executed), passes 5–8: F1 (all tensors a further 0.20 toward the parent),
  F2a (lower blocks `C_* L_12` a further 0.25), F2b (upper blocks `L_14 L_16 O_18` a further
  0.25), F4 (blank bias a further −0.25). In every parent step the blank bias is held at its
  kept value. No freeze yet; passes 9–10 are decided after these.
- **Caveat I keep in view:** both kept gains are small on practice, and H1's is in the
  direction of a donor that was trained on the Hertwig practice lines. H2's interval nearly
  touches 0 at line level; clustered by page it may contain 0.
- **Earlier sessions:** nothing read in this round named an edit, a kept decision or a result
  of an earlier session on this model.

## 2026-11-24T17:08Z — round 2 (after passes 5–8; freeze without a further pass)

- **Programmer:** `claude-fable-5-2`, same session, same settings.
- **Results of `program_02.py`** (stdout from the operator; decisions and digests checked in
  `log.jsonl`). All four against the kept state of pass 3 (CER 0.1364, Hertwig 0.1648,
  Schlussredaktion 0.1234, collapsed 21):

  | pass | edit | CER | Hertwig | Schlussred. | collapsed | Δ vs kept (95 % interval) | decision |
  |---|---|---:|---:|---:|---:|---|---|
  | 5 | F1: all tensors a further 0.20 toward the parent | 0.1418 | 0.1507 | 0.1378 | 23 | +0.0054 (+0.0018, +0.0093) | reverted (collapsed rose; CER rose too) |
  | 6 | F2a: `C_* L_12` a further 0.25 | 0.1396 | 0.1624 | 0.1292 | 22 | +0.0032 (+0.0008, +0.0058) | reverted (collapsed rose; CER rose too) |
  | 7 | F2b: `L_14 L_16 O_18` a further 0.25 | 0.1385 | 0.1510 | 0.1328 | 21 | +0.0021 (−0.0011, +0.0055) | refused by the rule |
  | 8 | F4: blank bias a further −0.25 | 0.1353 | 0.1624 | 0.1228 | 21 | −0.0011 (−0.0029, +0.0005) | refused by the rule |

  After every decision the digest is back at `9667acb5dd9d`, the state kept after pass 3.
- **Against the predictions.**
  - F1: predicted −0.001 (range −0.007 to +0.005), got +0.0054, just outside the range. I had
    the Hertwig gain right (−0.014) and the Schlussredaktion cost too low (+0.014 against a
    predicted +0.006 to +0.012). The cost is steeply convex: +0.006 for the first 0.25, +0.014
    for the next 0.15 of the way.
  - F2a / F2b: predicted that one of them might be kept, with a lean to F2a. Neither was, and
    the lean was wrong. Per-line counts against pass 3 (inline script on `readings/`):
    F2a Hertwig −8 edits (24 lines better, 13 worse), Schlussredaktion +43; F2b Hertwig −47
    (36 better, 6 worse), Schlussredaktion +70; F1 Hertwig −48, Schlussredaktion +107. So the
    upper blocks carry all of the Hertwig gain of the global step and about two thirds of its
    Schlussredaktion cost; the lower blocks carry cost and almost no gain. `D→d` enters the
    top-10 confusions in passes 5–7 (28–35 times): toward the parent the model writes
    Hertwig's lower-case convention on Schlussredaktion lines.
  - F4: predicted −0.001 with the interval containing 0; got exactly that.
- **Decision: no pass 9 or 10.** My round-1 plan allowed a refinement only of a class kept in
  passes 5–8, and nothing was kept. The candidates I can think of now (upper blocks with a
  smaller step, or with a sibling step added to pay back Schlussredaktion; blank −0.12) are
  all chosen after eight looks at the same 300 lines and would move CER by about the width of
  the interval. I leave the two passes unspent.
- **Wrote:** the round-2 paragraph of section 5, and `program_03.py` (syntax checked with
  `compile()`, not executed): it opens the session, prints the restored blank bias and digest,
  logs a note and calls `s.freeze()`. No pass.
- **What is frozen:** H1 (all 28 trained tensors 0.25 toward `hertwig-v2`, output rows by
  character, rows of `½ ƒ ̊` untouched) and H2 (blank bias −0.25, so +0.998 → +0.718 after
  both). Practice 0.1364 against 0.1456 (−0.0092); Hertwig −0.040, Schlussredaktion +0.005;
  collapsed 29 → 21.
- **What I do not claim.** The gain is on training-side lines and in the direction of a donor
  trained on a third of them. My held-out prediction is −0.003 (range −0.010 to +0.004), with
  the page-clustered interval more likely containing 0 than not. H2's practice interval
  (upper end −0.0002) is too close to 0 to stand on its own. H3 and F4 had negative point
  estimates and were refused; I make nothing of them.
- **Earlier sessions:** nothing read in this round named an edit, a kept decision or a result
  of an earlier session on this model.

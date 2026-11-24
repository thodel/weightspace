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

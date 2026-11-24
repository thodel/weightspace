# Notes — 20261124-bern-combined-v1-f2-b10-4


## 2026-11-24T17:05Z — round 0 (before any pass)

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

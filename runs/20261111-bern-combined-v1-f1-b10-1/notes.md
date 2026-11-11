# Notes — 20261111-bern-combined-v1-f1-b10-1

## 2026-11-11T08:55Z — round 0 (programmer: Claude Fable 5.1, `claude-fable-5-1`)

- **Settings:** Claude Code subagent, default reasoning effort; tools Read, Write, Edit, Bash. CPU-only
  reading scripts with `CUDA_VISIBLE_DEVICES=` and `.venv/bin/python` (`KrakenSubject(...).tensors()`,
  `.codec_map()`, numpy); no `forward`, `measure` or `open_session` was called; no `program_*.py` executed.
- **Read:** `PROGRAMMER_BRIEF.md`; `preregistration.md` (sections 1–4, 6–7 as the operator wrote them);
  `pack/readings_budget0.md`, `pack/c0_c1.md`, `pack/describe_*.md`, `pack/diff_*.md`,
  `pack/per_line_edits.json`, the key structure of `pack/diff_bern-hertwig-v2__bern-combined-v1.json`;
  `data/practice/bern-train-300.jsonl` and `.meta.json`; `wsp/ops.py`, `session.py`, `run.py`,
  `kraken_subject.py`, `subject.py`, `align.py`, `metrics.py`, `data.py`; `session.json`, `env.txt`.
  Nothing else: no other run, no docs/, protocol/, paper/, git history or held-out data. Nothing I read
  named an edit or a result of another programmer on this model.
- **CPU reads (no pass):** per-hand CER from `per_line_edits.json` reproduces c0_c1.md (subject H 0.2050,
  S 0.1184; parent H 0.1137 / S 0.2992; sibling H 0.2355 / S 0.1129). Parent beats the subject on 81/103
  H lines, loses on 174/197 S lines; sibling beats it on 58 S lines, loses on 38, ties 101; loses on 61/103
  H lines. 42 lines have edits ≥ half the reference for the subject, most equally bad for both donors.
  Output rows of every character in the practice text: cosine ≥ 0.97 across the three models; the
  "moved most" rare rows (`ü ö ä ſ · §`) do not occur in the practice references (umlauts are NFD
  combining marks `ͤ ͦ ̈`). Fine-tune directions τC = subject − parent, τS = sibling − parent: global cosine
  0.52; `O_18` 0.85–0.97, `L_16` 0.54–0.80, `L_14` 0.42–0.56, `L_12` weights 0.24–0.28, conv weights
  0.36–0.48; |τC|/|τS| = 1.16. Blank bias subject +0.998, parent +0.879, sibling +0.936. Codec: parent
  lacks `½ ƒ ̊`, sibling lacks `ƒ`.
- **Section 5 written** (H1 blank bias −0.20; H2 soup with the sibling α 0.5; follow-ups H1b, H2b/c, H3
  space+`/` bias, H4 toward the parent with the memorisation caveat, H5 L_12-only soup; stopping rule).
- **program_01.py written:** P1 C0 (stops before any edit if CER ≠ 0.1456 ± 0.0006), P2 H1, P3 H2, each
  measured then `keep()`/revert with the extra collapsed rule. Prints per-hand CER, del/ins/sub,
  confusions and the intervals. Does not freeze. Compiled with `py_compile`; `__pycache__` removed.
  Dry check (read-only, no session): the H2 arrays give the `ƒ` row unchanged, the blank row interpolated,
  the `e` row the exact mean; the blank bias is held if H1 was kept.

## 2026-11-11T09:12Z — round 1 (programmer: Claude Fable 5.1, `claude-fable-5-1`)

- **Read:** `stdout_program_01.log`, `log.jsonl`, `budget.json`, `readings/pass_001..003.jsonl` (my own
  passes), the practice list and its line images (sizes of the collapsed lines). Same restrictions as
  round 0; nothing else read.
- **Results of program_01:** C0 0.1456 reproduced (collapsed 29). H1 (blank −0.20) **kept**: 0.1444,
  Δ −0.0012 [−0.0023, −0.0002]; 16 lines better, 7 worse; del 1041→1002, ins 153→161, sub 383→401.
  H2 (sibling soup α 0.5) **refused**: 0.1429, Δ −0.0016 [−0.0076, +0.0033]; 68 better / 77 worse / 155
  same; the net −17 edits are one Hertwig line (`p0027/tr_2_tl_5`, −20, recovered from a collapse);
  collapsed 29→22 but the swings are convention swaps in both directions (`d↔D`, `u↔v`, `,↔/`).
  → H2b, H2c and H5 dropped: the children's difference is not a direction that helps.
- **Kept-state reading (pass 2):** 580 of 1564 edits are in the 29 collapsed lines, mostly small crops
  (80–320 px wide, median line 1472) that give an empty string for all three models; out of reach.
  Per hand: H subs `,→/` 12, `u→v` 11, `̈→ͤ` 9, `d→D` 8 (Schlussredaktion conventions applied to the
  Hertwig hand); H deletes `"` 16 of 17 (the `"` row: norm 3.03→4.27, bias −0.01→−0.15, cos 0.894 to the
  parent's — the one practice-relevant row with the label-noise signature). Space del/ins 159/55 overall
  (H 67/18, S 92/37); `/` 67/39 with 12 `,→/` subs (convention; left alone).
- **Section 5, Round 1 addendum written:** H1b (blank −0.20 more), H3 (space bias +0.30), H6 (`"` row
  from the parent, aligned by character; the parent's one-glyph classifier, not its recurrent stack),
  H4 (interpolate α 0.25 toward the parent, rows aligned, new rows and this session's edited bias rows
  held; memorisation caveat stated). Round 2: at most one follow-up (α 0.125 if H4 refused with H down
  and S up), then freeze.
- **program_02.py written:** passes 4–7 = H1b, H3, H6, H4 in that order, each measured then
  keep/revert with the collapsed rule (last kept collapsed read from `log.jsonl`: 29). Prints per-hand
  CER, del/ins/sub, space and `"` del/ins, confusions, intervals. Does not freeze. `py_compile` ok,
  `__pycache__` removed. Dry check (read-only, no session): `set_rows` replaces exactly the `"` row and
  bias; the H4 arrays hold blank/space bias and the `ƒ` row and move the shared rows by a quarter.

## 2026-11-11T09:25Z — round 2, last (programmer: Claude Fable 5.1, `claude-fable-5-1`)

- **Read:** `stdout_program_02.log`, `log.jsonl`, `budget.json`, `readings/pass_004..007.jsonl`. Same
  restrictions; nothing else.
- **Results of program_02:** H1b (blank −0.20 more, total −0.40) **kept**: 0.1410, Δ −0.0034 [−0.0055,
  −0.0016], 30 lines better / 9 worse, del 1002→941, ins 161→173, collapsed 29→26. H3 (space +0.30)
  refused, CER up (space deletions 147→158, not down: the bias does not reach the word-gap readings).
  H6 (`"` row from the parent) refused with output identical to the kept state — the glyph is lost in
  the features; nothing more on it. H4 (α 0.25 toward the parent) refused, Δ −0.0053 [−0.0116, +0.0002]:
  H 56 better / 7 worse (−120 edits, the parent's own training lines), S 24 better / 70 worse (+63);
  −30 of the net −57 on non-collapsed lines, −27 on collapsed ones; collapsed 26→21.
- **Section 5, Round 2 addendum written:** H1c (blank −0.20, total −0.60), H1d (−0.20 more, only if H1c
  kept with Δ ≤ −0.0025), H4b (α 0.125 toward the parent, the pre-registered follow-up; confound
  stated), then freeze.
- **program_03.py written:** P8 H1c; P9 H1d conditional; P9/10 H4b; `s.freeze()` at the end whatever
  the decisions (an unspent pass stays unspent if H1d is skipped). Last kept collapsed read from
  `log.jsonl` = 26. `py_compile` ok, `__pycache__` removed.
- **Where the session stands:** the only kept edits are blank-bias shifts on `O_18.lin.bias[0]`
  (−0.40 so far), practice CER 0.1456 → 0.1410, consistent per line, both hands (H 0.2050→0.1994,
  S 0.1184→0.1142). Donor edits did not pass: the sibling soup had no direction, the parent direction
  gains only on the parent's own training lines and costs Schlussredaktion. If H4b is kept, that
  practice gain carries the memorisation confound and the held-out decides.

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

# Notes — 20261111-bern-combined-v1-f1-b10-1

## 2026-11-11T08:40Z — round 0 (programmer: Claude Fable 5.1, `claude-fable-5-1`)

**Settings.** Claude Code subagent (`model: fable`), default reasoning effort; tools Read, Write,
Edit, Bash. All reading on the CPU (`CUDA_VISIBLE_DEVICES=`, `.venv/bin/python`); no forward pass,
no `measure`, no `open_session`; nothing written outside this folder; `program_01.py` compiled
with `py_compile` (pycache removed).

**What I read.** `PROGRAMMER_BRIEF.md`; the whole pack (`readings_budget0.md`, `c0_c1.md`, the
three `describe_*.md`, both `diff_*.md`, `per_line_edits.json`); `data/practice/bern-train-300.jsonl`
and `.meta.json` (keys, reference texts, line counts; no images opened); the harness sources
`wsp/ops.py`, `session.py`, `run.py`, `kraken_subject.py`, `subject.py`, `align.py`, `metrics.py`,
`data.py`; and the files already in this run folder (`preregistration.md`, `session.json`,
`env.txt`). Two CPU scripts: (1) the three models' codecs, output-layer biases and row norms
aligned by character against the practice character frequencies; (2) per-line analysis of
`per_line_edits.json` (dead lines, subject-vs-donor per hand, paired bootstrap widths with
`wsp.metrics.bootstrap_delta`). Nothing from other runs, docs/, protocol/, paper/, git history or
held-out.

**Disclosure.** Section 7 of `preregistration.md` (operator-written, in this folder) says that
session 2 kept an output-layer transplant from `hertwig-v2` which lost on held-out. I read it
because I had to open the file to fill section 5. My plan has no such transplant; my own reading
(the parent memorised the 103 Hertwig practice lines, 0.080 there vs 0.292 on Schlussredaktion)
argues against it independently, but the knowledge was in my hands and is recorded here.

**Findings before any pass.** Frequent-character biases and norms differ by at most ±0.06 across
subject/parent/sibling; the rows that moved most parent→subject cover 24 of 10,829 reference
characters (no leverage for a rare-row revert). Blank bias +0.998 vs +0.879 (parent) and +0.936
(sibling). Deletions 1,033 vs insertions 153 of 1,577 edits; 12 lines (216 ref chars, 207 edits)
are wrong for every model incl. the grandparent (unreadable crops) — CER without them 0.129.
Sibling vs subject paired Δ +0.0058, CI [−0.002, +0.013] (page-clustered [−0.005, +0.016]);
sibling better on 58/197 Schlussredaktion lines, worse on 61/103 Hertwig lines; per-line oracle
min(subject, sibling) 0.129. Parent vs subject Δ +0.095. A uniform 5 % reduction of edits would
give an interval of about ±0.0015 around −0.0038, so a consistent ~0.4-point gain is keepable.

**Plan (section 5).** P1 C0 (must give 0.1456 / collapsed 29, else stop). H1 blank bias −0.4
(predicted −0.3 to −0.8 pt); H1b a second −0.4 if kept (or H1b′ −0.8 at once if H1 refused with
Δ<0); H2 whole-network interpolation toward the sibling α=0.5, output rows by character, blank
bias held (predicted −0.3 to −1.0 pt, Schlussredaktion down, Hertwig uncertain); H2b α=0.25 if
H2 refused with Δ<0. `program_01.py` runs these (≤5 passes) and does not freeze. Follow-ups for
`program_02.py`, to be pre-registered individually: H3 space-row bias +0.25; H4 whole-network
α=0.2 toward the parent with a Schlussredaktion guard (reverted if that hand rises > 0.3 pt);
H5 block localisation of an H2 gain; H1c a third blank step if H1b's gain ≥ half of H1's. Extra
revert rule in the program: collapsed must not rise against the last kept state.

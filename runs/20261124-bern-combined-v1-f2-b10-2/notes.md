# Notes — 20261124-bern-combined-v1-f2-b10-2


## Round 0 — 2026-11-24 17:00 UTC (programmer; time corrected in round 1, first written as 17:05)

- **Model id:** `claude-fable-5-2` (Claude Fable 5.2), as stated in my system prompt. Headless
  Claude Code session, default reasoning settings as given by the operator; tools used: Read,
  Write, Edit, Bash. No sub-agents, no web.
- **Read:** `PROGRAMMER_BRIEF.md`; `preregistration.md`, `notes.md`, `session.json` of this
  folder (and its directory listing; `env.txt` was listed, not opened); the pack:
  `readings_budget0.md`, `c0_c1.md`, `per_line_edits.json`, `describe_bern-combined-v1.md`,
  both `diff_*.md` (the `describe_*`/`diff_*` JSON files and the two donor `describe_*.md` were
  not opened); `data/practice/bern-train-300.jsonl` and the head of `.meta.json`; harness source
  `wsp/ops.py`, `align.py`, `session.py`, `kraken_subject.py`, `subject.py`, `metrics.py`, and
  of `run.py` the outline and lines 196–272 (`RunSession`, `open_session`); of `data.py` only
  the `load` function and the first lines of `SOURCE_NOTES` after it (set descriptions). No line image was looked at.
- **Ran (CPU, read-only, scripts in the session scratchpad, no forward pass):**
  1. per-line analysis of `pack/per_line_edits.json` by hand;
  2. the three models' tensors: task vectors from the parent, their cosines, codec alignment.
  Also `py_compile` on `program_01.py`. No `program_*.py` was executed.
- **Findings** (details in section 5 of the pre-registration):
  - 40 lines with more than half the reference wrong hold 732 of the subject's 1,577 edits; 34
    of them fail for parent and sibling too. CER without them: 0.086.
  - Parent better than the subject on 81/103 Hertwig lines, worse on 174/197 Schlussredaktion
    lines. Sibling against subject on Schlussredaktion: better 58, worse 38.
  - Task vectors: ‖subject−parent‖ 85.4, ‖sibling−parent‖ 73.5, cosine 0.52 (L_12 0.24–0.28,
    O_18 weight 0.85, bias 0.97); ‖subject−sibling‖ 78.8.
  - The parent's codec maps to the subject's rows by identity; the sibling's does **not** (it
    lacks `ƒ`, so its later rows are shifted by one). `codec_map()` has no entry for the blank;
    the program adds row 0 → row 0 itself.
  - Practice references contain 22 no-break spaces, which `normalise` flattens.
- **Plan:** `program_01.py` = C0, then H1 (average with the sibling, α 0.5), H2 (quarter step
  toward the parent), H3 (blank bias, size by the D/I rule), each on the last kept state. It
  prints CER per hand, the paired interval per hand and the confusions for every pass. Passes
  5–10 are decided after its output. No freeze in program_01.
- **Earlier sessions:** the pre-registration (sections 1–4, 6, written by the operator) says
  that earlier sessions on this subject exist and that this run replicates their conditions; a
  code comment in `wsp/run.py` names a session id `20261106-bern-combined-v1-b10-1` in
  connection with a harness bug (`keep()` returned None). Neither names an edit, a kept
  decision or a result of such a session. `readings_budget0.md` lists phase-0 hypotheses for
  this model (blank bias to the parent's value, output-layer transplant, per-layer α); these
  are written before any session and are not results.

## Round 1 — 2026-11-24 17:05 UTC (programmer, `claude-fable-5-2`)

- **Read:** the operator's stdout of `program_01.py`; `log.jsonl` (decision events) and
  `readings/pass_001…004.jsonl` of this run, through two read-only scripts (no forward pass).
- **Results of program_01 (passes 1–4):**

  | pass | edit | CER | Hertwig | Schlussred. | collapsed | vs kept (95 %) | decision |
  |---|---|---:|---:|---:|---:|---|---|
  | 1 | C0 | 0.1456 | 0.2050 | 0.1184 | 29 | — | baseline, reproduced |
  | 2 | H1 sibling average α 0.5 | 0.1445 | 0.2015 | 0.1184 | 23 | −0.0011 [−0.0073, +0.0038] | refused |
  | 3 | H2 toward parent α 0.25 | 0.1381 | 0.1671 | 0.1247 | 22 | −0.0076 [−0.0147, −0.0015] | kept |
  | 4 | H3 blank bias −0.7 | 0.1346 | 0.1604 | 0.1228 | 18 | −0.0034 [−0.0067, −0.0006] | kept |

- **Predictions against outcomes:**
  - H1: predicted −0.3 points with Schlussredaktion down and Hertwig up. Got −0.1; Schluss-
    redaktion unchanged, Hertwig slightly down. Direction of the per-hand prediction wrong; the
    average is neutral. The two models are linearly connected (no damage), but averaging gives
    nothing on the hand both were trained on.
  - H2: predicted −0.4 (Hertwig −2 to −4, Schlussredaktion +0.5 to +1.5). Got −0.76, Hertwig
    −3.8, Schlussredaktion +0.63. Inside the predicted ranges.
  - H3: predicted −0.3, both hands alike, length ratio +0.005 to +0.02. Got −0.34, length
    ratio +0.018. But see below: the gain is not where I predicted it.
- **Where the gains sit** (lines split by whether more than half the reference was wrong at C0):

  | pass | failed@C0 H (14 lines, 388 chars) | failed@C0 S (26, 592) | ok@C0 H (89, 3,017) | ok@C0 S (171, 6,832) |
  |---|---:|---:|---:|---:|
  | 1 C0 | 296 | 436 | 402 | 443 |
  | 2 H1 (undone) | 264 | 437 | 422 | 442 |
  | 3 H2 | 263 | 406 | 306 | 520 |
  | 4 H3 | 244 | 389 | 302 | 523 |

  - H2 on readable lines: Hertwig −96 edits, Schlussredaktion +77. Net −19 of its −82.
  - H3 on readable lines: −1 edit. Its −37 come almost entirely from the failed lines. One of
    them (Hertwig p0027/tr_2_tl_5) is read properly after H2+H3 (28 → 6 edits); others emit more
    characters that partly match. I had predicted fewer narrow deletions on readable lines; that
    did not happen (deletions fell, insertions and substitutions rose by as much).
  - Schlussredaktion cost of H2 by kind: `D`→`d` 12 → 27, `/` deletions and space errors up.
- **Caveat for the claim.** Both kept edits pass the line-level rule, H3 narrowly. The
  Schlussredaktion cost of H2 is on a hand the parent never saw, so it should carry to held-out;
  the Hertwig gain may be recall of the parent's training lines. I expect the held-out gain to
  be smaller than the practice gain, and it may be nil or negative. The page-clustered interval
  of H3 may well include 0.
- **Plan (program_02, passes 5–6):** localisation of the parent step. L1: `L_16` and `O_18`
  back to the unedited subject's values, blank bias untouched. Then L2a (if L1 kept: front and
  `L_14` a further 0.2 toward the parent) or L2b (if not: conv stack and `L_12` back to the
  subject). The unedited subject file is opened read-only as the source of its own values.
  Predictions are in section 5 (round 1 paragraph) and in `s.hypothesis`. No second blank step
  and no second whole-network parent step, for the reasons given there. No freeze yet.
- **Earlier sessions:** nothing read in this round named an edit, a kept decision or a result
  of an earlier session on this model.

## Round 2 — 2026-11-24 17:08 UTC (programmer, `claude-fable-5-2`)

- **Read:** the operator's stdout of `program_02.py` only. No new script was run on the
  readings this round; the group counts come from the program's own output.
- **Results of program_02 (passes 5–6), against the kept state of pass 4 (CER 0.1346):**

  | pass | edit | CER | Hertwig | Schlussred. | collapsed | vs kept (95 %) | decision |
  |---|---|---:|---:|---:|---:|---|---|
  | 5 | L1: `L_16`, `O_18` back to subject | 0.1330 | 0.1709 | 0.1156 | 18 | −0.0017 [−0.0045, +0.0013] | refused |
  | 6 | L2b: conv stack, `L_12` back to subject | 0.1351 | 0.1715 | 0.1184 | 18 | +0.0005 [−0.0022, +0.0035] | refused |

- **Predictions against outcomes:**
  - L1: predicted −0.2 (Schlussredaktion −0.3 to −0.8, Hertwig +0.3 to +1.5). Got −0.17,
    −0.73, +1.06. As predicted, and as predicted not enough for the rule (I gave it 0.25).
  - L2b: predicted −0.1 (Schlussredaktion −0.2 to −0.6, Hertwig +0.3 to +2). Got +0.05, −0.44,
    +1.12. Inside the ranges; refused.
  - Neither cut separates the Hertwig gain from the Schlussredaktion cost. Both blocks carry
    both, in different proportions.
- **Reading across passes 1, 4, 5, 6** (readable lines only; table in section 5, round 2): the
  parent share buys Hertwig cheaply in the front and middle (73 edits for 23) and dearly in the
  late block (27 for 57). This is a post-hoc reading of four passes on the same 300 lines.
- **Plan (program_03, the last program):** M1 = late block at the subject plus front and
  middle at 0.45 parent, as one edit; M2 = front and middle at 0.60 parent, only if M1 is kept;
  then `s.freeze()`. Passes 9–10 stay unspent, and pass 8 too if M1 is refused. The program
  checks the restored state (CER of the kept pass, blank bias, and that `L_12` is
  0.75 subject + 0.25 parent) and stops without freezing if it is not the kept one.
- **What gets frozen if M1 is refused:** H2 + H3, i.e. 0.75 subject + 0.25 parent in all
  tensors (rows aligned by character, the three new rows unchanged) and the blank bias at
  +0.268. Practice CER 0.1346 against 0.1456, collapsed 18 against 29.
- **For the operator's controls:** the kept diff has parts of different kind (the parent mix
  and the single blank-bias entry; with M1 also a block-wise mix), so the split control applies.
  The blank step earned its practice gain on the 40 lines that fail wholesale at C0.
- **Earlier sessions:** nothing read in this round named an edit, a kept decision or a result
  of an earlier session on this model.

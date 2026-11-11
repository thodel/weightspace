# Pre-registration — 20261111-bern-combined-v1-f1-b10-1

Filled in **before the first pass** (PROTOCOL.md §2), 11.11.2026 on asteraix. This session
replicates `runs/20261106-bern-combined-v1-b10-1` (kraken session 2), which Claude Opus 5.5
programmed. It keeps that session's subject, practice set, held-out set, budget and keep rule,
with a different programmer. Sections 1–4 and 6–7 were written by the operator; section 5 was
written by the programmer.

1. **Subject:** `models/kraken/kraken-bern-combined-v1.mlmodel`.
   - Tensor digest `4836a9f494a9a62c2a743a0df0d0a848601a39ccee229964cd89687ddef8016c`, file
     sha256 `35e7eae281ea85e61847e22ab65a0295aeddb7ddbe49d93bcf7deae7f783a3a6`.
   - Trained from the 25.10 export, in which 44 of the 50 Hertwig pages carry another page's
     image (docs/HERTWIG_REVISIONS.md).
   - **Donors** (read-only, as in session 2):
     - `kraken-bern-hertwig-v2`, the parent. Tensor digest `f216564e4ddcf8f080556683626798cc2574bb90ee0a196b499a5d1ce9831572`,
       file sha256 `479392ab7899b831b3c3872dc0f178a894495757b2df2047f2b695c4a5dcc8f3`. Trained
       on the Hertwig training pages of the 22.10 export, so on all 103 Hertwig practice lines.
     - `kraken-bern-schlussredaktion-v1`, the clean sibling with the same parent. Tensor digest
       `a0a8e336c85ed4a2e6cde1ff6b6e72cd5191410dc2471a3ff966bc481063e366`, file sha256
       `d26eb02dfaf89f8f222165a1e5c982e38f19a915a5a744e4ad6c15f69b71f526`. Trained on the clean
       Schlussredaktion pages of 24.10, so on the 197 Schlussredaktion practice lines.
   - Output rows are aligned by character.
2. **Practice set:** `data/practice/bern-train-300.jsonl`, 300 lines from 146 pages (103
   Hertwig, 197 Schlussredaktion). List sha256
   `474414e3be95787da5f08f4bdcfc04adcfc024d8518b72cab8e885367977343f`, the same file as
   session 2.
   - **Side: training side** of both projects `combined-v1` was trained on, with the correct
     22.10 Hertwig images. It is labelled so as amendment A3 item 2 asks. A3 is proposed, not
     adopted; the training side is kept on purpose so that the conditions of session 2 hold.
   - Audit clean (`data/practice/overlap.json`).
3. **Held-out set:** `data/heldout/bern-eval.jsonl`, 545 lines: the 6 Hertwig test pages of the
   22.10 export and the 8 Schlussredaktion test pages. List sha256
   `f7b024e0fab59ae928cb7b0f28fe5a90d01823073ff3111d7d96f1b136d83227`.
   - Audit (`data/heldout/overlap-bern-eval.json`): all rows come from `_test` projects; no page
     is in any practice list or among the 168 training pages.
   - Scored **once**, after freeze, with `wsp.run heldout --set bern-eval`.
   - Paired comparisons, line-level and page-clustered (A4): against the unedited subject (C0 on
     held-out, `docs/baselines/heldout-bern-eval.json`), and against session 2's frozen model.
4. **Budget:** 10 passes. Pass 1 is C0.
   - **Keep rule:** PROTOCOL §3 as of 06.11.2026. The paired per-line bootstrap 95 % interval of
     CER(after) − CER(last kept), with 2,000 resamples and seed 20261106, must lie below 0
     (`min_effect` 0). It is enforced by `Session.keep()`. The programmer also reverts any edit
     that raises the collapsed count.
   - **No donor rule.** A3 is not adopted, and session 2 ran without it.
   - **Intervals for claims:** clustered by page (A4, adopted). They are computed by the operator
     after the session from `readings/`, with clusters `<project>/<page>`.
   - **Collapse detector:** the current one (A2). On every Bern set it gives the same counts as
     the legacy detector that session 2 used.
5. **Hypotheses:** written by the programmer, Claude Fable 5.1 (`claude-fable-5-1`), on
   11.11.2026 before the first pass, from the brief, the pack, the practice list and the harness
   source only (no forward pass; CPU-only reading of the three models' output layers and of
   `per_line_edits.json`).

   **What the reading says.** (a) The output-layer biases and norms of every character that
   occurs in the practice text differ by at most ±0.06 between subject, parent and sibling; the
   rows that moved most parent → subject (`· ſ ̶ ͠ ͛ ö ü ä § ꝝ ꝯ`) cover 24 of the 10,829
   reference characters. A rare-row revert therefore has no leverage on this practice set and is
   not a hypothesis here. (b) The one clear prior shift is the CTC blank: bias +0.998 against
   +0.879 (parent) and +0.936 (sibling), and the subject's errors are deletion-dominated: of
   1,577 edits, 1,033 are deletions and 153 insertions, 207 of them on 12 lines that every model
   (including the grandparent) gets entirely wrong (unreadable crops; CER without them 0.129).
   (c) The sibling is statistically indistinguishable from the subject on practice (paired
   Δ +0.0058, 95 % interval [−0.002, +0.013]), beats it on 58 of the 197 Schlussredaktion lines
   and loses on 61 of the 103 Hertwig lines; a per-line oracle of the two would read 0.129.
   (d) The parent reads the Hertwig practice lines at 0.080 (dead lines excluded) because it was
   trained on exactly those lines with the correct images; it reads the Schlussredaktion lines at
   0.292. Any edit toward the parent is therefore over-rewarded on the 103 Hertwig lines by
   memorisation, while its Schlussredaktion cost is measured honestly.

   **Donors.** The sibling (`schlussredaktion-v1`) was trained on the 197 Schlussredaktion
   practice lines (correct images), as was the subject, and never on Hertwig; a subject ↔ sibling
   interpolation gives neither side a memorisation advantage on practice and is the least
   confounded donor edit. The parent (`hertwig-v2`) was trained on all 103 Hertwig practice lines
   with correct images, the subject on the same pages with wrong images for 44 of 50; a gain from
   the parent on the Hertwig lines is partly memorisation transfer and must be discounted. No full
   transplant of a parent tensor block is planned; a parent-ward edit, if any, is a small
   whole-network shrinkage with a guard (H4).

   **Plan for the 10 passes.** Pass 1 is C0 (predicted 0.1456, Hertwig 0.2050, Schlussredaktion
   0.1184, collapsed 29; if it differs by more than 0.0005 or the collapsed count differs, the
   program stops without an edit). `program_01.py` then runs H1, H1b, H2 and H2b with branches on
   the keep decisions (at most 5 passes in all). The remaining passes are follow-ups decided from
   those results (H3–H5 below are the candidates), each written as a hypothesis with a prediction
   before its pass; the last program freezes. After every pass the program prints the per-hand
   CER (Hertwig / Schlussredaktion) computed from the pass's per-line edits, the paired interval
   against the last kept state, the collapsed count and the top confusions. My own extra revert
   rule: any edit that raises `collapsed` against the last kept state is reverted even if the
   harness would keep it.

   - **H1 — blank bias down (edit class: bias shift; tensor `O_18.lin.bias`, row 0 only;
     `add` with Δ = −0.4).** The subject learnt from mismatched image–text pairs on most Hertwig
     pages; CTC on such pairs rewards the blank everywhere, and the blank bias rose more here
     (+0.118) than in any other fine-tune of the family. Lowering it moves the blank-vs-character
     margin on every frame; characters whose frames were all taken by the blank come back.
     Prediction: practice CER −0.3 to −0.8 point (0.1456 → 0.138–0.143), both hands down in
     about the same proportion; the deletion confusions (` →∅`, `e→∅`, `n→∅`, `/→∅`) fall by
     10–25 %, insertions (`∅→ `, `∅→/`) rise by less than half of what deletions lose; length
     ratio 0.918 → 0.93–0.94; collapsed stays 29 (the 12 dead lines are dead at any blank bias).
     If refused: the deletions are not blank-margin effects, i.e. the missing characters are not
     represented in the recurrent features (or sit in the dead lines); the blank prior is not the
     obstacle, and no further blank edit is made unless the point estimate was negative (then one
     attempt at Δ = −0.8 total, H1b′).
   - **H1b — a second blank step (same class and tensor, Δ = −0.4 more, −0.8 in all), only if H1
     is kept.** Prediction: a further −0.1 to −0.4 point, smaller than H1's gain, with insertions
     now rising faster. If refused: the −0.4 state stands and the optimum lies between −0.4 and
     −0.8; conclude diminishing return and stop the blank sweep. If kept and its gain is at least
     half of H1's, one more step of −0.4 may be taken as a follow-up (H1c), otherwise not.
   - **H2 — whole-network interpolation toward the sibling, α = 0.5 (edit class: interpolation;
     all 34 tensors, the output layer aligned by character, `ƒ` row kept, blank bias held at its
     kept value so H2 does not undo H1).** Both children are fine-tunes of the same parent and
     share the Schlussredaktion data; their difference is, to first order, what the subject learnt
     from the mismatched Hertwig pages. Halving that component is a model soup of two nearly
     equal readers. Prediction: practice CER −0.3 to −1.0 point against the last kept state;
     Schlussredaktion −0.3 to −0.8 point (both soup members know this hand; soup effect);
     Hertwig between −0.5 and +1.5 point (the subject beats the sibling on 61 of the 103 Hertwig
     lines, so the midpoint may lose some of them); collapsed not up. If refused with a negative
     point estimate: try α = 0.25 (H2b). If refused with a non-negative point estimate: the two
     children are not linearly connected at the midpoint, or the noisy component carries real
     Hertwig ability that the practice lines reward; conclude that the damage is not separable by
     whole-network interpolation and leave this axis.
   - **H2b — α = 0.25 toward the sibling (same class and tensors), only if H2 was refused with a
     negative point estimate.** Prediction: −0.1 to −0.5 point. If refused: as for H2.
   - **Follow-up candidates (each gets its own hypothesis entry with a prediction before its
     pass, in `program_02.py` or later):**
     - **H3 — space-row bias up** (`O_18.lin.bias`, row of ` `, `add` +0.25): ` →∅` is the top
       confusion (168, against 54 insertions). Predicted −0.1 to −0.5 point; refused means the
       word gaps are not seen by the features.
     - **H4 — small shrinkage of the whole fine-tune toward the parent** (interpolation, all
       tensors, α = 0.2, output rows aligned by character, the three new rows kept): a
       regulariser against the noisy-label overfit. Predicted Hertwig −1 to −3 points (inflated
       by the parent's memorisation of those lines), Schlussredaktion +0.5 to +2 points, total
       −0.5 to +0.5: probably not keepable. Guard: it is reverted, whatever the harness says, if
       the Schlussredaktion CER rises by more than 0.3 point, because that part is the honest
       measurement and a Hertwig-only gain from the parent is suspect on held-out.
     - **H5 — localisation of an H2 gain by block** (conv stack vs recurrent stack vs output
       layer at the kept α): informative only; a block revert that leaves CER unchanged cannot be
       kept under the rule and is reverted.
     - **H1c** — a third blank step, as stated under H1b.

   **Overall expectation.** The Hertwig deficit is a data defect in the subject's training; what
   weight edits can remove is the blank prior and perhaps part of the noisy component by
   averaging with the clean sibling. I expect at most one or two kept edits and a practice gain of
   0.5–1.5 points, and I state in advance that a Hertwig gain from any parent-ward edit should be
   discounted on held-out. "Nothing kept" is an acceptable outcome.

   **Disclosure.** Section 7 of this pre-registration, which was in my run folder when I began,
   states that session 2 kept an output-layer transplant from `hertwig-v2` that lost on held-out.
   I read it. My plan contains no such transplant; my own reading (point d above) argues against
   one independently, but the fact that I knew this is recorded here and in `notes.md`.

6. **Controls** (as in session 2):
   - **C0:** pass 1 here, and `docs/baselines/bern-train-300.json` / `heldout-bern-eval.json`.
     Practice CER 0.1456, held-out 0.1933.
   - **C1:** the parent and the sibling on practice and held-out, from the same files. Parent
     0.2408 / 0.2749; sibling 0.1514 / 0.2036.
   - **C6 blind:** the budget-0 readings (docs/READINGS.md, phase 0).
   - **C5 sham:** for every kept edit that is reported as a gain on held-out. It is a random edit
     of the same tensors with the same Frobenius norm as the kept diff, 5 seeds, scored on
     practice and held-out, outside the budget.
   - **C4:** a trainer job; it needs an explicit OK and is not run here.
   - **C2 and C3:** per family, not per session.
7. **Programmer and operator:**
   - **Programmer:** Claude Fable 5.1 (`claude-fable-5-1`), as a Claude Code subagent
     (`model: fable`) with the default reasoning effort. Its tools are Read, Write and Bash
     (CPU-only reading, `py_compile`).
     - **Reading:** only `PROGRAMMER_BRIEF.md` in this folder and the files it lists: the pack
       (budget-0 readings as of phase 0, describe/diff of subject, parent and sibling, C0/C1),
       the practice list and the harness source. It did not read session 2's log, programs,
       notes or results, RESULTS R1–R3, or any report on session 2.
     - **Rounds:** it writes `program_NN.py` in rounds; the operator runs each in the sandbox
       (`wsp.run.sandbox_command`, CPU, no network) and returns its stdout.
     - **Difference from session 2:** session 2's programmer wrote one `program.py` in full
       before the run. Here the programmer may revise later programs after seeing results.
   - **Operator:** a Claude Opus 5.5 (`claude-opus-5-5`) session. It set up the run, runs the
     programs, scores held-out once, runs the controls and writes the comparison. It writes no
     hypotheses and no edits.
     - Before launching the programmer, it read session 2's pre-registration, which names that
       session's planned hypotheses, but not its log, notes or programs. It also read
       PROTOCOL.md and the amendments. Those say that session 2 kept an output-layer transplant
       from hertwig-v2, which lost on held-out. So **the operator knows session 2's kept edit**.
     - It passes none of this to the programmer. The brief names no edit and no result of
       session 2.

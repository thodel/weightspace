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
5. **Hypotheses** (programmer: Claude Fable 5.1, `claude-fable-5-1`; written 11.11.2026 before the
   first pass, from the brief, the pack, the practice list and CPU-only reads of the three models;
   no forward pass spent).

   **What I read in the pack and the weights.** The subject's errors are deletions: in a–z 669
   deletions against 48 insertions, space 168 : 54, `/` 70 : 37; blank bias +1.00 against the
   parent's +0.88 and the sibling's +0.94. The output rows of every character that occurs in the
   practice text are nearly the same in the three models (cosine ≥ 0.97, biases within ±0.1), so the
   output layer is not where the subject differs from its relatives. The rare rows that the diff
   lists as "moved most" (`ü ö ä ſ · §` …) do not occur in the practice references at all (the
   umlauts there are NFD combining marks `ͤ ͦ ̈`, whose rows barely moved), so a rare-row revert can
   change nothing here and is not a hypothesis. The two fine-tune directions τC = subject − parent
   and τS = sibling − parent have a global cosine of 0.52: 0.85–0.97 in `O_18`, 0.54–0.80 in
   `L_16`, 0.42–0.56 in `L_14`, 0.24–0.28 in the `L_12` weights, 0.36–0.48 in the conv weights. The
   two children learnt the same classifier and the same last layer and diverged most in the first
   BiLSTM; the subject's move is 1.16× the sibling's in norm. Per line (`per_line_edits.json`): the
   parent beats the subject on 81 of the 103 Hertwig lines (its own training lines) and loses on 174
   of the 197 Schlussredaktion lines; the sibling beats the subject on 58 Schlussredaktion lines,
   loses on 38, ties on 101, and loses on 61 of 103 Hertwig lines. 42 lines have ≥ half their
   reference length in edits for the subject, and most of those are as bad for both donors (very
   short or marginal lines): no edit will recover them, and they bound what any edit can show.

   **The donors and the practice set.** The *parent* (`hertwig-v2`) was trained on the 103 Hertwig
   practice lines with correct images; its 0.1137 there is a training-set score, so any Hertwig gain
   from an edit toward the parent is partly memorisation and need not transfer to held-out. It never
   saw Schlussredaktion (0.2992 on those lines). The *sibling* (`schlussredaktion-v1`) was trained on
   the 197 Schlussredaktion practice lines with correct images, as was the subject, so on those lines
   the two stand on equal footing; it never saw Hertwig, so a Hertwig gain from an edit toward the
   sibling cannot be memorisation.

   **Plan for the 10 passes.** P1 = C0 (must reproduce 0.1456; if not, the program stops before any
   edit). P2 = H1, P3 = H2, both in `program_01.py`. P4–P9: follow-ups from the list below, decided
   from the readings, each written as a hypothesis with a prediction in `notes.md` and in
   `s.hypothesis()` before its pass. Freeze by P10 at the latest; unspent passes may stay unspent.
   Every program reverts an edit that raises `m.collapsed` against the last kept state even if the
   rule would keep it.

   - **H1 — blank bias.** `add` on `O_18.lin.bias`, row 0 only, −0.20 (the parent sits 0.12 below
     the subject). *Prediction:* CER −0.1 to −0.5 points on both hands; deletions of ` e n i r t`
     fall, insertions rise by less; collapsed count does not rise. *If refused* (interval contains 0
     or CER up): the deletions are not a blank-threshold matter but sit in the features or in the
     unreadable lines; no further blank steps. *If kept:* **H1b**, a second step of −0.20
     (prediction 0 to −0.3 points; stop after a refusal).
   - **H2 — soup with the sibling.** `interpolate`, α = 0.5, on the 28 trained tensors (the six
     `bias_ih` are zero in all three models), output rows aligned by character, the `ƒ` row (subject
     only) unchanged; if H1 was kept, the blank bias keeps H1's value so that H2 does not partly undo
     H1. *Prediction:* Schlussredaktion −0.3 to −0.8 points (toward or beyond the sibling's 0.1129;
     a soup of two fine-tunes from one initialisation is usually no worse than the better one);
     Hertwig between −1 and +1 point (the subject's Hertwig signal is weak, 0.205 against 0.2355 for a
     model that never saw Hertwig, and averaging halves both its real Hertwig learning and its label
     noise); overall −0.2 to −0.6 points; collapsed unchanged. *If refused with CER up on both hands:*
     the two children are not linearly connected at α = 0.5 → **H2b** α = 0.25 (one pass, half the
     effect predicted). *If refused with Schlussredaktion down and Hertwig up:* the subject's Hertwig
     knowledge lies off the sibling's direction and only the parent carries Hertwig → H4. *If kept and
     Hertwig did not get worse:* **H2c** α = 0.75 is a candidate (predicted small further
     Schlussredaktion gain, Hertwig loss; likely refused).
   - **H3 — bias of the most-deleted characters.** `add` +0.3 on the rows of ` ` and `/` (deletion
     : insertion 168 : 54 and 70 : 37). *Prediction:* −0.1 to −0.4 points, confined to the confusions
     of these two characters. Decided after the confusions of the last kept state; dropped if H1 has
     already balanced them (insertions ≥ deletions) or if H1 was refused with insertions rising.
   - **H4 — toward the parent.** `interpolate`, α = 0.25, on the 28 tensors, rows aligned, the new
     rows `½ ƒ ̊` unchanged. *Prediction on practice:* Hertwig −2 to −4 points, Schlussredaktion +0.5
     to +1.5 points, net −0.3 to −1.0; **the Hertwig part is confounded** by the parent's training on
     these lines. Run only if passes are left after H1–H3; a kept H4 is a practice-set result whose
     held-out transfer is uncertain, and I say so in `notes.md`; held-out and the C5 sham decide.
     *If refused:* the parent's Hertwig memory does not survive averaging into the subject, or the
     Schlussredaktion loss outweighs it.
   - **H5 — layer-local soup (diagnostic).** If H2 was refused and a pass is left: `interpolate`
     α = 0.5 toward the sibling on the eight `L_12` tensors only (where the two fine-tunes diverge
     most). *Prediction:* a small effect of the sign H2 showed on Schlussredaktion, −0.2 to +0.3
     points. *If refused:* the difference between the children is distributed, not local; freeze.

   **Stopping.** I freeze when the planned hypotheses and their pre-registered follow-ups are
   exhausted, or after two consecutive refusals with no follow-up left on the list. **If nothing is
   kept** I conclude that the subject's deficit is spread over the recurrent stack, as the uniform
   per-tensor diff suggests, and is not removable by output-bias shifts or by linear combination with
   its relatives: nothing found, nothing kept.

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
     - What the operator knew about session 2 before the first pass is written in
       `docs/KRAKEN_S2_FABLE.md` §1, which the programmer does not read. The brief and this
       file name no edit and no result of session 2.
     - **Attempt 0 was discarded** before any pass. This is a deviation, also recorded in
       `summary.json`. An earlier wording of this section was visible to the first programmer
       subagent and contained information about session 2's outcome. That subagent's round-0 files were set
       aside unrun; after the session they are in `attempt0_discarded/`. The programmer of this
       session is a fresh subagent that did not see them.

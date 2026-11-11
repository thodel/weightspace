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
5. **Hypotheses:** *(written by the programmer before the first pass; see below)*

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
       subagent and said which edit session 2 kept. That subagent's round-0 files were set
       aside unrun; after the session they are in `attempt0_discarded/`. The programmer of this
       session is a fresh subagent that did not see them.

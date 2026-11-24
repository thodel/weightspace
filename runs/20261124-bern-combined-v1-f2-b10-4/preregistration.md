# Pre-registration — 20261124-bern-combined-v1-f2-b10-4

Filled in **before the first pass** (PROTOCOL.md §2), 24.11.2026 on asteraix. This session is a
replicate of kraken session 2 on `bern-combined-v1`: same subject, donors, practice set, held-out
set, budget and keep rule as the earlier sessions on this subject, with a different programmer.
Sections 1–4 and 6–7 were written by the operator; section 5 is written by the programmer.

1. **Subject:** `models/kraken/kraken-bern-combined-v1.mlmodel`.
   - Tensor digest `4836a9f494a9a62c2a743a0df0d0a848601a39ccee229964cd89687ddef8016c`, file
     sha256 `35e7eae281ea85e61847e22ab65a0295aeddb7ddbe49d93bcf7deae7f783a3a6` (checked by
     `wsp.run new`, 24.11.2026 16:55 UTC).
   - Trained from the 25.10 export, in which 44 of the 50 Hertwig pages carry another page's
     image (docs/HERTWIG_REVISIONS.md).
   - **Donors** (read-only):
     - `kraken-bern-hertwig-v2`, the parent. Tensor digest `f216564e4ddcf8f080556683626798cc2574bb90ee0a196b499a5d1ce9831572`,
       file sha256 `479392ab7899b831b3c3872dc0f178a894495757b2df2047f2b695c4a5dcc8f3`. Trained
       on the Hertwig training pages of the 22.10 export, so on all 103 Hertwig practice lines.
     - `kraken-bern-schlussredaktion-v1`, the clean sibling with the same parent. Tensor digest
       `a0a8e336c85ed4a2e6cde1ff6b6e72cd5191410dc2471a3ff966bc481063e366`, file sha256
       `d26eb02dfaf89f8f222165a1e5c982e38f19a915a5a744e4ad6c15f69b71f526`. Trained on the clean
       Schlussredaktion pages of 24.10, so on the 197 Schlussredaktion practice lines.
     - All digests checked on 24.11.2026 and equal to those of the earlier sessions on this subject.
   - Output rows are aligned by character.
2. **Practice set:** `data/practice/bern-train-300.jsonl`, 300 lines from 146 pages (103
   Hertwig, 197 Schlussredaktion). List sha256
   `474414e3be95787da5f08f4bdcfc04adcfc024d8518b72cab8e885367977343f`.
   - **Side: training side** of both projects `combined-v1` was trained on, with the correct
     22.10 Hertwig images. Labelled so as amendment A3 item 2 asks. A3 is proposed, not adopted;
     the training side is kept on purpose so that the conditions of the earlier sessions hold.
   - Audit clean (`data/practice/overlap.json`).
3. **Held-out set:** `data/heldout/bern-eval.jsonl`, 545 lines: the 6 Hertwig test pages of the
   22.10 export and the 8 Schlussredaktion test pages. List sha256
   `f7b024e0fab59ae928cb7b0f28fe5a90d01823073ff3111d7d96f1b136d83227` (checked 24.11.2026).
   - Audit (`data/heldout/overlap-bern-eval.json`): all rows come from `_test` projects; no page
     is in any practice list or among the 168 training pages.
   - Scored **once**, after freeze, with `wsp.run heldout --set bern-eval`.
   - Paired comparisons, line-level and page-clustered (A4): against the unedited subject (C0 on
     held-out, `docs/baselines/heldout-bern-eval.json`). The operator compares with the frozen
     models of the earlier sessions on this subject after the session, outside this file.
4. **Budget:** 10 passes. Pass 1 is C0.
   - **Keep rule:** PROTOCOL §3 as of 06.11.2026. The paired per-line bootstrap 95 % interval of
     CER(after) − CER(last kept), with 2,000 resamples and seed 20261106, must lie below 0
     (`min_effect` 0). It is enforced by `Session.keep()`. The programmer also reverts any edit
     that raises the collapsed count.
   - **No donor rule.** A3 is not adopted.
   - **Intervals for claims:** clustered by page (A4, adopted). They are computed by the operator
     after the session from `readings/`, with clusters `<project>/<page>`.
   - **Collapse detector:** the current one (A2). On every Bern set it gives the same counts as
     the legacy detector.
   - A5 (proposed, not adopted) concerns page-level adapter sessions and does not apply.
5. **Hypotheses** (written by the programmer before the first pass):

   *(to be filled in by the programmer)*

6. **Controls:**
   - **C0:** pass 1 here, and `docs/baselines/bern-train-300.json` / `heldout-bern-eval.json`.
     Practice CER 0.1456, held-out 0.1933.
   - **C1:** the parent and the sibling on practice and held-out, from the same files. Parent
     0.2408 / 0.2749; sibling 0.1514 / 0.2036.
   - **C6 blind:** the budget-0 readings (docs/READINGS.md, phase 0).
   - **C5 sham:** for every kept edit that is reported as a gain on held-out. A random edit of
     the same tensors with the same Frobenius norm as the kept diff (per tensor), 5 seeds, scored
     on practice and held-out, outside the budget. If the kept diff has parts of different kind,
     each part is also scored alone (split control).
   - **C4:** a trainer job; it needs an explicit OK and is not run here.
   - **C2 and C3:** per family, not per session.
   - **Serving path:** not run. It would need the frozen model registered on the share, and
     nothing is registered for this session.
7. **Programmer and operator:**
   - **Programmer:** Claude Fable 5.2, model id `claude-fable-5-2`, as a headless Claude Code
     session (`claude -p --model claude-fable-5-2`, tools Read, Write, Edit, Bash; Agent,
     WebFetch and WebSearch disallowed; no fallback model), default reasoning effort. One session
     for the whole run, resumed between rounds.
     - **Model verification:** *(filled in by the operator before the first pass)*
     - **Reading:** only `PROGRAMMER_BRIEF.md` in this folder and the files it lists.
     - **Rounds:** it writes `program_NN.py` in rounds; the operator runs each in the sandbox
       (`wsp.run.sandbox_command`, CPU, no network) and returns its stdout.
   - **Operator:** a Claude Opus 5.5 (`claude-opus-5-5`) session. It set up the run, runs the
     programs, scores held-out once, runs the controls and writes the comparison. It writes no
     hypotheses and no edits. What it knew before the first pass is recorded outside this folder,
     in a file the programmer does not read.

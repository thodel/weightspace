# Pre-registration — 20261124-bern-combined-v1-f2-b10-2

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

   Written 24.11.2026 16:59 UTC by the programmer (`claude-fable-5-2`), before pass 1.

   **What the pack and the weights say (no pass spent).**
   - 40 of the 300 practice lines have more than half their reference wrong for the subject.
     They carry 732 of its 1,577 edits; 34 of them fail in the same way for the parent and for
     the sibling, and 20 have 10 characters or fewer. I take these as lines no weight edit will
     read. Without them the subject's practice CER is 0.086, so the room for any edit is small.
   - The parent beats the subject on 81 of the 103 Hertwig lines (0.114 against 0.205) and
     loses on 174 of the 197 Schlussredaktion lines (0.299 against 0.118). The sibling is close
     to the subject on Schlussredaktion (0.113 against 0.118; better on 58 lines, worse on 38)
     and worse on Hertwig (0.236). Reading each hand with its own donor would give 0.113.
   - The sibling never saw Hertwig and reads it at 0.236; the subject, trained on Hertwig with
     44 of 50 pages mismatched, reads it at 0.205. So the subject's Hertwig loss looks like
     drift away from the parent while learning Schlussredaktion, with little help from its
     Hertwig pages, and not like a localised defect.
   - Task vectors from the parent (read-only script, rows aligned by character): subject
     ‖Δ‖ 85.4, sibling 73.5, cosine 0.52 over all tensors. The cosine is lowest in `L_12`
     (0.24–0.28) and highest in the output layer (weight 0.85, bias 0.97). Subject and sibling
     are 78.8 apart. All three models lie within one fine-tuning distance of each other.
   - Blank bias: subject +0.998, sibling +0.936, parent +0.879.

   **What the donors were trained on, and what that means here.** The parent was trained on all
   103 Hertwig practice lines with correct images; the sibling on all 197 Schlussredaktion
   practice lines. A gain on Hertwig from the parent, or on Schlussredaktion from the sibling,
   may therefore be recall of these very lines and need not hold on the held-out pages. The
   practice set cannot tell the two apart. Every pass is reported per hand; a gain on the hand a
   donor did *not* train on is the stronger evidence. The keep decision is the harness rule on
   all 300 lines plus the collapsed rule, with no extra condition of mine.

   **H1 (pass 2). Average with the sibling.**
   - Edit: `interpolate`, α = 0.5, toward `schlussredaktion-v1`; all 28 non-zero tensors (four
     conv blocks, three BiLSTMs, `O_18`); output rows aligned by character, the row of `ƒ`
     (absent in the sibling) unchanged.
   - Why: two fine-tunes of one parent with correlated task vectors; their average often reads
     the shared task better than either.
   - Prediction: practice CER −0.3 points (range −1.0 to +0.5); Schlussredaktion −0.5 to −1.0,
     Hertwig +0.5 to +1.5; collapsed unchanged within 2. Probability that the rule keeps it:
     about 0.35.
   - If it fails: the two fine-tunes are not linearly connected well enough at α = 0.5 (CER up
     on both hands), or the Hertwig loss outweighs the Schlussredaktion gain (then the sibling
     is of no use for the subject as a whole-network donor). No second sibling step unless the
     Schlussredaktion gain alone is clear.

   **H2 (pass 3). A quarter step back toward the parent**, on the last kept state.
   - Edit: `interpolate`, α = 0.25, toward `hertwig-v2`; same 28 tensors; rows aligned by
     character, the three new rows `½ ƒ ̊` unchanged.
   - Why: the drift reading above. Part-way points between a parent and its fine-tune usually
     keep most of the new task and recover part of the old one.
   - Prediction: practice CER −0.4 points (range −1.2 to +0.6); Hertwig −2 to −4 points,
     Schlussredaktion +0.5 to +1.5; collapsed unchanged or lower. Probability of keep: about
     0.35.
   - If it fails with Hertwig down and Schlussredaktion up by as much: the trade is linear and
     no point on this line helps. If Hertwig does not come down: the path is not smooth, and
     localisation (below) is the only parent edit left to try.

   **H3 (pass 4). Blank bias**, on the last kept state. No donor.
   - Edit: `add` on `O_18.lin.bias[0]` only. Size by a rule fixed here and computed from the C0
     readings over the lines with at most half their reference wrong: deletions D, insertions
     I. If D ≥ 2·I the step is −0.7; if I < D < 2·I it is −0.3; if D ≤ I the edit is skipped
     and the pass stays unspent.
   - Prediction: practice CER −0.3 points (range −0.8 to +0.3), both hands alike; deletions
     down, insertions up, length ratio up by 0.005–0.02; collapsed unchanged. Probability of
     keep: about 0.4.
   - If it fails: the deletions are not narrow blank wins; the blank bias is not an obstacle.

   **Plan for all passes.** Pass 1 is C0 (expected 0.1456, 1,577 edits, 29 collapsed; if not
   reproduced, the program stops without an edit). Passes 2–4 are H1–H3 in `program_01.py`,
   each on the last kept state and each followed by `measure` and `keep`/`revert`. Passes 5–10
   are follow-ups decided from these results; each will be written as a hypothesis with a
   prediction in `notes.md` and in `s.hypothesis` before its pass. The families I expect:
   - **F1, step size:** a second step of the same kind after a kept mix, or a half step after
     a refused one whose point estimate was below 0.
   - **F2, localisation of the parent step:** the same interpolation restricted to the conv
     stack and `L_12` (where subject and sibling task vectors agree least), or to `L_16` and
     `O_18`, if H2 shows a trade between the hands.
   - **F3, blank bias again:** a second step in the direction that helped, or on the new state
     after a kept mix.
   - Passes may stay unspent. If nothing is kept, the frozen model is the unedited subject.

   **Overall expectation.** The probability that at least one edit is kept is about 0.6. If
   edits are kept, I expect a practice CER of 0.135–0.143 at freeze, and on held-out a smaller
   gain or none, because of the donors' training overlap with the practice set.

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
     - **Model verification (operator, 24.11.2026 18:01 CET, before the first pass):** session id
       `b505bcce-e24b-47e4-8a07-58f89ac8a698`, started with `claude -p --model claude-fable-5-2
       --session-id … --permission-mode acceptEdits --allowedTools Read Write Edit Bash
       --disallowedTools Agent WebFetch WebSearch`, no `--fallback-model`, resumed between rounds
       with `--resume`. In its transcript (not committed) every assistant message of round 0 has
       `message.model = claude-fable-5-2` (23 of 23), and `modelUsage` lists only
       `claude-fable-5-2`. The programmer gave `claude-fable-5-2` as its own id. Each later round
       is checked the same way.
     - **Reading:** only `PROGRAMMER_BRIEF.md` in this folder and the files it lists.
     - **Rounds:** it writes `program_NN.py` in rounds; the operator runs each in the sandbox
       (`wsp.run.sandbox_command`, CPU, no network) and returns its stdout.
   - **Operator:** a Claude Opus 5.5 (`claude-opus-5-5`) session. It set up the run, runs the
     programs, scores held-out once, runs the controls and writes the comparison. It writes no
     hypotheses and no edits. What it knew before the first pass is recorded outside this folder,
     in a file the programmer does not read.

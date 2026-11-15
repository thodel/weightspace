# Pre-registration — 20261115-bern-combined-v1-f2-b10-1

Filled in **before the first pass** (PROTOCOL.md §2), 15.11.2026 on asteraix. This session is a
replicate of kraken session 2 on `bern-combined-v1`: same subject, donors, practice set, held-out
set, budget and keep rule as the earlier sessions on this subject, with a different programmer.
Sections 1–4 and 6–7 were written by the operator; section 5 is written by the programmer.

1. **Subject:** `models/kraken/kraken-bern-combined-v1.mlmodel`.
   - Tensor digest `4836a9f494a9a62c2a743a0df0d0a848601a39ccee229964cd89687ddef8016c`, file
     sha256 `35e7eae281ea85e61847e22ab65a0295aeddb7ddbe49d93bcf7deae7f783a3a6` (checked by
     `wsp.run new`, 15.11.2026 08:26 UTC).
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
     - All digests checked on 15.11.2026 and equal to those of the earlier sessions on this subject.
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
   `f7b024e0fab59ae928cb7b0f28fe5a90d01823073ff3111d7d96f1b136d83227` (checked 15.11.2026).
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

   Written by the programmer (`claude-fable-5-2`) on 15.11.2026, before any pass. All CER
   changes are on the 300 practice lines, in points (1 pt = 0.01 CER), against the last kept
   state.

   **What the pack and the weights say (no pass spent).**
   - The subject reads Hertwig at 0.2050 and Schlussredaktion at 0.1184. The parent reads
     Hertwig at 0.1137, the sibling Schlussredaktion at 0.1129. A per-hand switch between the
     donors would give 0.1131 on practice; that is the ceiling of what donors can give here, and
     it is a training-side ceiling: the parent was trained on all 103 Hertwig practice lines and
     the sibling on all 197 Schlussredaktion practice lines, both with correct images. A gain
     from a donor on practice is therefore partly recall of these very lines and is expected to
     be smaller on held-out.
   - 42 lines with ≥ 50 % errors carry 736 of the 1,577 edits, and the donors fail on most of
     them too (the 20 worst lines: subject 667 edits, parent 589, sibling 643). These lines are
     probably not readable by any edit; the reachable part is about 840 edits on 258 lines.
   - Weights (read-only script, no forward): the two fine-tunes moved in related directions.
     cos(subject − parent, sibling − parent) is +0.24 to +0.48 in the convs and `L_12`, +0.42
     to +0.63 in `L_14`/`L_16` weights, +0.85 in the output weight (rows by character) and
     +0.96 in the output bias. Subject and sibling differ by 0.17–0.29 relative L2 in the
     conv/LSTM weights and 0.09 in the output layer. Blank bias: subject +0.998, sibling
     +0.936, parent +0.879.

   **H1 — average with the clean sibling (donor interpolation, all tensors).**
   - Edit: `interpolate`, α = 0.5 toward `schlussredaktion-v1`, on all 28 conv/LSTM tensors
     that carry values and on `O_18.lin.weight|bias` with rows aligned by character (the row
     of `ƒ`, which the sibling lacks, stays the subject's).
   - Reason: both models are fine-tunes of the same parent with correlated deltas. The part of
     the subject's delta that the sibling does not share includes what the mismatched Hertwig
     pairs taught; averaging halves it while keeping the shared Schlussredaktion learning.
   - Prediction: total −0.3 to −1.0 pt; Schlussredaktion −0.3 to −1.0 pt; Hertwig between
     −1 and +1.5 pt (the sibling is 3 pt worse there); collapsed unchanged or lower. I put the
     chance that the keep rule is met at about 40 %.
   - If it fails: with CER up by more than 1 pt, the two models are not in one basin and no
     further whole-network averaging with the sibling is tried. With a change inside the
     interval, averaging neither helps nor hurts, and the subject's deficit against the sibling
     is not in a direction the sibling can remove.

   **H2 — a step back toward the parent (donor interpolation, all tensors).**
   - Edit: `interpolate`, α = 0.25 toward `hertwig-v2`, same tensors, rows by character (the
     rows of `½ ƒ ̊` stay as they are). Applied to the state kept after H1.
   - Reason: the Hertwig damage is the part of the subject's delta learnt from wrong image–text
     pairs; the parent is the model before that damage, but it reads Schlussredaktion at 0.299.
   - Prediction: Hertwig −1.5 to −4 pt, Schlussredaktion +0.5 to +2.5 pt, total between −1.0
     and +1.0 pt. The sign of the total is open; chance of a keep about 30 %. Any Hertwig gain
     is on lines the parent was trained on.
   - If it fails with Hertwig down and Schlussredaktion up: the Hertwig damage and the
     Schlussredaktion learning are not separable along this line, and only a block-restricted
     variant is still worth a pass. If Hertwig does not fall either: a quarter step does not
     reach the parent's Hertwig reading, and the line is dropped.

   **H3 — blank bias down (output-layer bias, one scalar).**
   - Edit: `add` −0.3 on `O_18.lin.bias[0]`, on the state kept after H1/H2.
   - Reason: deletions dominate (1,033 of 1,577 edits, length ratio 0.918), the blank has the
     highest bias of the layer and the subject's is the highest of the three models.
   - Prediction: −0.2 to −0.6 pt; deletions down by 40–100, insertions up by 20–60; length
     ratio up by 0.005–0.015; collapsed unchanged. Chance of a keep about 45 %. Much of the
     deletion mass sits in the unreadable lines, where a blank shift does nothing useful.
   - If it fails: with insertions up as much as deletions fall, the blank is calibrated and
     the deletions are not a threshold effect. With CER up clearly, a half dose (−0.15) is the
     only follow-up.

   **Plan for the 10 passes.**
   - Pass 1: C0. It must give CER 0.1456 (1,577 edits on 10,829 characters); otherwise the
     program stops with no further pass.
   - Passes 2–4 (`program_01.py`): H1, H2, H3 in this order, each followed by `measure` and
     then `keep`, or `revert` if the collapsed count rose against the last kept state.
     Structural edits come before the blank edit because the blank calibration depends on them.
   - Passes 5–10: follow-ups decided from the readings of passes 1–4, each written as a
     hypothesis with a prediction in `notes.md` and in the log before its pass. Candidates:
     - a second dose along a kept line (H1 or H2 kept: a further step; H3 kept: −0.15 more);
     - a half dose of a refused edit whose point estimate was below 0;
     - a block-restricted interpolation (convs only, LSTMs only, or the output layer only),
       toward the donor whose whole-network step gave the better per-hand reading;
     - single-row bias edits for the characters that the pass-1 readings show as deleted on
       readable lines (space, `/`), predicted at −0.1 to −0.3 pt each.
   - Passes may stay unspent. If nothing is kept, the frozen model is the unedited subject.

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
   - **Programmer:** Claude Fable 5.2, model id `claude-fable-5-2`, as a Claude Code subagent of
     type `wsp-programmer-f2` (`.claude/agents/wsp-programmer-f2.md`, frontmatter
     `model: claude-fable-5-2`, tools Read, Write, Edit, Bash), default reasoning effort. One
     subagent for the whole session, continued between rounds.
     - **Model verification (operator, 15.11.2026 09:5x CET, before the first pass):** the agent
       type could not be spawned as a subagent: agent files are read when a session starts, and
       the file was written during the operator's session. The programmer was therefore run as a
       headless Claude Code session with the same instructions and tools:
       `claude -p --model claude-fable-5-2 --session-id d89d5bf5-c617-447c-924e-83c697b77f14
       --allowedTools Read Write Edit Bash --disallowedTools Agent WebFetch WebSearch`, no
       `--fallback-model`, continued between rounds with `--resume` of that session id. In its
       transcript (`~/.claude/projects/…kraken-s2-fable52-923f0b/d89d5bf5-….jsonl`, not committed)
       every assistant message of round 0 has `message.model = claude-fable-5-2` (22 of 22), and
       the run's `modelUsage` lists only `claude-fable-5-2`. The programmer also gave
       `claude-fable-5-2` as its own id. Each later round is checked the same way.
     - **Reading:** only `PROGRAMMER_BRIEF.md` in this folder and the files it lists.
     - **Rounds:** it writes `program_NN.py` in rounds; the operator runs each in the sandbox
       (`wsp.run.sandbox_command`, CPU, no network) and returns its stdout.
   - **Operator:** a Claude Opus 5.5 (`claude-opus-5-5`) session. It set up the run, runs the
     programs, scores held-out once, runs the controls and writes the comparison. It writes no
     hypotheses and no edits. What it knew before the first pass is recorded outside this folder,
     in a file the programmer does not read.

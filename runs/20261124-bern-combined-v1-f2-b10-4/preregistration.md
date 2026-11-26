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

   Written 24.11.2026, about 16:59 UTC, by the programmer (`claude-fable-5-2`) before any pass.

   **What the pack and the weights say (no forward pass).**
   - The subject's deficit is on Hertwig. Practice CER per hand: subject 0.2050 / 0.1184
     (Hertwig / Schlussredaktion), parent 0.1137 / 0.2992, sibling 0.2355 / 0.1129. A model that
     read Hertwig like the parent and Schlussredaktion like the subject would be at 0.117, so
     the most any edit of this kind can give on practice is about −0.029.
   - Half of the subject's 1,577 edits sit in 30 lines, and all three models fail on most of
     them (40 lines have a line CER over 0.5). These are probably bad crops or misaligned
     references, and no edit will repair them. Without them the subject reads Hertwig at 0.133
     (parent 0.048 on the same lines) and Schlussredaktion at 0.065 (sibling 0.058).
   - Weight deltas from the parent: the subject's and the sibling's are of similar size in every
     tensor. Their cosine is 0.85 (weights) and 0.96 (bias) in the output layer, 0.54–0.80 in
     `L_16`, 0.42–0.70 in `L_14`, but only 0.24–0.41 in `L_12` and 0.36–0.48 in the conv
     weights. The two fine-tunes agree at the top and differ at the bottom.
   - Blank bias: subject +0.998, parent +0.879, sibling +0.936.
   - **Donors and the practice set.** The parent was trained on the 103 Hertwig practice lines
     with correct images, and the sibling on the 197 Schlussredaktion practice lines. So a
     practice gain from moving toward a donor is partly the donor's memory of these lines, and
     the held-out gain will be smaller than the practice gain. The subject was itself trained
     on the Schlussredaktion lines, and on the Hertwig texts with mostly wrong images.

   **Hypotheses in `program_01.py` (passes 1–4).**

   - **C0, pass 1.** Baseline. Predicted: CER 0.1456, 29 collapsed. If not, I stop.
   - **H1, pass 2: part of the way back to the parent.** Class: interpolate toward a donor.
     Tensors: all 28 trained tensors (`C_*`, `L_*` without the all-zero `bias_ih`, `O_18.lin`),
     α = 0.25 toward `hertwig-v2`; output rows aligned by character, the three rows the parent
     lacks (`½ ƒ ̊`) untouched.
     - Reason: the Hertwig loss came with the step parent → subject. Both hands' abilities lie
       on that line, and near the subject the Schlussredaktion loss should rise slowly (the
       subject is at its training optimum for that hand) while the Hertwig loss falls faster.
     - Prediction: Hertwig 0.205 → 0.165–0.185, Schlussredaktion 0.118 → 0.120–0.130, total
       **−0.006** (range −0.015 to +0.003). Collapsed stays at 29 ± 2.
     - If it fails (total not lower, or interval contains 0): the two hands are not separable
       along this line at this step. I then try a smaller or block-restricted step (F2) and do
       not go further out.
   - **H2, pass 3: lower the blank bias.** Class: bias shift (`add` on `O_18.lin.bias[0]`),
     −0.25, on the state kept so far.
     - Reason: deletions dominate (` →∅` 168, `e→∅` 78, `n→∅` 71; length ratio 0.918) and the
       subject's blank bias is the highest of the family.
     - Prediction: fewer deletions, more insertions, total **−0.002** (range −0.006 to +0.003).
       I expect the interval to contain 0 more likely than not: many of the deletions are in
       the broken lines, and the parent has the same length ratio (0.920) with the lower bias.
     - If it fails: the deletions are not a threshold effect of the blank. No further blank
       edits.
   - **H3, pass 4: average with the sibling.** Class: interpolate toward a donor. Same 28
     tensors, α = 0.30 toward `schlussredaktion-v1`, rows aligned by character (the sibling's
     row order differs from the subject's; `ƒ` untouched), on the state kept so far.
     - Reason: the sibling is the clean fine-tune for the hand that makes up 69 % of the
       practice characters, and averaging two fine-tunes of one parent often beats both.
     - Prediction: Schlussredaktion −0.003 to −0.008, Hertwig +0.000 to +0.010, total
       **−0.002** (range −0.006 to +0.004). If H1 was kept, the Hertwig rise should be smaller.
     - If it fails: the subject is already as good on Schlussredaktion as averaging can make
       it; the sibling is not used again except in a block-restricted follow-up.

   **Plan for passes 5–10 (follow-ups decided from these results).** Each is logged with
   `s.hypothesis` and a prediction before its pass, and written into `notes.md` first.
   - **F1, if H1 is kept:** continue the line search toward the parent (a further step of
     0.15–0.25), until a step is refused. Predicted smaller gain per step than H1.
   - **F2, if H1 is refused or after F1 stops:** block-restricted steps toward the parent, to
     find where the Hertwig damage sits: (a) lower blocks only (`C_*`, `L_12`), where the
     subject and the sibling differ most; (b) upper blocks only (`L_14`, `L_16`, `O_18`).
     Prediction: one of the two carries most of the H1 effect; I do not know which, with a
     slight lean to (a).
   - **F3, if H3 is kept:** one further step toward the sibling, or the step restricted to the
     blocks where F2 found no Hertwig effect.
   - **F4, if H2 is kept:** one further blank step of the same sign.
   - If nothing is kept after pass 4, I spend at most three more passes (F2 a, F2 b, and one
     smaller H1 step of α = 0.10) and then freeze the unedited subject. Unspent passes stay
     unspent.
   - I revert any edit that raises the collapsed count against the last kept state, also when
     the harness would keep it.
   - What I will not claim: a practice gain from a step toward a donor is not evidence of a
     held-out gain of the same size, for the reason given above.

   **Round 1 (24.11.2026, 17:04 UTC, after passes 1–4, before pass 5).**
   - Results so far: C0 0.1456 as predicted. H1 kept (−0.0076, interval −0.0147 to −0.0015;
     Hertwig 0.2050 → 0.1671, Schlussredaktion 0.1184 → 0.1247), inside the predicted ranges.
     H2 kept (−0.0017, interval −0.0032 to −0.0002); I had expected its interval to contain 0.
     H3 refused (−0.0015, interval −0.0045 to +0.0017): Schlussredaktion fell by 0.004 and
     Hertwig rose by 0.004, as predicted in direction, too small to pass the rule.
   - Kept state: CER 0.1364, Hertwig 0.1648, Schlussredaktion 0.1234, collapsed 21.
   - The H1 trade is broad, not a few lines: Hertwig 57 lines better / 7 worse (−129 edits, 22
     of them in one line), Schlussredaktion 25 better / 72 worse (+47 edits).
   - In every further step toward the parent the blank bias is left at its kept value, so the
     step does not undo H2.

   **Hypotheses in `program_02.py` (passes 5–8), each on the state kept so far.**
   - **F1, pass 5: a further 0.20 of the remaining way to the parent**, all 28 tensors (about
     0.40 of the way in total).
     - Prediction: Hertwig −0.010 to −0.020, Schlussredaktion +0.006 to +0.012, total **−0.001**
       (range −0.007 to +0.005). The first step was front-loaded for Hertwig and the
       Schlussredaktion cost should grow, so I expect the interval to contain 0 (kept with
       probability about 0.3).
     - If it fails: the optimum on the global line is near 0.25. No further global step.
   - **F2a, pass 6: lower blocks only** (`C_0 C_3 C_6 C_9 L_12`), a further 0.25 toward the
     parent.
     - Prediction: total **−0.001** (range −0.006 to +0.005); Hertwig −0.005 to −0.015.
   - **F2b, pass 7: upper blocks only** (`L_14 L_16 O_18`, blank bias untouched), a further
     0.25 toward the parent.
     - Prediction: total **−0.001** (range −0.006 to +0.005).
     - For F2a and F2b together: if the hands separate by depth, one of the two gives a
       Hertwig gain at a smaller Schlussredaktion cost than the global step and is kept; I lean
       slightly to F2a. If both fail (and F1 too), the Hertwig damage is spread over depth in
       the same proportion as the Schlussredaktion adaptation, and the line toward the parent
       is exhausted at the kept step.
   - **F4, pass 8: blank bias a further −0.25.**
     - Prediction: total **−0.001** (range −0.003 to +0.002); deletions down, insertions up
       (`∅→space` already rose from 54 to 67). Kept with probability about 0.35.
     - If it fails: the blank threshold is at its practice optimum near the kept value.
   - **Passes 9–10** are decided after these four: at most one refinement of whichever class
     was kept in passes 5–8, or nothing. Then freeze.

   **Round 2 (24.11.2026, 17:08 UTC, after passes 5–8; no further pass is planned).**
   - Results: all four edits of round 1 were undone.
     - F1 (all tensors a further 0.20 toward the parent): +0.0054 (interval +0.0018 to
       +0.0093), Hertwig 0.1648 → 0.1507, Schlussredaktion 0.1234 → 0.1378, collapsed 21 → 23.
       Predicted −0.001 (range −0.007 to +0.005): the result lies just above my range. The
       Hertwig gain was as predicted; the Schlussredaktion cost (+0.014) was larger than my
       range (+0.006 to +0.012).
     - F2a (lower blocks): +0.0032 (+0.0008 to +0.0058), Hertwig 0.1624, Schlussredaktion
       0.1292, collapsed 22. My lean to F2a was wrong: almost no Hertwig gain there.
     - F2b (upper blocks): +0.0021 (−0.0011 to +0.0055), Hertwig 0.1510, Schlussredaktion
       0.1328, collapsed 21. The whole Hertwig gain of the global step sits in `L_14 L_16
       O_18`, but so does most of the Schlussredaktion cost (among it `D→d`, a difference of
       convention between the hands, not of reading).
     - F4 (blank a further −0.25): −0.0011 (−0.0029 to +0.0005), as predicted; refused.
     - F1 and F2a were reverted by my collapsed check; the harness's rule would have refused
       both as well (CER rose, interval above 0).
   - Conclusions, as written before the passes: the optimum on the global line toward the
     parent is near the kept step of 0.25; the hands do not separate by depth in a way that an
     edit of whole blocks can use; the blank threshold is at its practice optimum near the kept
     value.
   - **Passes 9 and 10 stay unspent.** The round-1 plan allowed "at most one refinement of
     whichever class was kept in passes 5–8, or nothing", and nothing was kept. A new edit
     chosen now (for instance an upper-block step combined with a sibling step, or a smaller
     step between two refused ones) would be picked after seeing eight passes on the same 300
     lines, with effects of the size of the interval's width; I do not run it.
   - **`program_03.py` freezes the kept state**: H1 (all 28 trained tensors 0.25 toward
     `hertwig-v2`) and H2 (blank bias −0.25). Practice CER 0.1364 against 0.1456 (−0.0092),
     Hertwig 0.1648 against 0.2050, Schlussredaktion 0.1234 against 0.1184, collapsed 21
     against 29.
   - **Prediction for held-out** (scored once by the operator): a gain smaller than on
     practice, **−0.003** (range −0.010 to +0.004), because the parent was trained on the
     Hertwig practice lines and not on the held-out pages, while the Schlussredaktion cost
     should carry over in full. I expect the page-clustered interval on held-out to contain 0
     more likely than not. H2 alone is predicted at about −0.001 on held-out and not
     separable from 0.

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
       `24d97d0a-3dd9-4496-8e99-e9d8a46b4ae1`, started with `claude -p --model claude-fable-5-2
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

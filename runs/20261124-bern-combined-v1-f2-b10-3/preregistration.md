# Pre-registration — 20261124-bern-combined-v1-f2-b10-3

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

   Written 24.11.2026 about 17:00 UTC by the programmer (`claude-fable-5-2`), before pass 1.

   **What the pack and the weights say (no forward pass).**
   - *Where the errors are.* The subject makes 1,577 edits. 736 of them sit in 42 lines that it
     reads at ≥ 50 % CER, and the parent and the sibling fail on nearly all of the same lines
     (15 Hertwig lines with 298 edits, 27 Schlussredaktion lines with 438). These are short
     fragments or lines whose crop does not fit the text; I expect no weight edit to move them.
     That leaves about 400 Hertwig and 440 Schlussredaktion edits to work on.
   - *Where a donor is better.* On Hertwig the parent reads 0.114 against the subject's 0.205
     (311 fewer edits, about −0.029 of total CER if all of it were recovered at no cost). On
     Schlussredaktion the sibling reads 0.113 against 0.118 (41 edits, −0.004). The per-line
     oracle over all three models is 0.108 Hertwig / 0.098 Schlussredaktion, i.e. total ≈ 0.101.
     Both donors were trained on exactly the practice lines they are good at (parent: the 103
     Hertwig lines; sibling: the 197 Schlussredaktion lines; both with correct images), so a
     practice gain from moving toward a donor is partly a memorisation gain and will be
     smaller on held-out. The subject saw the same pages, the Hertwig ones mostly with wrong
     images.
   - *Geometry.* Subject and sibling are two fine-tunes of the parent, each with cosine ≥ 0.96
     to it per tensor. Their task vectors (child − parent) have cosine 0.24–0.48 in the conv
     stack and `L_12`, 0.42–0.70 in `L_14`, 0.54–0.80 in `L_16`, 0.85 / 0.97 in the output
     weight / bias. The subject is as far from the sibling as from the parent below `L_16`.
     The subject is better than the sibling on Hertwig (0.205 against 0.236), so the mismatched
     pages did not make it worse than a model with no Hertwig fine-tuning: what I see is mostly
     *forgetting* of the parent's Hertwig reading, not a localised defect.
   - *Blank.* Blank bias +0.998 (parent +0.879, sibling +0.936). Deletions dominate the
     confusions, but most deleted characters are in the 42 failed lines; the length ratio of the
     parent (0.920) equals the subject's (0.918). So the blank bias is not the main obstacle.

   **Hypotheses.** CER changes are on all 300 practice lines unless a hand is named. An edit
   is kept only by the harness's rule; I also revert when the collapsed count rises.

   - **H1 — partial return to the parent, all tensors.** Edit: `interpolate` with α = 0.30
     toward `hertwig-v2` on the 28 trained tensors (conv, LSTM, output; output rows placed by
     character, the rows `½ ƒ ̊` left as they are). Prediction: CER **−0.008** (range −0.003
     to −0.015); Hertwig −0.030, Schlussredaktion +0.003; collapsed 29 ± 2. If it fails with
     CER up: the path subject → parent is not flat at 30 %, or Schlussredaktion pays more than
     Hertwig gains; I then try a smaller step (H2c) and give more weight to block-wise edits.
   - **H2 — the step size along the same line (one pass, branch on H1).**
     - H2a, if H1 is kept: a further 2/7 toward the parent (total 50/50). Prediction against
       the kept state: −0.002 (range −0.008 to +0.006), Hertwig −0.020, Schlussredaktion
       +0.007; more likely refused than kept. If refused, the optimum is near 0.3.
     - H2b, if H1 lowered CER but was refused: α = 0.50 from the unedited subject. Prediction
       −0.010 (range −0.020 to +0.005).
     - H2c, if H1 raised CER: α = 0.12. Prediction −0.002 (range −0.006 to +0.002), expected
       refused; then whole-model interpolation toward the parent is closed.
   - **H3 — where the Hertwig gain sits (block-wise return to the parent).** From the state
     after H1/H2: `interpolate` toward the parent on a subset of blocks only, with a larger
     α (0.5–1.0). First candidate: the conv stack plus `L_12` (the blocks where subject and
     sibling disagree most, so the most hand-specific). Prediction: within ±0.004 of the
     whole-model step of the same α, more likely worse (blocks co-adapted); kept only if H1
     failed or if it adds ≥ 0.004. A failure says the forgetting is spread over depth.
   - **H4 — the sibling for Schlussredaktion.** `interpolate` toward `schlussredaktion-v1`,
     all tensors (rows by character, `ƒ` untouched), α = 0.2–0.3 from the kept state.
     Prediction: Schlussredaktion −0.003, Hertwig +0.004, total ±0.002: expected **refused**.
     The sibling's advantage is 41 edits and it has no Hertwig fine-tuning. I run it because
     a three-way mixture (subject, parent, sibling) may be flatter than either line.
   - **H5 — blank bias.** `add` −0.12 to −0.30 on `O_18.lin.bias[0]` (−0.118 is the parent's
     value). Prediction: −0.001 (range −0.004 to +0.003), fewer deletions, more insertions,
     interval contains 0: expected refused. Size and sign are set after I have read the
     deletion/insertion counts outside the failed lines in my own passes.
   - **H6 — output rows of rare characters back to the parent.** `set_rows` on the rows whose
     bias fell by > 0.5 (`ö ü ä ſ · § …`). The practice text spells umlauts with combining
     marks and holds almost none of these characters. Prediction: 0.000 ± 0.001, refused. Run
     only if passes are left and the readings show such a character missed.

   **Plan for the 10 passes.** Pass 1 C0. Passes 2–3 H1 and H2 (`program_01.py`). Passes 4–6:
   H3, H4 and H5 in the order the first readings suggest, each written as a hypothesis with a
   prediction in `notes.md` and in `s.hypothesis` before its pass. Passes 7–10: follow-ups
   decided from these results (finer α on whichever line worked, a combination of kept
   directions, H6), or left unspent. Total expectation: a kept gain of 0.005–0.015 on practice
   with probability about 0.6, all of it on the Hertwig lines; held-out gain, if any, about
   half of that, since the parent memorised the Hertwig practice lines. With probability about
   0.4 nothing passes the rule and I freeze the unedited subject.

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
       `017c1f65-a613-48a4-a101-69e49fb38156`, started with `claude -p --model claude-fable-5-2
       --session-id … --permission-mode acceptEdits --allowedTools Read Write Edit Bash
       --disallowedTools Agent WebFetch WebSearch`, no `--fallback-model`, resumed between rounds
       with `--resume`. In its transcript (not committed) every assistant message of round 0 has
       `message.model = claude-fable-5-2` (21 of 21), and `modelUsage` lists only
       `claude-fable-5-2`. The programmer gave `claude-fable-5-2` as its own id. Each later round
       is checked the same way.
     - **Reading:** only `PROGRAMMER_BRIEF.md` in this folder and the files it lists.
     - **Rounds:** it writes `program_NN.py` in rounds; the operator runs each in the sandbox
       (`wsp.run.sandbox_command`, CPU, no network) and returns its stdout.
   - **Operator:** a Claude Opus 5.5 (`claude-opus-5-5`) session. It set up the run, runs the
     programs, scores held-out once, runs the controls and writes the comparison. It writes no
     hypotheses and no edits. What it knew before the first pass is recorded outside this folder,
     in a file the programmer does not read.

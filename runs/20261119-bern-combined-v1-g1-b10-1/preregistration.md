# Pre-registration — 20261119-bern-combined-v1-g1-b10-1

Filled in **before the first pass** (PROTOCOL.md §2), 19.11.2026 on asteraix. This session
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
5. **Hypotheses** (written by the programmer before the first pass):

   - **Donor provenance and practice-set implications:**
     - The parent donor `kraken-bern-hertwig-v2` was trained on the clean 22.10 Hertwig training pages (including all 103 Hertwig practice lines; practice CER 0.1137 on Hertwig, 0.2992 on Schlussredaktion, 0.2408 overall).
     - The sibling donor `kraken-bern-schlussredaktion-v1` was fine-tuned from `hertwig-v2` on the clean 24.10 Schlussredaktion training pages (including all 197 Schlussredaktion practice lines; practice CER 0.2355 on Hertwig, 0.1129 on Schlussredaktion, 0.1514 overall).
     - The subject `kraken-bern-combined-v1` was fine-tuned from `hertwig-v2` on both hands, but 44 of its 50 Hertwig training pages carried mismatched images from the 25.10 export. As a result, its update vector from the parent ($\Delta_{\text{sub}} = W_{\text{sub}} - W_{\text{par}}$) combines a valid Schlussredaktion adaptation with severe label-noise corruption from mismatched Hertwig pairs (which inflated the CTC blank bias by $+0.118$, depressed character biases, and degraded Hertwig CER from $0.1137$ to $0.2050$). Because the two donors together saw all 300 practice lines cleanly, donor-based edits on this practice set benefit from direct training-set overlap on the practice lines; only the held-out evaluation (`bern-eval`, 545 lines from unseen test pages) tests true generalisation.

   - **Plan for the 10 passes:**
     - **Pass 1 (C0 baseline):** Run `s.measure("C0")` on the unedited subject. Predicted to reproduce practice CER `0.1456` (Hertwig `0.2050`, Schlussredaktion `0.1184`, collapsed `29`). Stop if it does not match.
     - **Pass 2 (H1 — CTC blank bias restoration to parent):**
       - *Edit class and tensors:* `set_rows` on `["O_18.lin.bias"]` (row 0, the CTC blank), setting `O_18.lin.bias[0]` from `+0.9976` back to the parent `hertwig-v2` value `+0.8792` ($\Delta = -0.1184$).
       - *Prediction:* Lowering the blank bias reduces deletion errors (the top confusions ` →∅`, `e→∅`, `n→∅`, `/→∅`, `i→∅`) across both hands at the cost of some insertions, yielding a net practice CER change of $-0.0020$ to $-0.0060$ (similar relative reduction on both Hertwig and Schlussredaktion) without raising collapsed lines.
       - *If it fails:* Conclude that the elevated blank bias in `combined-v1` is compensated by output row norms or recurrent representations and that shifting the blank threshold alone either adds as many insertions as deletions saved or does not clear the 95 % bootstrap interval.
     - **Pass 3 (H2 — Full output-layer character-aligned repair via parent–sibling midpoint):**
       - *Edit class and tensors:* `transplant` on `["O_18.lin.weight", "O_18.lin.bias"]` using the character-aligned average $0.5 W_{\text{par}} + 0.5 W_{\text{sib}}$ (via `wsp.align.codec_row_map` and `gather_rows`, keeping the subject's `ƒ` row which neither donor has, and using the sibling's rows for `½` and `̊`).
       - *Prediction:* Undoing the label-noise inflation of rare and umlaut rows and restoring clean classifier weights/biases across all 270 classes will reduce overall practice CER by $-0.0030$ to $-0.0080$ (primarily on Hertwig, $-0.0050$ to $-0.0150$, with Schlussredaktion changing by $0.0000$ to $-0.0030$).
       - *If it fails:* Conclude that the label-noise degradation on Hertwig lives primarily in the feature extractor and BiLSTM stack (`C_0`–`L_16`) rather than in the linear classifier `O_18`, or that `O_18` is tightly co-adapted to `L_16`'s corrupted features.
     - **Pass 4 (H3 — Full-network task-vector merge of clean parent and clean sibling, $\alpha = 0.50$):**
       - *Edit class and tensors:* `transplant` across all 34 tensors (`C_0` through `O_18`, with `O_18` rows aligned by character) to the midpoint of the two clean donors: $W = 0.5 W_{\text{par}} + 0.5 W_{\text{sib}}$ (preserving the subject's `ƒ` row and using the sibling's `½` and `̊` rows).
       - *Prediction:* Because `sibling` was fine-tuned from `parent` on clean Schlussredaktion data (relative L2 move $0.07$–$0.23$, cosines $>0.97$), the two models lie in a shared linearly connected basin. Replacing the noise-corrupted `combined-v1` weights with the clean midpoint $0.5 W_{\text{par}} + 0.5 W_{\text{sib}}$ is predicted to recover a large fraction of the parent's Hertwig accuracy while retaining most of the sibling's Schlussredaktion adaptation, lowering overall practice CER by $-0.0150$ to $-0.0300$ (Hertwig CER dropping from $\sim 0.205$ to $\sim 0.130\text{--}0.155$, Schlussredaktion CER staying near $\sim 0.120\text{--}0.135$, and collapsed lines dropping from $29$ toward $21\text{--}25$).
       - *If it fails:* Conclude that a $50/50$ linear average across the conv and BiLSTM layers suffers from internal feature mismatch or under-weights the Schlussredaktion adaptation (which accounts for $68.6\%$ of practice reference characters).
     - **Passes 5–10 (Follow-ups decided from the results of Passes 1–4, each written as a hypothesis with a prediction before its pass):**
       - In Round 1 we run Passes 1–3 (`program_01.py`) to test the localised output-layer repairs first and inspect their line-by-line readings.
       - In Round 2 (`program_02.py`) we will test H3 (the clean donor merge $W_{\text{par}} + \alpha (W_{\text{sib}} - W_{\text{par}})$) and, guided by the per-hand CER and collapsed counts, explore:
         1. Tuning the donor interpolation weight $\alpha \in [0.4, 0.7]$ (since Schlussredaktion is $68.6\%$ of practice characters, $\alpha \approx 0.55\text{--}0.65$ or layer-wise $\alpha$ with more parent in `L_12` where label-noise divergence is largest and more sibling in `L_16`/`O_18` may perform best).
         2. Blending the clean donor midpoint with the subject's clean features (`(1-\beta) W_{\text{clean-merge}} + \beta W_{\text{sub}}`) if `combined-v1` learned any useful shared representations not present in the linear span of `parent` and `sibling`.
         3. Layer-block decomposition (e.g. transplanting `L_12`–`L_16` vs `C_0`–`C_9` vs `O_18`) and fine-tuning `O_18.lin.bias[0]` (blank bias) on top of the best merged state.

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
   - **Programmer:** Gemini 4 Argon (`gemini-4-argon`), called through the Google API by `wsp.programmer`
     with the provider's default reasoning settings. It has the tools `read_file`, `list_dir`,
     `write_file`, `run_python` (CPU-only reading in the sandbox, on a throw-away copy of this
     folder) and `finish_round`. The tool layer lets it read only the files the brief lists and
     write only section 5 of this file, `program_NN.py` and `notes.md`.
     - **Reading:** only `PROGRAMMER_BRIEF.md` in this folder and the files it lists. It does
       not read any other run, report or plan.
     - **Rounds:** it writes `program_NN.py` in rounds. The operator runs each one in the
       sandbox and gives it the stdout; the conversation carries over between rounds.
     - **Record:** `programmer/transcript.jsonl` (every call and tool result),
       `programmer/usage.json` (tokens), `programmer/rounds.json`.
   - **Operator:** a Claude session. It set up the run, runs the programs, scores held-out once,
     runs the controls and writes the comparison. It writes no hypotheses and no edits. What it
     knew before the first pass is written in `docs/XVENDOR_PLAN.md` §6, which the programmer
     does not read. The brief and this file name no edit and no result of an earlier session.

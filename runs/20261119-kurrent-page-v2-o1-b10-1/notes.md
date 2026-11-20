# Notes — 20261119-kurrent-page-v2-o1-b10-1

## 2026-11-19 09:27 UTC — round 0 (before any pass)

Programmer: `gpt-6.2-astra`, OpenAI API, provider's default reasoning settings; operator options `{}`. Read the run brief, preregistration, notes and programmer configuration; listed this run folder. Consulted the budget-0 Markdown adapter reading, the practice summary in `docs/baselines/c0-kurrent-val-v2.json`, and the API/scoring code in `wsp/ops.py`, `wsp/lora_subject.py`, `wsp/session.py` and `wsp/pages.py`. No other run or test data inspected.

Wrote the pass plan and `program_01.py`. C0 must agree on all archived practice summary fields before H1 proceeds. H1 removes two high-contrast axes in the final layer; the lens does not tell us the signs of their coefficients on an actual page. The test is whether tails shorten across pages without a comparable change in clipped reading error. No program executed or result observed this round.

## 2026-11-19 12:01 UTC — after passes 1–2

Read the session log, both pass reports, the paired counts in `state.json`, and the runner's adapter setup. C0 reproduced all practice summary fields exactly. H1 was kept: CER 0.45151 → 0.37374 (−0.07777; paired 2,000-resample page interval [−0.17323, −0.00430]). The sampling unit is already the page; the log prints `ci95` rather than a separate cluster field. Caps 9 → 5, loops 12 → 10, length ratio 1.2264 → 1.1476. Truncated CER *also fell* (0.25639 → 0.24118), outside the predicted ±0.005. About 0.0626 of the 0.0778 CER drop was narrowing of the full-minus-clipped CER gap.

25 pages had fewer edits, 19 more, and 20 tied. The largest beneficiary accounts for 3,554 of the 8,418 edits saved; dropping it still gives ΔCER −0.0473 descriptively. This is not evidence of uniform improvement or of which layer-31 writer was responsible. Residual overrun remains (gap 0.1326). `program_02.py` tests the two already registered alternatives in order: H2 reduces late query updates, then H3 reduces late gate updates. Each compares against the last kept state, not necessarily C0; predictions and failure interpretations remain those in section 5. No new edit measured by the programmer.

## 2026-11-19 14:21 UTC — after passes 3–4

Read the logged measurements and decisions and the corresponding pass reports. Both trials were compared to H1 and reverted. H2 raised CER by 0.08725 (paired page 95% interval [0.00036, 0.18791]), raised clipped CER by 0.02758 and increased caps from 5 to 8. Although 28 pages improved and 24 worsened, the losses outweighed the gains. Late query attenuation at this dose harmed reading as well as stopping. H3 raised CER by 0.01601 (interval [−0.00393, 0.04207]); clipped CER rose by 0.00146, caps stayed at 5, and one fewer loop did not translate into fewer edits. Neither result justifies continuing the broad attenuation sweep.

Pass 5 (H4) instead follows H1: keep layer 31 as it is and project direction 2 out of layer 26's `mlp.down_proj` B and direction 0 out of layer 27's `mlp.down_proj` B, as one edit (indices zero-based). Among late writers outside layer 31 these are the two prominent negative end-token contrasts in the reading (z −2.10 and −2.80); that is a selection heuristic, not evidence of their actual contribution during generation. Prediction vs H1: CER −0.035, clipped CER unchanged within ±0.005, caps 5 → 3. Refusal means no demonstrated extension to that block; do not proceed layer by layer on the basis of one cap flip. `program_03.py` also reports page identities entering/leaving the cap and loop sets across the comparisons, without using that diagnostic to change the keep rule.

## 2026-11-19 15:30 UTC — after pass 5; decision to freeze H1

Read the H4 log entry and pass report. Relative to H1, H4 raised practice CER by 0.02489 (paired page 95% interval [−0.00099, 0.06557]) and truncated CER by 0.01481. It resolved none of H1's five caps and returned a sixth page (`008241_036296_0001_1211427`) to the cap. Two loops resolved and two appeared. H4 failed both keep criteria and was reverted.

H1 resolved four distinct baseline caps without introducing any. None of H2, H3 or H4 resolved even one of the remaining five. Given that pattern and the worse clipped reading under H4, there is no supported reason to vary another layer or dose on the same 64 pages. Stop with five passes unused. `program_04.py` freezes the last kept state (H1) without a measurement. The observed CER reduction belongs to these practice pages; transfer to unseen pages and specificity to the selected directions await the operator's held-out measurement and controls.

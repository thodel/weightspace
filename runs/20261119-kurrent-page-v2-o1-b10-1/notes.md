# Notes — 20261119-kurrent-page-v2-o1-b10-1

## 2026-11-19 09:27 UTC — round 0 (before any pass)

Programmer: `gpt-6.2-astra`, OpenAI API, provider's default reasoning settings; operator options `{}`. Read the run brief, preregistration, notes and programmer configuration; listed this run folder. Consulted the budget-0 Markdown adapter reading, the practice summary in `docs/baselines/c0-kurrent-val-v2.json`, and the API/scoring code in `wsp/ops.py`, `wsp/lora_subject.py`, `wsp/session.py` and `wsp/pages.py`. No other run or test data inspected.

Wrote the pass plan and `program_01.py`. C0 must agree on all archived practice summary fields before H1 proceeds. H1 removes two high-contrast axes in the final layer; the lens does not tell us the signs of their coefficients on an actual page. The test is whether tails shorten across pages without a comparable change in clipped reading error. No program executed or result observed this round.

## 2026-11-19 12:01 UTC — after passes 1–2

Read the session log, both pass reports, the paired counts in `state.json`, and the runner's adapter setup. C0 reproduced all practice summary fields exactly. H1 was kept: CER 0.45151 → 0.37374 (−0.07777; paired 2,000-resample page interval [−0.17323, −0.00430]). The sampling unit is already the page; the log prints `ci95` rather than a separate cluster field. Caps 9 → 5, loops 12 → 10, length ratio 1.2264 → 1.1476. Truncated CER *also fell* (0.25639 → 0.24118), outside the predicted ±0.005. About 0.0626 of the 0.0778 CER drop was narrowing of the full-minus-clipped CER gap.

25 pages had fewer edits, 19 more, and 20 tied. The largest beneficiary accounts for 3,554 of the 8,418 edits saved; dropping it still gives ΔCER −0.0473 descriptively. This is not evidence of uniform improvement or of which layer-31 writer was responsible. Residual overrun remains (gap 0.1326). `program_02.py` tests the two already registered alternatives in order: H2 reduces late query updates, then H3 reduces late gate updates. Each compares against the last kept state, not necessarily C0; predictions and failure interpretations remain those in section 5. No new edit measured by the programmer.

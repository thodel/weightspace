# Epic: an instruction file ("playbook") that lets other LLMs repair and improve kraken recognition models: proof of concept

Proposed 7 November 2026. Status: not started. Owner: Tobias Hodel. Related: paper/manuscript.md (Sections 7 to 9),
docs/CONTROLS_REPORT.md, docs/MODEL_FAMILY.md, docs/SEED_NULL.md, docs/ZOO_SURVEY.md, docs/READINGS.md.

## Question

Can what we now know about the inside of a kraken recogniser (CNN, three BiLSTMs, CTC) be written down as a documentation and
instruction file that lets other language models, small open ones and state-of-the-art ones, repair or improve a recognition model,
perhaps with a little manual feedback on a few lines the model has never seen?

## Verdict of the feasibility check (7 November 2026)

**Doable as a bounded proof of concept. A strong claim ("LLMs improve recognisers") is not yet supported, and the epic must be
built so that it can fail clearly.**

Evidence that it is plausible:

- Programmers of three sizes (Sonnet, Opus, Haiku) given a reading pack repaired planted defects: for the six corruptions that cost at
  least one point of held-out CER (1.3 to 90 points), a programmer with passes, a blind programmer and a fixed rule list all restored the
  model to within 0.22 points of the original (docs/CONTROLS_REPORT.md section 2). A diffuse defect (noise on every weight) was repaired
  to 83 % only by the programmer that could measure.
- The knowledge to put in a playbook now exists and was tested against a seed null: what fine-tuning leaves in the weights, the dead
  first-layer filters, the generic front end and individual recurrent stack, the failure of output-bias levers (the deleted character
  is a median 3.2 nats below the blank, 18 % of deletions have no blank frame to act on), additivity of single blocks and its failure for the
  whole network, the exchange moves that stabilised `latin_incunabula` (docs/MODEL_FAMILY.md, docs/RESULTS.md R4).
- The harness exists: `wsp.session` (budget, keep rule, paired and page-clustered intervals), `wsp.plant` (planted defects),
  `wsp.exchange`, `wsp.family`, `wsp.zoo`.
- A small, manually checked calibration set is realistic. Measured on our own line sets: 26 lines cover every one of the 109 characters of the
  Kurrent handwriting set at least once and 62 lines cover each at least three times; for the Bern set 26 and 66 lines (greedy set cover, NFD).

Evidence against, which the design must respect:

- No LLM has yet improved a real model on held-out data beyond a small interpolation toward a sibling (-0.39 points, page interval includes 0).
  The one real candidate so far came from a systematic grid, not a language model.
- A hand-written rule list did as well as the programmer on every localised defect. If the playbook only encodes rules, the right product is
  software, and the LLM adds nothing. The proof of concept must show where a language model adds something over rules and search, or say that it does not.
- Repairs depend on a donor of the same lineage: with an unrelated donor a zeroed row or block was not rebuilt (0 to 19 %). Weights
  alone do not give back lost knowledge ("removes obstacles, does not add ability").
- Sixty lines cannot resolve effects under about half a point of CER; the calibration set is a diagnostic for broken characters and large defects, not a test of small gains.

## Hypotheses

- H1 (repair): with the playbook, small and large models repair planted defects at least as well as the same models without it, and the playbook
  narrows the gap between small and large models.
- H2 (stabilise): with the playbook, a model finds exchange or interpolation moves that stabilise a specific model on unseen out-of-domain
  print, as the systematic grid did for `latin_incunabula`, within a budget of ten passes.
- H3 (adapt with feedback): a calibration set of 30 to 120 manually checked lines, covering the characters of the new material and unseen by the
  subject and its donors, lets a model with the playbook make a held-out improvement of at least one point that a non-LLM search over the same moves does not.
- H0 (null, reported equally): no arm beats the non-LLM baselines; then the deliverable is the playbook as documentation and the repair tool.

## Deliverables

1. **Playbook** `docs/playbook/KRAKEN_PLAYBOOK.md` (+ short variant for small models), written for a model that cannot see this repo: anatomy and block names;
   how to read a pack (norm ratios, cosines to the donor, zero blocks, per-character rows); a table of defect signatures and the repair for each;
   the move set (restore from donor, rescale, interpolate, exchange block, lossless rewrite); what not to do (output-bias levers, practice on training lines,
   donor trained on the practice lines, judging by line-level intervals); the measurement protocol; the API. Every statement links to the run or table that supports it.
2. **Tool layer** `wsp.agent`: JSON tools (`read_pack`, `get_tensor_stats`, `apply_move`, `measure`, `keep`, `revert`, `freeze`) with budget and keep rule enforced by the harness, so that models without a sandbox can work through tool calls; plus the existing write-a-program-in-advance mode.
3. **Calibration-set builder** `wsp.coverage`: greedy set cover over unseen pages so that every character of the target material occurs at least k times; audit that no line is in the subject's or the donors' training data; export for manual correction (eScriptorium or Transkribus) and re-import of the corrected lines; record the correction time.
4. **Benchmark** `docs/playbook/BENCH.md`: tasks T1 (planted defects: the 11 existing plus new ones with unrelated donors), T2 (stabilisation on the Track C pairs), T3 (adaptation to new material with a calibration set of 0, 30, 60, 120 lines, coverage design against random lines).
5. **Evaluation**: arms = no playbook, playbook, playbook plus calibration feedback, and the non-LLM baselines (rule list, coordinate search, exchange grid). Models: Claude Opus, Sonnet and Haiku; at least two open models of 8B parameters or less served on the shared card within its free memory (and the Qwen base of the adapter track if it fits); any further provider only if credentials exist. Record model id, date, effort and tokens.
6. **Report** `docs/PLAYBOOK_REPORT.md` and a section for the article (Section 9.3 or a follow-up paper).

## Rules and stop gates

- Calibration, practice and held-out lines are unseen by the subject and by every donor; audited before use. The playbook is written without reading held-out results; held-out sets are scored once per session.
- All intervals page-clustered. Effects below about one point of CER are reported as inconclusive.
- Edited models stay in the cache, are never registered or served; promotion is a human decision (compare the abbreviation-sign cost found for the R4 candidate).
- CPU for the benchmark; GPU only the shared card 0, within its free memory, never the trainer's card.
- Gate 1 (after the playbook and the tool layer): the playbook arm must repair at least the planted defects that the existing programmer repaired, with one SOTA model. If not, stop and fix the playbook.
- Gate 2 (after T1 and T2): if the playbook does not beat the no-playbook arm for at least one small model, or does not beat the non-LLM baselines on T2, report H0 and stop before T3.
- Gate 3: T3 needs the calibration set; Tobias or a project member supplies the manual corrections (estimate the effort first from the 26 to 66 line measurement above).

## Dependencies and risks

Depends on the merged branches (controls, seed null, zoo survey), Track C code, and, for small models, free GPU memory. Risks: gains stay near zero; the playbook overfits our two families; small models fail at code (measure the share of valid programs); API models change over time (pin ids, keep the transcripts); manual feedback costs time.

## Acceptance

A report that states, for each of H1 to H3 and with page-clustered intervals, whether the playbook helped, for which models, against which baselines, at what token and manual-correction cost, and the playbook file itself.

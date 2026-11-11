# Article for JDMDH: status and open items

State: 7 November 2026, late evening. Draft 3: `manuscript.md` (about 16,100 words without the reference list, 12 `[TODO]`; build with `./build.sh`). Draft 1 is `manuscript_v1_draft.md`.
Title: *What can be read from, and edited in, the weights of text recognition models? Readings and edits on a family of kraken models, and what they showed about our data*.

Draft 3 adds the experiments the internal review (`review/internal_review_2026-11-07.md`) and the fact-check (`review/factcheck_2026-11-07.md`) asked for: seed null (6.3), the 43-model comparison (6.4), planted defects with diffuse and unrelated-donor cases (7.6), practice sets that are not training data with six programmers (7.7), sham and averaging controls (7.8), global interpolation (7.9), block exchange and the `latin_incunabula` candidate (7.10), margins at deleted frames (8.2). Source reports: `docs/SEED_NULL.md`, `docs/ZOO_SURVEY.md`, `docs/MODEL_FAMILY.md`, `docs/CONTROLS_REPORT.md`, `docs/RESULTS.md` R4-R8.

## How the nine-part narrative maps to the draft

| # | Narrative part | Section | Status |
|---|---|---|---|
| 1 | Text recognition is established | 1 | drafted; check claims about each platform with the authors |
| 2 | Variety of models, many with kraken, even in the standard setup | 2 | drafted; 21 of 30 downloaded public models use the standard stack (zoo survey) |
| 3 | LLM can deal with weights ("white-boxing") | 3 | drafted; wording of "white box" to be agreed |
| 4 | Research questions: from comparison to adaptation | 4 | drafted |
| 5 | Existing research on explainability of text recognition | 5 | drafted; OpenAlex queries Q4-Q10 incomplete (quota), logged in `docs/SEARCH_LOG.md` |
| 6 | Comparison of models, with intermediate results | 6 | drafted; 43 models, seed null |
| 7 | Adaptability of weights, with intermediate results | 7 | drafted; controls run except C4 retrain |
| 8 | Both parts combined | 8 | drafted; margins measured |
| 9 | What we learned about the inner workings of LLMs | 9 | scoped down; 9.3 now reports the adapter track (branch `claude/adapters-4d972b`, `docs/ADAPTER_REPORT.md`, merged into main on 8 November) |

## Two interpretations to confirm

1. **Part 3, "white-boxing LLMs"**: read as using a language model to open up neural networks instead of evaluating them from outside. If opening up the LLMs themselves was meant, parts 3 and 9 change.
2. **Part 9**: the experiments study text recognition networks; the only language model is the programmer. Section 9.3 says so. Part 9 supports claims about LLM internals only if the adapter track is run.

## Summary of the results the article reports

- Readings: traces of fine-tuning; two survive a seed null (blank bias relative to other biases, rare rows); lineage recoverable in the zoo for fine-tunes that did not move far; front end generic, recurrent stack individual.
- Edits: 15 proposed edits in two sessions on training lines, no held-out gain; repeated on unseen practice lines one small gain (-0.19 to -0.39 points), an interpolation toward a sibling, not matched by 20 shams, page-clustered intervals include 0.
- C4 retrain on corrected data (branch `claude/c4-retrain`, merged): -0.55 (1e-5) and -0.99 points (5e-5) held-out, Hertwig lines only; at least as good as the best edit (Section 7.11).
- Positive control: repairs of localised defects by programmer, blind programmer and rule list alike; diffuse damage only by the programmer who measures; unrelated donor: no rebuilding.
- Two candidates from systematic block exchange (`latin_incunabula`): both read held-out Fraktur better (-0.2 to -2.0 points) but cost abbreviation signs or the own collection; neither qualifies under the pre-registered rule (RESULTS R9); none promoted.
- Data findings: a dataset export that paired 44 of 50 pages with wrong images (repaired); a held-out set that was half a printed book (replaced); 39 of 59 validation pages of the Reichenau incunable dataset are byte-identical to train pages.

## Decisions needed from you

1. Authors and corresponding author; wording on AI assistance (Section Declarations).
2. Data and code availability: repository and datasets are private; what is published, under which licence, where (Zenodo / HAL / GitHub).
3. C4 and the adapter track are done; whether to add the control on uncorrected pairs.
4. Spell out BLB, holding institutions, archival signatures (Section 2.2).
5. Length: the main text is about twice the 7,000 words an editor would expect; cut or split (data findings paper, harness report).
6. The playbook epic (`docs/epics/EPIC_LLM_PLAYBOOK.md`) is a candidate follow-up paper, not part of this one.

## How to submit (from the journal's pages, read on 7 November 2026)

- JDMDH accepts a PDF prepared in Word or LaTeX. Its template is used only after acceptance.
- Deposit the manuscript as a preprint (not as a journal article) in HAL, arXiv or Zenodo,
  create an Episciences account, then submit the archive identifier at jdmdh.episciences.org.
- Single-blind review: reviewer names are not shown to authors. A chief editor decides within
  5 days, appoints at least two reviewers within 7 days, and reviewers have 30 days.
- Original work only; an extended version of a workshop paper is allowed. English.
- Datasets must be described with access conditions. Orcid is strongly requested.
- No charges. Articles are CC-BY 4.0.
- There is a "Communications - Reports" route for software, datasets and large projects (short
  articles); not the right fit for this manuscript, but the harness itself could be a report.

## Build

`PANDOC=/path/to/pandoc>=2.11 ./build.sh` writes `build/manuscript.docx` and `build/manuscript.tex`.
No TeX and no SVG converter are installed on this server, so figures stay as SVG links in the Word
build; convert them (`rsvg-convert`) on a machine that has the tools, or let `build.sh` do it there.

## Open TODO markers (11)
Authors; BLB and archival signatures; the 5,441 versus 5,243 line count (trainer to answer); "many public models use the default setup" (now supported by the survey, wording to adjust); data and code availability (three places); AI statement confirmation; competing interests; funding; acknowledgements. Resolved in draft 3: the 9.55 % duplicate (two different line sets, 9.554 % and 9.551 %, coincidence), zoo survey, blank bias relative form, positive control, validation-based practice.

## Update 11 Nov 2026 (CET)

- Second-instance fact-check done (`paper/factcheck_draft3.md`); the three high and most medium findings are fixed in the manuscript (7.7 page intervals, RQ2 retraining sentence, latin v3 in 7.10/abstract/RQ2/conclusion, 7.8 step wording, table cross-references, k11 row, donor-rule wording, Appendix B rows, 43-vs-30 wording).
- Folded in: R13 (out-of-domain exchange, 7.10) and the phase-5 adapter session (9.3, abstract, conclusion).
- Still open from the fact-check: "0 of 28 edits" vs 27 in run logs; the low items 14-18 (page count 58 done; "three earlier statements" count, "six" vs "seven" attempts, 3-5 minutes, sd vs range, 9.3 "late layers"); the fact-check did not cover sections 1-6, 7.1-7.5, 8.1, 8.3, 10 and the appendices apart from B.
- Length is now about 19,100 words (target about 7,000).

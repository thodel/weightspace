# Article for JDMDH: status and open items

State: 11 November 2026.

- **Draft:** draft 3 of `manuscript.md`, about 18,200 words with front matter and without the reference list, 6 `[TODO]`; build with `./build.sh`. Draft 1 is `manuscript_v1_draft.md`.
- **Fact-check:** a second-instance fact-check of draft 3 (`factcheck_draft3.md`, 10 November) found five main problems, all fixed in `24d2102`:
  - page intervals in Table 10;
  - the RQ2 sentence on retraining;
  - `latin_incunabula` v3, missing from 7.10, the abstract, RQ2 and the conclusion;
  - the step wording in 7.8;
  - the Fable numbers, now on `main`.
- **Not yet in the text:** the phase-5 adapter session (Opus 5.5, merged) and its Fable 5.1 replicate (#25, open). Section 9.3 reports only the first adapter track.
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
| 7 | Adaptability of weights, with intermediate results | 7 | drafted; all controls run, C4 in 7.11, v3 in 7.10 |
| 8 | Both parts combined | 8 | drafted; margins measured |
| 9 | What we learned about the inner workings of LLMs | 9 | scoped down; 9.3 reports the first adapter track (`docs/ADAPTER_REPORT.md`); the phase-5 sessions are not yet in it |

## Two interpretations to confirm

1. **Part 3, "white-boxing LLMs"**: read as using a language model to open up neural networks instead of evaluating them from outside. If opening up the LLMs themselves was meant, parts 3 and 9 change.
2. **Part 9**: the experiments study text recognition networks; the only language model is the programmer. Section 9.3 says so. Part 9 supports claims about LLM internals only if the adapter track is run.

## Summary of the results the article reports

- Readings: traces of fine-tuning; two survive a seed null (blank bias relative to other biases, rare rows); lineage recoverable in the zoo for fine-tunes that did not move far; front end generic, recurrent stack individual.
- Edits: 15 proposed edits in two sessions on training lines, no held-out gain; repeated on unseen practice lines one small gain (-0.19 to -0.39 points), an interpolation toward a sibling, not matched by 20 shams, page-clustered intervals include 0.
- C4 retrain on corrected data (branch `claude/c4-retrain`, merged): -0.55 (1e-5) and -0.99 points (5e-5) held-out, Hertwig lines only; at least as good as the best edit (Section 7.11).
- Positive control: repairs of localised defects by programmer, blind programmer and rule list alike; diffuse damage only by the programmer who measures; unrelated donor: no rebuilding.
- Two candidates from systematic block exchange (`latin_incunabula` v1, v2): both read held-out Fraktur better (-0.2 to -2.0 points) but cost abbreviation signs or the own collection; neither qualifies under the pre-registered rule (RESULTS R9). v3 below.
- Fable 5.1 on the validation sessions (7.7): `xix-blb-v1` 0 of 9 edits kept; `combined-v1` half-way toward the sibling, practice 13.86 → 11.60, held-out −0.02.
- `latin_incunabula` v3 (RESULTS R12, now in 7.10): output layer moved half-way without the abbreviation rows. NewsEye −1.66 [−2.02, −1.30], MUFI share +0.014. It qualifies under a rule with minimum effects, which was set after R9's numbers. Serving path not worse. Registered and disabled.
- Exchange on out-of-domain material (R13): first-convolution exchange −0.35 to −1.07 points held-out, below the preregistered minimum. No exchange across lineages.
- Phase-5 adapter session (`docs/PHASE5_REPORT.md`, not yet in 9.3): one attention direction, practice −10.45 points, 733 held-out pages +0.24 [−3.5, +3.9]. A Fable 5.1 replicate kept the same edit with byte-identical readings, and one of three random shams passes the keep rule (#25).
- Data findings: a dataset export that paired 44 of 50 pages with wrong images (repaired); a held-out set that was half a printed book (replaced); 39 of 59 validation pages of the Reichenau incunable dataset are byte-identical to train pages.

## Decisions needed from you

1. Authors and corresponding author; wording on AI assistance (Section Declarations).
2. Data and code availability: repository and datasets are private; what is published, under which licence, where (Zenodo / HAL / GitHub).
3. C4, the adapter track and the phase-5 adapter sessions are done. Open: whether to add the control on uncorrected pairs, and whether 9.3 reports the phase-5 sessions.
4. Spell out BLB, holding institutions, archival signatures (Section 2.2).
5. Length: the main text (about 18,000 words) is more than twice the 7,000 words an editor would expect; cut or split (data findings paper, harness report).
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

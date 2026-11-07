# Article for JDMDH: status and open items

State: 7 November 2026, evening. Draft 2: `manuscript.md` (about 11,400 words including appendices and 18 `[TODO]`; build with `./build.sh`). Draft 1 is `manuscript_v1_draft.md`.
Working title: *What can be read from, and edited in, the weights of text recognition models? A pre-specified negative result with kraken, and what it showed about our data*.

**Draft 2 follows an internal review (`review/internal_review_2026-11-07.md`, recommendation: major revision as it stood) and a fact-check (`review/factcheck_2026-11-07.md`, 12 wrong / 17 overstated / 5 unsupported claims, all corrected or bounded in draft 2).** Main changes: part 9 retitled and scoped (no claim about LLM internals); the programmer is described as a single-shot code generator; "held jointly" and "interactions dominate" withdrawn; observation about conv versus recurrent withdrawn; six in-house models plus the public parent; data-quality material moved to appendices; protocol changes tabulated.

## How the nine-part narrative maps to the draft

| # | Narrative part | Section | Status | What it rests on | What is missing |
|---|---|---|---|---|---|
| 1 | Text recognition is established | 1 | drafted | Transkribus, eScriptorium, kraken, PyLaia, HTR-United, CATMuS, JDMDH special issue | check the claims about each platform with the authors |
| 2 | Variety of models, many with kraken, even in the standard setup | 2 | drafted | kraken 5 paper (48 models, 18 writing systems), gateway lists 42 third-party models, all seven in-house models share one architecture | a count of how many public models use the default architecture: needs the zoo survey (6.5) |
| 3 | LLM can deal with weights ("white-boxing") | 3 | drafted | the weight-space programming post (claims checked against the post) | none; wording of "white box" to be agreed |
| 4 | Research questions: from comparison to adaptation | 4 | drafted | the plan and protocol | none |
| 5 | Existing research on explainability of text recognition networks | 5 | drafted | `docs/RELATED_WORK.md`, 75 references in Zotero | rerun of the incomplete OpenAlex queries; read the 14 papers that were abstract-only |
| 6 | Comparison of models (kraken model zoo), with intermediate results | 6 | **partly** | five lineage pairs of the in-house models, baselines, drift figures | **the zoo-wide survey is not done** (brief written: `docs/BRIEF_ZOO_SURVEY.md`) |
| 7 | Adaptability of weights, with intermediate results | 7 | drafted | alpha sweep, two sessions, rewrite, collapse analysis | controls C2 (average), C4 (retrain), C5 (sham), C6 (blind) not run |
| 8 | Both parts combined | 8 | drafted | predicted-against-measured tables | strengthened once C2/C4 exist |
| 9 | What we learned about the inner workings of LLMs | 9 | **scoped down** | see below | the adapter track (QLoRA on Qwen-class models) has not been run |

## Two interpretations I made that you should confirm

1. **Part 3, "white-boxing LLMs".** I read it as using a language model to open up neural
   networks (white box) instead of evaluating them from outside (black box). If you meant
   opening up the LLMs themselves, part 3 and part 9 change.
2. **Part 9.** The experiments so far study text recognition networks. The only language
   model is the programmer. The draft says so (Section 9.3) and does not claim anything about
   LLM internals. Part 9 can only support such claims if the adapter track is run, or if you
   want it read as "what we learned about how an LLM works as a reader of weights"
   (Section 9.2, already drafted).

## Honest summary of the results the article would report

- Reading weights: informative about how fine-tuning moved a model, no repairable defect found.
- Editing weights: 15 proposed edits in two preregistered sessions, 3 kept, no held-out gain
  (differences -0.010 and +0.15 points, intervals contain 0).
- One edit that works: the lossless removal of 11,054,080 gradient-free weights.
- Data findings: a dataset export that paired 44 of 50 training pages with the wrong images
  (repaired), and a handwriting held-out set that was half a printed book (replaced).
- A mostly negative result for weight editing as an improvement method, a positive result
  for the protocol. Framing options are in the reviewer notes if they have been added.

## Decisions needed from you

1. Authors and corresponding author; whether the programmers (two Claude sessions) are
   described as a tool (the draft does) and what the journal's policy on AI assistance requires.
2. Data and code availability: the repository is private, the datasets are private. Which may be
   published, under what licence, and where (Zenodo / HAL / GitHub).
3. Whether to run the zoo survey (cheap, read-only, no forward pass) before submission. I recommend it:
   it is the only way to deliver part 6 as you described it.
4. Whether to run the missing controls C2 (average of siblings) and C4 (retrain on corrected data;
   needs an OK for a GPU job) and the adapter track. Without C2 and C4 a reviewer cannot tell
   whether the negative result is specific to weight editing.
5. Spelling out BLB, the holding institutions and the archival signatures (Section 2.2).

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

## What a JDMDH reviewer will still ask for (from the internal review)

1. A positive control (planted defects) and a validation-based practice set: both cheap, briefed in
   `docs/BRIEF_ARTICLE_EXPERIMENTS.md`. Without them the negative result cannot be interpreted.
2. The zoo survey (part 6 as you described it). Briefed in `docs/BRIEF_ZOO_SURVEY.md`.
3. Open data and code: a tagged public release (Zenodo), prompts, line lists with hashes, ideally the model files.
4. Length: cut to about 7,000 words in the main text; the appendices already take the forensic detail.
5. Humanities payoff: the article now says plainly that the data defects were found by ordinary QA, not by weight
   reading. Decide whether to lead with the data findings or with the zoo survey.

## Open TODO markers in the draft (18)
Authors; BLB and archival signatures; the 5,441 versus 5,243 line count; the claim about "many public models use the default
setup"; CPU confirmation; the blank bias relative to other biases; the zoo survey; the adapter track; the 9.55 % duplicate;
data and code availability (three places); the AI statement; competing interests; funding; acknowledgements; the pointer
to this file.

# Reading notes behind docs/RELATED_WORK.md

Raw per-paper notes from the literature review of 06.11.2026. One file per
reading batch, one section per paper, each with the same fields: model
analysed, question asked, assessment method, evidence that the method measures
what it claims, main finding, limits, relevance to weight-space edits.

**Provenance.** The notes were written by LLM reading agents working from the
open full text (pdftotext extracts of arXiv, HAL, ACL Anthology, CVF and
DergiPark copies). Verbatim quotes are marked with section or figure numbers;
the agents were told to mark anything they could not check as `UNVERIFIED` and
their own inferences as `(mine)`. The lead author spot-checked a sample of
numbers and quotes against the PDFs (see docs/RELATED_WORK.md, section 8) but
did not re-read every paper. Treat the notes as a finding aid, and check a
number against the paper before it goes into the manuscript.

**Access.** Papers behind a paywall or a bot challenge were not retrieved by
circumventing it. Their sections say "abstract only" or "metadata only".

| File | Batch | Subject |
|---|---|---|
| notes_A | first round | CTC and HTR analysis (six papers) |
| notes_B | first round | scene-text language prior (five papers) |
| notes_C | first round | VLM OCR heads and LRP (four papers) |
| notes_D | systematic search | HTR network analysis |
| notes_E | systematic search | VLM OCR, language prior, model editing |
| notes_F | systematic search | CTC behaviour and DH engines |
| notes_G | systematic search | explainable OCR and earlier unread papers |
| notes_H | literal queries | n-gram models, 2D-LSTM, Greek OCR, evaluation survey |
| notes_I | systematic search | unlearning, internal LM estimation, remaining candidates |
| notes_J | venue pass | recurrence ablation, adversarial text images |

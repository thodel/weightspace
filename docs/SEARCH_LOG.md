# Literature search log

Search run on 06.11.2026 (CET evening; log times in the tables below are UTC).
The purpose was to find prior work on analysing and explaining neural networks
for text recognition, and on how that work assessed what the network does with
text. Result summary and synthesis: [RELATED_WORK.md](RELATED_WORK.md).

## 1. Databases

| Source | How it was queried | Outcome |
|---|---|---|
| OpenAlex | REST API, relevance search (top 300 per query) and exact boolean filter on title and abstract | Filter retrieval complete for Q1 to Q3 only. Q4 stopped at 2,200 of 2,815 records and Q5 to Q10 failed, because the anonymous credit allowance ran out (HTTP 429, retry after about 2.5 h). For Q5 to Q10 only the relevance-ranked top 300 exist. |
| arXiv | API, abstract field, up to 1,000 per query | Complete for all ten queries. Plus the five phrases from the brief and the three engine names verbatim, all fields. |
| ACL Anthology | Full bibliography with abstracts (139,547 records) downloaded and evaluated locally | Complete. Few hits, as expected for an NLP venue index. |
| dblp | Full XML dump (4,007,530 records), evaluated locally on titles only, after the search API turned out to sit behind a bot challenge that was not bypassed | Complete on titles. Covers ICDAR, ICFHR, DAS, ICPR, IJDAR. No abstracts. |
| Crossref | REST API, 56 domain-by-method keyword pairs, 100 results each | Done. Relevance ranking only; abstracts mostly absent. |
| HAL | REST API, six queries | Done. |
| Zenodo | REST API, nine queries, first 50 hits each | Done but low yield: free-text search returns very large result sets. |
| Semantic Scholar | Bulk search API, anonymous | Failed. Rate limit (HTTP 429) for every query; no records retrieved. |
| IEEE Xplore | not queried | Blocked for automated access. IEEE proceedings were reached through dblp, Crossref and OpenAlex only. |
| Google Scholar | not queried | Blocked for automated access. |
| Scopus, Web of Science | not queried | No access from this machine. |

## 2. Queries

Ten boolean queries, written from the brief. Q1 to Q5 correspond to the five
phrases (HTR explainability, CTC attribution, text-line probing, OCR
interpretability, layer-wise relevance and CTC). Q6 and Q7 cover the DH
literature on kraken, PyLaia, Transkribus and eScriptorium. Q8 to Q10 were
added: implicit language model in HTR, weight editing or merging of OCR
models, and mechanistic analysis of OCR.

| Id | Query |
|---|---|
| Q1 | (handwriting OR handwritten OR HTR) AND (explainability OR explainable OR interpretability OR interpretable OR saliency OR attribution OR "relevance propagation" OR Grad-CAM) |
| Q2 | (CTC OR "connectionist temporal classification") AND (attribution OR saliency OR "integrated gradients" OR explanation OR explainability OR interpretability OR "relevance propagation" OR Grad-CAM) |
| Q3 | ("text line" OR "line recognition" OR "handwritten text recognition" OR OCR OR "text recognition") AND (probing OR "probing classifier" OR "linear probe" OR "hidden representations" OR "layer-wise analysis" OR "what does the network learn") |
| Q4 | (OCR OR "optical character recognition" OR "text recognition") AND (interpretability OR explainability OR mechanistic OR attribution OR "explain") |
| Q5 | ("layer-wise relevance propagation" OR LRP) AND (CTC OR "speech recognition" OR OCR OR handwriting OR "text recognition" OR "sequence-to-sequence") |
| Q6 | (kraken OR PyLaia OR Transkribus OR eScriptorium) AND (analysis OR "error analysis" OR interpretability OR visualization OR evaluation OR "model analysis") |
| Q7 | (HTR OR "handwritten text recognition") AND (historical OR manuscript OR archival) AND (explainability OR interpretability OR "error analysis" OR "model analysis" OR visualization OR "what the model") |
| Q8 | (HTR OR "handwritten text recognition" OR "handwriting recognition") AND ("implicit language model" OR "language prior" OR "context") AND (analysis OR ablation OR probing) |
| Q9 | ("model merging" OR "weight averaging" OR "task arithmetic" OR "model editing" OR "weight interpolation" OR "model soup") AND (OCR OR HTR OR handwriting OR "text recognition") |
| Q10 | (OCR OR "text recognition") AND ("attention heads" OR circuit OR "sparse autoencoder" OR "activation patching" OR "causal intervention" OR "neuron") |

Per-source counts, with the exact query string as sent, are in section 6.

## 3. Screening funnel

| Stage | Records |
|---|---:|
| Retrieved, all sources | 15,088 |
| After removing duplicates (DOI, arXiv id, normalised title) | 12,832 |
| Pass the automatic first screen (text-recognition term and an analysis term in title or abstract) | 5,571 |
| Top tier of the first screen (score 6 or more), titles reviewed by hand | 343 |
| Focused title list (text-recognition term and analysis wording in the title), all reviewed | 486 |
| Abstract-based list (strong method term in the abstract, plain title), all reviewed | 153 |
| ICDAR, ICFHR, DAS, IJDAR and similar venues not shown before, all reviewed | 106 |
| DH-venue list (CHR, JDMDH, DH conferences, HAL), all reviewed | 56 |
| arXiv, literal phrases from the brief, records not already found | 148 |
| Papers examined individually (shortlisted, then read) | 72 |
| of which read from open full text (method and evaluation sections) | 58 |
| of which read at abstract level only (paywalled or no open copy) | 12 |
| of which bibliographic metadata only | 2 |

The 58 full-text readings are 15 from the first round and 43 from the
systematic pass (38 new references, one already in the library, and four papers
listed earlier but not yet read). Some of the 58 were read in part only: one
benchmark paper skimmed, one book chapter grepped, and a few preprints whose
journal versions could not be compared.

**What was not covered.** A broad lower tier of 1,550 records (text-recognition
term in the title, generic analysis words anywhere) was reviewed only for its
first 230 entries. The focused title list and the abstract-based list were
designed to cover the same ground more strictly, so the unreviewed remainder
contains records with a text-recognition term in the title and only generic
words such as "layer" or "error" in the abstract. It is dominated by noise
(nuclear-reactor HTR, handwriting psychology, medical handwriting studies), but
it has not been looked at. Records matching a regular expression of medical,
forensic and attribution-of-authorship terms were dropped from the top tier
without being shown (43 of 168 in its second half).

## 4. Reading

Shortlisted papers were assigned to reading agents in ten batches (A to C in
the first round, D to J in the systematic pass). Each agent retrieved the open full text,
read the method and evaluation parts, and filled the same template. The
agents did not bypass paywalls, logins or bot challenges. The notes are in
[related_work/notes](related_work/notes/README.md); each reference in the Zotero
collection carries a child note with the same text.

## 5. Known limitations of the search

- Four databases that would normally be searched (IEEE Xplore, Google Scholar,
  Scopus, Semantic Scholar) are missing. Journal papers in Elsevier and Springer
  venues were found through OpenAlex, Crossref and dblp, so recall for them
  depends on those indexes.
- OpenAlex queries Q4 to Q10 are incomplete (section 1). They should be rerun
  once the credit allowance resets, ideally with a free API key.
- dblp was matched on titles only; a paper whose title does not mention text
  recognition would be missed there.
- Keyword screening favours English wording. Non-English DH literature (French,
  German, Spanish) is covered only through HAL.
- A paper that the search did not find cannot appear in the review. The claim
  in RELATED_WORK.md section 6 is therefore bounded by this search.

## 6. Query tables

**OpenAlex, fuzzy relevance search (top 300 per query)**

| Query | Query string | Hits (total) | Retrieved |
|---|---|---:|---|
| Q1_htr_explainability | `(handwriting OR handwritten OR HTR) AND (explainability OR explainable OR interpretability OR interpretable OR saliency OR attribution OR "relevance propagation" OR Gr...` | 209838 | 300 |
| Q2_ctc_attribution | `(CTC OR "connectionist temporal classification") AND (attribution OR saliency OR "integrated gradients" OR explanation OR explainability OR interpretability OR "releva...` | 116771 | 300 |
| Q3_line_probing | `("text line" OR "line recognition" OR "handwritten text recognition" OR OCR OR "text recognition") AND (probing OR "probing classifier" OR "linear probe" OR "hidden re...` | 23566 | 300 |
| Q4_ocr_interpretability | `(OCR OR "optical character recognition" OR "text recognition") AND (interpretability OR explainability OR mechanistic OR attribution OR "explain")` | 91797 | 300 |
| Q5_lrp_ctc | `("layer-wise relevance propagation" OR LRP) AND (CTC OR "speech recognition" OR OCR OR handwriting OR "text recognition" OR "sequence-to-sequence")` | 3080 | 300 |
| Q6_dh_engines | `(kraken OR PyLaia OR Transkribus OR eScriptorium) AND (analysis OR "error analysis" OR interpretability OR visualization OR evaluation OR "model analysis")` | 16998 | 300 |
| Q7_htr_historical_explain | `(HTR OR "handwritten text recognition") AND (historical OR manuscript OR archival) AND (explainability OR interpretability OR "error analysis" OR "model analysis" OR v...` | 14237 | 300 |
| Q8_htr_implicit_lm | `(HTR OR "handwritten text recognition" OR "handwriting recognition") AND ("implicit language model" OR "language prior" OR "context" ) AND (analysis OR ablation OR pro...` | 23838 | 300 |
| Q9_weight_editing_ocr | `("model merging" OR "weight averaging" OR "task arithmetic" OR "model editing" OR "weight interpolation" OR "model soup") AND (OCR OR HTR OR handwriting OR "text recog...` | 1065 | 300 |
| Q10_mechanistic_ocr | `(OCR OR "text recognition") AND ("attention heads" OR circuit OR "sparse autoencoder" OR "activation patching" OR "causal intervention" OR "neuron")` | 27851 | 300 |

**OpenAlex, exact boolean filter on title and abstract**

| Query | Query string | Hits (total) | Retrieved |
|---|---|---:|---|
| Q1_htr_explainability | `(handwriting OR handwritten OR HTR) AND (explainability OR explainable OR interpretability OR interpretable OR saliency OR attribution OR "relevance propagation" OR Gr...` | 4954 | 4954 |
| Q2_ctc_attribution | `(CTC OR "connectionist temporal classification") AND (attribution OR saliency OR "integrated gradients" OR explanation OR explainability OR interpretability OR "releva...` | 1425 | 1425 |
| Q3_line_probing | `("text line" OR "line recognition" OR "handwritten text recognition" OR OCR OR "text recognition") AND (probing OR "probing classifier" OR "linear probe" OR "hidden re...` | 296 | 296 |
| Q4_ocr_interpretability | `(OCR OR "optical character recognition" OR "text recognition") AND (interpretability OR explainability OR mechanistic OR attribution OR "explain")` | 2815 | 2200 (partial: rate limit hit) |
| Q5_lrp_ctc | `("layer-wise relevance propagation" OR LRP) AND (CTC OR "speech recognition" OR OCR OR handwriting OR "text recognition" OR "sequence-to-sequence")` | n/a | 0 (FAILED: anonymous rate limit exhausted; only the relevance-ranked top 300 from the fuzzy run exist) |
| Q6_dh_engines | `(kraken OR PyLaia OR Transkribus OR eScriptorium) AND (analysis OR "error analysis" OR interpretability OR visualization OR evaluation OR "model analysis")` | n/a | 0 (FAILED: anonymous rate limit exhausted; only the relevance-ranked top 300 from the fuzzy run exist) |
| Q7_htr_historical_explain | `(HTR OR "handwritten text recognition") AND (historical OR manuscript OR archival) AND (explainability OR interpretability OR "error analysis" OR "model analysis" OR v...` | n/a | 0 (FAILED: anonymous rate limit exhausted; only the relevance-ranked top 300 from the fuzzy run exist) |
| Q8_htr_implicit_lm | `(HTR OR "handwritten text recognition" OR "handwriting recognition") AND ("implicit language model" OR "language prior" OR "context" ) AND (analysis OR ablation OR pro...` | n/a | 0 (FAILED: anonymous rate limit exhausted; only the relevance-ranked top 300 from the fuzzy run exist) |
| Q9_weight_editing_ocr | `("model merging" OR "weight averaging" OR "task arithmetic" OR "model editing" OR "weight interpolation" OR "model soup") AND (OCR OR HTR OR handwriting OR "text recog...` | n/a | 0 (FAILED: anonymous rate limit exhausted; only the relevance-ranked top 300 from the fuzzy run exist) |
| Q10_mechanistic_ocr | `(OCR OR "text recognition") AND ("attention heads" OR circuit OR "sparse autoencoder" OR "activation patching" OR "causal intervention" OR "neuron")` | n/a | 0 (FAILED: anonymous rate limit exhausted; only the relevance-ranked top 300 from the fuzzy run exist) |

**arXiv API, abstract field**

| Query | Query string | Hits (total) | Retrieved |
|---|---|---:|---|
| Q1_htr_explainability | `( abs:handwriting OR abs:handwritten OR abs:HTR ) AND ( abs:explainability OR abs:explainable OR abs:interpretability OR abs:interpretable OR abs:saliency OR abs:attri...` | 200 | 200 |
| Q2_ctc_attribution | `( abs:CTC OR abs:"connectionist temporal classification" ) AND ( abs:attribution OR abs:saliency OR abs:"integrated gradients" OR abs:explanation OR abs:explainability...` | 92 | 92 |
| Q3_line_probing | `( abs:"text line" OR abs:"line recognition" OR abs:"handwritten text recognition" OR abs:OCR OR abs:"text recognition" ) AND ( abs:probing OR abs:"probing classifier" ...` | 31 | 31 |
| Q4_ocr_interpretability | `( abs:OCR OR abs:"optical character recognition" OR abs:"text recognition" ) AND ( abs:interpretability OR abs:explainability OR abs:mechanistic OR abs:attribution OR ...` | 249 | 249 |
| Q5_lrp_ctc | `( abs:"layer-wise relevance propagation" OR abs:LRP ) AND ( abs:CTC OR abs:"speech recognition" OR abs:OCR OR abs:handwriting OR abs:"text recognition" OR abs:"sequenc...` | 7 | 7 |
| Q6_dh_engines | `( abs:kraken OR abs:PyLaia OR abs:Transkribus OR abs:eScriptorium ) AND ( abs:analysis OR abs:"error analysis" OR abs:interpretability OR abs:visualization OR abs:eval...` | 29 | 29 |
| Q7_htr_historical_explain | `( abs:HTR OR abs:"handwritten text recognition" ) AND ( abs:historical OR abs:manuscript OR abs:archival ) AND ( abs:explainability OR abs:interpretability OR abs:"err...` | 14 | 14 |
| Q8_htr_implicit_lm | `( abs:HTR OR abs:"handwritten text recognition" OR abs:"handwriting recognition" ) AND ( abs:"implicit language model" OR abs:"language prior" OR abs:"context" ) AND (...` | 4 | 4 |
| Q9_weight_editing_ocr | `( abs:"model merging" OR abs:"weight averaging" OR abs:"task arithmetic" OR abs:"model editing" OR abs:"weight interpolation" OR abs:"model soup" ) AND ( abs:OCR OR ab...` | 11 | 11 |
| Q10_mechanistic_ocr | `( abs:OCR OR abs:"text recognition" ) AND ( abs:"attention heads" OR abs:circuit OR abs:"sparse autoencoder" OR abs:"activation patching" OR abs:"causal intervention" ...` | 23 | 23 |

**arXiv API, your five query phrases and kraken/PyLaia/Transkribus verbatim (all fields)**

| Query | Query string | Hits (total) | Retrieved |
|---|---|---:|---|
| L1 HTR explainability | `all:HTR AND all:explainability` | 1 | 1 |
| L2 CTC attribution | `all:CTC AND all:attribution` | 26 | 26 |
| L3 text line recognition probing | `all:"text line" AND all:recognition AND all:probing` | 0 | 0 |
| L4 OCR interpretability | `all:OCR AND all:interpretability` | 128 | 100 |
| L5 layer-wise relevance CTC | `all:"layer-wise relevance" AND all:CTC` | 0 | 0 |
| L6 kraken HTR | `all:kraken AND all:HTR` | 1 | 1 |
| L7 PyLaia | `all:PyLaia` | 3 | 3 |
| L8 Transkribus | `all:Transkribus` | 11 | 11 |
| L9 handwritten text recognition interpretability | `all:"handwritten text recognition" AND all:interpretability` | 4 | 4 |
| L10 handwritten text recognition explainable | `all:"handwritten text recognition" AND all:explainable` | 3 | 3 |

**ACL Anthology (full bibliography with abstracts, evaluated locally)**

| Query | Query string | Hits (total) | Retrieved |
|---|---|---:|---|
| Q1_htr_explainability | `(handwriting OR handwritten OR HTR) AND (explainability OR explainable OR interpretability OR interpretable OR saliency OR attribution OR "relevance propagation" OR Gr...` | 6 | 6 |
| Q2_ctc_attribution | `(CTC OR "connectionist temporal classification") AND (attribution OR saliency OR "integrated gradients" OR explanation OR explainability OR interpretability OR "releva...` | 1 | 1 |
| Q3_line_probing | `("text line" OR "line recognition" OR "handwritten text recognition" OR OCR OR "text recognition") AND (probing OR "probing classifier" OR "linear probe" OR "hidden re...` | 5 | 5 |
| Q4_ocr_interpretability | `(OCR OR "optical character recognition" OR "text recognition") AND (interpretability OR explainability OR mechanistic OR attribution OR "explain")` | 16 | 16 |
| Q5_lrp_ctc | `("layer-wise relevance propagation" OR LRP) AND (CTC OR "speech recognition" OR OCR OR handwriting OR "text recognition" OR "sequence-to-sequence")` | 1 | 1 |
| Q6_dh_engines | `(kraken OR PyLaia OR Transkribus OR eScriptorium) AND (analysis OR "error analysis" OR interpretability OR visualization OR evaluation OR "model analysis")` | 3 | 3 |
| Q7_htr_historical_explain | `(HTR OR "handwritten text recognition") AND (historical OR manuscript OR archival) AND (explainability OR interpretability OR "error analysis" OR "model analysis" OR v...` | 2 | 2 |
| Q8_htr_implicit_lm | `(HTR OR "handwritten text recognition" OR "handwriting recognition") AND ("implicit language model" OR "language prior" OR "context" ) AND (analysis OR ablation OR pro...` | 0 | 0 |
| Q9_weight_editing_ocr | `("model merging" OR "weight averaging" OR "task arithmetic" OR "model editing" OR "weight interpolation" OR "model soup") AND (OCR OR HTR OR handwriting OR "text recog...` | 0 | 0 |
| Q10_mechanistic_ocr | `(OCR OR "text recognition") AND ("attention heads" OR circuit OR "sparse autoencoder" OR "activation patching" OR "causal intervention" OR "neuron")` | 2 | 2 |

**dblp (full XML dump, evaluated locally; titles only)**

| Query | Query string | Hits (total) | Retrieved |
|---|---|---:|---|
| Q1_htr_explainability | `(handwriting OR handwritten OR HTR) AND (explainability OR explainable OR interpretability OR interpretable OR saliency OR attribution OR "relevance propagation" OR Gr...` | 17 | 17 |
| Q2_ctc_attribution | `(CTC OR "connectionist temporal classification") AND (attribution OR saliency OR "integrated gradients" OR explanation OR explainability OR interpretability OR "releva...` | 0 | 0 |
| Q3_line_probing | `("text line" OR "line recognition" OR "handwritten text recognition" OR OCR OR "text recognition") AND (probing OR "probing classifier" OR "linear probe" OR "hidden re...` | 1 | 1 |
| Q4_ocr_interpretability | `(OCR OR "optical character recognition" OR "text recognition") AND (interpretability OR explainability OR mechanistic OR attribution OR "explain") [title only]` | 3 | 3 |
| Q5_lrp_ctc | `("layer-wise relevance propagation" OR LRP) AND (CTC OR "speech recognition" OR OCR OR handwriting OR "text recognition" OR "sequence-to-sequence") [title only]` | 1 | 1 |
| Q6_dh_engines | `(kraken OR PyLaia OR Transkribus OR eScriptorium) AND (analysis OR "error analysis" OR interpretability OR visualization OR evaluation OR "model analysis") [title only]` | 0 | 0 |
| Q7_htr_historical_explain | `(HTR OR "handwritten text recognition") AND (historical OR manuscript OR archival) AND (explainability OR interpretability OR "error analysis" OR "model analysis" OR v...` | 0 | 0 |
| Q8_htr_implicit_lm | `(HTR OR "handwritten text recognition" OR "handwriting recognition") AND ("implicit language model" OR "language prior" OR "context" ) AND (analysis OR ablation OR pro...` | 0 | 0 |
| Q9_weight_editing_ocr | `("model merging" OR "weight averaging" OR "task arithmetic" OR "model editing" OR "weight interpolation" OR "model soup") AND (OCR OR HTR OR handwriting OR "text recog...` | 1 | 1 |
| Q10_mechanistic_ocr | `(OCR OR "text recognition") AND ("attention heads" OR circuit OR "sparse autoencoder" OR "activation patching" OR "causal intervention" OR "neuron") [title only]` | 0 | 0 |
| Q6_dh_engines:title | `kraken` | 27 | 27 |
| Q6_dh_engines:title | `PyLaia` | 0 | 0 |
| Q6_dh_engines:title | `Transkribus` | 8 | 8 |
| Q6_dh_engines:title | `eScriptorium` | 1 | 1 |

**Crossref**

| Query | Query string | Hits (total) | Retrieved |
|---|---|---:|---|
| pair | `handwritten text recognition explainability` | 850138 | 100 |
| pair | `handwritten text recognition interpretability` | 851402 | 100 |
| pair | `handwritten text recognition saliency` | 853860 | 100 |
| pair | `handwritten text recognition attribution` | 862860 | 100 |
| pair | `handwritten text recognition relevance propagation` | 1260878 | 100 |
| pair | `handwritten text recognition probing` | 897838 | 100 |
| pair | `handwritten text recognition visualization of what the network learns` | 2848089 | 100 |
| pair | `handwritten text recognition language model analysis` | 10659980 | 100 |
| pair | `HTR explainability` | 7028 | 100 |
| pair | `HTR interpretability` | 8293 | 100 |
| pair | `HTR saliency` | 11318 | 100 |
| pair | `HTR attribution` | 19921 | 100 |
| pair | `HTR relevance propagation` | 419593 | 100 |
| pair | `HTR probing` | 54863 | 100 |
| pair | `HTR visualization of what the network learns` | 2046520 | 100 |
| pair | `HTR language model analysis` | 9940597 | 100 |
| pair | `handwriting recognition explainability` | 576766 | 100 |
| pair | `handwriting recognition interpretability` | 578030 | 100 |
| pair | `handwriting recognition saliency` | 580483 | 100 |
| pair | `handwriting recognition attribution` | 589614 | 100 |
| pair | `handwriting recognition relevance propagation` | 987997 | 100 |
| pair | `handwriting recognition probing` | 624458 | 100 |
| pair | `handwriting recognition visualization of what the network learns` | 2581328 | 100 |
| pair | `handwriting recognition language model analysis` | 10430523 | 100 |
| pair | `OCR explainability` | 11066 | 100 |
| pair | `OCR interpretability` | 12332 | 100 |
| pair | `OCR saliency` | 15358 | 100 |
| pair | `OCR attribution` | 23960 | 100 |
| pair | `OCR relevance propagation` | 423630 | 100 |
| pair | `OCR probing` | 58901 | 100 |
| pair | `OCR visualization of what the network learns` | 2050465 | 100 |
| pair | `OCR language model analysis` | 9943840 | 100 |
| pair | `text recognition explainability` | 846308 | 100 |
| pair | `text recognition interpretability` | 847573 | 100 |
| pair | `text recognition saliency` | 850031 | 100 |
| pair | `text recognition attribution` | 859033 | 100 |
| pair | `text recognition relevance propagation` | 1257056 | 100 |
| pair | `text recognition probing` | 894009 | 100 |
| pair | `text recognition visualization of what the network learns` | 2844526 | 100 |
| pair | `text recognition language model analysis` | 10656759 | 100 |
| pair | `CTC explainability` | 7696 | 100 |
| pair | `CTC interpretability` | 8961 | 100 |
| pair | `CTC saliency` | 11986 | 100 |
| pair | `CTC attribution` | 20589 | 100 |
| pair | `CTC relevance propagation` | 420257 | 100 |
| pair | `CTC probing` | 55530 | 100 |
| pair | `CTC visualization of what the network learns` | 2047152 | 100 |
| pair | `CTC language model analysis` | 9941150 | 100 |
| pair | `text line recognition explainability` | 1323675 | 100 |
| pair | `text line recognition interpretability` | 1324939 | 100 |
| pair | `text line recognition saliency` | 1327374 | 100 |
| pair | `text line recognition attribution` | 1336382 | 100 |
| pair | `text line recognition relevance propagation` | 1730665 | 100 |
| pair | `text line recognition probing` | 1371146 | 100 |
| pair | `text line recognition visualization of what the network learns` | 3313966 | 100 |
| pair | `text line recognition language model analysis` | 11090643 | 100 |

**HAL**

| Query | Query string | Hits (total) | Retrieved |
|---|---|---:|---|
| Q1_htr_explainability | `(HTR OR handwriting OR "handwritten text recognition") AND (explainability OR explainable OR interpretability OR saliency OR attribution)` | 14 | 14 |
| Q2_ctc_attribution | `(CTC OR "connectionist temporal classification") AND (attribution OR saliency OR explainability OR interpretability OR "relevance propagation")` | 6 | 6 |
| Q4_ocr_interpretability | `(OCR OR "text recognition") AND (interpretability OR explainability OR mechanistic OR attribution)` | 17 | 17 |
| Q6_dh_engines | `(kraken OR PyLaia OR Transkribus OR eScriptorium) AND (analysis OR evaluation OR interpretability OR visualization)` | 75 | 75 |
| Q7_htr_historical_explain | `(HTR OR "handwritten text recognition") AND (historical OR manuscript) AND (explainability OR interpretability OR "error analysis" OR visualization)` | 3 | 3 |
| Q8_htr_implicit_lm | `(HTR OR "handwritten text recognition") AND ("language model" OR context) AND (analysis OR ablation OR probing)` | 18 | 18 |

**Zenodo (first 50 hits per query)**

| Query | Query string | Hits (total) | Retrieved |
|---|---|---:|---|
| Q6_dh_engines | `kraken HTR model analysis` | 1236341 | 25 |
| Q6_dh_engines | `PyLaia analysis` | 859364 | 25 |
| Q6_dh_engines | `Transkribus model analysis` | 1236039 | 25 |
| Q6_dh_engines | `HTR model interpretability` | 552064 | 25 |
| Q6_dh_engines | `handwritten text recognition explainability` | 273852 | 25 |
| Q6_dh_engines | `CTC attribution handwriting` | 41660 | 25 |
| Q6_dh_engines | `OCR interpretability` | 12765 | 25 |
| Q6_dh_engines | `HTR error analysis historical` | 981348 | 25 |
| Q6_dh_engines | `handwritten text recognition probing` | 273897 | 25 |

**Semantic Scholar bulk search**

| Query | Query string | Hits (total) | Retrieved |
|---|---|---:|---|
| Q1_htr_explainability | `(bulk boolean query)` | n/a | 0 (FAILED: anonymous rate limit, HTTP 429) |
| Q2_ctc_attribution | `(bulk boolean query)` | n/a | 0 (FAILED: anonymous rate limit, HTTP 429) |
| Q3_line_probing | `(bulk boolean query)` | n/a | 0 (FAILED: anonymous rate limit, HTTP 429) |
| Q4_ocr_interpretability | `(bulk boolean query)` | n/a | 0 (FAILED: anonymous rate limit, HTTP 429) |
| Q5_lrp_ctc | `(bulk boolean query)` | n/a | 0 (FAILED: anonymous rate limit, HTTP 429) |
| Q7_htr_historical_explain | `(bulk boolean query)` | n/a | 0 (FAILED: anonymous rate limit, HTTP 429) |
| Q9_weight_editing_ocr | `(bulk boolean query)` | n/a | 0 (FAILED: anonymous rate limit, HTTP 429) |
| Q10_mechanistic_ocr | `(bulk boolean query)` | n/a | 0 (FAILED: anonymous rate limit, HTTP 429) |


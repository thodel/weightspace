# Related work: how has the handling of text by recognition networks been assessed?

Status: first pass, 06.11.2026 (CET). Search was by web search, then every
paper below was verified against arXiv or Crossref metadata (title, authors,
year, venue). Fifteen papers were read from their full text (method and evaluation
sections in detail, key numbers spot-checked against the PDFs); the other
nine are marked as read from abstract or metadata only. This is
not a systematic review: no Scopus, Semantic Scholar or ACL/IEEE database
query was run, and the digital-humanities HTR literature (Transkribus,
eScriptorium, PyLaia, kraken users) was not searched beyond a first look.
"We found no paper that…" always means "in this search".

References live in the Zotero collection *WSP – analysis & explainability of
text-recognition networks* (below *Texterkennung (HTR/OCR)*, personal
library). Every item carries a child note with the reading summary used here.
A BibTeX export is in [`refs/wsp_related_work.bib`](../refs/wsp_related_work.bib).
Citations below use its keys.

## 1. Short answer

**Is there already a paper that analyses neural networks for text
recognition?** There are many partial analyses and no single one that does
what this project does. Four lines of work exist:

- *Formal analysis of the CTC criterion* (`zeyer_why_2021`, context in
  `hannun_sequence_2017`, `huang_less_2024`).
- *Behavioural analysis by swapping decoders or modules* (`liu_study_2015`,
  `baek_what_2019`, `fang_read_2021`, `bautista_scene_2022`,
  `puigcerver_are_2017`).
- *Probing for a language prior* (`zhang_can_2026`, `da_levenshtein_2022`).
- *Mechanistic work on vision-language models that read text*
  (`baek_how_2025`, `steinberg_where_2026`, `feucht_using_2026`).

We found no paper that reads the weights of a conv-BiLSTM-CTC line recogniser,
edits them and measures the effect. All assessment is through behaviour,
architecture swaps, attribution or hidden-state interventions.

**Is there an explainability paper about text recognition?** Yes, but each
covers a narrow case.

- `buoy_explainable_2023`: explainable CTC scene-text recognition with a
  built-in association map.
- `yousef_origaminet_2020`, Sec. 4.9: Integrated Gradients on a CTC page
  recogniser.
- `da_levenshtein_2022`: attention split between visual and textual tokens.
- `coquenet_towards_2022`: attention and prediction overlays for paragraph
  HTR.
- Neighbours: `pundy_transparency_2025` (writer identification),
  `azimi_improving_2023` (online handwritten characters, not read),
  `whitten_ai_2024` (digits and letters, abstract only), `hamdan_htr-jand_2024`
  (attention analysis, snippet only).

We found no survey of explainability for text recognition and no benchmark
that tests attribution methods on text recognisers. The general survey
`wan_survey_2025` lists attention heat maps as a standard local explanation.

## 2. How the assessment was done: methods in use

| Method | Used by | What it can show | Validity evidence in the paper | Main weakness |
|---|---|---|---|---|
| Formal counting and gradient analysis of the loss | `zeyer_why_2021` | Blank dominance and spike pattern follow from the CTC topology and frames per label, not from the input | Proofs, simulations, control losses | Toy models; says nothing about a trained or edited net |
| Decode with and without an explicit language model, vary data | `liu_study_2015` | How much context a BLSTM-CTC already carries (inferred) | None on the implicit LM itself | Inference from accuracy; depends on schedule, data diversity |
| Factorial module swap | `baek_what_2019` | Accuracy contribution of each stage; BiLSTM +2.5 / +4.5 points | Fixed training, five seeds | BiLSTM gain is receptive field, not shown to be language |
| Gradient-flow and LM-alone tests | `fang_read_2021` | LM as modest corrector; fails with several wrong characters | Indirect | LM's total share never isolated |
| Mask toggle on one set of weights | `bautista_scene_2022` | Context adds about 1 point on common words | Same weights, only the mask changes | Dictionary-like benchmarks |
| Start from corrupted or empty text | `da_levenshtein_2022` | Language is used; corrupted start falls below vision-only | Construction only | Oracle start; four examples for attention |
| Rarity-stratified accuracy grid, probes, swaps | `zhang_can_2026` | Failure on rare letter combinations is a lexical snap | Bootstrap for the gap; none for the mechanism | Probe uncontrolled; own CRNN row contradicts "CTC removes it" |
| Integrated Gradients per emitted character | `yousef_origaminet_2020` | Implicit character and line localisation | Visual only | No faithfulness test; blank and repeat frames dropped |
| Built-in association map, box-hit metric | `buoy_explainable_2023` | Spike frame to image region, agrees with boxes | 446 synthetic images with character boxes | No chance baseline; lenient hit test |
| Attention and prediction overlays, one intervention | `coquenet_towards_2022` | Reading order, blank as line spacing | One duplicated-line input | One image per figure |
| Saliency plus deletion/insertion against random | `pundy_transparency_2025` | Which ink drives writer embeddings | Quantitative, per backbone | One backbone's deletion scores are below chance level by our reading of its metric |
| LRP with the gate rule | `arras_explaining_2017`, `arras_explaining_2019` | Input relevance through LSTMs | Deletion test; 2019 adds a known-truth toy task | Toy and sentiment tasks; no CTC or image case; fails with more cells |
| Head scoring and head masking | `baek_how_2025` | Which VLM heads attend to answer-text patches | Random-head control | Scoring is not causal; synthetic task |
| Residual-stream subspace removal | `steinberg_where_2026` | Depth band where OCR information is routed | Random heads, random boxes | No random-subspace control; best of 14 picked on test data |

Four observations hold across the papers we read in full.

1. **Validation is thin.** Only `pundy_transparency_2025` tests an attribution
   against a random order, `buoy_explainable_2023` against ground-truth boxes
   (without chance level), and `arras_explaining_2019` against a known answer.
   The rest is visual agreement.
2. **"Implicit language model" is asserted, not tested.** `liu_study_2015`,
   `buoy_explainable_2023` and `coquenet_towards_2022` infer it from accuracy
   gaps. `baek_what_2019` credits the BiLSTM with receptive field instead, and
   `bautista_scene_2022` calls CTC "context-free". No paper intervenes on the
   recurrent layers of a CTC network.
3. **Blank frames are set aside.** `yousef_origaminet_2020` and
   `buoy_explainable_2023` drop blank and repeat outputs. `zeyer_why_2021`
   explains why: blank dominance comes from the CTC topology, so frame-level
   assessment belongs at the spike frames.
4. **The closest evidence for our architecture points both ways.**
   `zhang_can_2026` Table 2 gives the CNN-BiLSTM-CTC model the largest
   rare-composition gap of nine English backbones (−17.9 points against −10.5
   to −13.9 for the others; checked against the PDF). It is also the weakest
   model overall, so part of that gap is plain capacity. The paper nevertheless
   argues for the autoregressive decoder as the cause. It is a single-author,
   unreviewed preprint with a confounded architecture comparison; treat its
   mechanism claim as a hypothesis.

## 3. What this means for the project

These are proposals for the pre-registered protocol (docs/PROTOCOL.md), not
edits to it. Each ties a reading from the weights to a behavioural check that
the literature shows to be missing.

1. **Read at spike frames.** Any per-frame statistic (attribution, posterior
   mass, edit effect) should be reported on non-blank best-path frames and
   separately for blank frames (`zeyer_why_2021`).
2. **Never accept an attribution without a causal check.** If an attribution
   is used to choose an edit, run ink-only deletion and insertion against a
   random order (`pundy_transparency_2025`) and a character-box hit test with a
   chance baseline (`buoy_explainable_2023`). Use a white baseline and report
   its choice (`yousef_origaminet_2020`).
3. **Test the "implicit LM" claim directly.** Run a rarity-stratified grid
   (`zhang_can_2026`) on the held-out sets before and after each kept edit, and
   a no-LM versus n-gram decoding comparison (`liu_study_2015`). Add an
   ablation of one recurrent layer or direction, the BiLSTM analogue of the
   mask toggle (`bautista_scene_2022`).
4. **Use image-side counterfactuals.** A CTC net has no text input, so the
   analogue of the corrupted-start test (`da_levenshtein_2022`) is replacing
   one glyph with a rare one and measuring the pull toward a familiar word.
5. **Build a known-truth check for any relevance method.** Synthetic lines in
   which only chosen glyphs determine an output, as in the toy task of
   `arras_explaining_2019`. Expect LRP-style rules to report locality whether
   or not distant context is used.
6. **For the adapter track** (QLoRA on Qwen-family models): per-head OCR
   scores before and after training and masking of the top heads
   (`baek_how_2025`), and removal of the text-delta subspace at the OCR depth
   band (`steinberg_where_2026`). Both depend on transformer-only units, so
   they do not transfer to the kraken stack, and both need the random controls
   the originals lack: same-rank random subspace, depth-matched random heads.

## 4. Where this project would be new, and how to say it

The defensible claim is narrow: we found no earlier work that treats a trained
text-recognition network as an object to be read and edited in weight space,
with a forward-pass budget and a held-out check. The related fields are
interpretability by behaviour and attribution (sections 1 and 2) and model
merging (docs/PAPER.md). The claim must be rechecked with a systematic search
before submission; see section 5.

## 5. Open items before this goes into the paper

- Run a systematic search (Semantic Scholar, ACL Anthology, IEEE Xplore, ICDAR
  and DAS proceedings) for: "HTR explainability", "CTC attribution", "text
  line recognition probing", "OCR interpretability", "layer-wise relevance
  CTC", and the DH literature on kraken, PyLaia and Transkribus model analysis.
- Read in full: `puigcerver_are_2017`, `hamdan_htr-jand_2024`,
  `azimi_improving_2023`, `whitten_ai_2024`, `feucht_using_2026`. They were
  listed from title, abstract or snippet only.
- Re-verify numbers before quoting. The reading notes mark items that could
  not be checked in the text as UNVERIFIED (for example head-masking
  implementation in `baek_how_2025`, test-set sizes in `steinberg_where_2026`,
  OrigamiNet's backbone for its attribution figures).
- Several 2026 preprints (`zhang_can_2026`, `steinberg_where_2026`,
  `feucht_using_2026`) are not peer reviewed as far as we can tell.
- Two cited works are older than their arXiv date: `liu_study_2015` was
  presented at ICDAR 2015 and uploaded to arXiv in 2020.
- Graves et al. 2006 (CTC) and the kraken paper are already in the library
  under *Texterkennung (HTR/OCR)*; cite those records.

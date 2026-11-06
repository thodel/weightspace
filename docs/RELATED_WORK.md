# Related work: how has the handling of text by recognition networks been assessed?

Status: second pass, 06.11.2026 (CET). It replaces the first pass of the same
day. The search was run across several databases and is documented in
[SEARCH_LOG.md](SEARCH_LOG.md); coverage gaps are listed there and in
section 7. 72 papers were examined individually: 58 from open full text, 12 at
abstract level, 2 by metadata only. Every reference was verified against arXiv,
Crossref or HAL.

References are in the Zotero collection *WSP – analysis & explainability of
text-recognition networks* (below *Texterkennung (HTR/OCR)*, personal library),
each with a child note holding the reading summary. A BibTeX export is in
[`refs/wsp_related_work.bib`](../refs/wsp_related_work.bib); citations below use
its keys. The per-paper notes are in [related_work/notes](related_work/notes/README.md).
"We found no paper that…" means "in this search".

## 1. Short answer

**Is there a paper that analyses neural networks for text recognition?** There
are many partial ones. Almost all assess the network from outside, through its
outputs, and none opens up the weights of a conv-BiLSTM-CTC line recogniser.

- The closest is `sabir_implicit_2017`. It trains a CTC line recogniser (two
  fully connected layers, two BiLSTM layers) on synthetic English text and
  tests for an implicit language model with shuffled characters and with
  seen and unseen n-grams. It concludes that the network uses up to about five
  characters of context and reports a 2.4 point CER improvement over shuffled
  text on its synthetic test set.
- The only parameter-level reading of a recurrent text network we found is a
  master's thesis, `zervakis_multivariate_2019`: SVD of the output matrix of an
  LSTM, with ablation of single singular vectors. It is on MNIST digits and
  was read at abstract level only.

**Is there an explainability paper about text recognition?** Yes, each narrow.
`buoy_explainable_2023` (CTC scene text), `yousef_origaminet_2020` (Integrated
Gradients on a CTC page recogniser), `ty_scene_2023` (per-character
attribution for six scene-text models), `da_levenshtein_2022`,
`coquenet_towards_2022` and `hamdan_htr-jand_2024` (attention maps on HTR), and
several for isolated characters or online handwriting. One more,
`amirgaliyev_comparative_2026`, compares occlusion, LIME and KernelSHAP on a
CRNN-CTC model for Arabic manuscripts with deletion and insertion scores, but
it was read at abstract level only and must be read before it is cited. We
found no survey of explainability for text recognition. The survey
`beyene_survey_2026` covers output-level evaluation of OCR and says nothing
about model internals.

**How did earlier work assess what the network does with text?** Five ways
(section 3 has the detail):

1. Black-box input manipulation with an exact ground truth: shuffled or random
   text, perturbed text rendered to an image, rare letter combinations.
2. Swapping decoders or modules: external language models at decoding, removal
   of the recurrent layers with retraining.
3. Attribution maps, mostly judged by eye.
4. Mechanistic work on heads, neurons and hidden states, only on transformer
   vision-language models.
5. Weight-level operations on recognisers: averaging, unlearning and layer
   freezing, judged by accuracy.

## 2. What exists, by line of work

### 2.1 CTC behaviour

- `zeyer_why_2021` proves that blank dominance and spikes follow from the CTC
  topology and the ratio of frames to labels, not from the input. `li_reinterpreting_2020`
  reaches the spiky phenomenon from the gradient and simulates it on a bare
  output matrix, with a check on a CRNN by output shape only.
- `jung_blank_2023` counts blank frames in wav2vec2 CTC models: about 43 to
  46 percent of frames can be dropped before beam search without changing the
  transcript. Speech only.
- `zeyer_gradient-based_2026` computes input-gradient attributions through the
  CTC prefix score and validates them against gold word boundaries. For one
  streaming CTC model the native posteriors sit systematically late (word
  boundary error 357 ms against 127 ms for the gradient). Spike position is
  therefore not a reliable alignment. Speech only.

### 2.2 Context and the "implicit language model" in CTC line recognisers

No paper probes or edits the recurrent state of a trained recogniser. The
evidence is behavioural:

- `sabir_implicit_2017`: output-only probes, as above. Small effects, no
  confidence intervals, synthetic English, font-dependent.
- External language models. `tarride_revisiting_2024` and `tarride_improving_2024`
  (PyLaia: four conv layers, three recurrent layers, CTC) show that a
  character n-gram still lowers CER even though the network has recurrent
  layers (NorHand 9.72 to 8.23), and that the gain vanishes when CER is
  already below about 2 percent. `moysset_are_2018` and `liu_study_2015` find the same
  with other networks. A persisting gain shows that the recurrent layers do
  not saturate the language signal. It cannot show what they encode.
- Removing the recurrent layers and retraining. `coquenet_have_2019` reports
  test CER of 6.88 for CNN+BLSTM against 19.03 for CNN+dense on RIMES, and
  `al-azzawi_understanding_2026` finds larger cross-language transfer gains for
  a CRNN than for a CNN-only model. Both are retraining comparisons, not
  interventions on a trained network, and the second says "association rather
  than strict causation".
- `garrido-munoz_generalization_2025` finds that textual divergence between
  domains is the dominant factor in out-of-distribution error of eight HTR
  architectures, CTC models generalising better than autoregressive ones
  (average CER 40.4 against 48.7). The analysis is correlational.
- `kiessling_does_2024` is the nearest kraken-adjacent test of context. It
  zeroes the metadata token at test time. A Conformer that received century,
  script and language as input drops by only 0.01 points (91.14 to 91.13), so
  the gain may come from training with the token rather than from using it.
  The same paper reports 86.56 for kraken's CNN+LSTM against 90.32 for the
  Conformer on CATMuS Medieval, single runs.
- Language shift. `meyer_n-gram_2026` (transformer decoders) and
  `kaliosis_learning_2025` (a CRNN among the models; character-frequency loss,
  small effects) change the language prior through loss or decoding, not
  through the network.
- Earlier work separates the optical and the language side by design or by
  swapping components: `granet_separating_2018`, `villegas_optical_2015`,
  `tanha_crossing_2015` (abstract only).

### 2.3 Explainability for text recognition

| Paper | Object | Method | Check of the explanation |
|---|---|---|---|
| `yousef_origaminet_2020` | CTC page recogniser | Integrated Gradients per emitted character, white baseline | visual only |
| `buoy_explainable_2023` | ViT with CTC head, words | built-in association map | character boxes on 446 synthetic images, no chance level |
| `ty_scene_2023` | six scene-text models, words | Captum methods, per character | selectivity (deletion) only, also used to choose the method |
| `hamdan_htr-jand_2024` | CNN-BiLSTM-attention-CTC, lines | attention heat maps | none |
| `coquenet_towards_2022` | paragraph HTR | attention and prediction overlays | one duplicated-line input |
| `da_levenshtein_2022` | scene-text editor model | attention split over modalities | none |
| `azimi_improving_2023` | online isolated letters | LIME on time-series slices, two examples | none |
| `pundy_transparency_2025` | writer identification | saliency, point-specific maps | ink-only deletion and insertion against random |
| `amirgaliyev_comparative_2026` | CRNN-CTC, TrOCR, Nougat; Arabic manuscripts | occlusion, LIME, KernelSHAP | deletion and insertion AUC, AOPC, stability (abstract only) |

Isolated characters and digits (`whitten_ai_2024`, `dutta_understanding_2025`,
`gan_characters_2023`, `alothman_applying_2025`) and speech Transformers
(`sun_explainability_2024`) complete the picture. The survey
`wan_survey_2025` treats attention heat maps as a standard local explanation.

### 2.4 Language prior against visual evidence in VLM OCR

This is the most active area and its designs transfer to a CTC model, because
they need only rendered images and CER.

- `liang_visual_2026`: random mixed-case strings as a zero-prior test. DeepSeek-OCR
  falls from about 88 percent to 19.84 percent precision, a pipeline OCR system
  by 4.9 points.
- `karamolegkou_reading_2026`: perturb the text before rendering and score
  against the perturbed string (swap, shuffle, reverse, random, graded 5 to 25
  percent, words and characters). The clearest separation is at character
  level: VLMs worsen by 12.6 to 42.5 points on Greek, Tesseract and Kraken by
  2.8 to 6.1. Kraken is used as a black-box baseline; its internals are not
  analysed. The authors note that the effect could also be sensitivity to
  unfamiliar sequences.
- `shah_fade_2026` (random digits at graded visibility), `yao_par_2026`
  (glitches in highly predictable text with a shuffled control),
  `yang_cer-tpr_2026` and `haraguchi_text_2026` (abstract only),
  `peng_cc-ocr_2026` (per-factor slicing of a benchmark),
  `siddiqui_using_2020` (train-on-X, test-on-Y grid on a CRNN-CTC),
  `zhang_can_2026`, `bautista_scene_2022`, `fang_read_2021`.
- Mechanistic work, transformers only: `baek_how_2025` and `huangfu_retrieval_2026`
  (OCR heads; the latter compares head rankings before and after rank-32 LoRA
  training of Qwen2-VL-2B and Qwen3-VL-2B), `steinberg_where_2026` (subspace
  removal), `feucht_using_2026` (mean ablation with a random-head control),
  `yao_reading_2025` (probes on hidden states for correctness).

### 2.5 Weight-level operations on text recognisers

| Paper | Operation | Model | Evidence |
|---|---|---|---|
| `molina_rodriguez_ocr_2025` | average of task vectors of experts fine-tuned from one seed, then fine-tune on a new script | ViT with CTC loss, word crops | accuracy on in-domain, out-of-domain and new-script sets; large in-domain losses (Parzival .412 against .858 for a joint fine-tune); cosine check of task vectors |
| `kang_preserving_2026` | zero neurons by forget/retain activation ratio, then loss-guided fine-tuning, to unlearn writer identity | T5-encoder HTR, IAM words | writer-ID accuracy 0.00 on the forget set, membership-inference success 49 percent against 72.85 for the baseline; pruning alone raises CER to 58 to 67 |
| `sharma_trocr_2026` | freeze prefixes of encoder or decoder layers during fine-tuning | TrOCR | line-wise CER with significance tests; encoder must adapt, decoder tolerates freezing |
| `zervakis_multivariate_2019` | SVD of the LSTM output matrix, ablation of singular vectors per class | LSTM, MNIST | abstract only |
| `boros_investigating_2024` | scale activations of selected neurons in a forward pass | Llama2 and Mistral reading OCR-noisy text, not a recogniser | NER F1 |

`molina_rodriguez_ocr_2025` is the closest published precedent for the averaging
comparator in our plan (control C2). It uses a ViT, not a recurrent stack,
leaves the handling of the output layer for new alphabets unstated, and reports
single numbers.

### 2.6 Kraken and the DH literature

`kiessling_version_2025` gives the architecture of kraken 5 (three
convolutional layers, three BiLSTM layers of 200 units, a linear output layer,
VGSL-defined shapes, a many-to-many codec between labels and code points, no
language model) and no model analysis; the word "blank" does not occur in it.
The DH literature on kraken, PyLaia, Transkribus and eScriptorium consists of
evaluations, workflows and case studies. Two controlled input experiments
exist: `jacsont_impact_2023` (image pre-processing before training kraken
models; binarisation helps) and `chague_colored_2024` (background colour;
abstract only). We found no analysis of kraken model internals. The
humanities-facing statement of the folk model that a recogniser contains a
learned language component is in `schonhardt_automated_2026`, whose authors
include the owner of this repository.

## 3. How the assessment was done

| Method | Used by | What it can show | Validity evidence in the paper | Main weakness |
|---|---|---|---|---|
| Loss and gradient analysis of CTC | `zeyer_why_2021`, `li_reinterpreting_2020` | blank dominance and spikes follow from the loss | proofs, simulations | toy outputs; nothing on trained nets |
| Frame statistics of CTC emissions | `jung_blank_2023` | share of blank frames, run lengths | WER unchanged after collapse | speech; says nothing about content |
| Input gradients through CTC prefix score | `zeyer_gradient-based_2026` | where input evidence for a token sits | gold word boundaries | speech; not causal |
| Shuffled text, seen and unseen n-grams | `sabir_implicit_2017` | reach of implicit context | controls by construction | synthetic, small effects, no intervals |
| Perturb text, render, score against perturbed string | `karamolegkou_reading_2026` | prior reliance by ΔCER slope | exact ground truth, Wilcoxon with Holm | cannot separate prior from unfamiliar sequences |
| Zero-prior random text | `liang_visual_2026`, `shah_fade_2026` | drop when no prior helps | indirect | out-of-distribution glyph sequences |
| Glitch in predictable text with shuffled control | `yao_par_2026` | recitation against reading | shuffle control | famous quotes may be memorised |
| External LM fusion | `tarride_revisiting_2024`, `moysset_are_2018`, `liu_study_2015` | whether the network lacks language information | none | output level only |
| Recurrence ablation with retraining | `coquenet_have_2019`, `al-azzawi_understanding_2026` | contribution of the recurrent block | none | retraining, not intervention |
| Zero the context input at test time | `kiessling_does_2024` | whether context is used at inference | single control | single runs |
| Internal LM estimation by distillation | `yang_label-context-dependent_2025` | what CTC outputs imply about label context | WER after correction | speech; not weight-level |
| Rarity-stratified accuracy grid | `zhang_can_2026` | failure on rare combinations | bootstrap on the gap | mechanism untested |
| Attribution per emitted character | `yousef_origaminet_2020`, `ty_scene_2023` | which ink drives a character | selectivity, visual | circular or visual only |
| Attention maps | `hamdan_htr-jand_2024`, `coquenet_towards_2022` | where the model looks | none or one intervention | attention is not explanation |
| Deletion and insertion against random | `pundy_transparency_2025` | faithfulness of a saliency map | quantitative | one backbone below chance |
| Head scoring, masking, patching | `baek_how_2025`, `huangfu_retrieval_2026`, `feucht_using_2026` | which heads carry reading | random-head controls | transformers only |
| Hidden-state probes | `yao_reading_2025` | whether correctness is decodable | transfer across datasets | not a test of what is read |
| Task-vector averaging | `molina_rodriguez_ocr_2025` | whether averaged experts generalise | accuracy | ViT; in-domain losses |
| Activation-contrast neuron zeroing | `kang_preserving_2026`, `boros_investigating_2024` | which units carry a property | forget-set tests, membership inference | writer identity, not text |
| Train-on-X, test-on-Y | `siddiqui_using_2020`, `garrido-munoz_generalization_2025` | generalisation map | replicates, factor analysis | single task or correlational |

## 4. Observations that hold across the papers

1. **Nothing inside the recurrent layers.** No paper intervenes on the
   recurrent layers of a trained CTC recogniser. Two retrain without them, and
   the language-model evidence is external. "Implicit language model" is
   inferred from accuracy gaps.
2. **Validation of attributions is thin.** Only `pundy_transparency_2025`
   tests a saliency map against a random order, and on one backbone its own
   scores are below chance. `ty_scene_2023` uses one metric to pick and to
   compare methods. Attention maps are judged by eye.
3. **The strongest probes are black-box and exact.** Render a perturbed text
   and score against it. They need no access to the network and transfer to a
   CTC model. Their weakness is the same everywhere: a drop under perturbation
   mixes loss of the prior with out-of-distribution input.
4. **CTC structure is set aside or assumed.** Attribution papers drop blank
   and repeated outputs; `zeyer_why_2021` explains why; `zeyer_gradient-based_2026`
   shows that spike position can be systematically offset.
5. **Recurrent layers do not replace an external language model.** In the
   tests we read, an external character n-gram still helps a CNN-BiLSTM-CTC
   network, except when CER is already very low. Word-level models often hurt.
6. **The mechanistic literature is transformer-only.** Heads, residual
   streams and attention ratios have no counterpart in a conv-BiLSTM-CTC stack.
   The recipes transfer; the findings do not.
7. **Kraken is a black box in the literature.** It appears as a baseline, in
   training experiments and in an architecture description.
8. **The closest evidence for our architecture points both ways.**
   `zhang_can_2026` Table 2 gives the CNN-BiLSTM-CTC model the largest
   rare-composition gap of nine English backbones (−17.9 points against −10.5
   to −13.9), but it is also the weakest model overall, and the paper is a
   single-author, unreviewed preprint with a confounded comparison.

## 5. What this means for the project

These are proposals for the pre-registered protocol (docs/PROTOCOL.md), not
edits to it. Each ties a reading from the weights to a behavioural check that
the literature shows to be missing.

1. **Report CTC structure with CER.** Blank fraction, spike width and run
   lengths on non-blank frames before and after every kept edit
   (`jung_blank_2023`, `li_reinterpreting_2020`). If attribution is used,
   compare spike position with an input-gradient attribution through the CTC
   prefix score (`zeyer_gradient-based_2026`).
2. **Add a black-box probe suite for text handling.** Same lines, perturbed
   text, scored against the perturbed text: graded character and word swaps,
   shuffles, reversal and random strings (`karamolegkou_reading_2026`);
   zero-prior random lines (`liang_visual_2026`); seen and unseen n-gram
   strings for the reach of implicit context (`sabir_implicit_2017`); the
   rarity grid (`zhang_can_2026`); glitches in predictable text with a
   shuffled control (`yao_par_2026`). Report the ΔCER slope, not only the
   level.
3. **Check confidence after an edit.** Calibration can break without CER
   moving. Use the per-line confidence scores and temperature scaling of
   `tarride_improving_2024`; `ayllon_analysis_2024` is the calibration paper
   for HTR and was available as metadata only.
4. **Compare any claimed language gain with decoder-side language models.**
   A character n-gram fused at decoding time, weight about 1.5, is the
   baseline (`tarride_revisiting_2024`, `tarride_improving_2024`). The gain
   vanishes at low CER, so test on sets where CER is not already below 2
   percent. Estimate the internal language model before and after an edit by
   distillation (`yang_label-context-dependent_2025`).
5. **Control for components.** Zero or re-initialise one BiLSTM layer or one
   direction of a trained model and compare with a conv-only baseline
   (`coquenet_have_2019`, `moysset_are_2018`); the layer-prefix grid of
   `sharma_trocr_2026` is the training-time parallel.
6. **No attribution without a causal check.** Selectivity, ink-only deletion
   and insertion against a random order (`pundy_transparency_2025`), and a
   character-box hit test with a chance baseline (`buoy_explainable_2023`).
   Use a white baseline and state it (`yousef_origaminet_2020`). Build a
   synthetic known-relevance task for any relevance method
   (`arras_explaining_2019`).
7. **Averaging as a comparator.** Follow `molina_rodriguez_ocr_2025`: experts
   fine-tuned from one seed, task-vector average, cosine check of task
   vectors, compute-matched controls. Kraken codecs map labels many-to-many
   (`kiessling_version_2025`), so models must share a codec for the average to
   be defined; the alignment code in `wsp/align.py` exists for this.
8. **Weight-reading techniques with precedent.** SVD of the output matrix with
   singular-vector ablation per character (`zervakis_multivariate_2019`, to be
   read); activation-contrast selection of units followed by zeroing or
   scaling (`kang_preserving_2026`, `boros_investigating_2024`). Both are
   cheap to try on a kraken model and neither has been tested on a CTC line
   recogniser.
9. **A null condition for image-side sensitivity.** Pre-processing and
   background changes do move kraken accuracy (`jacsont_impact_2023`,
   `chague_colored_2024`), so an edit aimed at text behaviour should be
   checked for changes in that sensitivity.
10. **Adapter track (QLoRA on Qwen-family models).** Per-head OCR scores and
    masking before and after training (`baek_how_2025`, `huangfu_retrieval_2026`,
    the latter on the same family and LoRA rank 32); the text-hint reliance
    protocol of `yang_cer-tpr_2026` and the glitch protocol of `yao_par_2026`
    on base against adapter; subspace removal at the OCR depth band
    (`steinberg_where_2026`). All depend on transformer-only units and need
    the random controls the originals lack: same-rank random subspaces and
    depth-matched random heads.

## 6. Where this project would be new, and how to say it

The claim has to be narrower than the first pass suggested, because weight-level
operations on text recognisers do exist (section 2.5). The defensible claim is:
in this search we found no earlier work that treats a trained CTC line
recogniser as an object to be read and edited in weight space, with edits
proposed from a reading of the weights under a forward-pass budget and checked
on a held-out set. The nearest work averages the weights of word recognisers
(`molina_rodriguez_ocr_2025`), zeroes neurons to unlearn writer identity in a
transformer (`kang_preserving_2026`), freezes layers during fine-tuning
(`sharma_trocr_2026`) and applies SVD to an LSTM for digits
(`zervakis_multivariate_2019`). The claim must be rechecked with the missing
databases (section 7) before submission.

## 7. Coverage and limits

Details are in [SEARCH_LOG.md](SEARCH_LOG.md). The ones that matter here:

- Semantic Scholar, IEEE Xplore, Google Scholar and Scopus were not
  available. OpenAlex was exhausted partway through. Queries Q5 to Q10 are
  incomplete there and should be rerun.
- Fourteen papers could not be read in full: twelve at abstract level
  (`puigcerver_are_2017`, `amirgaliyev_comparative_2026`, `dutta_understanding_2025`,
  `gan_characters_2023`, `sun_explainability_2024`, `tanha_crossing_2015`,
  `villegas_optical_2015`, `chague_colored_2024`, `yang_cer-tpr_2026`,
  `haraguchi_text_2026`, `alothman_applying_2025`, `zervakis_multivariate_2019`)
  and two by metadata (`ayllon_analysis_2024`, `benaissa_why_2026`). The two
  that could change a conclusion are `amirgaliyev_comparative_2026` (a CRNN-CTC
  model explained with perturbation faithfulness metrics) and
  `zervakis_multivariate_2019` (weight-level analysis of an LSTM).
- Several 2026 preprints are not peer reviewed as far as we can tell
  (`zhang_can_2026`, `steinberg_where_2026`, `feucht_using_2026`,
  `huangfu_retrieval_2026`, `liang_visual_2026`, `karamolegkou_reading_2026`).
- Venues that the source text did not print were taken from the DOI or
  Crossref record. `sabir_implicit_2017` is the arXiv copy of an ICDAR 2017
  paper. `michael_evaluating_2019` carries ICDAR 2019 from the project brief.
- The first pass cited a blog attribution that was wrong; the blog is by
  Takehi, Teoh, Zhang and Khattab (`takehi_sparks_2026`).

## 8. Verification

Titles, authors, years and venues were checked against arXiv, Crossref or HAL
for every reference. Numbers and quotes in the reading notes were produced by
reading agents. The lead author checked a sample against the PDFs and found
no discrepancy: in the first pass the deletion scores in
`pundy_transparency_2025`, the CRNN gap in `zhang_can_2026` and the
attention-sum figure in `da_levenshtein_2022`; in the second pass the
five-character context and 2.4 point figures in `sabir_implicit_2017`, the
40.4 and 48.7 averages in `garrido-munoz_generalization_2025`, the Parzival
number and the averaging rule in `molina_rodriguez_ocr_2025`, the 19.84
precision in `liang_visual_2026`, the 357 and 127 ms in
`zeyer_gradient-based_2026`, the 91.14, 90.32 and 86.56 in
`kiessling_does_2024`, the 12.61 to 42.45 range in
`karamolegkou_reading_2026`, the 9.72 and 8.23 in `tarride_revisiting_2024`,
the 49 percent, 72.85 and 58.29 in `kang_preserving_2026`, and the 6.88 and
19.03 in `coquenet_have_2019`. Notes flag what the agents could not check.
Two internal inconsistencies in source papers are recorded in the notes (a
figure caption against its text in `garrido-munoz_generalization_2025`, and
version differences between the two TrOCR papers, where the later one should
be cited).

# Batch H: how four papers assessed what a text-recognition network does with its text input

Read from the arXiv PDFs in /tmp/claude-1000/-home-th/0dc8ad2a-3390-417b-9872-a7fd4cdb988d/scratchpad/search/H/ (pdftotext + grep). Quotes verbatim, <=25 words. "(mine)" marks my own reading; UNVERIFIED marks what I could not confirm from the text.

## H1. Tarride & Kermorvant, "Revisiting N-Gram Models: Their Impact in Modern Neural Networks for Handwritten Text Recognition" (arXiv 2404.19317)

- **Full text:** https://arxiv.org/pdf/2404.19317 (v1, 30 Apr 2024).
- **Authors, year, venue:** Solène Tarride and Christopher Kermorvant, TEKLIA, Paris; 2024. The PDF names no venue, so the venue is UNVERIFIED (arXiv cs.CV preprint). Funding is acknowledged to the Research Council of Norway (HuginMunin project).
- **Model analysed:**
  - PyLaia: "4 convolutional layers and 3 recurrent layers", CTC loss, softmax over characters (Sec. 3.2). Whether the recurrent layers are bidirectional is not stated in this paper (UNVERIFIED). Line images are resized to height 128 px and trained with early stopping after 80 epochs without improvement.
  - DAN: CNN encoder plus Transformer decoder, cross-entropy loss, character output, trained on lines and on paragraphs or pages.
  - Data: IAM (English), RIMES (French), NorHand v2 (19th/early-20th-c. Norwegian letters). Line test sets are 2,915 (IAM), 778 (RIMES) and 1,494 (NorHand, Table 3) lines. NorHand has 1,573 lines in Table 1; I did not resolve the difference.
- **Question asked:** "whether explicit language models, specifically n-gram models, still contribute to the performance of state-of-the-art deep learning architectures" (Abstract; the wording is ours from the abstract, not a quote). Framing: networks "implicitly" capture language statistics, so does an explicit LM still add anything?
- **Assessment method:**
  - Greedy decoding (argmax per frame) compared with CTC beam search fused with a KenLM n-gram LM (torchaudio ctc_decoder, lexicon constraints, LM rescoring on CPU). Only CER and WER are computed.
  - LM variants, all trained on the training transcriptions of each dataset ("auto-tuned", Sec. 6): character, subword (SentencePiece, vocabulary 1000, "SeparateSpaces" representation) and word level.
  - Tuned on NorHand only (Sec. 4; the split used is not stated, UNVERIFIED): order 1 to 6 (character and subword best at 6, word best at 3); LM weight by grid 0 to 5 in steps of 0.5 (best 1.5 for character and subword, 0.5 for word); smoothing None, Kneser-Ney or Witten-Bell (Kneser-Ney best "by a thin margin").
  - To make DAN usable with the CTC decoder, they "add fake CTC frames between frames to avoid the removal of duplicate characters" (Sec. 3.3).
  - Results are in Table 3 (lines) and Table 4 (pages or paragraphs).
  - There is no probing, ablation or perturbation of the network itself, and no analysis of the recurrent layers. The "implicit LM" is never measured directly. It is inferred only from whether adding an external LM still lowers CER/WER.
- **Evidence the method measures what it claims:** none stated. The only evidence is that CER/WER fall with an external LM. No test shows the network lacks, or has, an implicit LM; no seeds, confidence intervals or significance tests are reported (I found none, mine).
- **Main finding:**
  - A character n-gram LM improves nearly everything (numbers from Table 3). PyLaia on NorHand: CER 9.72 to 8.23, WER 27.78 to 21.65. PyLaia on RIMES: CER 4.57 to 3.79. DAN on IAM: CER 4.86 to 4.38.
  - Word LMs often hurt: "Word-based models, on the other hand, lead to a decline in performance, mainly due to challenges with out-of-vocabulary words." (Sec. 5.1)
  - Gains on RIMES are small for DAN, which already has low error rates.
  - DAN hallucinated on NorHand: "we observe some hallucinations from DAN on NorHand" (Sec. 5.1).
- **Limits:**
  - Stated:
    - The subword vocabulary size was not optimised.
    - The LM runs on CPU and slows PyLaia decoding about tenfold (Sec. 5.2, Table 5: 0.01 to 0.12 s/image).
    - LM parameters were tuned on NorHand only.
  - Mine:
    - Output-level only: a gain shows the explicit LM adds information the argmax path lacks, not what the recurrent layers do or do not encode.
    - The LM is trained on the same transcriptions the network was trained on, so it adds no new text knowledge. The gain is a re-weighting of training-set n-grams applied at decoding. This is a possible reason character LMs help most, but the paper does not test it (a hypothesis of mine).
    - Single runs. Table 3 shows non-monotone cases (for example DAN on RIMES: CER 3.22 to 3.22 with the character LM, WER 10.69 to 10.83), so the small differences are within unquantified noise.
    - The text says "Optimal parameters are optimized on the NorHand dataset" and then applies them to all three datasets. The split used for tuning is not stated.
- **Relevance to weight-space edits of a CTC line recogniser:** PyLaia is a conv + 3 recurrent layers + CTC line model very close to kraken's VGSL stack, and this paper is the baseline showing that its argmax path still gains CER/WER from an external character 6-gram (weight 1.5), so any claim that a weight edit "injects language knowledge" should be compared against that LM-fusion baseline, not only against the unedited network.

## H2. Moysset & Messina, "Are 2D-LSTM really dead for offline text recognition?" (arXiv 1811.10899)

- **Full text:** https://arxiv.org/pdf/1811.10899 (v1, 27 Nov 2018; arXiv comment "12 pages, 4 figures").
- **Authors, year, venue:** Bastien Moysset and Ronaldo Messina, A2iA SA, Paris; 2018. The text I read shows no journal or venue, so the venue is UNVERIFIED (arXiv preprint).
- **Model analysed:**
  - Five architectures: CNN, GNN (gated convolutions) and CNN-1DLSTM (Puigcerver-style; 9.6M parameters per Table 4) without or with bidirectional 1D-LSTM decoder layers, plus GNN-1DLSTM (Bluche, 799k parameters) and a 2DLSTM network (836k parameters; 3.3M for the 2DLSTM-X2 variant with doubled feature-map depth).
  - Trained with CTC; "No tuning of these parameters have been perform" (Sec. 3.1.2): RMSProp lr 1e-4, batch 8, early stopping after 20 non-improving epochs, dropout 0.5, line images at height 128 px.
  - Data: RIMES, IAM, READ (historical German handwriting, 14.8% OOV rate in the validation set), and MAURDOR handwritten and printed (French).
- **Question asked:** two questions. Do 2D-LSTM layers still beat 1D-LSTM or purely convolutional networks on hard data? And does recurrence act as a "latent" language model? "It can be seen as a “latent” language model." (Sec. 2.1)
- **Assessment method:**
  - Architecture ablation: remove the final 1D-LSTM layers (CNN-only and GNN-only variants) and compare CER/WER. Also vary the number of 1D-LSTM layers, feature-map depth, dropout, and the interface (max-pooling vs concatenation), Tables 14 to 17.
  - Dataset-transfer check (Sec. 3.4, Table 18): train on MAURDOR handwritten, test on RIMES validation. Result: "none of the tested models generalizes better than the others in a transfer setup."
  - External-LM test (Sec. 3.5, Tables 19 and 20), applied to the CTC posteriors with an FST and the Kaldi decoder:
    - LM types: character 5/6/7-gram and word 3-gram with vocabularies from 25k to 200k.
    - LM text: Wikipedia dumps for French and English; READ training data only for READ.
    - Posteriors are divided by character priors estimated from the training data, with prior weight 0.7.
    - "no effort was done in order to optimize the LMs". LM weight and beam are not described beyond the prior weight (UNVERIFIED). Greedy "best pred" is the no-LM baseline.
    - Reading: if recurrence had absorbed a language model, the external LM should help the LSTM nets less than the CNN-only net.
  - Gradient visualisation (Sec. 4, Figs. 2 to 4): gradient of 1 on the output(s) for one character at all horizontal positions, back-propagated to the input image with no parameter update. The absolute value ∂Out_i/∂In_xy is shown as an "attention map". Qualitative; Figs. 3 and 4 show two images (characters 'A' and 'L') for CNN, GNN-1DLSTM and 2DLSTM.
- **Evidence the method measures what it claims:** none stated for the "latent LM" reading. The LM test is purely behavioural. The gradient maps are single examples with no quantification, no sanity check, and no comparison to a perturbation test.
- **Main finding:**
  - No-LM ablation, MAURDOR handwritten test (Table 19): CNN CER 33.64% vs CNN-1DLSTM 14.29%. READ validation (Table 20): CNN 26.25%, 2DLSTM 7.97%. "LSTM layers ... are very important".
  - An external LM still helps every network, including the LSTM ones: "The use of a language model and of LSTM layers can therefore be considered complementary." (Sec. 3.5)
  - On READ the CNN-only net gains most from the LM (CER 26.25% to 4.88% with the word 3-gram; CNN-1DLSTM 12.39% to 4.64%), but the LSTM nets are not saturated either.
  - Authors' conclusion: 2D-LSTM gives "sharper attention maps" that make the network less sensitive to noise. They state this as an explanation, not as a tested result.
- **Limits:**
  - Stated: no hyper-parameter tuning; the 2DLSTM-X2 result is "not extensively tuned" while the rivals probably were; the LMs were not optimised.
  - Mine:
    - The text says both the LSTM and non-LSTM networks get "a similar improvement of about 20%" on MAURDOR. From Table 19 I compute roughly 32% relative CER reduction for the CNN with the 7-gram character LM and about 19% for CNN-1DLSTM, which does not match; I flag this as unchecked.
    - A smaller LM gain for recurrent nets would not by itself prove an implicit LM, because their baseline errors are mostly visual. A persisting gain does not show absence of one either.
    - There is no direct probe of the recurrent state (no context-length ablation, no scrambled-text test).
    - The gradient map shows spatial dependence, not linguistic dependence.
    - Single runs with no variance.
- **Relevance to weight-space edits of a CTC line recogniser:** it is the earliest ablation in this batch comparing a conv-only CTC net with conv + BiLSTM nets under identical external-LM fusion, so it gives a concrete protocol (remove or zero the recurrent block, then re-measure with and without an LM) for deciding whether an edit that targets the BiLSTM stack acts on language-like or on spatial-context behaviour.

## H3. Karamolegkou, Angleraud, Sagot & Clérice, "Reading or Guessing? Visual Grounding Failures of Vision-Language Models for OCR in Ancient Greek and Arabic Editions" (arXiv 2605.27750)

- **Full text:** https://arxiv.org/pdf/2605.27750 (v2, 7 Sep 2026; arXiv first posted 26 May 2026).
- **Authors, year, venue:** Antonia Karamolegkou, Nicolas Angleraud, Benoît Sagot, Thibault Clérice, Inria, Paris; 2026. ArXiv category cs.CL; the PDF names no venue (UNVERIFIED).
- **Model analysed:**
  - Five open-weight VLMs: LightOnOCR-1B, DeepSeek-OCR, Qwen3-VL-2B, Qwen3-VL-8B, olmOCR-2-7B.
  - Two "traditional" engines, labelled "CNN–LSTM" in Table 1: Tesseract v5 (grc and ara models) and Kraken v5. Kraken is the in-domain CLLG model for Greek and the pretrained OpenITI model for Arabic.
  - Data: 90 Greek pages (1,966 lines, 30 printed editions, CLLG) and 140 Arabic pages (3,197 line crops, OpenITI), plus rendered synthetic pages for the perturbation tests.
- **Question asked:** whether VLM OCR is visually grounded or driven by a linguistic prior ("do VLMs reproduce linguistically implausible text when it is clearly visible in the image", Sec. 1), compared with CNN-LSTM engines. Kraken is the in-domain baseline and gets none of the grounding analyses beyond the output-level ones.
- **Assessment method (RQ2 and App. C):**
  - Counterfactual perturbation (App. C.1): text is perturbed before rendering, so the rendered image shows the perturbed string and that string is the ground truth ("faithfully renders a modified input, which also serves as the ground truth", Sec. 4).
  - Test 1 (word level): swap or shuffle a proportion p = 5/10/25% of words, local window-3 shuffle, reverse, random. Test 2 (character level): the same operations on characters inside words, in Arabic on grapheme units.
  - Rendering and scoring: Greek rendered with lualatex in GFS Didot, Arabic in Amiri; scored as CER without spaces and diacritics; median paired ΔCER (clean vs perturbed per page, or per paragraph for Arabic), Wilcoxon signed-rank with Holm correction. Greek analysis is restricted to single-column pages. Control: Arabic glyph-form, ligature and rendered-length covariates "do not consistently predict the loss in accuracy" (App. C.1).
  - Key numbers (Sec. 4): under character perturbation VLMs rise +12.61 to +42.45 points (Greek) and +10.55 to +26.24 (Arabic); traditional systems +2.79 to +6.05 and +2.53 to +5.65. Word-order perturbations barely matter (VLMs at most +0.62 and +1.65). Kraken-OpenITI's +2.53 is significant even though it is the smallest.
  - Token-level "image gain" (App. C.2): teacher-force the model's own baseline transcription and compute gain(t) = log p(t | image, prompt) − log p(t | prompt), with the image-free pass using the same prompt. Substitutions are classed as perceptual, cross-script or lexical and compared with each other (not with correct tokens). Only the three Qwen-architecture models (Qwen3-VL-2B, Qwen3-VL-8B, olmOCR-2-7B) support it. LightOnOCR's path "yields no valid image-free comparison" and DeepSeek-OCR is unsupported (Limitations). The main text says the Greek benchmark, while Table 13 also lists Arabic.
  - Error analysis on real scans: word-level taxonomy (Tables 6 to 8). Real-word test: share of misreads that are attested lexicon words, Table 9. Greek 16.3 to 32.2% for VLMs vs 17.1% (Kraken) and 21.2% (Tesseract); Arabic 40.3 to 50.4% for VLMs vs 21.9% (Kraken) and 34.3% (Tesseract). The Greek lexicon (19,901 entries) is derived from the reference corpus, which is also the Kraken-CLLG training corpus, so the authors call the comparison "conservative" (appendix, "Real-word lexicon").
  - Failure-mode counters (Table 10): repetition, markup, off-script generation; Kraken and Tesseract score 0.0 on all of them in both languages.
  - Five inference-time interventions (RQ3; script-constrained decoding, length abstention, VCD, M3ID, text-only post-OCR correction), applied to the VLMs only.
- **Evidence the method measures what it claims:** partly. Controls stated: exact ground truth by construction; word-order vs character-level axes separate two prior types; Arabic rendering covariates. Against that, the authors write: "This pattern is consistent with prior-driven repair, but could also reflect greater sensitivity to unfamiliar character sequences." (Sec. 4) The image-gain analysis then shows prior reliance is model-specific (olmOCR lexical −0.31, p = .031; Qwen3-VL-8B +0.68, p < .001).
- **Main finding:** "accuracy and fluency alone are insufficient for assessing OCR reliability" (Abstract).
  - In-domain Kraken has the best CER/WER (Table 1: Greek 4.1/11.7, Arabic 3.1/18.2).
  - Aggregate accuracy does not separate VLMs from CNN-LSTM engines (out-of-domain Tesseract is worse than most VLMs).
  - Under character perturbation every VLM departs further from the displayed text than either traditional system.
  - Interventions often reverse direction between Greek and Arabic.
- **Limits:**
  - Stated:
    - Two scripts only.
    - The two real-scan benchmarks differ in sampling, segmentation, unit and normalisation, so absolute cross-language rates are not comparable.
    - Perturbation stimuli are rendered, not scanned.
    - Per-step distribution analyses cannot be run on CTC models: "a decoder's next-token distribution, which a CTC recogniser does not have" (Limitations, abridged).
    - Image gain exists for only three of the five VLMs.
  - Mine:
    - Kraken is a black box here. Its small perturbation sensitivity is read as less prior reliance, but no ablation or probe of its BiLSTM is run, and its training domain differs from the VLMs'.
    - Greek perturbation uses Kraken's CLLG-trained BLLA segmenter while the VLMs get full pages.
    - Arabic character permutation is only partly controlled (glyph forms change).
    - The "Kraken-CLLG trained on CLLG, tested on CLLG-derived data" overlap in the synthetic Greek test set is UNVERIFIED.
- **Relevance to weight-space edits of a CTC line recogniser:** the paired "perturb the text before rendering, score against the perturbed string" protocol (graded swap/shuffle at p = 5/10/25%, local, reverse, random) transfers directly to a kraken CNN + BiLSTM + CTC model, and if a weight edit changes language-prior reliance it should show up as a different ΔCER slope under character-level scrambling, which is the axis where this paper found the clearest separation (even Kraken moves by +2.5 to +6 points).

## H4. Beyene & Dancy, "A Survey of OCR Evaluation Methods and Metrics and the Invisibility of Historical Documents" (arXiv 2603.25761)

- **Full text:** https://arxiv.org/pdf/2603.25761 (v1, 26 Mar 2026).
- **Authors, year, venue:** Fitsum Sileshi Beyene and Christopher L. Dancy, Pennsylvania State University; 2026. Venue: FAccT '26 (ACM Conference on Fairness, Accountability, and Transparency, Montréal, June 25 to 26, 2026), per the PDF and arXiv journal_ref; the arXiv comment says this is the submitted version. DOI listed on arXiv: 10.1145/3805689.3806442.
- **Model analysed:** none. It is a PRISMA-style review of OCR and document-understanding evaluation (2006 to 2025). Retrieval: 80 papers from six databases, 78 after de-duplication; the number finally included is not in the text I extracted (it is in the PRISMA diagram, Fig. 1; UNVERIFIED). Extracted: Table 1 training data of 13 OCR systems, Table 2 comparison of systems, Table 3 seven benchmarks. A small case study (Sec. 4.2) runs Tesseract v5, Surya and olmOCR 2 on one page of The Weekly Advocate (1837).
- **Question asked:** how OCR and document-understanding systems are trained and evaluated and what evaluation misses on historical documents, especially Black-press newspapers. It asks nothing about what a network does with its text input.
- **Evaluation approaches it distinguishes:**
  - String-edit metrics CER/WER ("reduce document fidelity to just string-edit distance", Sec. 1). These cannot penalise layout errors: a scrambled column order still gets a perfect CER.
  - Structural, layout or reading-order metrics: "metric pluralism remains the exception instead of the standard" (Sec. 2.2).
  - Benchmark-level task success: DocVQA, DUDE and DUE (QA scored with ANLS), OCRBench v2, OmniDocBench (attribute-based, degradation factors), olmOCR-Bench (7,010 unit tests), CC-OCR, KITAB-Bench (human-in-the-loop).
  - Unsupervised evaluation without ground truth: the authors' own earlier framework [3], with a semantic coherence score, region entropy divergence and a textual redundancy score.
  - Qualitative failure-mode analysis (Table 4: geometric collapse, decoding instability, "corrective hallucination").
  - It also cites a GPT-4o study on Russian manuscripts reporting "59% of errors involved anachronistic character insertions" ("over-historicization", cited from [26]; not verified in the source).
- **Does it cover analysis of what the network does vs output-level metrics?** No. All evaluation approaches it covers are output-level (string, layout, task, qualitative). It contains no discussion of probing, perturbation or ablation of model internals, no language-model or implicit-LM analysis, and nothing on CTC or recurrent layers. Hallucination appears only as an observed failure type, not as something measured by attribution or grounding tests (my grep for interpretability, probing, attribution, faithful, perturb and ablation found only a synthetic-noise remark and the Hallucination rows in Tables 2 and 4).
- **What it says about historical documents:** historical and community-produced documents rarely appear in reported training data or benchmarks (Table 1: six of the 13 systems use the IIT-CDIP tobacco-litigation corpus; Table 3: only KITAB explicitly targets heritage material). Reported historical numbers are sparse (olmOCR 2: 82.3% on old math scans, 47.7% on general historical scans; TrOCR 5.7% CER after fine-tuning). Models robust to synthetic noise "often collapse under such structural degradations" as real microfilm artefacts (Sec. 5.2).
- **Assessment method:** PRISMA-based literature extraction; the case study is a single page, with no ground-truth transcription and no metric.
- **Evidence the method measures what it claims:** none stated for the case study; the survey counts are descriptive.
- **Main finding:** "Most evaluations emphasize character accuracy and task success on modern layouts" and rarely capture column collapse, typographic errors or hallucinated text (Abstract).
- **Limits:**
  - Stated: "We focused on a qualitative failure analysis instead of ground-truth transcriptions" (Limitations and Future Work); the case study is small and specific to 19th-century US Black newspapers; layout- and fidelity-aware metrics are left to future work.
  - Mine:
    - The review covers layout/document-understanding systems and VLMs. Handwritten-text recognition, line recognisers such as kraken and CTC models are barely represented. Tesseract, TrOCR and Donut are the main named examples.
    - Search strings are English and "OCR evaluation"-centred, and the included-paper list is not itemised in the text I read.
    - The paper itself says the full screening numbers are in Fig. 1 (not checked).
- **Relevance to weight-space edits of a CTC line recogniser:** it gives a citable statement that CER/WER-only evaluation of historical-document OCR hides what the model actually does (hallucination, layout), which supports reporting behavioural diagnostics for an edited kraken model beyond CER; it offers no method for analysing a network's text handling, so it cannot serve as a methodological precedent.

## Cross-paper observations
1. Only H1 and H2 test the "implicit LM" idea on CTC-BiLSTM line models, and both do it indirectly by fusing an external LM and watching CER/WER. Neither probes or ablates the recurrent state, so neither could show what the recurrent layers contain; both found the external LM still helps.
2. H3 is the only paper with a real behavioural probe of language-prior reliance (render-then-perturb with exact ground truth), and it includes Kraken, but as a baseline that it treats as the low-prior reference without examining its internals. H4 contributes only the point that CER/WER alone is not enough.

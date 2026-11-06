# Batch D notes: how each paper assessed what a recogniser does with its text input

Local copies and text extractions: /tmp/claude-1000/-home-th/0dc8ad2a-3390-417b-9872-a7fd4cdb988d/scratchpad/search/D/ (D1.raw.txt, D2.raw.txt, D4.raw.txt, D5.raw.txt, D7a.raw.txt, D7b.raw.txt, D8a.raw.txt).
Quotes are verbatim, section or figure given. "(mine)" marks my own inference. UNVERIFIED = could not be checked in an open source.

Summary of access: full text read for D1, D2, D4, D5, D7 (both versions), D8a. Abstract only for D6 and D8b. No open abstract and no full text for D3 (metadata only).

---

## D1. Implicit Language Model in LSTM for OCR

- **Full text:** https://arxiv.org/pdf/1805.09441 (arXiv v1, cs.CV, dated 23 May 2018; found via OpenAlex, which lists it as the open copy of the ICDAR 2017 DOI 10.1109/ICDAR.2017.361). The arXiv PDF itself does not name ICDAR; the ICDAR 2017 attribution comes from the DOI/OpenAlex record (UNVERIFIED in the PDF).
- **Authors, year, venue:** Ekraam Sabir, Stephen Rawls, Prem Natarajan (Information Sciences Institute, University of Southern California). ICDAR 2017 per DOI; arXiv 2018.
- **Model analysed:** Segmentation-free CTC line recogniser. Input is a 30 px high line, sliding window of 2 px width turned into a 60x1 vector, then 2 fully connected layers (60 units, ReLU), then 2 bidirectional LSTM layers (256 units each, 700 time-steps), CTC loss (Sec. III.C). Trained on synthetic English text: 32,180 unique sentences from "A Tale of Two Cities" and "Ivanhoe", font sizes 8 to 16, 6 fonts. Validation: Tom Sawyer sentences. Test fonts differ from training fonts (Comic Bold, Times, Arial Narrow). Latin script only. Training: lr 0.001, dropout 0.5, 1 epoch, 0.04% CER on train (Sec. III.D).
- **Question:** Does a CTC-LSTM trained for OCR learn an "implicit LM", and over how many characters of context does it act?
- **Assessment method:**
  1. Shuffled-characters experiment (Sec. IV.A): 3,742 sentences from Wuthering Heights, rendered in the test fonts, compared with the same sentences with characters randomly shuffled (same characters, random character LM). CER difference is attributed to the implicit LM.
  2. N-gram experiment (Sec. IV.B): test sets of fixed-length strings (2 to 7 letters, a-z, spaces removed, up to 10,000 samples each). "Seen" = n-gram frequency above 10 in the training sentences; "Unseen" = frequency 0; "Purely Unseen" = Unseen and all sub-n-grams Unseen too. Expectation: for Seen strings CER falls with length and plateaus where the context limit is reached; for Purely Unseen strings it should not improve.
  3. Per-character check and a control that fixes the share of the letter "e" (Sec. IV.B/C, Tables 3 and 4) because one font (Times) has a rogue character.
  - Controls: shuffled vs normal text with identical characters; Purely Unseen sets; training fonts excluded from test; background noise removed by using synthetic images. No weights or hidden states inspected, no repeats or seeds reported (mine, not stated).
- **Evidence the method measures what it claims:** Partly argued, not tested. The authors say any single CER is entangled: "it is not possible to isolate the contributions of implicit LM by a single measure of performance on any data" (Sec. II). They answer with several experiments that "together give a clearer picture". The Times result is an admitted counter-case: the "e" error stays around 72% regardless of context, and they "do not come across anything credible" as explanation (Sec. IV.C).
- **Main finding:** Shuffled characters give higher CER; the abstract states "2.4% CER improvement on our synthetic test set" (Table I: Arial Narrow 0.5% normal vs 2.9% shuffled; Comic Bold 2.4 vs 2.6; Times 7.2 vs 8.6). Seen n-gram CER stops improving beyond 5 characters, which they convert to "roughly 88 frames" for font size 16 on the widest font (Comic Bold). Conclusion: implicit LM uses up to 5 characters of context; it does not always help the current character.
- **Limits (stated):** Synthetic, single language (English), fixed font families; character context is only an approximation of frames ("though not perfect"); results are font-dependent (Times). The authors claim the inference "should hold good for any language" without testing.
- **Limits (mine):** CER differences are small (0.2 to 2.4 points) and no confidence intervals are given. The 5-character figure rests on a plateau read off Table II, not on a fitted statistic. Short strings of 2 to 7 letters have no spaces or word structure, so "context" here is letter-level only. Line-level real handwriting and word-level LM effects are not touched. Behaviour is inferred from outputs only.
- **Relevance:** This is the output-side recipe (same pixels, different text statistics, plateau of context) that can be run on any kraken CTC model to check whether a weight edit changes the implicit-LM reach, while the paper offers no parameter-level evidence on where the LM sits.

---

## D2. Understanding Cross-Language Transfer Improvements in Low-Resource HTR: The Role of Sequence Modeling

- **Full text:** https://arxiv.org/pdf/2605.05900 (arXiv v1, 15 pp.).
- **Authors, year, venue:** Sana Al-azzawi, Chang Liu, Nudrat Habib, Elisa Barney, Marcus Liwicki (Luleå University of Technology). arXiv preprint, cs.CV, 7 May 2026. No venue stated; code "will be publicly released upon acceptance".
- **Model analysed:** Two line-level models on one ResNet-style conv backbone (channels 64/128/256, column-wise pooling to a sequence): (a) CNN-only + CTC, about 5.8M parameters; (b) CRNN = same backbone + 3 stacked BiLSTM layers (256 hidden units per direction) + CTC, 10.1M parameters; (c) a widened CNN-only control (channels 86/172/344, about 10.4M). Data: Arabic KHATT, Urdu NUST-UHWR, Persian PHTD, K = 100, 500, 1000 target lines; single-script (J=0) vs multi-script (J=2, two auxiliary datasets). 110 x 1450 px input, AdamW, lr 5e-4, batch 16, augmentation, right-to-left transcriptions reversed. Three seeds reported (Sec. 3.5).
- **Question:** Are cross-language gains better explained by shared visual features (the CNN) or by sequence-level modelling (the BiLSTM)?
- **Assessment method:** Architecture ablation, not probing. Compute delta CER = CER(multi) - CER(single) for each architecture and K (Sec. 4). Control for capacity with the widened CNN (Table 2, only KHATT and PHTD). Qualitative error examples at K = 500 (Fig. 4). The BiLSTM is never inspected, perturbed or frozen; its role is inferred from the architecture difference.
- **Evidence the method measures what it claims:** Weak and partly conceded. The authors write the comparison "reflects a combined architectural effect rather than attributing improvements solely to sequence modeling" (Sec. 3.2) and that the relation should be read "in terms of association rather than strict causation" (Sec. 4.2). The capacity control helps against a pure parameter-count explanation. Figures show a single representative run; seeds said to agree but not tabulated.
- **Main finding:** CRNN gets larger and more stable delta CER (KHATT K=100: 26.7 to 19.9, delta -6.8; PHTD K=100: 31.4 to 18.3, delta -13.1) than the CNN-only model (29.6 to 25.8; 31.6 to 25.5). CNN-only sometimes shows positive delta (negative transfer, e.g. KHATT K=500/1000 +0.20/+0.30). Widened CNN did not close the gap (PHTD K=1000: -0.90 to +0.83). At K=100 on NUST-UHWR the two are nearly equal (-8.00 vs -7.90).
- **Limits (stated):** Association not causation; CRNN has more capacity; multi-script setting fixed at J=2; Arabic-script languages only; future work on transformers and other scripts.
- **Limits (mine):** The CNN-only model has a receptive field set by the ResNet, so "no sequence model" is not "no context"; CTC alone gives frame-level independence. The conclusion is about transfer across scripts that share letterforms and vocabulary, not about the LM as such. Delta CER differences at K=500/1000 are 1 to 3 points on one reported run. The text calls CNN-only gains "more modest" (-6.10 and -7.90), but on NUST-UHWR at K=100 that is essentially equal to the CRNN (-8.00), so the CRNN advantage is not uniform.
- **Relevance:** It tests the BiLSTM stack only by removing it, so it supports the view that the recurrent part carries what is shared across languages but gives no handle on which BiLSTM weights; a weight-space edit that touches only the CNN versus only the LSTM would be the matching, finer-grained version.

---

## D3. Analysis of the Calibration of Handwriting Text Recognition Models

- **Full text:** None open. OpenAlex/Unpaywall: closed, no repository copy; no arXiv or HAL record found (arXiv API title query and HAL API returned nothing). Springer page sits behind a client challenge, which I did not attempt to bypass. No abstract could be retrieved either, so this entry is metadata only.
- **Authors, year, venue:** Eric Ayllon, Francisco J. Castellanos, Jorge Calvo-Zaragoza (University of Alicante listing, PRAIG). Document Analysis and Recognition - ICDAR 2024, LNCS vol. 14805, pp. 139-155, Athens, 2024; DOI 10.1007/978-3-031-70536-6_9.
- **Model analysed:** UNVERIFIED.
- **Question:** UNVERIFIED (title: calibration of HTR models; OpenAlex keywords: model calibration, confidence estimation, out-of-distribution generalisation).
- **Assessment method:** UNVERIFIED.
- **Evidence the method measures what it claims:** UNVERIFIED.
- **Main finding:** UNVERIFIED.
- **Secondary evidence only:** D4 (Garrido-Munoz and Calvo-Zaragoza, same group) computes "the Expected Calibration Error (ECE) for each model, following the definition in [7]" (D4 Sec. 3.3), where [7] is this paper. So the ECE definition used in D4 can be taken to be this paper's; nothing more is verified.
- **Limits:** No text read.
- **Relevance:** Calibration is about output confidence, not about what the network computes with its text input, so for weight-space work it is at most a way to check that an edit leaves confidence usable; obtain the paper (author copy via Calvo-Zaragoza or the PRAIG page, or interlibrary loan) before citing details.

---

## D4. On the Generalization of Handwritten Text Recognition Models

- **Full text:** https://arxiv.org/pdf/2411.17332 (arXiv v2, 1 Jun 2025; OpenAlex marks it as the green OA copy of the CVPR DOI).
- **Authors, year, venue:** Carlos Garrido-Munoz, Jorge Calvo-Zaragoza (University of Alicante). CVPR 2025 per DOI 10.1109/CVPR52734.2025.01423 (the arXiv PDF does not state the venue). Code: github.com/carlos10garrido/HTR-OOD.
- **Model analysed:** Eight published HTR architectures trained from scratch, one source dataset at a time: CTC (CRNN of Puigcerver, VAN, CNN-SAN, HTR-ViT), sequence-to-sequence (ResNet + Transformer), hybrid (three models incl. Michael et al.). Seven datasets (IAM, Rimes, Bentham, Saint Gall, George Washington, Rodrigo, ICFHR 2016/READ), five languages, 336 OOD cases (train on A, test on B). Unified 94-character vocabulary, 500 epochs, early stop after 100 epochs without 0.1 gain, no LM or lexicon post-processing (Sec. 3.3). Synthetic 100K lines per language (English, French, Spanish, German, Latin) from WIT with 4,000 fonts.
- **Question:** What limits generalisation to out-of-distribution data, in particular how much is textual vs visual divergence, and how do models use synthetic data and "the internal LM"?
- **Assessment method:**
  1. Cross-domain CER and ECE matrix (Table 3, Sec. 4).
  2. Synthetic-data test: train on synthetic text in the target's language vs another language, compare (Fig. 4, Sec. 4.2). Used to quantify the influence of the "underlying language model".
  3. Proxy metrics: textual divergence = average KL-divergence over n-gram distributions, n = 1..5, between source and target transcriptions; visual divergence = reconstruction error (MSE) of a convolutional autoencoder trained on the source (about 33M parameters, 64 x 1024 input; Appendix 9).
  4. Factor analysis (Sec. 5) on: parameter count, ID/OOD CER, ID/OOD ECE, visual and textual divergence; 4 factors with eigenvalue >= 1, oblimax rotation.
  5. Regression from label-free proxies to OOD CER (Sec. 5.2, Fig. 7). The regressor type is not named in the main text (UNVERIFIED).
- **Evidence the method measures what it claims:** Correlational only. The factor loading for Factor 1 matches textual metrics (0.92 and 0.84) and correlates 0.62 with OOD error; Factor 2 (visual) correlates -0.3 with OOD error. No intervention separates the two sources, and no check that the autoencoder error measures what a recogniser finds hard. The language-swap synthetic experiment is the nearest thing to a controlled test of the LM influence.
- **Main finding:** "The most significant factor for generalization lies in the textual divergence between domains, followed by visual divergence" (Abstract). CTC models generalise best: average OOD CER 40.4 for CTC vs 48.7 for autoregressive (Sec. 4.1). Autoregressive models lose more when the synthetic language differs (relative drop 25% for AR vs 19% for CTC), read as "explicitly trained to compute an LM, unlike CTC models" (Sec. 4.2). More parameters do not help OOD (Fig. 5). OOD CER predicted with mean absolute deviation of 10.9 points; about 70% of cases within 10 points (Fig. 7).
- **Limits (stated):** Single-domain generalisation only; the authors say more study of the visual factor is needed (Sec. 6). No adaptation methods tested.
- **Limits (mine):** The Fig. 7 caption says "MSE of 10.9" while the text says MAE: inconsistent. Textual divergence is computed from ground-truth text, so it also tracks alphabet and vocabulary mismatch (a unified 94-symbol output means some symbols are never trained in a source domain). "Internal LM" is not measured inside the network; it is inferred from a CTC vs AR contrast, and CTC/AR differ in many other ways. Models are in the 2020-2023 literature at their original hyper-parameters, so the comparison mixes architecture with training recipe. One run per cell appears to be reported (UNVERIFIED).
- **Relevance:** It treats text prior as the dominant source of OOD error in CTC line models (a kraken CRNN is one of the compared families), motivating weight edits that change the text-side behaviour while leaving the visual stack alone, but it offers no parameter-level localisation.

---

## D5. Evaluating Sequence-to-Sequence Models for Handwritten Text Recognition

- **Full text:** https://arxiv.org/pdf/1903.07377 (arXiv v2, 15 Jul 2019).
- **Authors, year, venue:** Johannes Michael, Roger Labahn, Tobias Gruning, Jochen Zollner (University of Rostock; PLANET artificial intelligence GmbH). The arXiv PDF does not name ICDAR 2019 (UNVERIFIED in the PDF); the ICDAR 2019 attribution is from the task brief.
- **Model analysed:** Encoder: CNN + 3 BLSTM layers + a 1D conv output layer with CTC-compatible shape; decoder: single unidirectional LSTM, 256 units, 64-d learned character embeddings, attention layer size 128, beam width 16 (Sec. IV). Data: IAM, Bozen (READ 2016), StAZH. Adam, lr 0.001, 200 "runs" of 8,192 samples, line images at height 64 px, hybrid loss lambda = 0.5 (CE + CTC).
- **Question:** Which attention mechanism and positional encoding give a good alignment between input and output, and can the encoder keep an interpretable CTC output? The paper hopes attention "enables the networks to potentially model language structures" (Sec. I, motivation only).
- **Assessment method:** Only validation CER is compared. Six attention types (content, penalized, location, monotonic, chunkwise, hybrid), sinusoidal vs learned positional encodings, fixed (frozen CTC-pretrained) encoder vs hybrid-loss encoder (Sec. V-B, V-C, Tables III to V). Runs repeated three to five times with mean [min, max]. No attention map or alignment is plotted in the text: the only figure is the architecture diagram (Fig. 1). Alignment quality is judged indirectly from CER and from error symptoms (repeated snippets, missing end-of-sequence).
- **Evidence the method measures what it claims:** For "good alignment" the evidence is CER and described failure modes, e.g. with a frozen encoder purely content-based attention gives 64.92% CER because "the decoder tends to arrange correct or similar text snippets in a quite arbitrary order and repeats certain parts" (Sec. V-B). No direct inspection of alignments, so the attention-versus-language-model question is not tested.
- **Main finding:** A frozen CTC encoder with attention decoder does not beat the stand-alone encoder; with hybrid loss the decoder matches it (IAM validation CER about 3.5% for both, Table IV). Monotonic and chunkwise attention work best; positional encodings add nothing once hybrid loss is used; dropping CTC loss makes the encoder lose "its text-predictive power" but does not hurt final CER (Table V). The authors conclude the hybrid loss is "not necessary for the overall performance regarding HTR" (Sec. VI), and leave language-model integration to future work.
- **Limits (stated):** KWS and LM integration out of scope; whether an LM in the decoder beats CTC with a conventional LM is left open (Sec. VI).
- **Limits (mine):** The task brief's expectation of an attention/alignment analysis is not met in this paper: there is no attention visualisation or alignment measurement. Its useful content is structural: it shows a CNN + 3 BiLSTM encoder is a complete CTC recogniser and that a decoder can be learned on top of a frozen one only with difficulty.
- **Relevance:** The frozen-encoder result (decoder on fixed CTC encoder states fails without positional information) is evidence that BiLSTM states of such an encoder carry text-predictive content but little position information, a useful constraint for any edit read from or written to those states.

---

## D6. Multivariate analysis of the parameters in a handwritten digit recognition LSTM system

- **Full text:** None obtained: abstract only. DiVA (kth.diva-portal.org, record diva2:1415918, URN urn:nbn:se:kth:diva-271187) refused connections or timed out from this machine (curl and WebFetch), and the Internet Archive snapshot of the record returned HTTP 429. The abstract below comes from the uppsatser.se mirror (https://www.uppsatser.se/uppsats/fee042b1a0/), which points to DiVA for the PDF.
- **Authors, year, venue:** Georgios Zervakis. Master's thesis, KTH, School of Electrical Engineering and Computer Science (EECS), 2019 (OpenAlex W3014493027). Supervisor: not retrieved.
- **Model analysed (abstract):** An LSTM system for handwritten digit recognition on MNIST, a "balanced and simple" dataset. Architecture details (units, how an image is fed as a sequence) are UNVERIFIED.
- **Question:** "what in the network is responsible for the model arriving at a certain decision", i.e. interpretability of the model's behaviour from its parameters.
- **Assessment method (abstract only):** Starts with the final linear output layer. Applies PCA and SVD to its weight matrix; tries low-rank approximations of the output weight matrix and checks the importance of each singular vector per digit class. Then proposes "a framework for analysing the parameters of the hidden layer" together with an LSTM-based variational autoencoder implementation. Keywords: PCA, SVD, variational autoencoders. Which hidden-layer parameters (input, recurrent, gate matrices) and how they are analysed: UNVERIFIED.
- **Evidence the method measures what it claims:** UNVERIFIED. The reported test is functional: ablate a singular direction, observe the class accuracy.
- **Main finding (abstract):** "cutting off the fifth left and right singular vectors the model practically losses its ability to predict eights" (verbatim from abstract, with that spelling). The hidden-layer part is described only as a framework.
- **Limits:** Abstract only. (mine) Class-wise digit recognition has no text or sequence-context structure, so nothing carries over about language or context; closest in method, not in object.
- **Relevance:** It is the one item in this batch that works on weights: SVD of the output matrix with singular-vector ablation per class, an approach that could be tried on the CTC projection layer of a kraken model (per character); read the PDF before citing specifics.

---

## D7. Understanding Transformer-Based OCR for Medieval Manuscripts: A Systematic Ablation Study and Inspection Analysis (Research Square preprint v1) and its follow-up TrOCR for Medieval HTR: A Systematic Ablation Study with Cross-Dataset Validation

- **Full text:**
  - v1: https://www.researchsquare.com/article/rs-8109357/v1.pdf (17 pp., CC BY 4.0, posted 4 Dec 2025).
  - Follow-up: https://arxiv.org/pdf/2606.24302 (arXiv v1, 23 Jun 2026), which OpenAlex links to the Springer DOI 10.1007/978-3-032-36207-0_13. Springer's own page was not accessible (client challenge), so the arXiv copy is what was read; it may differ from the Springer version (UNVERIFIED).
- **Authors, year, venue:**
  - v1: Sachin Sharma, Federico Simonetta (Gran Sasso Science Institute), Research Square preprint, 2025.
  - Follow-up: Sachin Sharma, Michele Flammini, Federico Simonetta (GSSI). Crossref: Lecture Notes in Computer Science, series "Document Analysis Systems", published 2026 (Crossref gives 28 Aug 2026). The arXiv copy does not name the conference.
- **Model analysed:** TrOCR-base (ViT-base encoder, 12 layers; RoBERTa-base decoder, 12 layers; about 334M parameters) fine-tuned on the 13th-century Italian "Cortonese" manuscript I-CT 91 (2,517 lines; split 1,666/397/454; page-level split). Replication on READ-16. External references: a kraken CATMuS-Medieval model fine-tuned on the same data (CER 7.9 in follow-up), VAN, GatedLexiconNet. Latin-script medieval Italian.
- **Question:** Which fine-tuning choices (CLAHE, augmentation, layer freezing) matter for small historical sets; and whether Grad-CAM and decoder cross-attention maps can serve as quantitative error diagnostics.
- **Assessment method:**
  - Layer freezing (this is the nearest thing in the batch to a parameter-level manipulation). Follow-up: freeze the first K in {0, 3, 6, 9, 12} layers of the encoder or of the decoder, one module at a time, plus a combined enc 3 / dec 6; AdamW with One-Cycle schedule, 50 epochs, peak lr 5.5e-6, effective batch 16, weight decay 0.01, label smoothing 0.1. Line-wise CER (N = 454) compared with Wilcoxon signed-rank tests, Holm-corrected, Friedman omnibus test (Table 1). v1 froze only K in {0, 10, 12} encoder layers.
  - "Inspection analysis": token-by-token maps from the final decoder block under teacher forcing. Cross-attention: head-averaged weights reshaped to the patch grid. Grad-CAM: backpropagate the ground-truth logit for the same step, gradient-weighted sum over heads, ReLU (follow-up Sec. 5.2); in v1 the Grad-CAM target is the last encoder block (Sec. 4.7). Each map summarised by normalised entropy, Gini coefficient (and coverage in v1), correlated with token loss (v1: 1,949 tokens from 120 lines; Pearson), then used in a logistic-regression (elastic net, leave-one-out) classifier of correct vs wrong tokens (v1 Sec. 6). The follow-up shows four worked token examples (Table 3), no new statistics.
- **Evidence the method measures what it claims:**
  - Freezing: significance testing at line level; replication on READ-16 shows thresholds do not fully transfer (enc 3 / dec 6 is non-significant on Cortonese, significantly worse on READ-16).
  - Maps: v1 gives correlation with token loss (Grad-CAM entropy 0.66, Gini -0.55; attention entropy 0.27) and 66% balanced accuracy for token loss alone vs 67% for Grad-CAM entropy and 73% combined. Correlation with line CER is near zero (0.04). The follow-up explicitly drops the quantitative claim and calls it "qualitative", noting Grad-CAM can collapse to an all-zero map (Table 3, third row) and "attention weights are not explanations by default" (Sec. 1).
- **Main finding:**
  - Freezing: "freezing up to three encoder layers or six decoder layers does not significantly harm accuracy, while deeper freezing becomes progressively detrimental" (follow-up abstract). Table 1: enc 6 frozen +1.66 CER points (significant), enc 9 +5.79, enc 12 +8.43; dec 6 +0.34 (ns), dec 12 +2.23. The encoder (visual stack) must adapt; the decoder (language side) is more transferable. (mine: the decoder tolerance is evidence the pre-trained text prior in the decoder is reusable for Italian medieval text, though the decoder is not isolated from the encoder.)
  - Contradiction between versions: v1 says CLAHE gives a "12.9% relative CER reduction" and best CER 11.68%; the follow-up reports best CER 8.03%, CLAHE not needed (7.84% without), a different training schedule. Cite the follow-up, not v1, for results.
  - Inspection: attention is diffuse even on correct tokens, Grad-CAM follows stroke edges; in v1 Grad-CAM metrics correlate better with token loss than cross-attention does.
- **Limits (stated):** Single small manuscript plus READ-16 as "robustness evidence rather than proof of universal generalization"; the follow-up calls the diagnostics qualitative and notes degenerate Grad-CAM maps.
- **Limits (mine):** No language-content test: the inspection is about where on the image the model looks, not what text prior it uses. Freezing "prefixes" of layers is a coarse manipulation, with no per-layer or per-weight resolution. A TrOCR encoder-decoder with BPE tokens is not a CTC line model. v1 and follow-up disagree on the maps' definition and on headline numbers. Dataset not yet released.
- **Relevance:** Its freezing grid (encoder layers hurt, decoder layers tolerated) is the closest published parallel to deciding which part of a recogniser a weight edit can leave alone, but it is a training-time result on a transformer, not a read-out of what stored weights do.

---

## D8a. Impact of Image Enhancement Methods on Automatic Transcription Trainings with eScriptorium

- **Full text:** https://jdmdh.episciences.org/11569/pdf (open access; DOI 10.46298/jdmdh.10262).
- **Authors, year, venue:** Pauline Jacsont, Elina Leblanc (University of Geneva). Journal of Data Mining and Digital Humanities (JDMDH), special issue "Historical Documents and automatic text recognition", published 12 Sep 2023.
- **Model analysed:** Kraken recognition models trained with default settings (`ketos train -f alto ... -d cuda`), plus segmentation models (reported not to improve). Data: 19th-century Spanish chapbooks (printed, not handwritten); GT of 198 pages (transcribed with ABBYY FineReader, then manually corrected), split 80/10/10 into train, validation, test; low-resolution JPG with bleed-through (Sec. III).
- **Question:** Does changing the image input (binarisation, deblur, gamma) change recognition accuracy? Not about text handling.
- **Assessment method:** Same ALTO XML each time, only the images differ. Phase 1: one pre-processing at a time: OpenCV thresholding, Otsu, Niblack, Sauvola, kraken's built-in binarisation, deblur, two gamma corrections, versus no pre-processing. Phase 2: four combinations. Metric: per-character accuracy on the validation set and on a test set. Extra qualitative check on a few lines of the worst chapbook (Table 1). Single training per condition; no repeats or seeds stated (mine: none mentioned).
- **Evidence the method measures what it claims:** None stated beyond the held-out test split; no confidence intervals or repeated runs. Figure 3 values are scrambled in text extraction, so I do not report per-method numbers (UNVERIFIED which value belongs to which method).
- **Main finding:** "binarisation methods give better results than the other, and that the combination of several techniques did not significantly improve the transcription prediction" (Abstract); Niblack was chosen for the corpus (Sec. IV). Authors note a different project with a high-resolution 16th-century Latin manuscript found no gain from enhancement (Sec. V).
- **Limits (stated):** One corpus of printed, low-quality material; they call for larger and more varied corpora.
- **Limits (mine):** Pure input-perturbation at training time; nothing about what the network does with text. The model is retrained each time, so it speaks to data preparation, not to a fixed network's behaviour. No statistical treatment.
- **Relevance:** Evidence that kraken recognisers are sensitive to image pre-processing on poor scans, which matters when a weight edit is evaluated on images preprocessed differently from training.

---

## D8b. Do (colored) backgrounds matter? An experiment on artificially augmented ground truth for handwritten text recognition applied to historical manuscripts

- **Full text:** Abstract only. HAL record hal-04622805 (CSDH/SCHN 2024, COMM; PDF at https://inria.hal.science/hal-04622805/file/CSDH%202024%20-%20Do%20%28colored%29%20backgrounds%20matter_-2.pdf) is behind an anti-bot challenge ("Making sure you're not a bot!"), which I did not bypass. A second HAL record, hal-04450004 (31 Jan 2024), has the same title and is behind the same challenge. No Zenodo copy found. Abstract and metadata obtained from the HAL search API.
- **Authors, year, venue:** Alix Chague, Hugo Scheithauer (HAL: Inria). CSDH/SCHN 2024: "Sustaining Shared Futures", 19 Jun 2024; earlier HAL deposit dated 31 Jan 2024.
- **Model analysed (abstract):** Transcription models trained with Kraken (keyword in HAL record) on the IAM Handwriting Database (original, greyscale) and an enhanced version in which colored backgrounds taken from real historical blank pages are added. Architecture and training hyper-parameters: UNVERIFIED.
- **Question (abstract):** Does background color matter, and can an augmented IAM serve for historical handwriting?
- **Assessment method (abstract only):** "We train various transcription models playing on the composition of trainset and validationset using the original and enhanced IAM-Database." Tested against original and enhanced test sets and against a test set of real historical documents. Also an estimate of environmental cost. Number of runs, seeds, metrics: UNVERIFIED.
- **Evidence the method measures what it claims:** UNVERIFIED.
- **Main finding (abstract):** "though the transcription engine proves robust to color changes, this technique could be used to bring up to speed older grayscale datasets" (abstract, as quoted from HAL); environmental impact of the enhanced data is minor.
- **Limits:** Abstract only. (mine) Input perturbation on the image side, content of the text untouched by design, so it can show invariance to a nuisance factor but says nothing on text handling.
- **Relevance:** A ready-made controlled input perturbation (same text, same lines, only background color changed) that could be used as a null condition to check that a weight edit targeting text behaviour does not alter background robustness.

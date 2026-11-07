# Notes batch G: how each paper assessed what a text-recognition network does with its input

Working files: /tmp/claude-1000/-home-th/0dc8ad2a-3390-417b-9872-a7fd4cdb988d/scratchpad/search/G/ (g1 ... g10 subfolders; PDFs and pdftotext output).
Access summary: full text read for G1, G5 (arXiv v1 only), G6, G7, G8. Abstract only for G2, G3, G4, G9, G10. Springer, SSRN and Elsevier pages returned bot challenges / redirects; I did not try to get past them.
Tobias' list gave "(arXiv or author copy)" for G1 and an IJDAR title for G5; both checked against Crossref/OpenAlex.

## G1. Ty & Atienza, "Scene Text Recognition Models Explainability Using Local Features" (STRExp)

- Full text: https://arxiv.org/pdf/2310.09549 (arXiv v1, 14 Oct 2023; OpenAlex lists it as green OA copy of the ICIP paper). Whether it is identical to the IEEE version: UNVERIFIED. 5 pages, read in full.
- Authors, year, venue: Mark Vincent Ty, Rowel Atienza (Univ. of the Philippines); 2023 IEEE International Conference on Image Processing (ICIP), pp. 645-649 (Crossref, doi 10.1109/ICIP49359.2023.10222406).
- Model analysed: six existing scene-text recognisers: ViTSTR, PARSeq, TRBA, SRN, ABINet, MATRN (Sec. 4.2). Scene-text images (CUTE80, SVT, IIIT5k, SVTP, IC03, IC13, IC15 test sets). Training data and checkpoints not described in the paper (UNVERIFIED). Word-level crops, not lines.
- Question asked: "Why do STR models work?" (Sec. 1), i.e. which input regions drive a recogniser's output.
- Assessment/explanation method: take existing attribution methods from Captum (Integrated Gradients, GradientSHAP, DeepLift, Saliency, Input x Gradient, Guided Backprop, Deconvolution, KernelSHAP, Feature Ablation, LIME, Shapley; Sec. 4.2). Baseline "global" use collapses the output sequence to one score (mean of outputs, "R"). STRExp instead explains each predicted character c_k separately (target = that character at position k) and then combines the local maps with the global map (Sec. 3.2, Fig. 2). The best base method per dataset is chosen by the same selectivity metric. Hyperparameters, the combination rule E(.), the segmentation used to define "features", and the value used to hide removed segments: not stated (UNVERIFIED).
- Evidence the method measures what it claims: one automatic metric, "selectivity" (Montavon et al.): remove highest-scored segments first and plot STR accuracy / confidence against fraction removed; lower area under the curve = better (Sec. 4.1). Plus qualitative judging by the authors: "greener areas inside and near the text", fewer red areas on text (Sec. 4.3, Fig. 4). No human study, no ground-truth character boxes, no sanity checks (e.g. randomised weights).
- Main finding: STRExp "generally has the lowest selectivity accuracy and lowest selectivity confidence in the majority of cases" (Sec. 4.2). Headline qualitative claim: previous methods give attribution far from the text (false positives) and low attribution on the text (false negatives) (Fig. 1).
- Limits: stated: none (no limitations section). Mine: the same selectivity metric is used to pick the best base method and to compare methods (circular); selectivity can reward maps that are merely out-of-distribution-damaging when pixels are removed; "attribution on the text region" is taken as correct without checking that it is on the right characters; results are only plotted, not tabulated, and no variance is given.
- Relevance to kraken-type CTC line recogniser: this is the closest method precedent for per-output-symbol attribution (explain character k, not the whole string) with a deletion-type faithfulness check; but it is word-level scene text with attention/transformer decoders, none of them CTC (mine: from architecture knowledge, not stated in the paper), so mapping character k to a CTC frame range and to a line crop is not addressed.

## G2. Dutta & Muppalaneni, "Understanding character recognition using Grad-CAM and predicting faithfulness through human evaluation"

- Full text: abstract only. Springer page returns a bot "Client Challenge"; OpenAlex: closed. Abstract content below comes from a library-catalogue record as summarised by a web search result, not read by me directly: UNVERIFIED wording.
- Authors, year, venue: Prarthana Dutta, Naresh Babu Muppalaneni; Evolving Systems, vol. 16 (Crossref: online 2025-05-17; catalogue: June 2025, issue 2, pp. 1-18).
- Model analysed: per the catalogue summary, a CNN recognising handwritten Assamese digits (UNVERIFIED; digits, not text lines).
- Question asked: whether Grad-CAM explanations of a character/digit classifier are faithful, judged by humans (from title).
- Assessment/explanation method: Grad-CAM heatmaps; human evaluation of faithfulness (from title/summary). Study design, number of raters, rating scale, baselines: UNVERIFIED (not seen).
- Evidence the method measures what it claims: "predicting faithfulness through human evaluation" suggests a human-rating study; whether any automatic faithfulness metric or ground truth exists: UNVERIFIED.
- Main finding: UNVERIFIED.
- Limits: unknown. Mine, from the title alone: human judgement of "does the heatmap look right" measures plausibility, not faithfulness to the network.
- Relevance to kraken-type CTC line recogniser: low and indirect; isolated digits with a Grad-CAM on a CNN classifier, no sequence output, no CTC. Do not cite for line/word transcription.

## G3. Amirgaliyev, Amilbek, Yedilkhan, Tleuken, "Comparative Analysis of Explainable Artificial Intelligence Methods for Optical Character Recognition of Arabic Historical Manuscripts"

- Full text: abstract only. SSRN abstract page returned a Cloudflare challenge (HTTP 403); not bypassed. Abstract text taken verbatim from Crossref (doi 10.2139/ssrn.7664871, type posted-content, 2026). Not peer-reviewed as far as the record shows (SSRN preprint).
- Authors, year, venue: Beibut Amirgaliyev, Nurbolat Amilbek, Didar Yedilkhan, Aidana Tleuken; 2026; SSRN Electronic Journal. (Title as given by Crossref/OpenAlex; the task text had it truncated.)
- Model analysed (abstract): "CRNN-CTC, TrOCR-base, TrOCR-large, and Arabic Nougat", "trained on RASAM dataset" (Arabic/Maghrebi manuscripts). Best CER: TrOCR-large, 0.1618. Input unit (line vs word) not stated in the abstract: UNVERIFIED (RASAM is line-level as far as I know from the dataset, not from this paper).
- Question asked: "evaluate the quality and consistency of explanation for various classes of neural networks" and use XAI "to diagnose and eliminate weaknesses in recognition models".
- Assessment/explanation method (abstract): "Occlusion Sensitivity, LIME, and KernelSHAP". Window sizes, superpixels, baselines: UNVERIFIED.
- Evidence the method measures what it claims (abstract): "Deletion AUC, Insertion AUC, AOPC, Spearman correlation, and stability index". These are perturbation faithfulness metrics plus inter-method agreement and stability; no human study or ground truth mentioned.
- Main finding (abstract): "LIME and KernelSHAP methods showed the greatest mutual consistency"; occlusion "most informative for transformers, while surrogate methods are most valuable for CRNN architectures"; maps drove changes to the CRNN-CTC baseline ("SE-attention, positional coding, diacritics augmentation") that "increased Char F1 index by 9.7%".
- Limits: stated: none in abstract. Mine: agreement between LIME and KernelSHAP is expected (both are local surrogate/Shapley estimators) and is not evidence of correctness; whether the +9.7% Char F1 was against a control change not motivated by the maps: UNVERIFIED.
- Relevance to kraken-type CTC line recogniser: the only paper in this batch that puts a CRNN-CTC through a perturbation-faithfulness comparison on historical-manuscript text; read the full text if a copy appears. Whether it works on whole lines or words is unknown from the abstract.

## G4. Benaissa, Bahri, El Allaoui, "Why Did the AI Read It that Way? Explainable OCR for Transparent and Accountable Multilingual Administrative Document Processing"

- Full text: abstract only, and not even that: Springer chapter page not reachable (bot challenge on the Springer host), Crossref and OpenAlex carry no abstract, OpenAlex marks it closed. Everything below except bibliographic data is UNVERIFIED.
- Authors, year, venue: Ali Benaissa, Abdelkhalak Bahri, Ahmad El Allaoui (Crossref); Lecture Notes in Networks and Systems, 2026, book chapter pp. 124-131, doi 10.1007/978-3-032-28094-7_13. The title as given by OpenAlex is complete (the task text had it truncated).
- Model, question, method, evidence, finding, limits: not available. Eight pages; from the title only: explainable OCR for administrative documents in several languages (mine, inferred from title, unconfirmed).
- Relevance to kraken-type CTC line recogniser: unknown. Do not cite beyond the bibliographic entry until the text is read.

## G5. Azimi, Chang, Gold, Karabina, "Improving accuracy and explainability of online handwritten character recognition"

- Full text: https://arxiv.org/pdf/2209.09102 (arXiv v1, 14 Sep 2022, titled "Improving Accuracy and Explainability of Online Handwriting Recognition"). The IJDAR version (Crossref: published online 2023-12-19) was not accessible; differences between v1 and the journal version: UNVERIFIED. An NRC repository copy returned HTTP 410.
- Authors, year, venue: Hilda Azimi, Steven Chang, Jonathan Gold, Koray Karabina; IJDAR 2023 (Crossref), doi 10.1007/s10032-023-00456-5.
- Model analysed: 16 classifiers (FCN, ResNet, LSTM, BiLSTM, InceptionTime, XceptionTime, XCM etc., plus optimised variants and ensembles) on OnHW-chars: single letters a-z (lower, upper, combined) written with a sensor-fitted pen, 13 channels of time series (accelerometers, gyroscopes, magnetometers, force) from 119 writers. Online isolated characters.
- Question asked: why are the models' predictions made as they are, and why does an ensemble make sense (Sec. 1: "provide some level of explainability").
- Assessment/explanation method: (a) "failure space" analysis: compare which test samples each of the 16 models gets wrong (Sec. 4.2.1, Figs. 5-6). (b) LIME for time series ("lime-for-time"): series divided into 20 slices, importance of each slice per channel from LIME; top 30 channel/slice cells examined; shown for two single examples (Sec. 5, Figs. 7-8). Perturbation/surrogate settings otherwise not stated.
- Evidence the method measures what it claims: none stated. No faithfulness metric, no ground truth, no human study. Two examples (Optimised LSTM correct "B"; InceptionTime correct vs XceptionTime wrong on "M" predicting "A") are described.
- Main finding: channels 6, 8 and 12 carry most weight: "These results are consistent among most of the models and most of their predictions." (Sec. 5). Models that disagree "derive importance from different parts of the MTS", used as motivation for ensembling.
- Limits: stated: "it would be interesting to run a deeper analysis on the explainability of models" (Sec. 6, future work, paraphrase of closing remarks). Mine: explanation is used to justify an architectural choice (ensembling), not validated; two examples cannot support "consistent among most"; which sensor channel matters says nothing about what in the written shape matters.
- Relevance to kraken-type CTC line recogniser: negligible; isolated online letters on pen-sensor time series, no CTC, no lines. Useful only as an example of LIME-on-slices used without any validation of the explanation.

## G6. Hamdan, Rahiche, Cheriet, "HTR-JAND: Handwritten Text Recognition with Joint Attention Network and Knowledge Distillation" (attention analysis only)

- Full text: https://arxiv.org/pdf/2412.18524 (v1, 24 Dec 2024; banner: "THIS PAPER IS CURRENTLY UNDER REVIEW AT IEEE TRANSACTIONS"). Read Sec. III-B (architecture), Sec. VI-B, VI-C, VI-E, VII for attention claims.
- Authors, year, venue: Mohammed Hamdan, Abderrahmane Rahiche, Mohamed Cheriet; arXiv preprint 2024.
- Model analysed: CNN with FullGatedConv2d + Squeeze-and-Excitation blocks, BiLSTM, then a "Combined Attention" (multi-head self-attention fused with "Proxima Attention"), CTC loss plus auxiliary classification loss; Teacher 1.5M and Student 0.75M parameters via knowledge distillation; T5 post-correction. Lines/text images on IAM, RIMES, Bentham (also Saint Gall, Washington). Line level, CTC.
- Question asked: "To analyze the behavior and decision-making process of our model, we employ various visualization techniques" (Sec. VI-E).
- Assessment/explanation method: three visualisations, no computed quantity beyond plotting: (1) Fig. 5 "attention heatmaps" for sample line images, "warmer colors indicating stronger attention weights"; (2) Fig. 6 "class probabilities heatmap for character alignment" on RIMES (diagonal = correct, off-diagonal = confusions), described as attention allocation; (3) Fig. 7 frame from an animation of "dynamic focus shifts" (animation on GitHub, not checked). Which layer's attention is plotted, how it is mapped to image pixels, averaging over heads: not stated (UNVERIFIED).
- Evidence the method measures what it claims: none stated. No perturbation test, no ground-truth alignment, no human study. The ablation (Table V, Sec. VI-C) covers distillation, curriculum, ensemble, multi-task and lexicon correction, and does not remove the attention module as far as the text shows.
- Main finding (authors' reading of one heatmap): "The model shows a distinct focus on character-specific features, especially ascenders and descenders" and "bright spots at word boundaries suggest the model has learned to recognize spaces" (Sec. VI-E1). They also say the "intensity of attention correlates with the model's character-level recognition confidence" (Sec. VI-B) without a measurement.
- Limits: stated: none for the attention analysis. Mine: attention weights are read as explanation of the CTC output though the CTC head sits after BiLSTM and attention; Fig. 6 is a class-probability/confusion map relabelled as attention; claims like "learned to recognize spaces" and "retains context" are from single examples; the Table IV T5 error-correction results are unrelated to what the network attends to.
- Relevance to kraken-type CTC line recogniser: shows the typical (weak) way HTR papers talk about "what the model attends to" on lines: a heatmap plus narrative with no test. Line-level and CTC, but the attention layer is an add-on that kraken's VGSL stack does not have.

## G7. Whitten, Wolff, Papachristou, "An AI Architecture with the Capability to Explain Recognition Results"

- Full text: https://arxiv.org/pdf/2406.08740 (v2, 3 Jul 2024; 4 pages, read in full).
- Authors, year, venue: Paul Whitten, Francis Wolff, Chris Papachristou (Case Western Reserve University); arXiv 2024 (no venue given in the source).
- Model analysed: not a deep text recogniser. A committee of per-"property" classifiers (MLP or SVM from scikit-learn): each applies a hand-designed transform (circle, ellipse, crossing, endpoints, enclosed region, stroke, Harris corners) to the image, then an inference engine votes. Plus one "unexplainable" flow on the raw image. MNIST digits and EMNIST characters (isolated).
- Question asked: how to give explanations "in plain terms" without losing accuracy; how to quantify explainability of a decision.
- Assessment/explanation method: by construction: each vote is tied to a named geometric property; output text lists which properties voted for each class (Sec. III-A). Metric Ex_d = effectiveness-weighted fraction of votes from explainable flows (Eq. 7), with X_j set by hand to 1 for property flows and 0 for the raw-image flow (Sec. III-B). New per-class weight E_PARS = P x ACC x R x S (Eq. 8), compared with AUC, F1, Cohen's kappa, MCC etc. (Table II).
- Evidence the method measures what it claims: accuracy comparison only (Table II: explainable-only vs explainable+unexplainable; MNIST/EMNIST, MLP/SVM). The explanations themselves are shown as one sample matrix (Table V, letter "S" in Fig. 2). No human evaluation, no test that the named property actually drives the decision.
- Main finding: adding the opaque flow raises accuracy "two to ten percentage points" typically (Sec. IV-A), at the cost of lower Ex_d; E_PARS is the best effectiveness weight; reported up to 98.0% on MNIST.
- Limits: stated: "Results center on explainability and are not meant to compete with the latest unexplainable recognition techniques" (Sec. 1). Mine: Ex_d just counts how much weight went to flows labelled explainable; it says nothing about whether a human understands or trusts the rationale; the properties are not shown to be what a neural net would use.
- Relevance to kraken-type CTC line recogniser: only as the contrasting design (explain by construction with hand-made properties vs post-hoc on a trained net). Isolated digits/letters only, no sequences.

## G8. Feucht, Krojer, Wang, Abrahamsen, Wallace, Bau, "Using OCR Heads to Verbalize Image Semantics"

- Full text: https://arxiv.org/pdf/2609.18823 (v2, 2 Nov 2026). Read abstract, Sec. 2, 3, 4.2, appendices B/C by grep.
- Authors, year, venue: Sheridan Feucht, Benno Krojer, Sarah Wang, Henry Abrahamsen, Byron C. Wallace, David Bau (Northeastern; one independent); arXiv 2026. (Author list per the PDF header; Tobias' note says "Feucht et al.")
- Model analysed: four vision-language models: Qwen3-VL-2B, Qwen3-VL-8B, Molmo2-7B, Llava-Next-34B (Sec. 2, App. A/B). Task: "Word:" prefill transcription of a single random English word (single vocabulary token) placed on an ImageNet background. Not a CTC or line recogniser; autoregressive VLMs.
- Question asked: "How do VLMs map from pixels to semantics?" via the narrow case of OCR.
- Assessment/explanation method: (1) head scoring: apply logit lens to each attention head's output at the last position and take the mean probability of the target word (Eq. 1); (2) causal test: progressively mean-ablate top-scoring heads and measure OCR accuracy, against random late-layer heads of equal number (Fig. 3; random baselines "100 images across 10 seeds"); (3) attention-redirect: force these heads to attend to non-text image tokens and read what is output; (4) "verbalization lens": collapse the heads' parameters into one matrix applied to hidden states, combined with logit lens; evaluated on COCO object confidence and token-level localisation (Table 1) and by concept editing scored by an LLM (o4-mini scoring prevalence, specificity, coherence; 256 ImageNet images, Sec. 4.2).
- Evidence the method measures what it claims: yes, causal: "mean-ablating the top 10% of heads ranked by OCR scores fully degrades task performance" (Sec. 2.2) while random late-layer heads of the same count are "less effective". Editing is the second causal test. Scoring of edits relies on an LLM judge (mine: not human-checked as far as I saw; not verified).
- Main finding: heads causally needed for reading words are general "verbalization" heads: pointed at a bird wing they output "feathers" (Fig. 1); image tokens already carry linguistic information from layer 0.
- Limits: stated: head-finding rests on "a small synthetic OCR task" (Sec. 2.3); late-layer concentration "might be a confound" (Sec. 2.2). Mine: single-word English stimuli on photo backgrounds; no handwriting; "reads the word" here means decoder-side retrieval of image-token content, which has no counterpart in a CTC net with no decoder.
- Relevance to kraken-type CTC line recogniser: a method model (ablate components, then test necessity with a matched random control) rather than a result about HTR; applies to autoregressive VLMs on single words, not to line transcription or CTC.

## G9. Puigcerver, "Are Multidimensional Recurrent Layers Really Necessary for Handwritten Text Recognition?"

- Full text: abstract only (OpenAlex: closed; IEEE not opened). Abstract verbatim from OpenAlex/IEEE metadata.
- Authors, year, venue: Joan Puigcerver; 14th IAPR International Conference on Document Analysis and Recognition (ICDAR), 2017, pp. 67-72 (Crossref), doi 10.1109/ICDAR.2017.20.
- Model analysed (abstract): a model "that relies only on convolutional and one-dimensional recurrent layers", compared with MDLSTM networks for offline handwritten text recognition. Datasets, depth, CTC use: not in the abstract (UNVERIFIED here).
- Question asked: whether two-dimensional recurrent layers are needed for offline HTR.
- Assessment/explanation method (abstract): the paper observes that MDLSTM layers "extract features visually similar to those of convolutional layers". How this is shown (feature-map visualisation?) is not in the abstract: UNVERIFIED. The main comparison is by accuracy and speed.
- Evidence the method measures what it claims: not assessable from the abstract; the visual-similarity claim is stated as an observation motivating the study, not as a tested explanation.
- Main finding (abstract): equal or better results than the best MDLSTM architecture, "runs significantly faster", and random distortions as augmentation "dramatically improves the accuracy". Its closing line: "Probably not."
- Limits: not available.
- Relevance to kraken-type CTC line recogniser: it is the architectural ancestor of conv + 1D BiLSTM line models (line-level), but the only analysis of internal behaviour in the abstract is one qualitative observation about learned features. Whether kraken's VGSL default follows this paper is not stated here and should be checked in kraken docs, not in this note.

## G10. Gan, Chen, Hu, Leng, Wang, Gao, "Characters as graphs: Interpretable handwritten Chinese character recognition via Pyramid Graph Transformer"

- Full text: abstract only. Elsevier DOI redirected to a "Redirecting" page; OpenAlex/Semantic Scholar: closed, no abstract in the metadata. What I have is a web-search summary and the code README (https://github.com/ganji15/PyGT-HCCR); treat the description below as UNVERIFIED against the publisher abstract. A search hit on arXiv 2309.03072 turned out to be an unrelated paper (Jungo et al., character segmentation) and was discarded.
- Authors, year, venue: Ji Gan, Yuyan Chen, Bo Hu, Jiaxu Leng, Weiqiang Wang, Xinbo Gao (Crossref); Pattern Recognition, vol. 137, article 109317, 2023.
- Model analysed: Pyramid Graph Transformer (PyGT): handwritten characters turned into skeleton graphs; graph attention over all nodes plus graph convolution over neighbours; pyramid of shrinking resolutions. Online and offline handwritten Chinese character recognition (isolated characters; README mentions MNIST and HCCR data).
- Question asked: interpretable recognition via the graph representation (title). The abstract-level claims about what "interpretable" means and how it is assessed: UNVERIFIED.
- Assessment/explanation method / evidence: not available. Mine, from the title only: interpretability here is probably built into the representation (nodes = skeleton points) rather than post-hoc.
- Main finding: per the search summary, outperforms 2D-CNN, RNN/1D-CNN and ViT baselines (UNVERIFIED).
- Limits: unknown.
- Relevance to kraken-type CTC line recogniser: low; isolated Chinese characters with a structural input representation, not line transcription.

## Cross-cutting observations (for the related-work section)

1. In every full text read (G1, G5, G6, G7), the assessment of "what the network does with its input" stops at either a heatmap/importance plot judged by eye (G5, G6) or one deletion-style metric (G1). Only G8 tests necessity causally (ablation with a matched random control), and it does so on VLMs, not on HTR networks. None of these papers runs a human study that I could read; the one with a human faithfulness study (G2) is closed.
2. Only G6 (HTR-JAND) is a line-level CTC handwriting model with an attention claim, and its claim rests on unlabelled heatmaps and no ablation of the attention module. G3 (CRNN-CTC on RASAM manuscripts, perturbation metrics) is the one potentially strong precedent, but I could only read its abstract.

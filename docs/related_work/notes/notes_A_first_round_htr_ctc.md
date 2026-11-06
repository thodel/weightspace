# Notes A: how six papers assessed what a CTC-style recogniser does with its text input

Sources: the extracted files in this directory. Quotes are verbatim and short. "UNVERIFIED" = not checkable in the supplied text. Paper 5 (Coquenet thesis) was re-extracted in full from the PDF (`pdftotext -layout`, 183 pages, saved at `../coq/thesis.txt`), so section and figure numbers below come from the whole thesis, not only the first 30 pages.

---

## 1. Zeyer, Schlüter, Ney 2021, "Why does CTC result in peaky behavior?" (arXiv 2105.14849)

- **Model analysed.** No trained recogniser. Toy models on synthetic inputs: a single-softmax-layer "FFNN" with no bias (Def. 4.8, p_t(s|x)=softmax(x_t W)), a "memory model" with a free logit table per frame (Def. 4.15), and a small LSTM (Sec. 6, Sim. 6.1). Target strings are tiny ("a", "abc", the phoneme sequence "p ih ng"). Inputs are hand-built one-hot-like frames (Ex. 4.9, Ex. 5.1). Language and script are irrelevant. The motivation is speech and the cited handwriting work (Bluche).
- **Question asked.** Why does CTC training produce "peaky" outputs, where blank dominates and each label fires in a single frame? And does that follow from the loss rather than from the data?
- **Assessment method.** Formal and symbolic, not an inspection of a trained net.
  - Counting argument over the alignment set. Def. 3.5: a model is peaky if all Viterbi alignments are peaky. Lemma 3.11 and Cor. 3.12 give blank's share of alignment frames: "dominant label is s̊ = B for T ≥ 5".
  - Gradient analysis (Remark 4.3) and a proof that gradient descent from uniform initialisation reaches a peaky local optimum (Thm 4.12).
  - Numerical simulations: Sim. 4.13 (T=16, "p_t(s=B|x) > 88% for all t, and 100% error rate"), Sim. 4.16 (memory model, T=100, ">93%"), Sim. 6.1 (LSTM, average blank posterior against T/N, Fig. 4). Fig. 2 plots loss surfaces and negative-gradient maps over the parameters.
  - Comparison losses: a label-prior "hybrid" loss (Def. 7.1) and a generative loss. These are the controls showing the effect is a property of the CTC criterion.
  - Some proofs were done with SymPy. Code is published (footnote 1, returnn-experiments repository).
- **Evidence the method measures what it claims.** The claim is proved, and the same constructed example is used to show the optimum has "0% error rate" (Remark 4.11). Simulations match the theorems. Controls with other losses do not go peaky (Sim. 7.5). Fig. 4 shows the peaky behaviour appears for T larger than about 20 with N=3 and is absent below that. There is no experiment on a real, trained handwriting or speech model in the supplied text.
- **Main finding about text input.** The dominance of blank is fixed by the label topology and the ratio T/N alone: "independent from the input x" (Remark 3.10). Gradient descent from a uniform start then drives posteriors to spikes. Local-context models "cannot learn the peaky alignment because [they have] only local context" (Sec. 4). The authors also argue that a topology with an unrestricted blank is better than one with silence only between words (Sec. 5).
- **Limits.**
  - Stated: Remark 4.10 says the synthetic inputs are "optimistic" because real audio has dominating silence, which would add to peakiness.
  - Not stated, but visible: all demonstrations are synthetic and tiny, so nothing here shows that a trained line recogniser has particular spike positions. The theory concerns training dynamics from uniform initialisation, not the behaviour of an already trained or edited net.
  - The final sentences of the Conclusions are cut off in both the raw and the layout extraction ("Even with the blank label ..."), so the authors' closing remarks on limits are UNVERIFIED.
  - I found no explicit limitations section.
- **Relevance to weight-space edits.** Because blank dominance and spike timing are set by the CTC topology and by T/N, any edit that changes character shapes or contexts will be read out through spiky, blank-heavy frames. Frame-level attributions (and perturbation checks) therefore need to be taken at the spike frames, not averaged over all frames.

---

## 2. Liu, Wang, Huo 2020, implicit vs explicit LM information in DBLSTM-CTC handwriting recognition (arXiv 2008.01532)

- **Model analysed.** A deep BLSTM with a CTC output layer: 5 BLSTM layers of 240 cells each (120 per direction). Features are 50-dimensional PCA vectors from a 30 px sliding window with a 3 px shift on lines normalised to 60 px height (Sec. II.A, III.A). Output is a 79-class softmax with 52 letters, 10 digits, 15 punctuation marks, space and blank. Training data are up to 1M rendered lines from a Microsoft in-house online-ink corpus (English), subsets of 10K to 1M lines (Table I). The test set is the rendered IAM-online test set. IAM-offline serves only for tool debugging. There is no convolutional front end.
- **Question asked.** How much language-model information is already implicit in the BLSTM, and does an explicit LM still help as training data grow? The authors say they "learn both local character image dependency ... and long-range contextual dependency for implicit language modeling" (Abstract).
- **Assessment method.** Purely behavioural, by comparing decoders.
  - Best-path CTC decoding, which "only leverages the implicit language model information", against WFST decoding (Kaldi) with character n-grams (n = 3, 4, 5, 8, 10) and a word trigram with a 200k-word lexicon. Prior scaling alpha = 0.2.
  - The comparison is repeated across nine training-set sizes. Metrics are CER and WER (Table V, Fig. 1).
  - Learning-rate schedule effects are controlled in Table IV.
  - LMs were built with SRILM on 500 of 4495 LDC2008T15 documents.
- **Evidence the method measures what it claims.** None stated that the implicit LM is actually probed. The implicit LM is inferred only from the equivalence "no-LM decoding performs about as well as an n-gram of order k". There is no probing of hidden states, no scrambled or out-of-vocabulary text test, and no ablation of context. Table IV shows that learning-rate scheduling changes the no-LM numbers a lot (1M lines: 16.2/54.2 without reduction vs 11.8/40.1 with six reductions), so the "implicit LM strength" depends on the training schedule.
- **Main finding.** No-LM decoding matches a 3-gram, then a 4-gram, then beats a 4-gram as data grow, "but cannot surpass the WFST-based decoding with a character 5-gram LM even when about 1M lines" are used (Sec. III.B). At 1M lines: no LM 11.8/40.1 (CER/WER), char 8-gram 7.0/33.5, word trigram 6.0/18.0 (Table V). The small CER gain and the large WER gain from the lexicon "shows clearly the power of lexical constraints". Both the character modelling and the implicit LM improve with more data, and the two cannot be separated. The authors suggest the implicit LM is weak because of data: "only about 44k unique text lines out of about 1M lines" (Sec. IV).
- **Limits.**
  - Stated: only the data-diversity caveat above and a future-work list (pre-processing, better LM, discriminative sequence training, more representative data).
  - Not stated:
    - One model family, one script (English) and one in-house corpus. Training data are rendered online ink, so the domain is synthetic relative to the IAM-online test images.
    - The char 3-gram row has worse WER than no-LM decoding (e.g. 67.1 vs 40.1 at 1M), which suggests a decoder or lexicon-free artefact, not a pure LM effect.
    - The explicit LM changes the decoder as well as adding information (WFST, prior-scaled likelihoods).
    - The 44k unique lines out of 1M make memorisation of whole sentences a likely contributor to the implicit LM (my inference).
- **Relevance to weight-space edits.** If a weight-space edit alters character shapes or context use, a no-LM versus n-gram-LM decoding comparison is a cheap behavioural proxy for how much context the edited net carries, but it cannot say which weights encode it.

---

## 3. Yousef & Bishop 2020, OrigamiNet (arXiv 2006.07491), Sec. 4.9 Model Interpretability

- **Model analysed.** A fully convolutional CTC recogniser (the paper's backbones are VGG, ResNet and the gated "GTR", Table 2) with the OrigamiNet module, which up-scales the feature map vertically, down-scales it horizontally and unfolds a 2D page into one long 1D sequence of length L2 for CTC (Fig. 1b, Sec. 4.4). UNVERIFIED: the text of Sec. 4.9 does not say which backbone or dataset Figs 2 and 3 use (the figure shows IAM-style English paragraph text). IAM paragraphs were scaled to 500x500 for the ablations (Sec. 4.8). Latin script, English text (IAM, 1539 pages, 657 writers).
- **Question asked.** "what does the model actually learn?" (Sec. 4.9). Specifically, whether the model implicitly localises characters and segments lines without ever being given localisation.
- **Assessment method.** Integrated Gradients (path-integrated gradients, citing Sundararajan et al.) ensembled with a SmoothGrad variant.
  - Attribution target: "the attribution of the corresponding output neuron in the 1D prediction map fed to CTC", for one random character per text line.
  - Blank and repeated-label outputs are discarded: "we found their attribution maps to be global and uninformative".
  - IG baseline: an all-white image. "Using white baselines produced much sharper attribution maps than black ones".
  - IG steps: "We used 50 steps to approximate the integral".
  - SmoothGrad: they introduce "SmoothGrad-Abs", which averages absolute values of the maps, because standard SmoothGrad is noisy and the squared variant "suppresses most of the signal". They use "5 noisy images". The noise level is not stated in the supplied text (UNVERIFIED).
  - Fig. 2: heat-maps for single characters over 8 images. Fig. 3: for every character, Otsu-threshold the map, mark its centre of mass, colour markers by text line, and read off the line segmentation.
  - Behavioural side tests (not attribution): touching lines (seam-carved, 6.5% CER), warped lines (6.2% CER) (Sec. 4.7), and a horizontally flipped IAM with line-flipped ground truth that "managed to achieve nearly the same CER" (Sec. 4.10).
- **Evidence the method measures what it claims.** None quantitative. There is no deletion or insertion test, no comparison with ground-truth boxes, no sanity check with randomised weights, and no second attribution method. The only validation is visual (the maps look sharp, the markers form lines) and the white-versus-black baseline observation, which the authors themselves use as evidence of IG's sensitivity ("showing how sensitive IG is to the choice of the baseline"). The baseline was chosen because it gave sharper maps, i.e. a visual criterion.
- **Main finding.** "the model does indeed implicitly learn good character-level localization" (Sec. 4.9), and the centre-of-mass scatter "represents a very good implicit line segmentation". The flipped-IAM result is read as evidence that the model learns reading order from data.
- **Limits.**
  - Stated: multi-column flow is not learned directly (Sec. 4.10). IG baseline sensitivity is acknowledged.
  - Not stated:
    - Qualitative only, with 8 images shown and one randomly chosen character per line.
    - Blank and repeat frames, which are most of the CTC output, were excluded from the analysis.
    - There is no test of what the attribution says about letter shape versus context. It shows where the model looks, not what it uses.
    - Only IAM was visualised, and IAM lines are easy to segment (the authors say so in Sec. 4.7).
    - IG with a white baseline attributes against "no ink", so it measures sensitivity to ink presence, not to ink shape.
- **Relevance to weight-space edits.** It gives a concrete recipe for attributing a single emitted character (IG with a white baseline, 50 steps, noise-averaged, blanks dropped) and it is a warning that attribution maps need a second, causal check (for example masking the attributed ink) before they are trusted on an edited net.

---

## 4. Buoy, Iwamura, Srun, Kise 2023, "Explainable CTC-based scene text recognition" (J. Imaging 9(11):248)

- **Model analysed.** ViT-CTC: pretrained vision transformers (DeiT-Small/Medium/Base, CaiT-Small) as a 2D feature extractor on 224x224 inputs (14x14 feature map). A linear layer gives 2D class scores, a softmax over height and class jointly, then marginalisation over height gives the 1D distribution fed to CTC (Eqs 3-5, Fig. 2). No recurrent layer. Trained on 8.5M synthetic images (MJSynth, SynthText, SynthAdd, SynthTiger) and fine-tuned on 2.4M real labelled images. Evaluated on IIIT, SVT, IC13, IC15, SVTP, CUTE. Character set: 37 classes (case-insensitive letters, digits, blank). Latin scene text, words not lines. Maximum 14 characters because the width is 14 (Sec. 4).
- **Question asked.** Can a 1D CTC decoder give a character-to-image-region link ("explainable 2D spatial relationship between the predicted characters and corresponding image regions") like cross-attention does in a Transformer decoder?
- **Assessment method.**
  - "Association map" (Eq. 6): along the best-path alignment A*, AM(i,j)=1 if the joint 2D probability U(i,j,ind(a_j)) is at least a threshold alpha and a_j is not blank. So blank frames are excluded by construction.
  - "Alignment evaluation metric" (Eqs 7-8): AEM_k = 1 if the estimated character region R_k intersects the ground-truth box GT_k at all (non-empty intersection, IoU deliberately not used). AEM_text is the mean over characters.
  - Baseline: cross-attention maps of a same-backbone Transformer decoder, summed over heads, normalised, thresholded at beta (Eq. 9).
  - Data: 446 synthetic SynthTiger images with character-level boxes (Sec. 2.2.3). Only samples "correctly recognized by both" models are scored (Sec. 3.6).
  - Also: qualitative maps for several alpha (Fig. 10, two images where other text intrudes), a curved-text comparison at alpha=0.8 and beta=0.5 (Fig. 12).
- **Evidence the method measures what it claims.** Ground-truth character boxes are the only external check (Sec. 3.6). The authors report that for alpha <= 0.95 "the average AEM of the association map remains above 98% accuracy" and that the cross-attention AEM is stable across beta (curves in Fig. 11, no numbers in the text). No random-region or whole-image baseline for AEM is reported (UNVERIFIED that none exists in the figures). No deletion, insertion or masking test is described. The maps are the model's own readout (marginalisation is part of the forward pass), so they are exact in that sense, but there is no test that ink under an association region is what drives the character.
- **Main finding.** Association maps align with cross-attention maps and with character boxes, e.g. in Fig. 10 the model links the right characters to the lower word ("ATHLETIC", "LONDON") and not to the intruding upper text. Separately, the Transformer-decoder variants are consistently more accurate; the authors say that is "due to their ability to capture character dependencies through implicit language modeling that is absent in a CTC decoder" (Sec. 3.4).
- **Limits.**
  - Stated (Sec. 4): maximum 14 characters; no vertical or strongly rotated text; "the character locations they generate exhibit low resolution" because of the 16x16 patch receptive field.
  - Not stated:
    - The implicit-language-model explanation of the accuracy gap is asserted, not tested (UNVERIFIED as an experiment).
    - AEM is a lenient hit test on a coarse 14x14 grid, so high scores are possible with coarse regions (no chance level given).
    - Evaluation set is synthetic and small (446 images) and includes only correctly recognised words.
    - It covers only spike frames (non-blank best-path frames), so it says nothing about what blank frames do.
    - A ViT mixes information globally, so a high-probability cell does not prove the underlying pixels are the cause.
- **Relevance to weight-space edits.** It shows a way to ask "where does emitted character k come from": take only best-path non-blank frames and test the associated region against character boxes. A kraken VGSL net collapses the height before the BiLSTMs, so the 2D association map cannot be reproduced there; only the horizontal (frame) position of each spike could be compared with character boxes.

---

## 5. Coquenet 2022, "Towards End-to-end Handwritten Document Recognition" (arXiv 2209.15362, PhD thesis)

- **Models analysed.**
  - SPAN (Ch. 4.3): fully convolutional, no recurrence, standard CTC on a flattened 2D feature map ("recurrence-free FCN"), paragraph level.
  - VAN (Ch. 4.4): an FCN encoder, a vertical hybrid attention that picks one line per step, and a decoder with one LSTM layer (256 cells, state carried from line to line) and a 1x1 convolution producing N+1 CTC outputs per frame. 2.7M parameters.
  - DAN (Ch. 5.3): a Transformer-based autoregressive decoder, not CTC. It appears here only for its attention visualisation.
  - Data: RIMES 2011 (French), IAM (English), READ 2016 ("450 pages of Early Modern German"), plus RIMES 2009 and READ 2016 at page level for DAN. No external language model; best-path decoding only.
- **Question asked.** Can the network learn line alignment and reading order without segmentation labels, and does seeing the whole paragraph help recognition? (Sec. 4.3.3, 4.4.4, 4.4.5.2, 5.3.4.4.)
- **Assessment method.**
  - Visualisation only for attention and alignment.
    - SPAN prediction overlay: red = character prediction, transparency = blank, rows shown separately (Fig. 4.10, Sec. 4.3.3.2, p. 90-91).
    - VAN: attention weights of each iteration shown as transparency of red, replicated across the row because they are computed only along the vertical axis, rescaled to image height (Fig. 4.16, Sec. 4.4.4, p. 107-108).
    - DAN: last mutual-attention layer projected on the image, intensity range 0.02 to 0.25 (Fig. 5.6, 5.7, Sec. 5.3.4.4, p. 139-140).
  - One input intervention (Fig. 4.17, p. 109): a text line from a validation image duplicated so that two lines share vertical positions.
  - Indirect behavioural comparisons:
    - Pre-training with recognition-only versus recognition and alignment (Table 4.3).
    - Paragraph-level versus line-level CER for the same VAN, with and without interline characters in the metric (Tables 4.13-4.15). Example: RIMES 2011 test CER 3.08 at line level against 2.15 at paragraph level.
    - A DAN ablation table (Table 5.6).
- **Evidence the method measures what it claims.** No faithfulness test of the attention maps. The only causal evidence is the duplicated-line input of Fig. 4.17, which behaves as the vertical-attention hypothesis predicts: "some text is predicted twice ... and other is ignored". The authors also check the interline-character confound in the metric (Table 4.13), which is a measurement control for the line-versus-paragraph comparison.
- **Main finding.**
  - "the VAN has learned the reading order, from top to bottom". "the DAN is not trained using any segmentation label, it performs a kind of implicit segmentation".
  - SPAN: "text line predictions are totally aligned, or aligned by blocks; the lines are well separated by blank labels, which act as line spacing labels" (Sec. 4.3.3.2).
  - The VAN's gain at paragraph level is attributed to implicit segmentation, more context, and the LSTM, which "may have a positive impact acting as a language model" (Sec. 4.4.5.2, p. 114). This is explicitly hedged and not tested. The same passage says "Multiple factors can explain this result".
- **Limits.**
  - Stated: the VAN "cannot handle the case where lines would overlap vertically" (Sec. 4.4.4). SPAN cannot handle upward-inclined lines and is "limited to single-column multi-line text images" (Sec. 4.3 conclusion). The reshaping or prediction offset is attributed to the receptive field, e.g. "shift that can occur between the prediction and the text" (Sec. 4.3.3.2).
  - Not stated:
    - One sample image per visualisation, from the validation or test sets, with no quantitative alignment measure.
    - The line-versus-paragraph comparison is confounded (different splits in the compared works, pre-training procedure, annotation noise).
    - The language-model idea for the VAN LSTM is a conjecture with no test.
    - The DAN is not a CTC model, so its attention maps do not transfer to CTC blank and repeat behaviour.
- **Relevance to weight-space edits.** The duplicate-line input is a template for a cheap intervention test of a hypothesised mechanism, and the finding that blank labels act as spacing and the LSTM state may carry cross-line context suggests an edit to the recurrent weights could change context use in ways that attention-style pictures cannot show.

---

## 6. Pundy, Peer, Kleber 2025, transparency techniques for writer identification and verification (arXiv 2506.16331)

- **Model analysed.** Not a text recogniser. Embedding networks for writer identification (ResNet18/20/50, triplet loss with margin 0.3, cosine distance, Adam lr 0.01) and writer verification (SigNet-style Siamese with the same backbones, contrastive loss, lr 0.001). Input: random 400x400 px snippets with at least 2% black pixels (200 px for ICDAR2013), 4-fold cross-validation. Data: CVL (1604 pages), Firemaker (1000), ICDAR2013 (1000), offline contemporary handwriting, split so that writers do not overlap between train and test. Scripts and languages are not stated in the supplied text (UNVERIFIED). Performance in Table 1 (e.g. CVL ResNet18 mAP 86.41).
- **Question asked.** Which ink in a snippet does the network use to identify or compare writers, and do the highlights agree with what forensic examiners compare (the same letter at different positions)?
- **Assessment method.**
  - Pixel-wise saliency (Kobs et al.): gradient of the cosine distance between the embedding of the image and that of a base image, "a white image" (a blank sheet), normalised to [0,1]. SmoothGrad-like noise: "n image variants are created by applying a random mask of white pixels"; "The value for n is set to 4".
  - Point-specific saliency (Zhu et al.): decomposes cosine similarity over last-layer feature-map positions of a query and a retrieved image (needs GAP and an FC layer without bias).
  - Quantitative: deletion and insertion curves following Hu et al., modified so only black pixels are changed; deleted pixels become white; the insertion base is white; the similarity is cosine, floored at zero; the saliency curve is compared with a random-order curve (Eq. 3). The "score" is the share of steps at which the saliency order beats random, so higher is better. That is my reading of Eq. 3; the text says pixel-level maps "perform better" with higher numbers, which is consistent.
  - Qualitative: two CVL pages, one where retrieval works and one where it does not (Table 3). The check is whether occurrences of the same letter ("y", "f") get similar highlights (Figs 3-10).
- **Evidence the method measures what it claims.** Deletion and insertion scores against random order are the faithfulness test (Table 2, 9 network-dataset combinations for WI, 9 for WV). The qualitative check is anchored in the examiner practice cited from Christlein.
- **Main finding.**
  - Insertion scores are mostly above 90% (the networks identify with few pixels). Deletion scores vary strongly with the backbone: pixel-level WI deletion is 89.7/12.9/55.6 (CVL, ResNet18/20/50), 97.2/15.6/51.3 (Firemaker) and 82.3/15.1/72.0 (ICDAR2013).
  - Pixel-level maps give similar highlights for repeated letters on the page where retrieval works but not on the page where it fails. Point-specific maps "do not display similar highlightings with reliable frequency".
  - Last-layer feature-map size affects whether a highlight can be assigned to one character (ResNet18 coarse, ResNet20 concise).
- **Limits.**
  - Stated: the intensities are only relative within a snippet, because of per-snippet normalisation. The point-specific map's deletion score uses only the overall map, because manual point selection is infeasible. Future work proposes single-character networks.
  - Not stated:
    - ResNet20's pixel-level deletion scores are 12.9, 15.6 and 15.1, which on the authors' own metric means the saliency order is mostly worse than random, yet the conclusion generalises that pixel-wise maps "perform better".
    - The qualitative evidence rests on two pages and two letters, and it is eyeballed.
    - Network dependence ("differing highlighting patterns") means the maps describe the network as much as the handwriting.
    - Deleting individual ink pixels is off-distribution for the networks.
    - The deletion and insertion tests alter ink only, which cannot separate shape from stroke thickness or density.
- **Relevance to weight-space edits.** It supplies a template for a quantitative check of any saliency map on handwriting (ink-only deletion and insertion against a random order, white base image), and it shows that the same method can fail on one backbone, so the check must be repeated for each edited model.

---

## Cross-paper summary (for the related-work paragraph)

| Paper | What it asks about text | Method type | Causal or faithfulness check |
|---|---|---|---|
| Zeyer 2021 | Why blank and spikes dominate | Proof, simulation on toy inputs | Proofs plus control losses; no trained OCR |
| Liu 2020 | Implicit LM in BLSTM-CTC | Decode with and without explicit LM, scale data | None on the implicit LM itself |
| Yousef 2020 | Character and line localisation | IG (white baseline, 50 steps) with SmoothGrad-Abs | None quantitative; flip and distortion CER as side tests |
| Buoy 2023 | Character-to-region link in CTC | Built-in association map; box-hit metric (AEM) | Ground-truth boxes on 446 synthetic images; no chance baseline |
| Coquenet 2022 | Alignment and reading order | Attention and prediction overlays; one duplicated-line input | One intervention; rest visual |
| Pundy 2025 | Which ink drives writer embeddings | Two saliency methods | Deletion and insertion against random order |

Observations that recur:
1. Only Pundy (deletion and insertion against random) and, partly, Buoy (ground-truth character boxes, on synthetic data) test an attribution against anything external. Yousef and Coquenet rely on visual agreement, and the only perturbation experiments are Coquenet's duplicated line and Yousef's flipped-text training.
2. Claims about "implicit language modelling" (Liu, Buoy, Coquenet) are inferences from accuracy differences or are explicitly hedged; no paper probes context use directly. Blank and repeated-label frames are either excluded from attribution (Yousef, Buoy) or treated as spacing (Coquenet), and Zeyer's result is that their dominance is set by the topology, not by the input.

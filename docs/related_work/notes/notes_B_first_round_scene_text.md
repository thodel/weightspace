# Notes B: how five STR papers assessed visual evidence vs language prior, and module contributions

Reading basis: `pdftotext` (flow mode) of the PDFs in this directory, because the `.txt` files are two-column garbled for paper 1. Quotes are verbatim from that text. Section/table numbers are as printed in each paper. "UNVERIFIED" = not checkable in the text I read. "(mine)" = my own observation or arithmetic, not a claim of the authors. Paper 1 is a single-author preprint (arXiv v1, double-blind, code "[anonymised]"), so it is not peer-reviewed as far as I can tell.

## 1. Zhang 2026, "Can Scene Text Recognition Read Rare Compositions?" (arXiv 2609.00816)

- **Model analysed.** Ten off-the-shelf public checkpoints (Sec. 2, 4.1, Table 1): CRNN (8.4M, CNN-BiLSTM-CTC), TRBA, ViTSTR, ABINet, PARSeq (23.8M), MAERec-B, SVTRv2 (19.8M, CTC), CLIP4STR-B/H, GOT-OCR-2.0. Evaluated on the six standard benchmarks (7,248 crops; 6,541 with `^[a-z]+$` labels enter the analysis, Sec. 3). Cross-script: SVTRv2-ch, PP-OCRv5, EasyOCR-Arabic (Table 3). Mechanism probes are run on PARSeq only (Sec. 4.4, App. G, I). Training data is whatever each release used (App. J, not read in detail). Mitigations fine-tune PARSeq (4,000 iterations, lr 1e-5; App. E).
- **Question.** Is the ~96% benchmark accuracy hiding a failure on text whose letter combinations are unusual, and is that failure missing visual capacity or a lexical prior in the decoder? Sec. 4.4: "If instead a prior is applied on top of the visual evidence, the wrong predictions should look like nearby training-set strings."
- **Method (the probes, exactly).**
  - *Stratified bucket diagnostic (Sec. 3.2).* Each test label gets two ranks against a reference corpus (MJSynth; Union14M-L as a check): whole-word frequency rank, and the rank of its rarest character trigram (max over trigrams). Per-benchmark quintiles on log10(rank) give a 5x5 grid. Gap = accuracy(q5/q5) - accuracy(q3/q3). The grid is built from labels only; nothing is perturbed.
  - *Probe (i), error inspection.* PARSeq's q5/q5 errors are read and labelled by mode (Table 4: lexical-snap, whole-word, char-edit, char-noise), e.g. VEICHLES -> VEHICLES.
  - *Probe (ii), confidence when wrong.* PARSeq mean top-1 softmax on wrong predictions per cell: 0.933 in q5/q5 vs 0.890 in q3/q3 (Table 12, App. G).
  - *Probe (iii), layer-wise linear probe.* L2 logistic regression on frozen PARSeq states predicting "is this sample in q5/q5?", 80/20 split stratified by (benchmark, label), class-weighted. Encoder block-k token AUC 0.600 (block 0) -> 0.704 (block 11); decoder mean-pool AUC 0.801 (Table 11).
  - *Probe (iv), attention entropy.* Ratio of decoder self- to cross-attention entropy, first decoder layer, by cell: ~0.27-0.34 in the common-trigram row, 0.38-0.44 in the rare-trigram row (Fig. 5, App. I). Plus a manual commit-vs-abstain split of 50 wrong predictions per script (Fig. 6, App. L).
  - *Interventions.* 6x encoder scale-up (CLIP4STR-B -> H), 16 non-architectural mitigations (temperature, entropy penalty, distilGPT-2 rescoring, LoRA on the AR decoder MLP, char-augmentation fine-tuning), AR -> CTC swap (PARSeq vs SVTRv2), and a confidence-routed ensemble.
  - *Controls.* Paired bootstrap on n=474 (10,000 resamples, App. F); marginal-effect decomposition showing the trigram effect only acts when the word is rare (-7.6 pt at word q5 vs -0.1 pt at q1, App. A); corpus swap with Pearson r=0.99 on model ordering (Sec. 3.2); same-recipe CLIP4STR-B vs H.
- **Evidence the method measures what it claims.**
  - For the *gap*: the controls above, and a stated pre-attempt showing disjoint n-gram splits are infeasible (Sec. 3.1).
  - For the *mechanism*: only "converging" indirect probes. The paper says it "localises" the failure to the AR decoder, but no probe has a control. There is no random-label or text-identity control for the linear probe, no NAR/context-masked run of PARSeq, and no probe of CTC models.
  - The authors themselves say: "With one exception, the q5/q5 comparisons in this paper should be read as directional rather than significance-tested." (App. F)
- **Main finding on text handling.** The failure is a lexical prior over visual evidence: errors "snap" to the nearest English word and are *more* confident when wrong in the stress cell. They claim it is located in the autoregressive decoder, not the encoder, and not fixable by vision scaling (86.9 -> 86.5, p=0.73). Only AR -> CTC (SVTRv2) clears the noise floor (+2.5 pt, p=0.02, n=474). Residual after the CTC swap: "inherits the same end-of-string lexical attractor" (App. D). Adding an external LM raised aggregate accuracy but not q5/q5 (Sec. 5.4).
- **Limits stated.** The +1.3 pt best non-architectural gain is "a descriptive mean, not a significant gain (p = 0.14)"; four intervention families untested (large-LM rescoring, byte-level tokenisation, retrieval, RL objectives; Sec. 6); universality claimed "only for the direction" (Sec. 4.3); the absolute q5/q5 level depends on the reference corpus (Sec. 3.2).
- **Limits I see (mine).**
  1. Architecture attribution is confounded. SVTRv2 vs PARSeq differs in encoder, training recipe and decoder, not just AR vs CTC. Their own Table 2 has CRNN (CNN-BiLSTM-CTC) with the *largest* gap (-17.9 pt; q5/q5 74.7), so CTC alone does not remove the problem, and a BiLSTM-CTC net is the case closest to kraken. The paper does not discuss this.
  2. The layer-wise probe is confounded. q5/q5 membership is a function of the label text; decoder states carry the decoded characters, so higher decodability there is expected without any "prior". The probe compares different pooling: an encoder "[CLS] token" vs a decoder mean-pool. PARSeq's encoder has no [CLS] token (Bautista & Atienza, Sec. 3.1: "without the classification head and the [CLS] token"), so what was probed at the encoder is UNVERIFIED.
  3. The attention-entropy ratio is reported for one layer. PARSeq's decoder is a single layer, so "first layer" is the only layer. An entropy ratio is not an attention "budget"; no variance is given.
  4. Counts do not line up. The text says "We inspect the 474 q5/q5 wrong predictions", but 474 is the cell size; at 85.0% accuracy (Table 2) wrong predictions are ~71 (my arithmetic). In q3/q3 (n=347, 99.1%) the "confidence when wrong" baseline of 0.890 rests on ~3 errors (my arithmetic).
  5. Per-trigram CTC gains (Table 8) use trigrams with >=5 occurrences in n=474.
  6. 16 mitigations plus ensembles are tested against one corner cell of n=474; no multiplicity correction is stated (UNVERIFIED).
  7. The "same across architectures" claim for the mechanism rests on qualitative inspection (App. D, UNVERIFIED which models).
  8. Labels restricted to lowercase alphabetic, which drops 9.75% of the benchmark union.
- **Relevance to weight-space edits of a CTC line recogniser.** Its stratify-by-label-rarity grid is a model-agnostic read-out you can apply before and after an edit to the BiLSTM stack, but its own CRNN row (largest gap, CTC with BiLSTM) means "CTC removes the prior" cannot be assumed and the mechanism must be tested directly on the recurrent layers.

## 2. Da, Wang, Yao 2022, "Levenshtein OCR" (arXiv 2209.03594, ECCV 2022)

- **Model analysed.** LevOCR: a visual feature extractor (modified ResNet plus transformer units, giving V and an initial visual prediction via position attention), a character-level Transformer "textual module", and a 6-layer Vision-Language Transformer (VLT) that takes concatenated visual+textual tokens with modality embeddings; action heads for deletion, placeholder insertion, token prediction (Sec. 3). 6 transformer units, 8 heads, D=512, 109M parameters (Sec. 4.2, 4.9). Trained on MJSynth+SynthText only; tested on IC13, SVT, IIIT, IC15, SVTP, CUTE (Sec. 4.1). Trained by imitation learning with corrupted-text augmentation (Eq. 6-10).
- **Question.** Does the model use language knowledge beyond its visual prediction, and which modality supports each edit decision? Sec. 4.3: "LevOCR indeed adopts linguistic information, not just relies on visual features."
- **Method.**
  - *Initial-sequence manipulation (Sec. 4.3, Table 1).* The trained model is run from four starting texts: its own visual prediction, an empty sequence, the ground truth with random one-character edits (30% replace / 30% insert / 40% delete, digits untouched), and the ground truth itself; 1-3 iterations.
  - *Attention-sum read-out (Sec. 4.8, Fig. 3).* "Sum attention weights of different modalities in the top layer" over visual vs textual tokens, for four hand-picked images.
  - *Architecture ablation (Sec. 4.5, Table 2).* Without the LevT deletion/insertion losses ("w/o LevT"), and CNN vs ViT backbones.
  - *Qualitative traces (Sec. 4.6, Table 3).* Six hard images, three starting sequences.
  - *Controls.* The "w/o LevT" variant and the visual-only prediction (LevOCR_VP) as baselines. No model without a textual module is trained (none stated).
- **Evidence the method measures what it claims.** None stated beyond the construction. The attention sum is presented as modality contribution ("the pixels near 'n' ... contribute the most"), but attention weight mass is not shown to equal causal contribution (mine). The claim that the empty-start run shows language use rests on the gain from iteration 1 to 2: Table 1 AVG goes 90.64 -> 92.51 (+1.87) -> 92.63, whereas the text says "greatly increased 2.1%" (mismatch, mine).
- **Main finding on text handling.**
  - Table 1 AVG: visual-only 90.65; from own visual prediction 92.81 (#3); from empty 92.63 (#3); from corrupted GT 85.25; from GT 96.91. Ordering GT > VP > Emp > Rand. The authors read this as: a better text input gives a better result, so language is used.
  - Fig. 3 (a): visual 0.737 vs textual 0.263 for "snout". For the blurred/corrupted cases (c), (d) the text says attention shifts to text. Reading the figure at low resolution, I get textual/visual = 0.555/0.445 for (c) and 0.5043/0.4957 for (d); the figure legibility makes this UNVERIFIED.
  - Quote, Sec. 4.8: "LevOCR relies more on visual information than on textual information (0.737 vs. 0.263)."
- **Limits stated.** Only efficiency/size (36 ms per iteration, 109M parameters; Sec. 4.9). No limit about the analysis method.
- **Limits I see (mine).**
  1. The GT start is an oracle that leaks the answer, and it is not a measure of language knowledge.
  2. The corrupted start (85.25) is *below* the visual-only result (90.65), i.e. wrong text input actively drags accuracy down. That is direct evidence of text-prior influence, but the authors do not frame it so; they frame it as "85% noisy texts can be corrected".
  3. Even from the exact ground truth the model is wrong 3.1% of the time (96.91), so the text channel is not a clean copy path.
  4. Fig. 3 is four selected examples; (d) is almost 50/50 and cannot support the "more attention to text" claim. How heads/layers are aggregated is UNVERIFIED.
  5. No input-image perturbation (blur/occlusion at test time), no text-free ablation, and the benchmark vocabulary is dictionary-like, so no rare-word test.
- **Relevance to weight-space edits of a CTC line recogniser.** Its "feed corrupted text, see whether output follows text or image" design cannot be reproduced on a CTC net with no text input channel, but the lesson to copy is to also report accuracy *below* the vision-only baseline as evidence of prior pull, which for a BiLSTM-CTC net means perturbing the *image* at the character level (e.g. swapping a glyph for a rare one) and measuring pull toward a familiar word.

## 3. Fang et al. 2021, ABINet "Read Like Humans" (arXiv 2103.06495, CVPR 2021)

- **Model analysed.** ABINet: vision model (VM; CNN + 2-3 Transformer units with position attention; Table 1, sizes SV/MV1/MV2/LV, e.g. 19.6M-23.5M parameters) -> probability vectors -> bidirectional cloze network (BCN, 4 layers, 8 heads, D=512) as language model (LM) -> gated fusion (Eq. 6) with iterative re-feeding (M iterations). Trained on MJ+ST; self-training on Uber-Text. LM pre-trained on text of MJ/ST ("PLMin") or WikiText-103 ("PLMout") (Sec. 3.2.1, 4.1, Table 2).
- **Question.** Whether an LM inside a recogniser learns real linguistic rules at all ("whether and how well the LMs learn character relationship is unknowable", Sec. 1), and how much it helps under noisy vision input.
- **Method.**
  - *Gradient flow (Sec. 4.2.2, Table 2).* Block gradient between VM and LM (BGF, default) vs allow it (AGF), with pretrained VM; plus pre-training LM on training text vs WikiText-103.
  - *Stand-alone LM as spelling corrector (Table 4).* Text-only test: 20,000 random items, a character added/removed in 20%, replaced in 60%, unchanged in the rest; BCN vs SRN. Reported: BCN 82.8 char / 41.9 word vs SRN 78.3 / 27.6 (top-5 accuracy).
  - *Top-5 probability visualisation (Fig. 5).* BCN on "today" with "-oday" and "tod-y" inputs, then with error characters.
  - *Iterative correction (Table 5, Fig. 6, Fig. 7).* Iterations 1-3 in training and testing, with qualitative before/after.
  - *Bidirectionality (Table 3).* SRN-U, SRN, BCN on SV and LV VMs at similar parameter count.
  - *Controls.* Matched VMs and parameter counts in Table 3; BGF vs AGF with the same VM.
- **Evidence the method measures what it claims.**
  - The claim that BGF "enforces model to learn linguistic knowledge inevitably" is supported only indirectly. The text-only test (Table 4) evaluates a separately trained LM; the AGF-trained LM is never probed.
  - The explanation for AGF's loss ("overfitting occurs in LM as the VM helps to cheat in training") rests on an unreported observation: "We also notice that the training loss of AGF reduces sharply to a lower value." No curve is shown.
  - Overall "none stated" beyond these.
- **Main finding on text handling.**
  - AGF loses 0.9 pt average (Table 2; PVM 92.3 vs PVM+AGF 91.4 in the table as I read it). LM pre-training on the in-domain training text gives nothing (91.6 vs 91.7 no pre-training); WikiText-103 pre-training helps (92.5).
  - Iterating the LM adds only +0.4 (SV) / +0.3 (LV) average, with "little gains on IIIT" and the largest gains on IC15/SVT/SVTP (up to +1.3 and +1.0 on SVT; Table 5).
  - From Fig. 5: the LM is useful when the context is mostly clean, and "if there are multiple error characters, it is hard for BCN to restore correct text".
  - So the paper's picture: vision carries the bulk; the LM is a modest, context-dependent corrector.
- **Limits stated.** Multiple errors defeat the LM; length misalignment (fixed padding mask) is a known weakness, which iteration alleviates (Sec. 3.2.3, 4.2.2); gains saturate beyond 3 iterations (Fig. 6).
- **Limits I see (mine).**
  1. There is no VM-only vs VM+LM row in Tables 1-5 (Table 6 not checked, UNVERIFIED), so the LM's total contribution to accuracy is never isolated; only increments.
  2. The gate G (Eq. 6) that decides vision vs language per character is never analysed.
  3. No image perturbation and no rare/out-of-vocabulary split; MJ/ST text has a fixed dictionary-like vocabulary (Zhang, Sec. 3.1, quotes an "88 K-word MJSynth vocabulary"), so "learned linguistic rules" may be lexicon memory.
  4. Overlap between the 20,000 spelling test items and training text is UNVERIFIED.
  5. Seeds/variance not stated in what I read (UNVERIFIED).
- **Relevance to weight-space edits of a CTC line recogniser.** ABINet shows that a *separable* LM block with a probability-vector interface can be tested stand-alone (Table 4-style text-only spelling test), whereas in a BiLSTM-CTC stack the "LM" is entangled with the visual features and no such interface exists, so an edit's effect on the implicit LM has to be inferred from image-side counterfactuals.

## 4. Bautista & Atienza 2022, PARSeq (arXiv 2207.06966, ECCV 2022)

- **Model analysed.** PARSeq-S: 12-layer ViT encoder (no [CLS] token), single-layer Transformer decoder with position-query/context-key attention; trained with Permutation Language Modeling (K=6 permutations; Sec. 3.1-3.2, 4.4). One weight set decodes in three ways by changing only the attention mask: left-to-right AR, non-autoregressive (NAR, context = [B] only), iterative cloze refinement (Sec. 3.3, Table 2). Trained on synthetic (MJ+ST) or on real data (COCO-Text, RCTW17, Uber, ArT, LSVT, MLT19, ReCTS, TextOCR, OpenVINO; Sec. 4.1). Tested on the six benchmarks (7,672 samples) plus ArT, COCO, Uber (Table 5).
- **Question.** Can one internal LM provide both context-free and context-aware decoding and bidirectional refinement, so that an external LM (ABINet, SRN) is unnecessary? Also how much the language context matters relative to the image: "the primary input signal in STR is the image, not the language context" (Sec. 4.4).
- **Method.**
  - *Permutations vs decoding (Table 3, Sec. 4.4).* K in {1, 2, 6, 12, 24}; word accuracy for AR, NAR, and "cloze acc." = one refinement iteration with the *ground truth as the initial prediction*.
  - *Decoding-scheme toggle on the same weights.* K=6, 94-char: AR 93.34, NAR 92.22, cloze 94.81 (Table 3). Refinement iterations vs latency (App. H, Fig. 11).
  - *Qualitative failures (App. A, Tables 7-8).* A left-to-right AR model adds a suffix (terrifying -> terrifyingly, last-char confidence 0.72), and LTR/RTL AR models disagree on "open" (LTR "open" vs RTL "opell").
  - *External-LM audit (App. B, Tables 9-10).* ABINet parameters/FLOPS by module (LM = 34.48% of parameters, 13.65% of FLOPS), and ABINet's LM given the ground-truth label as input: 50.44% combined word accuracy.
  - *Robustness proxies.* Charset-size evaluation (Table 4), and occluded/oriented datasets (Table 5) as dataset-level proxies; no synthetic occlusion.
  - *Controls.* Same-weights mode toggle; K sweep; training hours per K.
- **Evidence the method measures what it claims.** The mode toggle is a clean architectural control (same weights, mask only), so AR-vs-NAR (+1.1 pt, 93.34 vs 92.22) is a reasonable estimate of what the context path adds on this benchmark union. For "image is primary" there is no direct test; it is stated as intuition. The appendix anecdotes are single examples. Seeds/variance for Table 3 are UNVERIFIED (Table 4 reports ±0.1 values).
- **Main finding on text handling.** In a joint image+context decoder, context adds roughly one point over context-free NAR decoding on the standard benchmarks, and a cloze pass seeded with the true text (94.81) still errs on about 5% of words, i.e. the image can override a correct text context. With K=1 (plain AR training) cloze with GT context drops to 71.14 and NAR to 0.01: the AR-trained model's character prior cannot be used without preceding context. App. A shows the prior winning over the image in single cases (suffix addition; direction-dependent reading).
- **Limits stated.** None specific to the context analysis in what I read (the Sec. 4.4 argument is presented as supportive). Conditional independence of an external LM from the image is given as a design flaw (Sec. 1).
- **Limits I see (mine).**
  1. The benchmark union is dominated by common words, so a ~1 pt context contribution says nothing about rare or non-word text. Zhang's rarity grid (paper 1) exposes exactly what this aggregate hides.
  2. The "50% wrong even with perfect VM" argument against ABINet's LM (App. B) conflates a cloze LM, which cannot see the character it predicts, with a fusion module; a masked-position LM scoring 50% on exact word reconstruction is not itself an error (mine).
  3. The cloze (GT-start) accuracy is an oracle, as in LevOCR.
  4. "Context-free" is used for CTC models (Sec. 2: "output characters are conditionally-independent of each other"), which ignores context that CTC networks acquire through recurrent feature layers.
  5. The Table 7/8 "confidence" examples are n=1.
- **Relevance to weight-space edits of a CTC line recogniser.** PARSeq's mask-toggle is the only experiment in this batch that turns the context path off in a *trained* network without retraining; for a BiLSTM stack the nearest analogue is zeroing or re-initialising a recurrent layer (or one direction) and comparing to a vision-only (conv+CTC) baseline.

## 5. Baek et al. 2019, "What is wrong with scene text recognition model comparisons?" (arXiv 1904.01906, ICCV 2019)

- **Model analysed.** A four-stage framework: Transformation (None/TPS) -> Feature extraction (VGG/RCNN/ResNet) -> Sequence modelling (None/2-layer BiLSTM, hidden 256) -> Prediction (CTC/Attn). Full factorial 2x3x2x2 = 24 combinations including CRNN (None-VGG-BiLSTM-CTC), RARE, Rosetta, STAR-Net, etc. (Sec. 3, 4, Fig. 4). Trained on MJ+ST (14.4M images), 300K iterations, model selected on a union of IC13/IC15/IIIT/SVT training sets; evaluated on 9 real datasets (8,539 images, alphanumerics only), five random seeds averaged (Sec. 4.1).
- **Question.** Which module (transformation, feature extractor, sequence model, prediction) contributes how much accuracy, speed and memory under identical data and training; what failures remain.
- **Method.**
  - *Marginal module analysis (Sec. 4.4, Table 2).* For each module option, the mean accuracy over all combinations containing it, split into regular and irregular benchmarks.
  - *Trade-off frontiers (Sec. 4.3, Fig. 4-5).*
  - *Qualitative module-upgrade examples (Fig. 7).* Images recognised only after a given module is added.
  - *Failure analysis (Sec. 4.5, Fig. 6).* 644 of 8,539 images (7.5%) are wrong in all 24 combinations; label-noise audit.
  - *Controls.* Same train/validation/test data, optimiser, iterations; reproduction sanity check of four existing models, CRNN, RARE, GRCNN and FAN w/o Focus Net (appendix Table 3); training-data study showing MJ+ST 84.1 vs MJ-only 80.0 vs ST-only 75.6 (Sec. 4.2).
- **Evidence the method measures what it claims.** The factorial design with fixed training is a strong control for *accuracy attribution*, and the paper says the modules contribute "similarly under all combinations" from frontier ordering. It does not test whether any module does *language* modelling: the statements about context are qualitative (Fig. 7). Interactions between modules are not reported (the table has only main effects). Variance across the five seeds is not shown in the main-text Table 2 (UNVERIFIED for the supplement).
- **Main finding on text handling.**
  - Table 2 marginal means: BiLSTM None -> BiLSTM 85.1 -> 87.6 (+2.5) regular, 65.2 -> 69.7 (+4.5) irregular. ResNet over VGG +3.8 / +7.1; Attn over CTC +1.7 / +2.6; TPS +1.1 / +3.4. So in this framework the feature extractor is the biggest accuracy lever and the sequence model the second.
  - The authors credit BiLSTM with receptive-field adjustment, not lexical knowledge: "BiLSTM leads to better context modeling by adjusting the receptive field" (Fig. 7 discussion, Sec. 4.3). Implicit language modelling is credited to *attention*: "Attention including implicit character-level language modeling finds missing or occluded character".
  - They conclude of the failure analysis: "current methods do not extensively exploit contextual information to overcome occlusion."
- **Limits stated.** Failure modes are listed as open problems (calligraphic fonts, vertical text, special characters, occlusion, low resolution, label noise: 1.3% / 6.1% / 24.1% mislabelling depending on strictness, Sec. 4.5).
- **Limits I see (mine).**
  1. None -> BiLSTM changes parameters (+2.7M) and the receptive field along the sequence axis as well as adding recurrence, so the +2.5/+4.5 cannot be read as "LM benefit". The authors' own explanation is receptive field and ignoring stray crops.
  2. Attn vs CTC conflates alignment mechanism with language modelling.
  3. Marginal means can hide interactions (e.g. BiLSTM may matter more under CTC), and CTC-with-BiLSTM is only 2 of the 24 cells per feature extractor.
  4. All models are 2-layer BiLSTM; kraken's 3-layer stack is outside this grid.
  5. Evaluation is word accuracy on dictionary-like benchmarks, so the language prior never shows up separately from visual accuracy.
- **Relevance to weight-space edits of a CTC line recogniser.** This is the best available evidence that removing the sequence stage costs ~2.5-4.5 points on a CTC pipeline, but it attributes that to context windowing, so any claim that the BiLSTM stack is an "implicit LM" has to be established with an own test (e.g. a rare-trigram split) rather than cited from this paper.

## Cross-paper summary (mine)

- **Method types.** Only two of the five manipulate the *text input* of a trained model (LevOCR corrupted/empty initial text, PARSeq mask toggle), and both evaluate on dictionary-like benchmarks. The other three infer module roles from architecture swaps (Baek, ABINet) or correlational probes on a single model (Zhang). None intervenes on the recurrent layers of a CTC network.
- **Mechanism attribution is thin everywhere.** Zhang's decoder claim rests on an uncontrolled linear probe and a one-layer attention ratio; his own Table 2 puts CRNN (CNN-BiLSTM-CTC) at the worst gap. Baek explains BiLSTM by receptive field; PARSeq/ABINet treat CTC as context-free.
- **Anchors for the related-work text.** Prior-over-evidence is documented anecdotally in PARSeq (App. A) and quantitatively only by Zhang's rarity-stratified grid. Oracle-text numbers (LevOCR 96.91, PARSeq cloze 94.81) are upper bounds, not measures of language use.

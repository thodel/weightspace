# Batch C notes: how each paper assessed what a network does with text input

Source files read: 2505.15865 (pdftotext re-extraction, both raw and -layout, because tables were garbled), 2602.22918.raw.txt and .txt, 1706.07206.txt and pdftotext re-extraction, 1909.12114.txt and pdftotext re-extraction. Section/table numbers are as printed in the papers. Anything marked UNVERIFIED was not found in the extracted text. Items marked "(my inference)" are mine, not the authors'.

## 1. Baek, Chang, Ryu, Lee 2025 (EMNLP), arXiv 2505.15865 -- OCR heads

**Model analysed.** Multi-image LVLMs. Qwen2-VL-7B-Instruct and InternVL2-8B-Instruct carry the main experiments (Sec. 4, Tables 1-4); Qwen2-VL-2B appears in the head-distribution figures (Fig. 3, Fig. 7). Transformer decoders with attention heads (layer-head ids such as L5_H19, Fig. 5).

**Question asked.** Do LVLMs contain a special class of attention heads that read text out of image patches, distinct from LLM "retrieval heads"? RQ1 "How can we identify the OCR heads?"; RQ2 "How do OCR heads differ from existing retrieval heads?" (Sec. 1). Hypothesis: "OCR heads should selectively attend to visual patches corresponding to characters or words and directly guide text extraction from images."

**Assessment method.**
- Data (Sec. 3.1-3.3): the Passkey and Needle-in-a-Haystack tasks are rendered as 2 to 12 images. Rule-based bounding boxes are made for every character of the answer. Boxes are scaled for resizing. Patch tokens with IoU > 0.1 with an answer box become "Evidence Patch Tokens" E. "they are not provided as input to the model during inference."
- OCR score (Sec. 3.4, Eq. 2): same token-level recall as the Wu et al. 2024 retrieval score. A head gets credit for a generated answer token w when (1) w is part of the answer and (2) the input patch with the highest attention for w "must belong to the set of Evidence Patch Tokens E". Score = |g_h ∩ k| / |k|.
- Head selection (Sec. 3.5): 1,200 sampled instances. A head is an OCR head if its score exceeds 0.1 in at least 10% of instances (120 or more) and its mean over all 1,200 also exceeds 0.1.
- Characterisation (Sec. 4): Jaccard overlap of the OCR-head set with the retrieval-head set per score bin (Table 1), and "activation frequency" versus mean score (Fig. 4) to call heads "static" or "dynamic".
- Downstream tests (Sec. 5):
  - CoT scores (Table 2).
  - Head masking (Sec. 5.2, Table 3). Top 5, 10 or 20 OCR heads, retrieval heads (RH), or the same number of random heads. Random: "mean ± variance across 5 selections". Tasks: DocVQA, MP-DocVQA, NQ, HotpotQA; F1.
  - Sink-token redistribution (Sec. 5.3, Eq. 4, beta = 0.4, Table 4).
- Case study (Sec. 6.1, Fig. 6): single-character "passkeys" (0-9, a-z), top-5 heads per character.

**Evidence that the method measures what it claims.**
- Identification is by an attention-argmax criterion only. It is a scoring method, not an intervention. The authors say so: "scoring-based approach for efficiently detecting attention heads with unique capabilities" (Sec. 7).
- The causal evidence is the masking experiment (Table 3). Qwen2-VL-7B, 20 heads masked: DocVQA 76.3 (baseline) -> 59.7 (OCR) / 67.1 (RH) / 75.1 (random). MP-DocVQA 69.1 -> 47.7 / 62.4 / 67.7. On NQ the effect reverses: RH 36.4, OCR 56.8, random 57.3 (baseline 58.0). InternVL2-8B, 20 heads: DocVQA 39.1 -> 24.2 (OCR) vs 38.2 (RH) vs 38.3 (random).
- This is a double dissociation: OCR-head masking hurts image-text VQA, RH masking hurts text QA. It uses a random-head control.
- UNVERIFIED: how heads were masked (zeroing the head output, attention set to zero, mean-ablation?). I found no description in the text.
- The Sec. 6.1 claim "OCR heads mechanistically encode visual similarity" rests on overlap of top-5 head lists for look-alike glyphs (for example '1' and 'i' share four heads). No statistic or control is given.
- Table 4 (sink redistribution) gains are small and mixed in sign: Qwen +0.3/+0.6, RH -0.1/+0.3. No variance is reported.

**Main finding.** Heads that put their top attention on the evidence patches of the answer string exist, and they are numerous: "a large number of heads are activated" (abstract). Overlap with retrieval heads is low: Jaccard 0.347 (Qwen2-VL-7B) and 0.133 (InternVL2-8B) in the 0.1-or-higher bin (Table 1). They sit in middle-to-late layers (Sec. 6.2: for Qwen2-VL-7B, 10 OCR heads in layer 14 and 11 in layer 22). Masking them selectively degrades DocVQA-type tasks.

**Limits stated.** "many open-source LVLMs support only a single image input or rely on thumbnail-style training settings"; passkey/NIAH only "up to 8k samples" (Limitations).

**Limits I see.**
- The test is a copy-from-image task on synthetic rendered text (digits and short sentences). Attention at an evidence patch is not shown to be where character identity is computed. The score cannot separate "locates the text" from "reads the text".
- Only two model families, one size each. No effect sizes are given for per-head OCR scores.
- Masking was done on top-k by score. Only random heads serve as a control. There is no control for matched layer depth.
- The text of Sec. 5.2 says masking RH lowers MP-DocVQA by "7.3%"; Table 3 gives 69.1 -> 62.4, which is 6.7 points. Minor inconsistency.
- InternVL2 NQ/HotpotQA: random masking sometimes hurts as much as RH masking (HotpotQA, 10 heads: RH 28.2, random 29.2), so the "RH matters for text QA" half is weaker there.
- No test on handwriting or noisy text. Nothing about training or fine-tuning.

**Relevance.**
- (a) CTC BiLSTM: does not transfer. The method needs attention heads and an attention argmax over patch tokens. A kraken line model has neither. The nearest analogue is comparing CTC frame posteriors against character boxes, which is not in the paper (my inference).
- (b) QLoRA adapters: it transfers directly to Qwen-style VLMs. Compute per-head OCR scores before and after adapter training and mask the top heads, to test whether the adapter changes or relies on the OCR-head population (my suggestion; the paper analyses only frozen models).

## 2. Steinberg & Gal 2026, arXiv 2602.22918 -- OCR routing bottleneck

**Model analysed.** Five dense VLMs from three families: Qwen3-VL-2B, -4B and -8B Instruct (DeepStack injection), Phi-4-multimodal-instruct and InternVL3.5-4B (single-stage projection) (Sec. 3.2, App. A.1). Interventions on the residual stream of the LLM decoder.

**Question asked.** "where does this optical character recognition (OCR) information enter the language processing stream?" (abstract). Secondary question: is the OCR signal separable from other visual capabilities (counting, spatial reasoning, general VQA)?

**Assessment method.**
- Counterfactual pairs (Sec. 3.1, App. A.4): EgoTextVQA images with text regions inpainted away (boxes expanded 10%, 50 pairs spot-checked). Activation difference at layer l: Δh_l = h_l(original) − h_l(inpainted) (Eq. 1). 315 images for PCA; disjoint evaluation set (size UNVERIFIED).
- Intervention (Sec. 3.4, Eq. 2): at inference project out the top-N PCs of the deltas, "We fix α=1 throughout (full projection removal)." Forward hooks on layer outputs; "Hooks affect all tokens (visual and text) during both prefill and decode." Single layers or layer ranges (names such as pca_L17-19_pc5).
- Metric (Sec. 3.6): normalised substring match for OCR accuracy. Footnote: a VLM judge on a subset gave 97% agreement with it. Collateral damage is measured on CountBench, EmbSpatial and RealWorldQA. Greedy decoding, full datasets.
- Complementary head ablation (Sec. 3.5, 4.4): rank 720 heads by an attention-to-text-region selectivity ratio. Zero the output projections of the top 93.
- Cross-dataset transfer (Sec. 4.5, Table 3): directions learned on EgoTextVQA applied to OCRBench and InfoVQA.
- Aggregation of the activation deltas over token positions before PCA: UNVERIFIED, not stated.

**Evidence that the method measures what it claims.**
- Causal intervention (subspace removal) with an outcome metric. The authors position it explicitly: "Our work builds on these descriptive and correlational analyses by adding causal intervention" (Sec. 2).
- Controls:
  - Head ablation: 93 random heads, mean of 10 draws, "degrades CountBench by 6.8±2.1pp" with OCR −2.1 ± 1.5pp, versus the OCR-selective heads: OCR −8.2pp, CountBench +5.0pp (Sec. 4.4).
  - Inpainting controls (App. A.5, Table 5): LaMa, SDXL and blur give PC1 variance 72.9%, 71.2%, 68.4%. Random non-text boxes give 12.3% variance and −0.1pp OCR.
  - Retention tasks. A layer outside the bottleneck has little effect (App. C.1 for InternVL).
  - Cross-layer and cross-dataset checks (Fig. 3-4, Table 3).
- Not found: a control that removes a random subspace of the same rank, or any confidence interval or seed variation for the PCA interventions. Test-set sizes are not stated. UNVERIFIED.

**Main finding.** The layers at which removing the delta subspace most reduces OCR accuracy depend on the architecture. Qwen3-VL-4B: L16-L20 (about 50% depth). Phi-4: L3-L9. InternVL3.5-4B: L2-L3 (Table 4). Maximum OCR drop for Qwen3-VL-4B on EgoTextVQA is 69.6 pp at L18 (Table 3). The authors say PC1 explains 72.9% of the delta variance at L17 (Sec. 4.2). In Qwen3-VL-4B, removal at L17-19 raises CountBench by 4.3 pp (best multi-layer: +6.9 pp), which they read as OCR "interfering" with counting.

**Limits stated.**
- All five models are dense and in the 2B-8B range; "Our claims regarding modularity and scaling should be strictly contextualized within this evaluated regime" (Limitations).
- Depends on "artifact-free text removal to isolate OCR signals"; unreliable for dense documents.
- Retention covered only three benchmarks. The 8B model shows a −14pp RealWorldQA drop.
- On the PC1 figure: "this measures variance in deltas rather than providing direct evidence about the degree of superposition" (Sec. 2).
- Cross-dataset results "test method transferability ..., not bottleneck transferability" (Sec. 4.5); the peak layer moves from L18 (EgoTextVQA) to L30 (OCRBench) for Qwen3-VL-4B.
- "We hypothesize that the OCR pathway competes for representational capacity in the residual stream" (Sec. 5): the interference reading is a hypothesis.

**Limits I see.**
- Variance share is not necessity. PC1 holds 72.9% of delta variance, yet removing PC1 alone at L17 gives only a 32.0 pp OCR drop; PC5 gives 53.6 pp and PC10 gives 54.6 pp (Table 6). So the "low-dimensional" claim describes the deltas, not the minimal subspace needed to read.
- The intervention hits all token positions, including text-prompt tokens, so it cannot say that the removed subspace is where image text enters the stream.
- The best configuration (L17-19_pc5, labelled "Best") was picked from 14 configurations (Table 6), judged by CountBench gain on the same evaluation set. That is selection on the test data. Not corrected.
- Tables 1/2 and Table 6 disagree for the Qwen3-VL-4B single-layer rows. Examples: pca_L17_pc3 is 94.1 in Table 2 and −0.4 pp in Table 6; pca_L16_pc5 is 94.1 in Table 2 and +4.8 pp (96.2) in Table 6. Multi-layer rows agree (L16-20_pc3 98.3; L17-19_pc5 95.7).
- The Sec. 4.3 paragraph on the 8B model repeats the 4B sentence "L17-19_pc5 intervention improves CountBench (+4.3pp)..." next to different 8B numbers (+2.9 pp, −14 pp RWQ in Table 1).
- Effects near layer 18 (RWQ collapse of −13 pp at L18 only) show that one-layer removal results are sensitive to layer choice.

**Relevance.**
- (a) CTC BiLSTM: the recipe (matched original/text-removed pairs, delta PCA, project the subspace out at a chosen layer, measure CER) can be applied to a kraken stack in principle. Hooks would go on the conv outputs and the LSTM hidden states per time step. The paper's concepts (residual stream, depth band tied to the DeepStack versus single-stage fusion) do not exist in a conv-LSTM-CTC stack. For a line recogniser the "no text" counterfactual is a blank line, so the contrast is the whole input (my inference). It transfers as a method, not as a finding.
- (b) QLoRA adapters: this is the closest fit. Comparing hidden-state deltas with and without the adapter, and testing whether subspace removal at the OCR depth band (about 50% depth for Qwen3-VL) undoes the adapter's gain, is a causal check one can run on any adapted Qwen-family model. The depth band is architecture-specific, so it must be re-measured for the target model.

## 3. Arras, Montavon, Müller, Samek 2017, arXiv 1706.07206 -- LRP for LSTMs, sentiment

**Model analysed.** One bidirectional LSTM (one hidden layer, embedding and hidden size 60) trained on five-class Stanford Sentiment Treebank sentiment (model from Li et al. 2016). 46.3% five-class and 82.9% binary test accuracy on full sentences (Sec. 3). Input is word embeddings; the classifier reads the final hidden state of each direction (Appendix).

**What the method explains.** Which input words support or oppose a chosen target class: word-level relevance scores R_d summed over embedding dimensions (Sec. 2, 4). Not internal mechanisms: it attributes the output to the input, layer by layer.

**Assessment method.**
- Method, Sec. 2.2: LRP for linear layers with an epsilon stabiliser (0.001) and a bias parameter δ (default 0, "approximately conservative"; LRP_cons with δ = 1).
- Rule for multiplicative gates: "we set Rg = 0 and Rs = Rj". The gate (sigmoid, in [0,1]) gets zero relevance and the source neuron gets all of it. Justification: "the gate neuron decides already in the forward pass how much of the information contained in the source neuron should be retained".
- Baseline: sensitivity analysis (squared partial derivatives) (Sec. 2.1).
- Evaluation:
  - Qualitative heatmaps (Fig. 1-2) and most/least relevant words for "very positive" over 2,210 test sentences (Table 1).
  - Quantitative word deletion (Sec. 4.3, Fig. 3): sentences with 10 or more words (1,849). Delete up to 5 words, "for deleting a word we simply set its word embedding to zero". Most-relevant first for 1,014 initially correct sentences; least-relevant first for 835 initially wrong ones. Random deletion averaged over 10 runs (std < 0.016). Accuracy tracked.
  - Relevance distribution over sentence position and per direction (Sec. 4.4).

**Evidence that the method measures what it claims.**
- Perturbation-based faithfulness test, "pixel flipping" style, with a random baseline and the SA baseline. LRP gives the steepest decline/incline and "SA performs even worse than random deletion" when deleting least-relevant words of wrongly classified sentences (Sec. 4.3, Fig. 3).
- No ground-truth explanation, no human study, no other attribution method (occlusion is not compared), no second model or dataset.
- The deletion operation (zeroing an embedding) puts the LSTM off-distribution. This is acknowledged for removal in the 2019 chapter, not in this paper.

**Main finding.** LRP with the gate rule gives signed word relevances that the bi-LSTM's decision follows under deletion better than gradient sensitivity does. Negations flip sign depending on the following word (Sec. 4.1). Most relevance falls on the second half of a sentence, and the left and right encoders differ in smoothness (Sec. 4.4).

**Limits stated.**
- "can only be “as good” as the neural network itself" (Sec. 4.1); a weak 46.3% model may mean mediocre explanations.
- Future work: "applying the proposed technique to other recurrent architectures such as character-level models or GRUs" (Conclusion).

**Limits I see.**
- Single model and task. Only SA as a competitor, a weak baseline (the 2019 chapter shows occlusion matches LRP on the same test).
- Faithfulness is judged by deletion effects only. Nothing is said about what the gates or cell states represent or store.
- Gate rule drops all relevance to gates; the 2019 chapter shows where this fails.
- The target is a single sentence-level class score from the last hidden state. There is no sequence-output or CTC case.

**Relevance.**
- (a) CTC BiLSTM: it transfers in principle, and it is the only one of the four papers that handles LSTMs. Kraken's recurrent layers are standard gated LSTMs, so the "gate gets zero" rule applies. The deletion-based check would become "blank out columns or ink, watch the character posterior" (my inference). What is missing: the explained target would be the logit of a character at a time step (or a CTC path), which the paper does not cover. The paper also does not cover the conv front-end, and nothing in the paper has been tried on image input.
- (b) QLoRA adapters on VLMs: does not transfer. There is no attention or softmax rule in this paper, only LSTM/GRU products.

## 4. Arras, Arjona-Medina, Widrich, Montavon, Gillhofer, Müller, Hochreiter, Samek 2019, arXiv 1909.12114 -- Explaining and Interpreting LSTMs

**Model analysed.** (i) The same SST bi-LSTM as the 2017 paper (Sec. 5.1). (ii) Minimal one-memory-cell LSTMs (17 trainable parameters) trained on toy addition and subtraction, 50 models per task (Sec. 5.1). (iii) A modified LSTM trained to predict return in a one-dimensional "Moneybag/Coins" reinforcement-learning environment (Sec. 5.2, Fig. 6-7).

**What the method explains.** Input-position relevance (one value per time step) for LSTM outputs, via LRP extended to gated products and the accumulator (Sec. 3). The chapter also asks which LSTM designs make explanations easier (Sec. 4).

**Assessment method.**
- Theory, Sec. 3.2: Deep Taylor Decomposition of the gate-times-signal product. The nearest root point is (z_g, 0), so "Rg = 0". Four rules compared (Table 1): LRP-all (signal takes all), LRP-prop, LRP-abs, LRP-half. Caveat: "the saturation effect of the hyperbolic tangent can create a mismatch" (exact only if tanh is replaced by identity or ReLU).
- Accumulator analysis, Sec. 3.3: relevance through c_k = f·c_(k−1) + a_p carries a product of forget gates, so "only the few most recent time steps will receive relevance."
- Selectivity (Sec. 5.1, Fig. 5): word removal on SST. 1,849 sentences, 1,014 correct and 835 wrong. Removal by discarding the word (zeroing an embedding gave similar results). Compared: Gradient, Gradient×Input, two Occlusion variants, Contextual Decomposition (CD), the four LRP variants, random (10 runs, std < 0.02).
- Fidelity (Sec. 5.1, Table 2): a synthetic task with known answer. Inputs contain two relevant numbers n_a, n_b among zeros. Target is n_a + n_b or n_a − n_b. The relevance should correlate linearly with the numbers (+1 for n_a, −1 for n_b in subtraction). Statistics over 2,500 test points per model, mean over 50 models.
- RL case study, Sec. 5.2: do different product rules find the Moneybag event?

**Evidence that the method measures what it claims.**
- Two kinds: perturbation (selectivity) and ground-truth comparison (fidelity). The authors justify the second: "By using a synthetic toy task we can avoid problems of disentangling errors made by the model from errors made by the explanation."
- Results: on SST deletion, LRP-all, occlusion and CD are on par; LRP-prop/abs/half are "almost equivalent to random" (Fig. 5). On subtraction, only Gradient×Input and LRP-all give correlations near +1/−1: LRP-all 98.5 and −99.3, relevance share 99.3%. Occlusion gives 99.0 and −69.0; CD gives −25.9 and −50.0 (Table 2).
- Caveat on the evidence: "we certainly can not deduce that every method that passes the subtraction test is also appropriate to explain any complex nonlinear prediction task".

**Main finding.** The signal-take-all rule is the best of the tested product rules for standard LSTMs. LRP-all degrades when more memory cells are used, "gates might also be used by the standard LSTM to store the input numbers' value" (Sec. 5.1). In the RL case, "no relevance is assigned to the event Moneybag" under LRP-all, because that information acts through the input gate; allowing relevance through gates (LRP-prop, LRP-half) finds it but leaks relevance into the cell (Sec. 5.2).

**Limits stated.** Conclusion: the more interpretable LSTM variants and "their interplay with various LRP propagation rules" need controlled experiments (future work). Sec. 5.2: relevance "is not completely conserved when it is propagated through the gates". Relevance to far-away steps decays with the forget-gate product (Sec. 3.3).

**Limits I see.**
- The decisive fidelity evidence comes from one-cell toy models. It degrades with more cells, i.e. for realistic models the ground-truth check says LRP-all is not reliable.
- The RL environment shows an explanation rule blind to gate-mediated information. Whether that happens in a trained text-recogniser LSTM is not tested.
- No image or CTC case. Theory assumes the simplified accumulator.

**Relevance.**
- (a) CTC BiLSTM: it is the most useful of the four for a kraken-type stack. It supplies a ground-truth style test (synthetic task with known relevance) that could be copied: synthetic lines where only certain glyphs determine a particular output. It also gives warnings. The forget-gate product means relevance from distant columns is exponentially small, so LRP-all will report locality whether or not the net uses distant context (my inference). A gate-mediated dependence (a gate encoding what a character is) would be invisible to LRP-all.
- (b) QLoRA adapters on VLMs: does not transfer; no attention or adapter content. Only the evaluation pattern (perturbation plus synthetic ground truth) is reusable.

## Cross-paper observations

1. The two VLM papers are about where text-reading is localised in a transformer: heads in Baek et al. (identified by an attention score, causal evidence only from head-masking), a residual-stream subspace in Steinberg & Gal (found by activation differences, tested by projection). Neither says anything about how a recurrent CTC stack reads. Both depend on transformer-specific units (heads, residual stream, depth bands). The two LRP papers handle LSTMs but only attribute the output to the inputs; they do not look inside, and they have never been tried on image input or sequence-output CTC.
2. Validation strength differs. Baek et al.: score, then masking with random-head control. Steinberg & Gal: intervention with random-head and random-box controls, but no random-subspace control and the headline configuration chosen on the evaluation set. Arras 2017: word deletion only. Arras 2019: adds a synthetic ground-truth check, which is the only place a method is compared with known truth, and it shows rule failures that deletion alone did not reveal.

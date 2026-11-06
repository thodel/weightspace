# Batch E: how each paper assessed what a text recogniser does with its input

Working copies (pdftotext) are in /tmp/claude-1000/-home-th/0dc8ad2a-3390-417b-9872-a7fd4cdb988d/scratchpad/search/E/ . Quotes are verbatim, section/figure numbers given. "(mine)" = my own judgement, not the authors'. UNVERIFIED = I could not confirm it in the text.

Summary table (assessment type):

| # | Paper | Assessment type | Open full text read |
|---|---|---|---|
| E1 | DeepSeek-OCR, Visual Merit or Linguistic Crutch | black-box input corruption (semantic disruption, zero-prior random text) | yes |
| E2 | PAR | black-box glitch-in-familiar-text test + last-layer attention ratio + inference-time RoPE/attention editing | yes (ACL Anthology) |
| E3 | Retrieval to Recognition | attention-head discovery + zero-ablation + value patching + LoRA before/after comparison | yes |
| E4 | FADE | black-box visibility gradient on digit strings | yes (ACL Anthology) |
| E5 | Latent Representation Probes | MLP/transformer probes on hidden states or attention to predict answer correctness | yes |
| E6 | OCR Quest (model editing) | weight-space task-vector averaging of fine-tuned ViT-CTC recognisers; accuracy only | yes |
| E7 | CC-OCR v2 | black-box per-factor slicing of benchmark scores | yes (skimmed) |
| E8 | Human Psychophysics for STR | black-box train-on-X/test-on-Y manipulation grid on CRNN-CTC vs Attn | yes |

All eight had open full text. None was abstract-only.

---

## E1. Visual Merit or Linguistic Crutch? A Close Look at DeepSeek-OCR

- **Full text:** https://arxiv.org/pdf/2601.03714 (v2).
- **Authors, year, venue:** Yunhao Liang, Ruixuan Ying, Bo Li, Hong Li, Kai Yan, Feiteng Fang, Qingwen Li, Min Yang, Okamoto Satoshi, Zhe Cui, Shiwen Ni (first two marked equal contribution). arXiv:2601.03714v2 [cs.CL], 8 Jan 2026. Venue as given: arXiv preprint only (no conference named).
- **Model analysed:** DeepSeek-OCR (Tiny, Small, Base resolution modes; the paper ties Tiny to fewest vision tokens) plus 11 other OCR/VLM systems "spanning 125M-72B parameters" (Sec. 4.1): dots.ocr, Qwen2.5-VL (7B; 72B in Fig. 4), GOT-OCR, MonkeyOCR, SmolDocling, Nougat, HunyuanOCR, MinerU, PaddleOCR-v5 (models listed in Tables 6-7; the Table 6-7 end-to-end/pipeline column headings are scrambled in the PDF text, so classes follow the Sec. 4.2 prose: Nougat, HunyuanOCR, Qwen2.5-VL end-to-end; PaddleOCR-v5, MinerU pipeline).
- **Question asked:** Is DeepSeek-OCR's OCR score driven by visual evidence or by language priors in its LLM decoder ("Visual merit or linguistic crutch", Sec. 1)?
- **Assessment method:** Purely black-box input perturbation, no internals.
  - Base set: Fox benchmark, 112 English pages, 600-2500 text tokens (Sec. 2.1). Clean baseline = ground-truth text re-rendered to image (text2png).
  - RQ1 sentence level ("distort"): key nouns, verbs, phrases replaced by absurd look-alike words ("butterfly" to "breadflutter"), re-rendered. Precision per token-length bin.
  - RQ2 word level (Sec. 3.1): Swap (5% or 10% of words, two letters swapped), Shuffle (5% or 10% of words, letters fully shuffled), Zero-Prior Random Text ("2–10 random letters, mixed case", same length distribution as Fox).
  - RQ3: same clean vs zero-prior comparison on 11 other models. RQ4: QA vs VQA on Fox pages with 3 annotated questions each. RQ5: context stress test up to 12,000 tokens.
  - Metric: "OCR precision" (exact definition not given in the parts I read; UNVERIFIED). Controls: the text2png clean render and the original Fox images; pipeline OCR as a contrast class. No repeated runs or confidence intervals seen (mine).
- **Evidence the method measures what it claims:** Indirect only: the drop from natural to random text. Authors call random text a "truth serum" and say it forces systems to "rely solely on visual recognition capabilities" (Sec. 4.1). Supporting contrast: PaddleOCR-v5 loses only 4.9 points (to 89.53) while end-to-end models lose 40-60 (Sec. 4.2). Mechanistic check: none (no attention, probing or ablation).
- **Main finding:** DeepSeek-OCR Tiny precision falls from about 88% (derived: 76.7 + 11.2 from Sec. 2.2; text2png baseline) to 19.84% on zero-prior text (Table 4); Small 42.12, Base 61.70. Sentence-level distortion costs -11.2 points avg in Tiny, -3.6 Small, -0.6 Base (Sec. 2.2). All end-to-end models collapse on random text; pipeline PaddleOCR-v5 does not (MinerU also fails, but for a detector reason, Sec. 4.2). VQA accuracy is near chance (~20%, 4 options).
- **Limits:**
  - Stated: "We have not extensively benchmarked the model's performance on structured or high-entropy data types" (Limitations).
  - (mine) Random mixed-case 2-10 letter strings change the visual statistics (no word shapes, odd capital patterns, different line breaks), so the drop mixes absence of language prior with out-of-distribution glyph sequences. No intermediate control such as real-word but rare strings, or random text at Base resolution with more tokens varied independently.
  - (mine) "conclusively proves" (Sec. 3.2) overstates; the evidence is correlational. Tiny/Small/Base differ in vision tokens AND encoder resolution, so the fewer-tokens-means-more-prior-reliance reading is confounded with resolution (the authors' own phrase: "sufficient visual resolution reduces dependence", Sec. 2.2).
  - (mine) The 13-model comparison mixes model sizes, training data and architecture classes; end-to-end vs pipeline is not isolated from having a decoder LM or not.
- **Relevance:**
  - (a) kraken CTC line recogniser: The zero-prior test transfers unchanged, because it needs only rendered images and CER. It would measure how much a conv-BiLSTM-CTC model relies on the implicit context of its BiLSTM (natural-text vs random-string lines, same writer/font); it is the cheapest first control for a weight-space paper. Works on any recogniser, black-box.
  - (b) QLoRA adapters on Qwen VLMs: Run the same natural vs scrambled/random rendering test on base vs adapter to see whether adaptation raised or lowered prior reliance; Qwen2.5-VL 7B and 72B are already in their table as baselines.

---

## E2. PAR: Training-Free Positional Perturbation and Attention Recycling for Faithful OCR

- **Full text:** https://aclanthology.org/2026.acl-long.1065.pdf (open ACL Anthology PDF). No arXiv version found (arXiv API title and abstract queries returned nothing; absence of evidence only).
- **Authors, year, venue:** Yao Yao, Manwen Liao, Weitian Zhang, Zuchao Li, Hai Zhao (Shanghai Jiao Tong Univ., HKU, Wuhan Univ.). "Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 23258–23273, July 2–7, 2026". Code: github.com/Zoeyyao27/PAR-for-Faithful-OCR.
- **Model analysed:** DeepSeek-OCR, Qwen3-VL-2B and 8B, InternVL3.5-8B (open, with intervention); GPT-4o and Claude (the text says Claude-3.5-Sonnet, Table 2 says Claude-Sonnet-4.5; inconsistent, mine) only black-box in Table 2. Attention analysis on DeepSeek-OCR and Qwen3-VL-8B.
- **Question asked:** When visual evidence conflicts with a familiar text, does the VLM "recite" rather than "read" ("Linguistic Priors Hallucination"), and does visual reliance decay with generation length?
- **Assessment method:**
  - **GlitchText Probing Dataset construction (Sec. 2.1, App. A):**
    1. Familiarity-driven selection. Chinese: Chinese Poetry Dataset (73,281 ancient poems). English: English Quotes Dataset (2,510 goodreads quotes). "we employ DeepSeek-OCR (Wei et al., 2025) to calculate the Perplexity (PPL) of candidate texts"; rank ascending (low PPL = familiar); keep top-500 Chinese and top-200 English (Table 1: max PPL 2.40 and 5.69; mean length 180.76 characters and 146.31 words).
    2. Adversarial glitch injection: exactly 20 glitches per instance. Chinese: replace characters with visually confusable ones from the Confused_Chinese dictionary (e.g. 兵 to 乒), choosing the 20 highest visual-similarity replacements. English: SpaCy POS tagging, lemminflect morphological change of verbs and nouns (go to went; apple to apples), Gestalt pattern matching similarity, keep top-20. Note (mine): the Introduction describes the English glitches as "keyboard proximity interference (Typos)", but Sec. 2.1 and App. A.1 describe morphological changes; the latter is what was built.
    3. Image synthesis: black text on white, SimHei for Chinese, DejaVuSans for English, character-level (Chinese) or word-level (English) line breaking, high-resolution PNG.
    4. Shuffle control variant: character (Chinese) or word (English) order randomised, "to rigorously disentangle the contribution of visual perception from linguistic prediction" (Sec. 2.2).
    5. English subset in the PAR experiments is left-padded with 200 words of glitch-free text (Sec. 4).
  - **Metrics:** CER; Identification rate (glitches transcribed as the glitch, "visual faithfulness"); Correction rate (glitches corrected back to the original, "intensity of linguistic hallucination"). Iden and Cor need not sum to 100%.
  - **Attention probe (Sec. 2.3, 3.1):** last-layer attention for each generated token t; Image-to-Text Focus Ratio R_focus = sum of attention over image tokens divided by sum over historical text tokens. "This metric serves as a proxy for the model's instantaneous reliance on visual evidence relative to linguistic context." Per layer and head heatmaps classify models as "Decay" (DeepSeek-OCR, InternVL) or "Late-Fusion" (Qwen3-VL).
  - **Intervention (PAR):** Positional Perturbation (phase noise delta_pos added to RoPE angle, shallow-middle layers, e.g. 0-5) and Foveal Attention Recycling (penalise outlier text-token attention, refill salient image patches with Gaussian kernel, deep layers). Fixed parameters stated: perturbation frequency omega=1, outlier multiplier k=3, tau_rise=100, sigma=20, kernel window 80 (App. C); layer choice per model in Table 7.
  - **Controls:** Shuffle variant; padding-length sweep (200/500 tokens) to push glitches later; ablations (w/o PP, w/o FAR, random noise, average refill, constant/linear scheduler, layer shift); baselines VCD (training-free contrastive decoding) and TVC-7B (training-based); OmniDocBench for side effects.
- **Evidence the method measures what it claims:** Partial. The shuffle control is the main evidence that Cor tracks language prior: Cor falls sharply when order is shuffled in all models (Sec. 2.3), e.g. DeepSeek-OCR Chinese Iden 44.81 to 82.39. R_focus is explicitly "a proxy"; no test that R_focus causally predicts faithfulness beyond the intervention results. In the English columns of Table 2 the pdftotext column order is scrambled, so I did not rely on those numbers (UNVERIFIED).
- **Main finding:** All six models show a Cor drop under shuffling; GPT-4o's CER rises on Shuffle (0.2283 to 0.5587), read by the authors as dependence on context for weak fine-grained vision. Visual attention share decays with generation length. PAR lowers CER (e.g. Qwen3-VL-8B English 4.10 to 2.91; Chinese 0.0729 to 0.0617) and on Qwen3-VL-8B Chinese raises Iden 58.68 to 69.86 and lowers Cor 37.87 to 26.90 (Sec. 4). The smaller Qwen3-VL-2B has lower CER than 8B at long context, read as stronger LM head means stronger prior.
- **Limits:**
  - Stated: "applying PAR to a completely new architecture might require a preliminary diagnostic run" to pick layers; operates at layer, not head, level (Limitations).
  - (mine) Familiarity is defined with the same model family that is later tested (DeepSeek-OCR perplexity), so selection is partly circular; no human or external-LM familiarity check.
  - (mine) Chinese glitches are visually confusable characters (a perception test), English glitches are grammatical-form changes that remain valid words (closer to a normalisation test); the two languages measure different things, and the paper's "Ident/Cor" label hides this.
  - (mine) Ablation effects are small on DeepSeek-OCR Chinese (CER 0.0959 PAR vs 0.0981 w/o FAR vs baseline 0.1000, Tables 2 and 5); no seeds or intervals seen. The Linear scheduler gives the highest Iden (50.56) but the worst CER (0.1002), so Iden alone can be gamed by making the model less accurate.
  - (mine) Quotes and poems are short, famous, and probably in pretraining data; this is a memorisation test, less a general language-prior test.
- **Relevance:**
  - (a) kraken CTC: The GlitchText design (insert a few visually plausible deviations into highly predictable text, compare to a shuffled-order control, count identification vs auto-correction) transfers as a test-set design for a CTC line recogniser, e.g. with historical orthographic variants. The attention ratio, RoPE perturbation and attention recycling do NOT transfer: a conv-BiLSTM-CTC stack has no text-history attention and no autoregressive decoding.
  - (b) QLoRA on Qwen VLMs: Directly applicable. Qwen3-VL is a tested architecture, Iden/Cor can be computed on base vs adapter, and the R_focus probe works on the adapted model; PAR could be stacked on an adapter since it modifies only inference.

---

## E3. From Retrieval to Recognition: How Vision-Language Models Become OCR Specialists

- **Full text:** https://arxiv.org/pdf/2609.21543 (v1).
- **Authors, year, venue:** Yuanxiang Huangfu, Hanmeng Zhong, Linqing Chen, Jeffrey Tiong Jee Hui (PatSnap Co., LTD., Suzhou). arXiv:2609.21543v1 [cs.CV], 18 Sep 2026. Venue as given: "Under review as a conference paper at ICLR 2027".
- **Model analysed:** Specialised OCR models GLM-OCR (16 layers x 16 query heads), MinerU2.5, PaddleOCR-VL-1.6 for head validation; general VLMs Qwen2-VL-2B (336 query heads; main) and Qwen3-VL-2B (28 layers, 448 query heads; replication) for the retrieval-head overlap; and authors' own LoRA-specialised Qwen2-VL-2B and Qwen3-VL-2B.
- **Question asked:** Does OCR specialisation build a new reading circuit in attention heads or reuse the text retrieval/copy circuit already present?
- **Assessment method:**
  - **OCR-head discovery (Sec. 3):** For each correctly transcribed token, the visual evidence set E(z) = image-patch positions whose rectangles have IoU > 0.1 with the annotated region of that character, table cell or formula glyph. Greedy decode, then replay the generated sequence with its own prefix (replay agreement 100% text, ~99.96% formulas, App. A). Only tokens wholly inside a maximal exact-match span of prediction and reference are scored. A head gets an "OCR hit" at step t if its global attention argmax falls in E(z_t). Head score = hits / reference tokens, averaged over samples; head active if score > 0.1; candidate if mean score > 0.1 and activation frequency >= 0.1.
  - **Data and folds (Sec. 4.1):** shared OmniDocBench subset of 1,200 instances (400 each text, table, formula); folds A and B (160 each) for independent discovery, held-out fold C (80) for causal test.
  - **Causal test:** "Principal free-generation interventions zero head output only at answer positions" (App. A). Damage: NED increase (text), TEDS decrease (table), CDM decrease (formula), against five random head sets matching head count in every layer and excluding the consensus top-20. Example: GLM-OCR table-head ablation TEDS drop 0.903 vs 0.032 random; formula CDM drop 0.846 vs 0.016.
  - **Retrieval/copy heads:** independently found from text-only needle-in-a-haystack (Wu et al. 2024 criterion); compared by all-head Spearman, top-20 overlap, Jaccard, containment. Cross-task ablation both ways. POPE as specificity control (ablating overlap heads does not hurt object-presence judgements beyond random, Sec. 5.3).
  - **Counterfactual value patching (Sec. 5.4):** minimal image pairs differing in one glyph (184 text, 105 table, 128 formula pairs); keep recipient attention fixed, replace only the Value at the target evidence positions with the donor's; measure donor-directed logit shift (4.476 text, 2.425 table, 2.498 formula vs 0.658, 0.812, 0.168 matched random; non-target positions ~0).
  - **Specialisation (Sec. 6, App. B):** "rank-32 LoRA training (alpha 64) for two epochs on 12,000 mixed OCR examples"; run the same discovery before/after; compare top-20 identity, all-head rank correlation, score concentration, cross-ranked ablations.
- **Evidence the method measures what it claims:** Stated and reasonably strong for the claim that these heads matter causally and selectively: held-out fold, layer-matched random controls, POPE specificity control, source-position control (non-target patch gives ~0). The authors explicitly state the limits of the value-patching result: "We therefore do not view the overlap heads as sufficient on their own to emit the copied token." (Sec. 5.4), and cite that ablation importance depends on replacement value (Li and Janson 2024; Sec. 2). Attention argmax alone is only used for discovery, not as evidence.
- **Main finding:** Sparse stable OCR heads (top-20 sets overlap at least 19/20 across disjoint folds); Qwen2-VL OCR heads overlap retrieval/copy heads (13/17/15 of top-20 for text/table/formula; Spearman 0.846/0.886/0.677); LoRA specialisation keeps 17-20 of the top-20 heads (all-head rho 0.874-0.942) but redistributes causal weight ("circuit conservation with causal reweighting", Sec. 6.4).
- **Limits:**
  - Stated: no dedicated limitations section found (UNVERIFIED that none exists); the paper itself notes the pre/post comparison is "incorporating both parameter adaptation and prompt differences rather than isolating parameter updates alone" (Sec. 6.1), and that the Qwen3-VL checkpoint "does not uniformly improve every task" (text NED 0.0155 to 0.0168, table TEDS 0.7855 to 0.9192, App. B).
  - (mine) The OCR-hit criterion scores only tokens in exact-match spans, so by construction it excludes exactly the tokens where a language prior might override vision; it cannot say anything about hallucination or prior-driven errors.
  - (mine) Zero-ablation of whole head outputs is a coarse intervention; no mean or resampling ablation reported. Only 2B-scale general models are used for the overlap claim; the three specialised models are not compared to their own base models.
  - (mine) Evidence sets use box-IoU > 0.1 against image patches; with dense text one patch can overlap several characters, so a hit is a coarse location test.
  - (mine) The LoRA used for the specialisation analysis is the authors' own small run (12k examples, 2 epochs), not a production fine-tune.
- **Relevance:**
  - (a) kraken CTC: The head-discovery and head-ablation machinery needs attention heads and does not transfer to a conv-BiLSTM-CTC stack. The idea of an alignment-based "evidence set" (CTC forced alignment gives frame-to-character spans) plus held-out zero-ablation of units against layer-matched random sets could be re-created at the level of BiLSTM hidden units or conv channels, but would be a new method (mine).
  - (b) QLoRA on Qwen VLMs: Closest existing recipe for asking what an adapter changed inside the model: same Qwen2-VL-2B/Qwen3-VL-2B family, LoRA rank 32, before/after head-identity and causal comparison; transfers as is to Qwen-family decoders. It compares head rankings and not adapter weight deltas (mine), so it does not measure the adapter matrices directly.

---

## E4. FADE: Probing the Limits of VLMs on fine-grained OCR

- **Full text:** https://aclanthology.org/2026.alvr-main.23.pdf (open ACL Anthology PDF). No arXiv version searched beyond one API title query (nothing found).
- **Authors, year, venue:** Deep Shah, Nehal Kathrotia, Sanket Badhe (Google LLC). "Proceedings of the 4th Workshop on Advances in Language and Vision Research (ALVR), pages 249–259, July 3, 2026".
- **Model analysed:** Black-box: Gemini 3.0 Pro and Flash, Claude 4.5 Sonnet, Gemma 3 27B; baseline: a UNet segmentation network followed by an unnamed "OCR engine" (engine UNVERIFIED).
- **Question asked:** Where does zero-shot VLM OCR break when text has no semantic anchor and visibility falls?
- **Assessment method:** Synthetic benchmark (Sec. 3, 4):
  - Nine-digit random numeric strings, white digits, overlaid at image centre on COCO backgrounds (480x640 RGB), alpha-blended at alpha 0.8, 0.5, 0.3, 0.2.
  - 2,600 training images for the UNet, 1,000 test images evaluated at each alpha (4,000 inferences per model).
  - One fixed zero-shot prompt (Fig. 4, returns the number in tags, "None" allowed). Metrics: exact match and CER. Extra condition: images cropped to the watermark box with 15 px padding (App. B). Confusion matrix for Gemini 3 Pro at alpha 0.3 (restricted to equal-length predictions).
  - UNet: depth 16-32-64-128 with 256 bottleneck, Adam, lr 0.001, batch size 32.
  - Controls: alpha 0.8 as upper-bound baseline; white-only digits so contrast is not a confound; uniform digit distribution (Fig. 2); centred position.
- **Evidence the method measures what it claims:** The design claim is that random digits remove semantic predictability: "existing benchmarks often entangle visual perception with language priors" (Abstract); the Conclusion says FADE "provides a unique framework to disentangle pure visual grounding from semantic and linguistic predictability" (Sec. 6). No test is given that the VLMs do not use some prior over nine-digit strings (format, phone-number-like patterns); no human baseline; no experiment that changes the language prior while keeping the glyphs.
- **Main finding:** Accuracy collapses at low alpha (e.g. Gemini Flash 3.0 exact match 0.955 at alpha 0.8 vs 0.233 at 0.2; Claude Sonnet 4.5 0.672 to 0.040; Gemma 3 0.768 to 0.062; UNet 0.948 to 0.478; Table 1). Cropping helps only modestly (Flash 0.233 to 0.289 at alpha 0.2). Confusions such as 9 to 0 (52) and 3 to 8 (51) at alpha 0.3.
- **Limits:**
  - Stated (Limitations): "the watermark vocabulary is restricted to numerical digits"; text always at image centre.
  - (mine) The authors' reading of the confusion matrix as "a perceptual collapse driven by dataset priors" (Sec. 5.4) is speculation; confusions between similar-stroke digits (9/0, 3/8) are equally what a pure perception failure produces, and no prior-free vs prior-laden comparison is run.
  - (mine) Inconsistent mechanism claims: Sec. 5.1 says the UNet result "validates our hypothesis that global attention mechanisms of current general-purpose VLMs fails", while the Conclusion says cropping suggests the bottleneck is "the visual encoder's inability to register sparse signals rather than a failure of global attention". The paper lists the encoder-vs-LLM question only as future work (Sec. 7.1).
  - (mine) Comparing a trained, task-specific UNet (trained on 2,600 in-distribution images) with zero-shot VLMs is not like-for-like; single prompt, no repeats, no CI, no open-weights model analysed internally.
- **Relevance:**
  - (a) kraken CTC: A random-digit, graded-visibility test is cheap to replicate on a line recogniser (digit lines with contrast gradient) and mirrors E1's zero-prior idea; the UNet-plus-OCR pipeline result is the one place where a segmentation-then-recogniser design beat generalist VLMs, which is modest support for modular recognisers. Black-box, so transfers to any recogniser.
  - (b) QLoRA on Qwen VLMs: Usable only as an external behavioural benchmark for base vs adapter; nothing in it probes hidden states or weights, and the authors leave the encoder-vs-LLM split open.

---

## E5. Reading Between the Lines: Abstaining from VLM-Generated OCR Errors via Latent Representation Probes

- **Full text:** https://arxiv.org/pdf/2511.19806 (v1).
- **Authors, year, venue:** Jihan Yao, Achin Kulshrestha, Nathalie Rauschmayr, Reed Roberts, Banghua Zhu, Yulia Tsvetkov, Federico Tombari (University of Washington; Google). arXiv:2511.19806v1 [cs.CV], 25 Nov 2025. Venue: arXiv preprint (no conference named; the layout looks like a CVPR template but this is not stated; UNVERIFIED).
- **Model analysed:** Qwen2.5-VL-7B-Instruct and Gemma 3-12B-it (open weights), on OCRBench v2, SEED-Bench-2-Plus (free-form), HierText (with Gemini-2.5-Pro-synthesised QA, human validated) and EgoTextVQA (outdoor, under 1 minute clips).
- **Question asked:** "Can VLMs know when they can't see?" (Abstract): are uncertainty/correctness signals present in hidden states or attention patterns even when output logits are miscalibrated?
- **Assessment method:**
  - **Probes (Sec. 4):** features = hidden states averaged over output tokens per layer, or attention weights averaged over output tokens. Designs: (1) Concatenated probe, 4-layer MLP over all layers; (2) Visual-focused probe, attention averaged over visual token positions, then 4-layer MLP on L*H features; (3) Ensemble probe, one MLP per layer, top K=5 layers by validation abstention accuracy, majority vote; (4) a 4-layer transformer encoder on full attention.
  - **Labels:** soft correctness label from each dataset's evaluation script (binary cross-entropy with soft labels); "Since OCR labels are continuous rather than binary, we modify the standard metrics to handle soft correctness labels." (Sec. 5.2). Metrics: effective reliability, abstention accuracy, reliable accuracy, abstention precision. Random 6:2:2 split per dataset; threshold tau chosen on validation.
  - **Baselines:** token probability, verbalised confidence, self-consistency (5 samples, temperature 1.0), prompt-to-abstain, VLM judge, R-Tuning, SVAR (attention ratio to visual tokens), Contextual Lens (cosine of image and answer hidden states).
  - **Analyses:** four synthetic uncertainty sources made from HierText (white-box occlusion, Gaussian noise sigma=5, unanswerable questions, injected low-confidence prefix), zero-shot transfer of probes; cross-dataset transfer (Table 2); per-layer probes (Fig. 3).
- **Evidence the method measures what it claims:** Partly stated. The claim is that probes capture "general uncertainty" rather than dataset-specific patterns, supported by transfer to the four synthetic sources (Sec. 6.1: "if probes decode genuine uncertainty signals from latent representations, they should generalize across different uncertainty sources", average 0.825) and the rise-then-fall across layers (peak around layers 15-20 for Qwen, 20-30 for Gemma, Sec. 6.3). No control probe on shuffled labels or a selectivity/control task is described in the sections I read (UNVERIFIED that none exists).
- **Main finding:** Ensemble hidden-state probe is best on average (+7.6% abstention accuracy over best baseline; e.g. Qwen OCRBench v2 0.789 vs best baseline 0.768 VLM Judge); visual-focused attention probe second; on video only LRP gives meaningful abstention. Cross-dataset generalisation is mixed: self-consistency (training-free) generalises best overall; the visual attention probe is best among trained methods (Table 2; avg 0.636).
- **Limits:**
  - Stated: training-based methods need training-data diversity; cross-dataset results are weaker (Sec. 6.2). No separate limitations section found (UNVERIFIED).
  - (mine) The probe predicts whether the final answer will be judged correct; it does not tell whether the network used visual evidence or a language prior, so it cannot answer E1/E2-style questions. A probe can read any correlate of difficulty (question type, text length, image resolution).
  - (mine) Gains over token probability are modest in absolute terms (Qwen avg A-Acc 0.750 vs 0.669 for token probability, Table 1); no repeated runs or intervals. A random 6:2:2 split of questions may place the same image in train and test (OCRBench v2 and EgoTextVQA have several questions per image or clip; not stated either way; UNVERIFIED).
  - (mine) The "Ensemble" selects layers on the validation set and so adds a tuned choice.
- **Relevance:**
  - (a) kraken CTC: The generic recipe (extract per-layer activations, train a small classifier to predict line-level CER above a threshold, pick best layers) is architecture-agnostic and could use BiLSTM layer outputs, with CTC frame posteriors as the "token probability" baseline; the attention-based probes do not transfer. Would show whether correctness is linearly decodable, not what the network reads.
  - (b) QLoRA on Qwen VLMs: Qwen2.5-VL-7B is one of their two models, so the code path (hidden-state probes at output tokens, layer sweeps) applies directly to a base vs adapter comparison, e.g. to see whether an adapter moves the layer where error signals peak.

---

## E6. The OCR Quest for Generalization: Learning to recognize low-resource alphabets with model editing

(Most relevant to a paper on weight-space operations on a text recogniser.)

- **Full text:** https://arxiv.org/pdf/2506.06761 (v2). Model card (checkpoints): https://eaudedata.github.io/ODAOCR/model_card/model_card.html (given in footnote 1; not opened).
- **Authors, year, venue:** Adrià Molina, Oriol Ramos Terrades, Josep Lladós (Centre de Visió per Computador, Universitat Autònoma de Barcelona). arXiv:2506.06761v2 [cs.LG], 18 Jun 2025 (first submitted 7 Jun 2025). Venue as given: "Preprint submitted to Pattern Recognition" (arXiv comment: "Preprint (under review) For Journal"). Whether it has since been published: UNVERIFIED.
- **Model analysed:** A single architecture family: ViT encoder "as described in [3]" (Atienza, Vision Transformer for fast and efficient scene text recognition, 2021, i.e. ViTSTR) with one extra encoding layer carrying positional encoding; "we train the transformer using a CTC [13] loss function" (Sec. 4.2). Model size (parameters) not stated in the text I read (UNVERIFIED). NOT a kraken model, NOT a VGSL/BiLSTM stack, NOT a VLM. Word-level recognition crops.
- **Question asked:** Can averaging independently fine-tuned expert models improve out-of-domain generalisation and then speed up incorporation of an unseen alphabet, compared with centralised pre-training on the same data? Not a question about what the network does with text.
- **What exactly is edited (answer to the lead question):**
  - Edited object: all weights, as one vector theta in R^m ("we consider a model as vector of parameters", Sec. 3.3). Full fine-tuning, not LoRA or adapters.
  - Operation: task-vector (task-arithmetic, Ilharco et al.) averaging, equivalent to FedAvg/Reptile with T=1: tau_n = U^k(d_n, theta_0) - theta_0 per sub-domain; "we then average the respective task vectors of each sub-domain to incorporate it to the pretrained model" (Sec. 3.3, Eq. 2). Merged model = theta_0 + mean(tau_n). Equation 2 can be repeated T times (iterated = Reptile; Fig. 7 sweeps T and k with k x T = 75).
  - Seed theta_0: 75 epochs on HierText (11,639 Latin images; 749,802 words) because task arithmetic needs a "reasonably good seed" (locality; Sec. 4.2; random-init distributed training worse, Table 3).
  - Experts d_n: the seven training sets individually: IAM, Esposalles, Parzival, George Washington (handwritten); MLT19 Latin, COCO-Text, TextOCR (scene).
  - Variants (Table 2): Avg (HW, Scene) = two task vectors; Avg (Ind.) = seven vectors; Avg (Orth.) = drop the Parzival and MLT vectors because "models trained on MLT19 and Parzival exhibited non-orthogonality via cosine distance calculation" (Sec. 5.1).
  - After merging, the merged model is fine-tuned (30 epochs, rather than the 75 used otherwise) on a new alphabet. How the output layer or charset is extended for scripts not in the seed (Arabic, Chinese, Japanese, Korean, Bangla, Hindi, Borg/Copiale/Vatican ciphers) is NOT described in the sections I read (UNVERIFIED). This matters: merging needs aligned parameter shapes.
  - No per-layer/module-level editing, no negation, no sparse or TIES-style merging, no adapters.
- **Assessment method (evaluation):**
  - Metric: "accuracy" on word crops (word-level exact match presumably; not defined in the text read; UNVERIFIED). No CER reported in the main text (the paper says supplementary material has further metrics).
  - In-domain: IAM, Esposalles, Parzival, GW (handwritten); MLT, COCO-Text, TextOCR (scene). Out-of-distribution: SVT, IIIT5K, WordArt, FUNSD, Historical Maps, TotalText, SROIE, AMR (gas-meter digits). Transfer targets: MLT Arabic/Chinese/Japanese/Korean/Hindi/Bangla, Copiale, Borg, Vatican.
  - Baselines: zero-shot HierText model; Table 1 grid of fine-tunes on each dataset ("FT-Best": best per column, not one model, "an upper bound", Table 2 caption); joint fine-tune on HW + Scene (centralised); ensemble of Wick et al. [62]; distributed from random init; for transfer, centralised fine-tune from Z.
  - Hyperparameters: Adam, lr 1e-5, batch 128, 75 epochs (30 for new alphabets); RTX 3090/A40; ~3,000 GPU hours overall.
  - Ablations: leave-one-out of task vectors, drop handwritten, drop non-English, drop historical (Table 4); data sub-sampling (Fig. 5, 6); k vs T at fixed k x T = 75 (Fig. 7).
- **Evidence the method measures what it claims:** Only accuracy comparisons against baselines; "generalisation" is defined operationally as accuracy on the OOD and new-script sets. No analysis of internal representations, language prior, or what the edit changes; "model editing" here is the authors' framing for task-vector averaging. The cosine non-orthogonality check on task vectors is the single weight-space diagnostic, and its numbers are not reported in the text I read.
- **Main finding:** Table 2: weight-averaged models slightly improve several OOD sets versus the joint fine-tune (e.g. WA .648-.684 vs .585; AMR .715-.738 vs .706; SVT +.01 to +.03 and IIIT +.01 to +.02) but lose heavily on in-domain handwritten sets (Parzival Avg(Ind.) .412 vs FT(HW,Scene) .858; Esposalles .606 vs .963; GW .523 vs .753; IAM .708 vs .773). Table 3 (new alphabets, accuracy): Baseline / Centralised / Distributed-from-Z: Arabic .131 / .175 / .422; Bangla .260 / .115 / .462; Korean .282 / .214 / .374; Japanese .116 / .095 / .197; Chinese .020 / .010 / .073; Hindi .470 / .384 / .514; Borg .382 / .000 / .505; Vatican .549 / .465 / .591; Copiale .825 / .794 / .838. The summary column "x delta" is 0.69 / 1.62 / 1.04 for centralised / distributed-from-Z / distributed-random (meaning not defined; UNVERIFIED, plausibly a mean relative factor against baseline). The authors claim "up to three times the performance" (Conclusions); Fig. 7: "task arithmetic is not all you need" (Sec. 5.2), too many rounds over-specialise.
- **Limits:**
  - Stated: gains plateau beyond ~60% of each sub-domain's data (Sec. 5.2 and Ethical Concerns); merging needs a good common seed (locality); non-orthogonal vectors (MLT, Parzival) had to be left out.
  - (mine) The Conclusion says the approach retains "high in-domain accuracy", but Table 2 shows large in-domain losses for the merged models (numbers above); the authors compare against an "upper bound" of the best per-dataset specialists.
  - (mine) The "three times" ratios start from very low baselines (Arabic .131, Chinese .020), and the absolute Chinese accuracy of .073 is not usable. Distributed-from-random beats distributed-from-Z on Copiale (.930 vs .838), so "consistently" is not exact.
  - (mine) Tables report single numbers; no seeds, no intervals, no significance tests visible. Fig. 5 mentions a standard deviation but the source of variance (runs vs scripts) is not stated (UNVERIFIED).
  - (mine) Architecture is a ViT, so any claim about linear mode connectivity or task-vector additivity does not carry over automatically to conv-BiLSTM models; no experiment on recurrent recognisers, on LoRA deltas, or on a language-prior question.
  - (mine) Experts are trained from the same seed with the same optimiser and a small learning rate (1e-5), which is the regime where task arithmetic is known to work; kraken-style fine-tunes with a changed codec would violate shape alignment.
- **Relevance:**
  - (a) kraken CTC line recogniser: This is the closest published precedent for editing a text recogniser in weight space: task-vector averaging of experts, merged-then-fine-tuned onto new scripts, with CTC loss. It does NOT cover an LSTM stack, a line-level recogniser, VGSL, or any analysis of the language prior, and it leaves the output-layer/charset handling for new alphabets unstated. The operation itself (theta_0 + mean of fine-tuned deltas) is architecture-agnostic in principle and could be applied to kraken checkpoints that share one base model and codec; the evaluation (accuracy on OOD sets and on new scripts, centralised baseline from the same data, k x T compute-matched controls) can be copied.
  - (b) QLoRA adapters on VLMs: Only an analogy; there are no adapters here. The adapter equivalent of their task vector is the LoRA delta (BA); averaging adapters is a natural extension but untested in this paper. Their finding that merged vectors can be non-orthogonal (cosine check) is a transferable diagnostic.

---

## E7. CC-OCR v2: Fine-Grained Attribution of LMM Failures in Real-World Visual Document Understanding

(Skimmed.)

- **Full text:** https://arxiv.org/pdf/2605.03903 (v2).
- **Authors, year, venue:** Chunyi Peng, Zhipeng Xu, Yuqi Xiong, Zulong Chen, Junhao Ji, Qing Liu, Zhenghao Liu, Zubao Qin, Bing Zhao, Jianqiang Wan, Jun Tang, Zhibo Yang, Shuai Bai, Dayiheng Liu, Maosong Sun (Northeastern University, Alibaba/Qwen Team, Tsinghua). arXiv:2605.03903v2 [cs.CL], 10 Aug 2026. Venue: the ACM template placeholder "Conference acronym 'XX" is still in the file; no real venue given.
- **Model analysed:** 17 LMMs, as named in the paper: e.g. Qwen3.7-Plus, Qwen3.6-Plus, Claude Opus 4.8, GPT-5.5, Gemini 3.1 Pro, Seed 2.0 Pro, Kimi K2.6, Qwen3.5-9B (on-device); full list in its Sec. 3 and App. A.8.
- **Question asked:** Why do models with similar aggregate OCR scores fail under different document conditions?
- **Assessment method:** Benchmark with factor annotations. 7,093 samples, 16 subtasks, 74 scenarios, five tracks (recognition, parsing, grounding, extraction, QA), ten document categories, 32 recognition languages. Each sample is manually annotated with ten document factors (Imaging Method, Image Quality, Lighting and Screen, Natural Interference, Adversarial or Misleading Factors, Font and Text Form, Mixed Layout, Geometry and Arrangement, Layout Structure, Language; Sec. 2.2). Failures are "attributed" by slicing each model's score per factor (Fig. 3 radar-style plots, Sec. 4.3) and by 13 hand-picked qualitative failure types (Sec. 4.4). Metrics: multi-set micro-F1 (recognition), normalised edit similarity and TEDS (parsing), mIoU/Acc@0.5 (grounding), field-level micro-F1 (extraction), normalised Levenshtein or containment match (QA). Deterministic decoding, same prompts.
- **Evidence the method measures what it claims:** The text says it estimates "the contribution of each factor while controlling for task, document category, and other document factors" (Sec. 2.2), but the estimator (regression, matching, stratification) is not named in the main text (UNVERIFIED whether the appendix specifies it; I did not find it). Annotation quality: primary annotator plus reviewers and adjudication (App.).
- **Main finding:** Rankings shift across document types and factors; similar aggregate scores hide different weaknesses (e.g. Claude Opus 4.8 weak under smearing and stamp overlays, dot-matrix and curved text, East Asian text; GPT-5.5 weaker on several non-Latin scripts). Qualitative labels for failure types include "Seal Hallucination" and "Low-resource-to-English Drift", glossed as "insufficient visual evidence can be overridden by dominant language priors" (Sec. 4.4).
- **Limits:**
  - Stated: none found as a dedicated section (UNVERIFIED).
  - (mine) "Attribution" is correlational slicing of black-box scores by annotated input factor; it says which input conditions hurt, not whether the network failed because of vision or language prior. The prior-override claim is an interpretation of examples, not a test.
  - (mine) Factors co-occur and are annotated by humans; the controlled estimate is unspecified; no repeated runs.
- **Relevance:**
  - (a) kraken CTC: The factor-annotation idea (tag each line with smearing, stamp, curved, script, etc. and report per-factor scores) transfers directly to a test set of historical lines; black-box, works for any recogniser.
  - (b) QLoRA on Qwen VLMs: Same black-box per-factor reporting could show where an adapter helps or hurts; it contains no analysis of weights or activations.

---

## E8. Using Human Psychophysics to Evaluate Generalization in Scene Text Recognition Models

- **Full text:** https://arxiv.org/pdf/2007.00083 (v1).
- **Authors, year, venue:** Sahar Siddiqui, Elena Sizikova, Najib J. Majaj (New York University), Gemma Roig (Goethe University Frankfurt), Denis G. Pelli (New York University). arXiv:2007.00083v1 [cs.CV], 30 Jun 2020. Venue as given: "Preprint. Under review."
- **Model analysed:** Two scene-text recognisers trained from scratch per condition: a CRNN with CTC loss (Shi et al., PAMI 2016, ref. [32]) and a seq2seq attention model (Shi et al., CVPR 2016, ref. [33]) with LSTM decoder. The recurrent layer type of the CRNN is not stated in the text beyond the citation (UNVERIFIED in the paper; the cited CRNN is conv plus recurrent plus CTC). Optimisers: Adadelta (CTC), Adam (Attn), batch 32, up to 100 epochs, early stopping on test word accuracy; input 100x32 (CTC) and 280x32 (Attn).
- **Question asked:** What is each model's "domain", i.e. the range of stimuli it can read (word length, font, occlusion, noise, flankers)? "testing models till they break."
- **Assessment method:**
  - Synthetic word images from 22,878 lower-case English dictionary words, 2-7 letters, 70:30 train/test split; white-on-black, Courier by default, greyscale 100x11 px pre-rendered (Sec. 4.2-4.3).
  - Manipulations (Sec. 4.1): bottom-up and top-down occlusion (0-100% in 10% steps), five fonts (Courier, Arial, Comic Sans MS, Bookman Old Style, Courier CTT Bold Italic), three random flanker symbols, salt-and-pepper noise p=0.15.
  - One model of each type is trained on each of 11 training sets (all words, three length-restricted sets, bottom-up, top-down, flankers, 3 fonts, 5 fonts, noise, all manipulations) and tested on all (the train-X/test-Y heatmap, Table 1). Detailed sweeps on 100 images per sub-category, 5 runs, mean and standard deviation (Figs. 2-5).
  - Metrics: word accuracy, character accuracy, word-length accuracy, edit-distance accuracy (Sec. 4.4). Human comparison: taken from the psychophysics literature ([15] etc.); the authors did not run human observers in this paper (as far as I found; UNVERIFIED).
- **Evidence the method measures what it claims:** Controlled single-factor manipulation with matched train/test grid and replicate runs is the design's evidence. "Generalisation" is defined as accuracy off-diagonal. No test of lexical/language-prior reliance (words are all real dictionary words in train and test, so a word-level lexical prior is possible and untested; mine).
- **Main finding:** Both models do best on the manipulation they were trained on and generalise poorly elsewhere. CTC generalises better to new word lengths, handles flankers ("The CTC model handles flankers perfectly", Sec. 5), tolerates noise and occlusion better; the Attn model "attempts to align" the input sequence with the output and drops flanked characters (e.g. "model" read as "ode"). Training with noise improves occlusion robustness by about 18% in both. Both fail on new fonts and tolerate less occlusion than humans do.
- **Limits:**
  - Stated: none as a section; the Conclusion states "Our results reveal the models' limited ability to generalize."
  - (mine) Tiny synthetic task (100x11 px, 2-7 letters, English dictionary words, single trained model per cell apart from the 5 evaluation repeats) and 2016-era architectures. Early stopping uses the test-set word accuracy (Sec. 4.3), a mild leak. The text on occlusion is ambiguous: "both models perform better with top-down than with bottom-up occlusion" yet "tolerate about 1.5 times bottom-up more than top-down occlusion" (Sec. 5); I could not resolve which direction is meant. No language-prior or hidden-state analysis.
- **Relevance:**
  - (a) kraken CTC line recogniser: Closest architecture class (conv-recurrent-CTC) and a ready design for a behavioural domain map: train on a manipulation, test on all others, report word, char and edit-distance accuracy; its finding that CTC copes with flankers while the attention decoder does not is relevant to why a CTC model behaves differently from VLM decoders in E1/E2. Black-box, transfers directly (it was done on a CTC network).
  - (b) QLoRA on Qwen VLMs: The train-on-X/test-on-Y grid can be applied to adapters (train an adapter per manipulation, test across), but the paper itself says nothing about VLMs or adapters.

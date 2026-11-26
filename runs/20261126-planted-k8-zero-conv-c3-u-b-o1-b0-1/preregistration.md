# Pre-registration — 20261126-planted-k8-zero-conv-c3-u-b-o1-b0-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (b), **GPT-6.2 Astra rerun** of `runs/20261107-planted-k8-zero-conv-c3-u-b-b0-1` (Sonnet 5.5). Written before the first pass; the program is `runs/planted-xv/o1/s01/b/program.py`, copied unchanged.

1. **Subject:** `k8-zero-conv-c3.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: conv block C_3 (weights and bias) set to 0; `runs/planted/plant.json`) — tensor digest `df5577f17d54eec6a967ac38de6e09b18aefd467b1b80041c119ece5631304db`, file sha256 `80df9c7b020b643618164105826e3d7ee088addaad390b2e24232a5e6aa4ebd4`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: `kraken-bern-hertwig-v2`, **an unrelated donor** (other lineage, trained on Bern 1528 data, never on BLB); the parent is not available to this arm.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 0 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. Budget 0: edits are accepted unmeasured (`keep_unmeasured`).
5. **Hypotheses (predicted direction and magnitude):**

Model: `gpt-6.2-astra` (GPT-6.2 Astra via OpenAI API).
Settings: provider-default reasoning; tools: read_file, list_dir, write_file (automatic py_compile), finish_round. Zero evaluation passes.

**H1 — erased second convolution.** Both `C_3` tensors are zero: changing the image cannot change the signal that reaches `C_6`. This explains all 300 empty readings without positing a blank-logit defect. Replace only the erased kernel and bias. Use Bern's second convolution as a stroke-feature prior, not as the missing weights: match its input channels by the signed similarity of first-layer filters, and estimate its output correspondence from the spatial weight patterns with which `C_6` consumes each channel. Keep all intact Kurrent weights.

**Prediction:** fewer than 300 empty lines and a partial CER reduction, roughly 5–25 percentage points from 100%. The diagnosis is much more certain than the magnitude: the unrelated model need not encode the same stroke features, even after channel matching. A return to normal reading accuracy is not predicted.


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** GPT-6.2 Astra (`gpt-6.2-astra`), called through the OpenAI API by `wsp.programmer` (docs/XVENDOR_PLAN.md), one fresh conversation per run, provider default reasoning settings, tools read_file/list_dir/write_file (py_compile) only; read only `runs/planted-xv/o1/PROGRAMMER_BRIEF_UNRELATED.md`, its pack (`runs/planted-xv/o1/s01/pack/*`) and the API files; not told which defect was planted; never shown the Sonnet or Fable runs, their programs or results (7 API calls, 52786 input / 1740 output tokens). Budget 0. Operator: a Claude Opus 5.5 (claude-opus-5-5) session; it knows the defects and the Sonnet and Fable results, set up the runs and does the held-out scoring and the write-up; it wrote no hypothesis and no edit and passed nothing of this to the programmers.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).

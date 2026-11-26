# Pre-registration — 20261126-planted-k6-lstm16-x0.5-u-b-o1-b0-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (b), **GPT-6.2 Astra rerun** of `runs/20261107-planted-k6-lstm16-x0.5-u-b-b0-1` (Sonnet 5.5). Written before the first pass; the program is `runs/planted-xv/o1/s07/b/program.py`, copied unchanged.

1. **Subject:** `k6-lstm16-x0.5.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: last BiLSTM block (L_16, all weights and biases) x0.5; `runs/planted/plant.json`) — tensor digest `a339a718a392342d8f0c47c62505ed6b3f9c3f008b0ba5480471f534d1de55f1`, file sha256 `3c7da1f6f6ae42d70ccd49f5cea8b35dc781168bc70b46f557f7af9c3320fae5`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: `kraken-bern-hertwig-v2`, **an unrelated donor** (other lineage, trained on Bern 1528 data, never on BLB); the parent is not available to this arm.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 0 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. Budget 0: edits are accepted unmeasured (`keep_unmeasured`).
5. **Hypotheses (predicted direction and magnitude):**

Model: `gpt-6.2-astra` (GPT-6.2 Astra via OpenAI API). Reasoning effort: provider default.
Tools: read_file, list_dir, write_file (including py_compile), finish_round. No model execution or evaluation passes.

**H1 — the last BiLSTM was attenuated as a block.** In both directions, L16's recurrent weights have RMS 0.154–0.157 versus 0.299–0.310 at L14; its recurrent biases are 0.424–0.438 versus 0.931–0.934. Its input weights show a similar reduction. Weaker final character evidence fits the 0.828 length ratio, widespread deletions and 24 collapsed lines better than damage to a particular character row.

**Ship one scale edit:** amplify the six nonzero L16 tensors together, preserving the subject's features and gate balance. Use ×2 unless a nearby hundredth-step attenuation factor leaves a distinctive common float32 rounding signature in both directions, in which case invert that factor. Leave the blank and the rest of the head alone: deletion counts alone do not establish a blank-bias offset, and the unrelated donor cannot provide replacement features.

**Prediction:** fewer collapsed lines and deletions, with CER falling from 0.1993 by roughly 0.08–0.15 absolute. The exact gain and original scale cannot be inferred from RMS alone.


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** GPT-6.2 Astra (`gpt-6.2-astra`), called through the OpenAI API by `wsp.programmer` (docs/XVENDOR_PLAN.md), one fresh conversation per run, provider default reasoning settings, tools read_file/list_dir/write_file (py_compile) only; read only `runs/planted-xv/o1/PROGRAMMER_BRIEF_UNRELATED.md`, its pack (`runs/planted-xv/o1/s07/pack/*`) and the API files; not told which defect was planted; never shown the Sonnet or Fable runs, their programs or results (8 API calls, 46670 input / 1462 output tokens). Budget 0. Operator: a Claude Opus 5.5 (claude-opus-5-5) session; it knows the defects and the Sonnet and Fable results, set up the runs and does the held-out scoring and the write-up; it wrote no hypothesis and no edit and passed nothing of this to the programmers.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).

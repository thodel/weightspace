# Pre-registration — 20261126-planted-k8-zero-conv-c3-b-o1-b0-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (b), **GPT-6.2 Astra rerun** of `runs/20261107-planted-k8-zero-conv-c3-b-b0-1` (Sonnet 5.5). Written before the first pass; the program is `runs/planted-xv/o1/s09/b/program.py`, copied unchanged.

1. **Subject:** `k8-zero-conv-c3.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: conv block C_3 (weights and bias) set to 0; `runs/planted/plant.json`) — tensor digest `df5577f17d54eec6a967ac38de6e09b18aefd467b1b80041c119ece5631304db`, file sha256 `80df9c7b020b643618164105826e3d7ee088addaad390b2e24232a5e6aa4ebd4`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: parent `kraken-german-xix-v1`, a lineage donor; it was never trained on BLB data, so the donor rule holds.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 0 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. Budget 0: edits are accepted unmeasured (`keep_unmeasured`).
5. **Hypotheses (predicted direction and magnitude):**

Model: `gpt-6.2-astra` (GPT-6.2 Astra via OpenAI API); reasoning effort: provider default.
Tools used: read_file, write_file (automatic py_compile), finish_round. Budget: zero passes.

**H1 — erased second convolution.** Both `C_3.co.weight` and `C_3.co.bias` have zero norm, whereas both parent tensors are nonzero. No image content can pass through this block. That explains why all 300 lines decode to empty strings without requiring a defect in the blank row (its bias differs from the parent's by only +0.113).

**Edit:** transplant just those two tensors together from the parent. Neighboring convolution weights remain closely aligned with the parent's (cosines 0.993 and 0.967), supporting reuse of the parent's filters in their original channel order. Preserve the child's other tensors, including its expanded output layer.

**Prediction:** most lines will decode to nonempty text; practice CER should fall from 1.00 to roughly 0.10–0.30 (70–90 percentage points lower). The exact recovery is uncertain because the donor lacks the child's tuning of this block.


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** GPT-6.2 Astra (`gpt-6.2-astra`), called through the OpenAI API by `wsp.programmer` (docs/XVENDOR_PLAN.md), one fresh conversation per run, provider default reasoning settings, tools read_file/list_dir/write_file (py_compile) only; read only `runs/planted-xv/o1/PROGRAMMER_BRIEF.md`, its pack (`runs/planted-xv/o1/s09/pack/*`) and the API files; not told which defect was planted; never shown the Sonnet or Fable runs, their programs or results (7 API calls, 50608 input / 929 output tokens). Budget 0. Operator: a Claude Opus 5.5 (claude-opus-5-5) session; it knows the defects and the Sonnet and Fable results, set up the runs and does the held-out scoring and the write-up; it wrote no hypothesis and no edit and passed nothing of this to the programmers.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).

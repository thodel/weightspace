# Pre-registration — 20261126-planted-k2-negate-row-n-b-o1-b0-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (b), **GPT-6.2 Astra rerun** of `runs/20261107-planted-k2-negate-row-n-b-b0-1` (Sonnet 5.5). Written before the first pass; the program is `runs/planted-xv/o1/s13/b/program.py`, copied unchanged.

1. **Subject:** `k2-negate-row-n.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: output row of `n` (weights and bias) negated; `runs/planted/plant.json`) — tensor digest `2daff3da912c8f98552bdde696a9c565cfc6eff847652c81a836d5735660d372`, file sha256 `df3b7f92a0e392169f348393a5fdd0c8db10016ef7dad6edd6d6ed02ef347286`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: parent `kraken-german-xix-v1`, a lineage donor; it was never trained on BLB data, so the donor rule holds.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 0 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. Budget 0: edits are accepted unmeasured (`keep_unmeasured`).
5. **Hypotheses (predicted direction and magnitude):**

Model: `gpt-6.2-astra` (GPT-6.2 Astra, OpenAI API via the operator's tool layer).
Settings: provider default reasoning; read_file (including parallel reads), write_file with automatic py_compile, finish_round. Zero evaluation passes.

**Hypothesis:** The `n` output weight row has its sign reversed. Its cosine with the aligned parent row is −0.991 despite similar norms (8.33 vs 7.96). This accounts for both kinds of `n` error: contexts that should suppress `n` now emit it (7,667 insertions), while genuine `n` often decodes as `m` or `r`.

**Edit:** Negate only the child's `O_18.lin.weight` row indexed by its codec for `n` (`set_rows`). This restores positive alignment without discarding the child's learned adjustments. The bias delta of −0.142 is not compelling evidence of bias damage; leave all biases and other weights alone.

**Predicted effect:** The excess `n` insertions should largely disappear and the decoded/reference length ratio should approach one. I expect an absolute CER reduction of roughly 0.60–0.70 from 0.7456, not a measured result.


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** GPT-6.2 Astra (`gpt-6.2-astra`), called through the OpenAI API by `wsp.programmer` (docs/XVENDOR_PLAN.md), one fresh conversation per run, provider default reasoning settings, tools read_file/list_dir/write_file (py_compile) only; read only `runs/planted-xv/o1/PROGRAMMER_BRIEF.md`, its pack (`runs/planted-xv/o1/s13/pack/*`) and the API files; not told which defect was planted; never shown the Sonnet or Fable runs, their programs or results (6 API calls, 35951 input / 810 output tokens). Budget 0. Operator: a Claude Opus 5.5 (claude-opus-5-5) session; it knows the defects and the Sonnet and Fable results, set up the runs and does the held-out scoring and the write-up; it wrote no hypothesis and no edit and passed nothing of this to the programmers.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).

# Pre-registration — 20261126-planted-k3-swap-rows-e-n-b-o1-b0-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (b), **GPT-6.2 Astra rerun** of `runs/20261107-planted-k3-swap-rows-e-n-b-b0-1` (Sonnet 5.5). Written before the first pass; the program is `runs/planted-xv/o1/s12/b/program.py`, copied unchanged.

1. **Subject:** `k3-swap-rows-e-n.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: output rows of `e` and `n` swapped (weights and bias); `runs/planted/plant.json`) — tensor digest `e04ecfb521c6e879805ac6017fbecdaa4545f1e7e315de4804ea722f62b5ba9c`, file sha256 `c0bc09e5c852501c31d1fd20b48a9a2495071d13fa61edaf919573fb89b580e3`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: parent `kraken-german-xix-v1`, a lineage donor; it was never trained on BLB data, so the donor rule holds.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 0 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. Budget 0: edits are accepted unmeasured (`keep_unmeasured`).
5. **Hypotheses (predicted direction and magnitude):**

Model: `gpt-6.2-astra` (GPT-6.2 Astra via OpenAI API). Reasoning effort: provider default.
Tools: read_file, write_file (with py_compile), finish_round. No model runs or measurement passes.

**Hypothesis:** The child's `e` and `n` output logits have exchanged labels. The two reciprocal substitutions account for 2,201 of 3,144 edits, while the output/reference length ratio is still 0.969. Both rows have cosine near 0.36 with their own parent rows; the next most changed row has cosine 0.881. This points to a transposition of learned rows rather than a general loss of letter recognition.

**Edit:** Use `set_rows` to exchange the child's `e` and `n` weight vectors and their corresponding bias entries, identified through its codec. This inverts a full logit swap without discarding the child's adaptation. Leave the blank and every other parameter alone.

**Predicted effect:** Most reciprocal substitutions disappear. Their count corresponds to 0.171 CER, so I expect roughly a 17 percentage point absolute decrease from 24.43% to around 7%, allowing for alignment changes and genuine recognition errors. The weight exchange should account for nearly all the gain; exchanging the biases completes the permutation with little incremental CER change.


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** GPT-6.2 Astra (`gpt-6.2-astra`), called through the OpenAI API by `wsp.programmer` (docs/XVENDOR_PLAN.md), one fresh conversation per run, provider default reasoning settings, tools read_file/list_dir/write_file (py_compile) only; read only `runs/planted-xv/o1/PROGRAMMER_BRIEF.md`, its pack (`runs/planted-xv/o1/s12/pack/*`) and the API files; not told which defect was planted; never shown the Sonnet or Fable runs, their programs or results (6 API calls, 35997 input / 852 output tokens). Budget 0. Operator: a Claude Opus 5.5 (claude-opus-5-5) session; it knows the defects and the Sonnet and Fable results, set up the runs and does the held-out scoring and the write-up; it wrote no hypothesis and no edit and passed nothing of this to the programmers.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).

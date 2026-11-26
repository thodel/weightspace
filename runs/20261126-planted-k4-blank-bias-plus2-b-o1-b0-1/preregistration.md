# Pre-registration — 20261126-planted-k4-blank-bias-plus2-b-o1-b0-1

Planted-defect control (docs/BRIEF_ARTICLE_EXPERIMENTS.md item 2), arm (b), **GPT-6.2 Astra rerun** of `runs/20261107-planted-k4-blank-bias-plus2-b-b0-1` (Sonnet 5.5). Written before the first pass; the program is `runs/planted-xv/o1/s03/b/program.py`, copied unchanged.

1. **Subject:** `k4-blank-bias-plus2.mlmodel` (a corrupted copy of `kraken-german-xix-blb-v1`: blank bias +2.0; `runs/planted/plant.json`) — tensor digest `339161afc6ffa62ee51b2d9e5ce3dd0bbd0c3f3cd7939c6300bfa46500dc993b`, file sha256 `2727c5266842c02e67e61b779bf0d900f981af6137e12447ed7f279c833c268c`; the uncorrupted model has tensor digest `f35e075ad6d3e7286d9611b20e04a46f167a39e18f4f7b26ed2c73d3e33947dd`.
   Donor: parent `kraken-german-xix-v1`, a lineage donor; it was never trained on BLB data, so the donor rule holds.
2. **Practice set:** `data/practice/blb-train-300.jsonl`, 300 lines, sha256 `4a803b49ab336144225623d091f24dbf4359df2087d29580b8790b9e09d88a5b`. Source: the four BLB train projects (train side). **These are the subject's own training lines** (the review's second objection); the validation-based sets are item 3 of the same report.
3. **Held-out set:** `data/heldout/blb-eval-hand.jsonl`, 2272 lines, 29 objects, sha256 `a784ae084058c22bb3afb18165c0bcbc6f41ce3b13f5f22ea9fb277acb9f5091`; audit `data/heldout/overlap-blb-eval-hand.json` (0 shared documents with any practice set).
4. **Budget:** 0 passes (pass 1 = C0 baseline). Keep rule: PROTOCOL.md §3, paired 95 % interval below 0, `min_effect` 0. Budget 0: edits are accepted unmeasured (`keep_unmeasured`).
5. **Hypotheses (predicted direction and magnitude):**

Model: `gpt-6.2-astra` (GPT-6.2 Astra, OpenAI API).
Settings: provider-default reasoning; tools used: read_file, write_file (automatic py_compile), finish_round. Zero evaluation passes.

**H1 — excess CTC blank bias.** The blank bias is +3.232, up 2.113 from the parent, whereas the blank weight row changed by about as much as an ordinary shared row. The decoded text is 5.9% short and all ten leading confusions are deletions, across spaces and several common letters rather than one character pair. Subtract 2.0 from `O_18.lin.bias[0]` only: this puts the blank intercept near its parent value without replacing the child's learned visual classifier. I predict fewer deletions (especially of spaces), a length ratio closer to one, and a net CER decrease of roughly 1–3 percentage points from 7.89%; some extra insertions are possible. The exact excess cannot be inferred from the bias difference alone.

No donor transplants or other edits: the relatively large *fractional* change in the small C_3 bias is not by itself evidence of a damaged feature channel, and the practice confusions do not implicate a particular nonblank row.


6. **Controls:** the other three arms on the same model; C0 on the corrupted model and the uncorrupted model on both sets (`docs/CONTROLS_REPORT.md`). The corrupted model's CER before the session is the baseline of this session.
7. **Programmer:** GPT-6.2 Astra (`gpt-6.2-astra`), called through the OpenAI API by `wsp.programmer` (docs/XVENDOR_PLAN.md), one fresh conversation per run, provider default reasoning settings, tools read_file/list_dir/write_file (py_compile) only; read only `runs/planted-xv/o1/PROGRAMMER_BRIEF.md`, its pack (`runs/planted-xv/o1/s03/pack/*`) and the API files; not told which defect was planted; never shown the Sonnet or Fable runs, their programs or results (5 API calls, 25279 input / 998 output tokens). Budget 0. Operator: a Claude Opus 5.5 (claude-opus-5-5) session; it knows the defects and the Sonnet and Fable results, set up the runs and does the held-out scoring and the write-up; it wrote no hypothesis and no edit and passed nothing of this to the programmers.
8. **Outcome classes (fixed before any session):** repaired = held-out CER of the frozen model recovers at least 80 % of the corruption's held-out damage (CER_corrupted − CER_original) and the paired interval of final vs corrupted lies below 0; partly = 20–80 %; not repaired = under 20 % or CER above the corrupted model's. A corruption with damage under 0.5 points on held-out is a *null corruption* (nothing to repair; any change is reported as a harm or neutral).

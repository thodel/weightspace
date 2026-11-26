# Pre-registration — 20261126-planted-adapter-p3-alpha2-b-g1-b0-1

Planted-defect control on an adapter (docs/ADAPTER_REPORT.md, stage A3), arm (b), **Gemini 4 Argon rerun** (pack `q01`, attempt 1) of `runs/20261107-planted-adapter-p3-alpha2-b-b0-1` (Sonnet 5.5). Written before the first pass; the program is `runs/planted-xv/g1/q01/b/program.py`, copied unchanged.

1. **Subject:** `p3-alpha2`, a corrupted copy of `qwen3.5-4b-german-kurrent-xvi-xix-page-v2` (`plant.json`: {"module": "all (config)", "lora_alpha": 256, "digest": "b08cc2e90f19a89471c77ac2a4ae2125bebdbb6ac682fa5a8cd910e9689926f0"}); tensor digest `fb4f55ee7386abc7292e29bbf2a24f502fc4079cb4732934cbead229b300504c`. The tensor digest does not cover `lora_alpha` (config), so p3/p5 share the clean adapter's tensor digest; the file digests in `plant.json` differ. Source adapter file digest `f921f0c2a56976b1…`. Donors: siblings `…-page-v1`, `…-blb-page-v1` for statistics only (independent A initialisations).
2. **Practice set:** `data/practice/kurrent-val-32.jsonl`, 32 pages (every second page of the 64-page practice draw of the registered val side), sha256 `3d1311f00140575ecd44c81aaba7979011fc8cd963f157dfbe3644bdbaecf31d`. Not training data of v2 (docs/PHASE5_PLAN.md §2; audit `data/practice/kurrent-val-64.audit.json`).
3. **Score set:** `data/practice/kurrent-val-holdout-64.jsonl`, 64 val pages disjoint from the 64 practice pages, sha256 `6300181905a39a800e7227494ad621236e6b17ed10a2b912ba47cb0c440fa886`; the test side stays closed. Not a true held-out set (same documents/hands as the practice draw for 56 of 60 documents), a positive-control score set.
4. **Budget:** 0 passes (pass 1 = C0). Keep rule: PROTOCOL.md §3 on pages (paired page interval below 0, pages at the cap not up). Passes: vLLM batched, greedy, cap 3,072, **not repeatable run to run** (A0): the keep rule sees that noise.
5. **Hypotheses (predicted direction and magnitude):**

# Hypotheses — pack `q01`, task (b) (budget 0)

- **Model ID:** `gemini-4-argon` (Gemini 4 Argon, called through the Google API by the operator's tool layer)
- **Settings:** provider default reasoning settings; tools used: `read_file`, `write_file`, `finish_round`

## Diagnosis

1. **Global `lora_alpha` / scale doubling across all 128 modules:**
   - The adapter config has `lora_alpha = 256` with `r = 64`, giving an update scale of `lora_alpha / r = 4.0`.
   - Both healthy sibling adapters (`v1` and `blb1`, trained with the same base and recipe) have `lora_alpha = 128` (`r = 64`), giving a scale of `2.0`.
   - Consequently, the total Frobenius norm of $\Delta W$ across all modules is `234.99`, whereas the siblings have total norms of `127.35` (`v1`) and `135.65` (`blb1`) — a ratio of $234.99 / 131.50 = 1.787$.
   - Examining the per-module table across all 128 modules (`mlp.{gate,up,down}_proj` in layers 0–31 and `self_attn.{q,k,v,o}_proj` in layers 3, 7, 11, 15, 19, 23, 27, 31): **every single module** has a norm ratio to the sibling median tightly clustered between `1.60` and `1.85` (median `1.79`), with completely healthy effective ranks (`16.6` to `48.2`), normal $\sigma_1 / \|\Delta W\|$ proportions, and smooth layer-to-layer profiles of $\|B\|$ and $\|A\|$. No individual module, layer, or rank-1 direction is an outlier relative to the rest of the adapter.
   - Because $\Delta W$ is over-scaled by a factor of $2\times$ across the entire language tower, generation destabilizes into repetition loops: in the C0 baseline reading, 29 of 32 pages run to the 3,072-token cap (`loops: 28`, `length_ratio: 2.55`), inflating page-level CER to `2.5029` (and `truncated_cer` to `0.8243`).

## Hypothesis & Edit Plan

- **H1 (Global scale repair — edit class: `scale`):**
  - **Hypothesis:** `q01` suffers from a planted global scale defect where `lora_alpha` was doubled from `128` to `256` (or equivalently all updates $\Delta W$ are scaled by $2\times$ relative to the healthy recipe scale of `2.0`). Since the harness saves tensor weights while keeping `adapter_config.json` unchanged (`lora_alpha = 256`), scaling every module's `lora_B.weight` tensor by `factor = 0.5` restores the effective update $\Delta W = (\text{lora\_alpha} / r) \cdot (0.5 B) \cdot A$ to the exact `lora_alpha = 128` scale (`2.0`), bringing the total $\|\Delta W\|$ from `234.99` to `117.50` and per-module sibling ratios to $\approx 0.89–0.92$.
  - **Edit:** `s.edit("scale", b_names, hypothesis=h1, factor=0.5)` where `b_names` are all 128 `*.lora_B.weight` tensor names.
  - **Predicted effect:** Large reduction in page-level CER on the practice set (predicted $\Delta\text{CER} \approx -2.2$ to $-2.4$, bringing CER from `2.503` down to the normal $\sim 0.10–0.30$ range) and a drastic drop in pages running to the token cap (`at_cap` falling from `29/32` to $\le 3/32$).


6. **Outcome classes (fixed before any session):** let N be the repeat noise of the clean adapter on the score set (|CER difference of two readings|, at least 3 points) and D = CER(corrupted) − CER(clean) on the score set. D < max(N, 3 points): *null corruption* (below the pass noise). Else *repaired* = CER(final) recovers at least 80 % of D and the paired page interval of final vs corrupted lies below 0; *partly* = 20–80 %; *not repaired* = under 20 %.
7. **Programmer:** Gemini 4 Argon (`gemini-4-argon`), called through the Google API by `wsp.programmer` (docs/XVENDOR_PLAN.md), one fresh conversation per run, provider default reasoning settings, tools read_file/list_dir/write_file (py_compile) only; read only `runs/planted-xv/g1/PROGRAMMER_BRIEF_ADAPTER.md`, its pack (`runs/planted-xv/g1/q01/pack/*`) and the API files; not told which defect was planted; never shown the Sonnet or Fable runs, their programs or results (5 API calls, 101523 input / 2049 output tokens). Budget 0. Operator: a Claude Opus 5.5 (claude-opus-5-5) session; it knows the defects and the Sonnet and Fable results, set up the runs and does the held-out scoring and the write-up; it wrote no hypothesis and no edit and passed nothing of this to the programmers.

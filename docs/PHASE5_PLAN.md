# Phase 5 plan: the Qwen3.5-4B Kurrent page adapter (asteraix, 07.11.2026)

Status: decisions taken 07.11.2026 (§7); work under way. It covers phase 5 of PLAN.md §5:
"the same deliverables as 1–4 for one adapter". It was written from the
harness as it stands (`wsp/lora_subject.py`, PROTOCOL §7) and from
training-atr-models at `43fd73b`. Several facts below correct what PLAN.md and
DESIGN.md assumed, and §1 lists them.

## 0. In one paragraph

The target is **`qwen3.5-4b-german-kurrent-xvi-xix-page-v2`**, job
`20261029T133027Z-…`, the only Qwen Kurrent page arm with a clean held-out
number. On `german-kurrent-xvi-xix-v1/test` (733 pages, 70 documents) it reads
**CER 0.4181**. Its `truncated_cer` is **0.2502**, its `length_ratio` 1.181, and
**49 of 733** pages run to the 3,072-token cap (HELDOUT_BOUNDARY.md, #341). About
17 of its 42 points are text written past the page, and that is the defect. The
adapter has rank 64 over 128 language-tower modules, so it has at most 8,192
enumerable directions. The session looks for the directions that keep the model
from stopping and edits only the adapter. The edit has to beat two cheaper
fixes: a decode-time repetition guard and a short retrain.

## 1. What the material actually is (checked 07.11.)

| assumption (PLAN.md / DESIGN.md) | measured | consequence |
|---|---|---|
| rank 16 | **r = 64, α = 128** (scale 2), F32, peft 0.20.0. Read off the sibling `qwen3.5-4b-german-xix-v2` from the same recipe; v2 itself still has to be read | up to 64 directions per module |
| language and vision tower adapted | **language tower only**: `mlp.{gate,up,down}_proj` in all 32 layers, and `self_attn.{q,k,v,o}_proj` in the **8** full-attention layers. The Gated-DeltaNet layers and the vision tower are untouched | 128 modules. An edit cannot reach the linear-attention layers or the vision tower |
| `lm_head` not adapted | confirmed | edits cannot touch the EOS logit directly, only the residual stream that feeds it |
| the trainer's eval is a vLLM pass | **it is transformers `generate`**, greedy (`do_sample=False`). It uses the prompt cut from the training render, `fit_pixels(max_pixels)` and the processor's stop ids (`engines/vlm_train_svc/evaluate_qlora.py` l.373) | a vLLM pass is a **proxy** and must be shown equivalent first (§3, gate G1) |
| the adapter is on HF | **only v1 is on HF** (`dh-unibe/…-page-v1`), and v1 trained on the held-out documents (`crossed`). **v2 is only on UBELIX scratch**, which purges after 30 days, so around **28.11.2026** | copy it first (§3, P5.0) |
| `wsp.stats.adapter_directions` exists | it does not | write it in P5.1 |
| card 1 is held by the bar_mcp embedding server (18.6 GB) | at 07.11. both cards showed 0 MiB | the rule does not change: card 0 only (§5) |
| training-atr-models#396 is a proposal | **merged** (`43fd73b`, 06.11.): `GET /gpu` and `GET /health` carry `waiting` | a session checks it before every pass (§5) |

The vLLM venv exists: `~/.venvs/vllm` with vllm 0.29.0, transformers 5.20 and
torch 2.13. The base `Qwen/Qwen3.5-4B` is in the HF cache. The wsp venv has no
vLLM, and it does not need one: the forward pass runs as a subprocess in the
vLLM venv.

## 2. The sets

- **Held-out:** `eval_sets/german-kurrent-xvi-xix-v1/vlm-pages-test`, 733 pages,
  `draw_md5` `ae8bf1fd…`, frozen on the share and readable here. It is scored
  once, at the end, the same way the trainer scores it: transformers, greedy,
  cap 3,072. The v2 job's `eval_report.german-kurrent-xvi-xix-v1-test.raw.jsonl`
  holds per-page held-out readings and must not reach the programmer.
- **Practice (recommended): the registered `val` side** of the same set. It has
  509 pages from 60 documents (`draw_md5` `4ade0622…`). The trainer withholds
  these documents from every run and scores nothing on them. These are unknown
  hands, so the defect should show at the held-out rate. On kraken, train-side
  pages served as practice. For a page VLM, train pages are partly memorised and
  may not loop at all.
  - Draw **64 pages by document** with a seed, stratified by reference length so
    that long pages are represented. Long pages are where the cap is hit.
  - The size is fixed for the session and logged. 64 is a guess until P5.2 has
    timed a pass.
  - **Alternative:** train-side pages from corpus `4e2abd840de3`. They sit on
    UBELIX scratch too, under the same purge.
- **C0 over all 509 val pages**, outside any budget, as in phase 1. It gives the
  at-cap rate and the `truncated_cer` before the practice draw is fixed.
  Selecting practice pages on C0 output is allowed and preregistered. The
  held-out number decides in the end, not the practice number.

## 3. Steps

**P5.0 Secure the material.** CPU only, needs UBELIX read access.
- Copy the v2 adapter directory into `~/atr-cache/wsp/adapters/` with the
  `ubelix_*` tools or scp. Take the whole directory, never only the
  safetensors file.
- Copy its `job.json` and the training manifest too, then record the sha256
  values in `models/DIGESTS.md`.
- Archiving it on the share or uploading it to HF is the trainer's act, not
  this repo's. Raise it there, because scratch purges around 28.11.

**P5.1 Harness.** Runs on h1 or here, no GPU, tests green.
- `LoraSubject` gets a per-module scale (default `lora_alpha / r`). Scaling
  writes into `B`, so the saved adapter stays an ordinary peft directory and
  rank 64 is kept.
- `directions(module)` already exists. Add `wsp.stats.adapter_directions`: the
  top singular values per module, energy per layer, and the rank actually used.
- Add page metrics to `wsp.metrics`: an at-cap count (`finish_reason == "length"`),
  `truncated_cer`, `length_ratio`, and a page-scale loop detector based on
  `wsp.collapse.degenerate_repeat`. The keep rule and the bootstrap work per
  page instead of per line.
- `wsp.run` accepts an adapter directory as a subject.

**P5.2 Forward pass.** Card 0, in `wsp/vllm_forward.py`.
- Run as a subprocess in `~/.venvs/vllm`, one process per pass, so the card is
  free between passes.
- Use the same inputs as `evaluate_qlora`: the prompt string from the training
  render, the run's `max_pixels`, greedy decoding, the processor's stop ids and
  `max_tokens` 3,072. Log finish reasons and token counts.
- Apply the adapter with `enable_lora` (`max_lora_rank` 64) when vLLM 0.29
  supports LoRA on Qwen3.5's attention and MLP modules. Otherwise merge
  `W += s·B·A` into a bf16 copy of the base for each pass. The copy is about
  9 GB, written under `~/atr-cache/wsp/`.
- **Gate G1, engine equivalence.** On 32 val pages, compare the vLLM no-op
  pass with `evaluate_qlora` on the same pages.
  - **Thresholds:** the CER between the two readings is ≤ 0.5 points, and the
    at-cap sets are identical.
  - **Why the at-cap test matters:** greedy loops are sensitive to numerics, so
    the defect itself may differ between engines.
  - **If G1 fails,** passes run in transformers. They are slower, and the
    budget is in passes, so the budget stays the same.

**P5.3 C0 and the preregistration.**
- Run C0 on the 509 val pages and draw the practice set.
- Run the overlap audit on three pairs:
  - practice against test (disjoint by registry, still checked);
  - practice against the training manifest;
  - the v2 run's own validation draw against practice.
- Write `preregistration.md` with the hypotheses of P5.4 and their predicted
  effects.

**P5.4 Reading, budget 0.** No forward pass.
- SVD of ΔW per module, and how energy is spread over layers and module kinds.
- Logit lens without a pass: map the top output directions of `down_proj` and
  `o_proj` ΔW in the late layers through the base's final norm and `lm_head`.
  Which tokens do they promote or suppress?
- Hypotheses to test:
  - **H1:** a direction that lowers `<|im_end|>` or EOS. The adapter learned
    not to stop.
  - **H2:** directions that promote line-break and hyphen tokens (`¬`, `\n`).
    These are the repetition loops on long pages.
  - **H3:** a loop direction written by a few MLPs, as in Takehi et al., Exp 8.
- Each hypothesis names its modules, its edit and its predicted effect on the
  at-cap count and on CER.

**P5.5 Session.** Budget 10, then 25 if 10 shows a lead.
- **Edit classes:**
  - per-module scale (α);
  - direction subtraction on a module's `B` columns, which keeps the rank;
  - transplanting one module from a sibling adapter of the same base. The only
    candidate is v1, and only for reading, because v1 is `crossed` on held-out.
- **Keep rule:** that of phase 2 (PROTOCOL §3), on the page bootstrap. As a
  second condition, the at-cap count on practice must not rise.

**P5.6 Controls and comparators.**
- **C0:** the adapter as it is.
- **C1:** the base without the adapter. Does the base loop as well? If it
  does, the defect may not be in ΔW at all.
- **Sham:** a random direction of equal norm, subtracted from the same modules.
- **α-sweep:** the whole adapter at 0.75, 0.9 and 1.1.
- **Decode comparator:** `repetition_penalty` 1.05, or a no-repeat n-gram
  guard. This is the cheapest fix of all, and a weight edit that does not beat
  it is not worth shipping.
- **C4:** a short QLoRA continuation through `train_spec`. This is a trainer job
  and needs explicit approval.

**P5.7 Held-out, once.**
- Score the 733 test pages with `evaluate_qlora` and the edited adapter. Two
  places can do it:
  - on card 0 here, in the trainer's `vlm-train` venv; this may take hours,
    so time it in P5.2;
  - on UBELIX with `rescore_on_shared_draw.py --adapter <dir> --heldout`.
    This needs the edited adapter copied to UBELIX, which is a write outside
    this repo, so ask first.
- Reference: 0.4181 / 0.2502 / 49 at cap.
- `docs/PHASE5_REPORT.md` and a row in `docs/RESULTS.md`.

## 4. Done when

A `runs/<session>/` directory holds the log, the diff (a sparse diff of the
adapter), the summary and one held-out score. The comparator table includes
C0, C1, the sham, the α-sweep, the decode guard and, if approved, C4. The
report says whether a weight-space edit fixes stopping better than a decode
setting. A negative answer is a result too.

## 5. The cards

- **Card 0 only, never card 1.**
- **Before every pass:**
  - `nvidia-smi` shows at least 12 GB free on card 0, and `GET /gpu` on the
    trainer has `waiting` set to null;
  - if a waiting job could use card 0, pause.
- **VLM headroom:** a VLM trainer job needs 24 GB on a card. Size vLLM's
  `gpu_memory_utilization` so that 24 GB stay free on card 0. That is roughly
  0.45 of 46 GB for weights plus the KV cache of a 64-page batch, to be measured
  in P5.2. Alternatively, run only while `waiting` stays null.
- **One process per pass:** nothing holds the card between passes.
- **Environment:** `TMPDIR=~/atr-cache/tmp`. Writes go only under
  `~/Repo/weightspace` and `~/atr-cache/wsp`. Nothing goes to the share, the
  registry, HF or the trainer, and there are no service restarts.

## 6. Risks

- vLLM 0.29 may lack LoRA for Qwen3.5's hybrid layers. The fallback is a merge
  per pass, which costs about a minute and 9 GB per pass.
- G1 can fail on exactly the long pages that matter. The fallback is
  transformers passes and a smaller practice set.
- The defect may sit in the base and not in ΔW (C1 says). Takehi et al.'s best
  edit was on base weights; here the subject is the adapter. Whether edits may
  touch base weights is an open decision, and the plan's answer is no.
- `lm_head` is not adapted, so an EOS fix has to go through the residual
  stream. It is weaker and riskier than a bias edit.
- The v2 adapter is lost if it is not copied before about 28.11.

## 7. Decisions (taken 07.11.2026, on the plan's recommendations)

1. **Practice set: the registered `val` side** of `german-kurrent-xvi-xix-v1`.
   64 pages are drawn by document, with a seed, stratified by reference length,
   after C0 over all 509 pages. The test side is not opened before the final score.
2. **Edit scope: the adapter only.** Base weights are read, and the base without
   the adapter is run as control C1, but they are never edited. An edit is a
   change to `lora_A` / `lora_B` or to a per-module scale, and the result is
   again a rank-64 peft directory.
3. **Held-out path: here, card 0, through the trainer's own `evaluate_qlora`**
   (transformers, greedy, cap 3,072, `max_pixels` 2,097,152, the run's prompt).
   Nothing is copied to UBELIX.
4. **C4 retrain: deferred.** It is submitted only if a session keeps an edit,
   and then through `train_spec` with a separate go-ahead, because it is a
   trainer job.
5. **The v2 adapter is copied into `~/atr-cache/wsp/adapters/`.** Archiving it
   on the share is the trainer's act.
   - **Copy blocked (07.11.):** the adapter is at
     `/scratch/network/users/th19c587/runs/trained/qwen3.5-4b-german-kurrent-xvi-xix-page-v2`
     (`ubelix_record`, Slurm 20544433). It is not on the share:
     `trained-ubelix/` holds only `…-page-v1` and the BLB arms. This box's UBELIX
     key is restricted to the results probe (`atr_results_probe_only`), so the
     copy needs someone with a login shell before the purge, about 28.11.
   - **What does not need v2 goes ahead with v1 as a stand-in:** the harness,
     the vLLM pass, gate G1 and the C0 machinery. v1 is
     `trained-ubelix/qwen3.5-4b-german-kurrent-xvi-xix-page-v1`, the same recipe.
     v1 trained on the registered documents, so its readings on `val` measure
     the engine, not the defect, and are never reported as practice numbers.

## 8. Status (07.11.2026)

| step | state | where |
|---|---|---|
| P5.0 copy v2 | done 07.11. Tobias copied it from UBELIX scratch to the share (`trained-ubelix/…-page-v2`), and the phase-1 session copied it from there. sha256 `1b70ce4a…`, identical in both places (`models/DIGESTS.md`). Registration stays the trainer's act | `~/atr-cache/wsp/adapters/` |
| P5.0 v1 stand-in | copied from the share. sha256 `52f94e10…` (`adapter_model.safetensors`) | `~/atr-cache/wsp/adapters/` |
| P5.1 harness | done. `LoraSubject` has peft's scale, QR-based SVD, `scale_module`, `remove_direction` and `vllm_runner`. `wsp.pages` holds the page metrics and a numpy Levenshtein (0.08 s for a 5,000-character page). `wsp.card` checks card 0 and #396's `waiting`. Tests: `tests/test_pages.py`, `tests/test_lora_subject.py` | `wsp/` |
| P5.2 render | done for the 509 val pages. Prompt, stop ids `[248046, 248044]` and `fit_pixels` come from the trainer's own code (`vlm-train-tf5`, transformers 5.17) | `wsp/vlm_render.py`, `~/atr-cache/wsp/render/kurrent-val/` |
| P5.2 vLLM pass | works. vLLM 0.29 applies the rank-64 LoRA to Qwen3.5 with `enable_lora`, so no merge is needed. `ninja` from the vLLM venv has to be on `PATH`. Loading takes about 2.5 min per process | `wsp/vllm_forward.py` |
| P5.2 reference | `evaluate_qlora`, with its argv built by the trainer's `evaluate_cmd` from v2's `job.json` | `wsp/vlm_reference.py` |
| G1 (v1, 32 val pages) | **failed**: see below. Passes run through `evaluate_qlora` (`wsp.lora_subject.reference_runner`) | `docs/baselines/g1-kurrent-val-32-v1.json` |
| reference repeat | deterministic: 32 of 32 pages byte-identical, same at-cap pages | the same file, `reference_repeat` |
| session runner | `wsp.run new` takes an adapter directory and `--data-root`. Sessions on pages use `unit="page"`, and the keep rule gains "pages at the cap must not rise". `heldout --set kurrent-test` runs the trainer's scorer once | `wsp/run.py`, `wsp/session.py` |
| P5.3 practice draw | done, see below | `data/practice/kurrent-val-64.*`, `scripts/draw_kurrent_practice.py` |
| P5.3 C0 | running: v2 on all 509 val pages in 16 chunks of 32, card checked between chunks. About 17 minutes per chunk | `scripts/c0_kurrent_val.py` → `docs/baselines/c0-kurrent-val-v2.json` |
| P5.3 preregistration | after C0 | — |
| P5.4 reading | done for v2, and for v1 as a method test | `docs/readings/adapter_kurrent-page-v2.*`, `…-v1.*` |

**Practice draw.** 64 pages from all 60 val documents: one page each, and three
from each of the two 160-page documents. The seed is 20261107, the side's
`draw_md5` is `4ade0622…`, as registered, and the sha256 is `2cc5278e…`.

**Overlap audit.** There is no document in common with the 70 test documents,
and none with v2's training manifest (1,191 documents). Six practice pages are
on v2's own *validation* split, which the run scored but never trained on. The
trainer withholds reserved documents from training only.

**Reading v1 (method test only, not a v2 finding).** Energy sits mostly in the
MLPs: `gate_proj` 47 %, `up_proj` 28 %, `down_proj` 15 %. Attention takes 11 %,
nearly all of it in `q_proj`. No direction dominates. The median module needs 35
of its 64 directions for 90 % of its energy, and its top four hold 32 %. Through
the logit lens, the output direction that lowers the end-of-turn token most is
direction 0 of `layers.31.self_attn.o_proj` (z −3.95). The same direction
suppresses the newline (−4.97) and `¬` (−7.14). Directions 0 of
`layers.29/30.mlp.down_proj` raise all three. A candidate for H1 and H2, to be
checked on v2 before anything is preregistered.

**G1, engine equivalence (v1 stand-in, 32 val pages stratified by length).**
vLLM 0.29 against `evaluate_qlora`: 12 of 32 pages identical, CER between the
two readings 7.2 %, and the at-cap sets differ. One page
(`008241_036296_0001_1211427`) loops to the cap under transformers and stops
under vLLM. Most first differences are single-token flips spread along the
page, a few within the first characters. That points to kernel numerics and
image resizing, not to a setup error. A greedy loop is sensitive to exactly
this, so vLLM cannot stand in for the scorer. Passes therefore run in
transformers, at about 0.5 min per page plus 3 minutes of loading.
transformers lacks the fast Gated-DeltaNet kernels here
(`flash-linear-attention`), so it uses its reference implementation. A
64-page pass takes about 35 minutes, and a budget of 10 about 6 hours. The
reference run twice is byte-identical, so a difference between two passes is
the edit's, not noise.

**Reading v2 (budget 0).** The structure is v1's:
- Energy: `gate_proj` 47 %, `up_proj` 28 %, `down_proj` 15 %, `q_proj` 7 %.
- The median module's rank at 90 % energy is 35 of 64.
- The layers with most energy are 27, 23, 0, 31 and 3.

Through the logit lens, the output directions that lower the end-of-turn token
most are:
- direction 0 of `layers.31.self_attn.o_proj` (z −3.60), which also suppresses
  the newline (−5.52) and `¬` (−8.06);
- direction 0 of `layers.31.mlp.down_proj` (−3.32);
- direction 0 of `layers.27.mlp.down_proj` (−2.80).

These are the H1 candidates for the preregistration. The lens ignores
everything after the module, so whether removing one of these directions
shortens a page is a pass's question.

# Adapter report — weight-space programming on a QLoRA adapter of Qwen3.5-4B (asteraix, 07.11.2026)

Branch `claude/adapters-4d972b`, on `origin/claude/article-controls-8803eb` plus the earlier phase-5 harness commit (`d42f662`: `LoraSubject`, page metrics, vLLM and `evaluate_qlora` passes, gate G1, practice draw, budget-0 reading of v2; it was never merged and the brief did not know of it, so it is included here). Card 0 only, never card 1; nothing written to the share, the registry, Hugging Face or the trainer; the trainer's venvs were only *run* (as `evaluate_qlora` and the trainer's code are), never modified, and nothing was installed anywhere. Edited adapters live under `~/atr-cache/wsp/` and are not exported. Peak cache use of this track: about 9 GB of adapter copies and planted variants plus 2 GB of renders (base model weights are the existing HF cache).

**Correction to the brief.** The adapters have rank **64** and `lora_alpha` **128** (scale 2), not rank 16; they cover the language tower only (`mlp.{gate,up,down}_proj` in all 32 layers, `self_attn.{q,k,v,o}_proj` in the 8 full-attention layers; the Gated-DeltaNet layers, `lm_head`, embeddings and the vision tower are not adapted): 128 modules, 340 MB in F32.

## 0. The forward pass, exactly

Two engines were built; they are **not** equivalent, and that is the main result of stage A0 (§A0.3).

- **Scorer engine (used for the real session, A4, and the held-out readings):** `evaluate_qlora` of the trainer (transformers 5.17, `vlm-train-tf5`), base in bf16 on card 0, adapter applied beside the frozen weights by peft, one page at a time, greedy (`do_sample=False`), `max_new_tokens` 3,072, the processor's stop ids `[248046, 248044]`, prompt "Transcribe the handwritten text in this image exactly as written." cut from the training render, `max_pixels` 2,097,152 (page), `seed` 42. Deterministic: the same adapter in two separate processes gives byte-identical text (32 of 32 pages in G1; 10 of 10 here). A pass of 32 practice pages costs 20–60 min depending on contention and how many pages loop (a page at the cap takes 230–260 s, about 13 tokens/s, because `flash-linear-attention` and `causal_conv1d` are not installed and the Gated-DeltaNet layers run their reference implementation), plus about 3 min to load. Memory: 9.7 GB peak allocated, 10.4 GB on the card.
- **Fast engine (used only for the planted-defect control, A3):** vLLM 0.29.0 (`~/.venvs/vllm`), `enable_lora` (`max_lora_rank` 64), same prompt, stop ids, `max_pixels`, greedy, cap 3,072, `max_num_seqs` 16, `gpu_memory_utilization` 0.35 (16 GB; card 0 held other jobs). `wsp/vllm_server.py` loads the base once (≈3.5–4 min) and reads adapters in turn; a batch of 64 pages takes 5.8–6.7 min (32 pages 3.5–5 min), 6–10× faster per page than the scorer. **Not repeatable**: the same adapter and batch gave 19 of 64 identical pages on a repeat (CER 0.350 vs 0.407, pages at the cap 4 vs 7), and with `max_num_seqs` 1 still only 2 of 16 pages identical (CER between the readings 5.3 %). The server checks free memory before loading, the trainer's `waiting` (#396) before every job, and exits when idle.
- A pass = one fixed batch of pages: 32 (A3, A4) or 64 (score set of A3). Pass metrics are `wsp.pages`: CER, `truncated_cer` (reading clipped to the reference's length), length ratio, pages at the cap (`finish_reason == "length"`), page loops.

## A0 — Feasibility and the forward pass

### A0.1 The adapters the trainer produced (share `trained/`, read-only; HF `dh-unibe/*` is private and was not downloaded — same files)

| adapter | base | r / α | modules (targets) | size | trainer's metric (own eval draw) |
|---|---|---:|---|---:|---|
| `qwen3.5-4b-german-kurrent-xvi-xix-page-v2` (cache copy; scratch on UBELIX) | Qwen3.5-4B | 64 / 128 | 128 | 340 MB | val CER 0.237 (159 pages); held-out 733 pp.: **0.418**, `truncated_cer` 0.250, length ratio 1.18, **49 at cap** |
| `…-xvi-xix-page-v1` (HF; trained on the held-out documents) | Qwen3.5-4B | 64 / 128 | 128 | 340 MB | not clean (crossed with test documents) |
| `…-xvi-xix-blb-page-v1` | Qwen3.5-4B | 64 / 128 | 128 | 340 MB | 0.215, `truncated_cer` 0.199 (168 pages); loops on 8 of 157 eval pages; 3 of 276 Transkribus pages |
| `throughput-qwen35-4b-a40` | Qwen3.5-4B | 64 / 128 | 128 | 340 MB | 0.337 (benchmark run) |
| `olmocr2-7b-german-kurrent-xvi-xix-blb-page-px4m-native-v1` | olmOCR-2-7B | 64 / 128 | 7 target kinds | 646 MB | 0.166 / 0.162 |
| `throughput-qwen3vl-8b-page-a40` | Qwen3-VL-8B | 64 / 128 | | 698 MB | 0.74 (throughput test) |
| `vlm-xix-effective-batch-01-*` (7 runs) | Qwen3-VL-4B | 64 / 128 | | 529 MB | 0.136–0.301 (line/block arms) |

Training data of the Qwen3.5-4B Kurrent arms: 20 datasets of the `german-kurrent-xvi-xix` recipe (v2: corpus `4e2abd840de3`, 11,891 train pages, 1 epoch, effective batch 2, lrate 2e-4, `paged_adamw_8bit`), blb-v1 adds the Laßberg BLB handwriting projects. No Apertus adapter exists on the share.

**First target: kept as in PLAN.md, `…-page-v2`.** It is the only Qwen3.5-4B Kurrent adapter with a clean held-out number (the October v1 and the BLB arm trained on documents that are in the test draw or partly overlap), it has the largest recorded symptom (0.418 vs 0.250 truncated; 49 of 733 pages at the cap) and it fits the card. It was copied by the earlier session, sha256 `1b70ce4a…` (`models/DIGESTS.md`).

### A0.2 Loading, and what was broken

`LoraSubject` (from the earlier commit) loads the real adapter and runs; three things had to be fixed or added: (i) saving an edit from the read-only cache copy produced read-only config files (chmod after copy); (ii) the session sandbox (`unshare -Un`, uid mapped to nobody, `HOME` = run folder) made the card/trainer check fail (loopback down, home lookup wrong) — `wsp.card` now reads a flag kept by a watcher outside the sandbox (`python -m wsp.card watch`); (iii) the session runner needed a way to pass through the vLLM server (`wsp.run new --spool`). New: `wsp/hf_pass.py` (the trainer's engine, one base load, several adapter runs, `peft` / `merged` / `merged32` modes), `wsp/vllm_server.py` + `vllm_client.py` + `lora_subject.server_runner`, `wsp/adapter_compare.py`, `wsp/adapter_pack.py`, `wsp/plant_adapter.py`.

### A0.3 Equivalence: merged weights vs the peft-wrapped model (10 practice pages, `data/practice/kurrent-val-64`, spread over reference length 126–4,292 characters; `docs/baselines/a0-equivalence-v2.json`)

| comparison | identical pages | CER between the readings | same pages at the cap |
|---|---:|---:|---|
| peft (my runner) vs C0 by `evaluate_qlora` itself | 10 / 10 | 0 | yes |
| merged (bf16 `merge_and_unload`) vs merged32 (ΔW added in float32, cast once) | 10 / 10 | 0 | yes |
| **peft vs merged** | **3 / 10** | 0.387 (dominated by two loop pages) | **no** |

The merge itself is correct and rounding-robust (two ways of merging agree exactly). But the merged and the peft-wrapped model give different text: seven pages differ by 0.7–2 % CER with the first differing character at position 66–1,027; one page loops to the cap with peft (2,138 characters) and stops merged (195); one loops to the cap under both with different text. So the strict criterion "same text on 10 pages" is **not met**, and the reason is numerical: bf16 arithmetic in two different orders. It matters for the whole track: *whether a page loops is a chaotic event that a numerically tiny change flips.* The same holds, more strongly, between engines: vLLM vs `evaluate_qlora` on the 61 practice pages with a C0 reading: 10 of 61 identical, CER between 0.248, pages at the cap 4 vs 9 (earlier gate G1 on v1: 12 of 32, 0.072, at-cap sets differ). Consequence: a weight edit whose effect is a few pages ceasing to loop cannot be told from numerical noise by one pass, and merging the edited adapter into the base for serving (`merge_loras.py`) changes the very behaviour the edit targeted.

### A0.4 Memory and time (card 0, shared)

| engine | batch | load | generate | memory |
|---|---|---:|---:|---|
| `evaluate_qlora` / `hf_pass` | 1 page at a time | 3 min (adapter swap 1–6 s) | 8 s for a short page; 230–260 s per page at the cap; 10 pages 586–896 s; 134 pages 10,620 s (3 h) | 9.7 GB allocated, 10.4 GB on the card |
| vLLM server | 16 sequences | 3.4–4.1 min | 64 pages 404–420 s; 32 pages 211–326 s | 16 GB (0.35 of the card) |

Free memory on card 0 varied between 11 and 36 GB (a `ketos` job of a sibling session, a long `evaluate_qlora` C0 of the earlier phase-5 session, and the neighbours); each load was preceded by `nvidia-smi` and `wsp.card`, nothing was killed, and no out-of-memory occurred. It fits; a 4-bit load or a smaller target was not needed.

## A1 — Readings, budget 0 (`docs/readings/adapter_compare.json`, `docs/readings/adapter_kurrent-page-v2.md`)

ΔW = (α/r)·B·A was never formed: norms and inner products come from the 64×64 Gram matrices; singular directions from QR of the factors. Four Qwen3.5-4B adapters:

| adapter | ‖ΔW‖ over all modules | median per-module ‖ΔW‖ | median effective rank (of 64) | median rank for 90 % energy |
|---|---:|---:|---:|---:|
| v2 | 117.5 | 10.2 | 34.0 | 35 |
| v1 | 127.4 | 11.0 | 34.8 | 35 |
| blb-v1 | 135.7 | 11.7 | 36.1 | 36 |
| throughput | 21.5 | 1.9 | 26.3 | 35 |

- **Layer profile (v2):** energy by kind `gate_proj` 47 %, `up_proj` 28 %, `down_proj` 15 %, `q_proj` 7 %; the layers with most energy are 27, 23, 0, 31, 3 (table in `adapter_kurrent-page-v2.md`). By module kind the median ‖ΔW‖ is largest for `gate_proj` (14.0) and `q_proj` (11.0), smallest for `k_proj`/`v_proj` (3.2).
- **Healthy siblings are tight:** per module, v2's norm is 0.80–0.93 of the median of v1 and blb-v1 (v1/blb1 0.90–1.00), uniformly smaller (v2 trained 5,946 steps against 7,571). **No module is an outlier by norm.** A scale defect of the planted kind would stand out at once (§A3).
- **Adapters of one base are nearly orthogonal:** cosine of ΔW between v2 and v1 0.109 globally (median 0.057 per module, max 0.335), v2–blb1 0.100, anything–throughput 0.03; no A matrix is shared (0 of 128 modules identical). Different training runs write different low-rank subspaces. **Consequence: there is no donor** — a module of a sibling cannot be transplanted, averaged or interpolated; only statistics transfer. The donor rule of the proposed PROTOCOL amendment is therefore moot for adapters, and PLAN.md's "transplant one module from a sibling adapter" and "averaging of sibling adapters" are not available.
- **Logit lens (no pass):** the output directions of `down_proj`/`o_proj` that most lower the end-of-turn token (`<|im_end|>`): layer 31 `o_proj` direction 0 (z −3.60; also suppresses newline −5.5 and `¬` −8.1), layer 31 `down_proj` direction 0 (−3.32), layer 27 `down_proj` direction 0 (−2.80); the ones that raise it: layer 30 `down_proj` directions 0 (+2.85) and 2 (+2.27). The promoted/suppressed tokens are mostly multilingual noise tokens, not text characters, so the lens is a weak reading.
- **What the trainer records as the symptom:** `cer` vs `truncated_cer` (the gap is text written past the page), `length_ratio`, and the count of answers that ran to `max_new_tokens` (`at_cap`, "the CER is a floor"). On the page models: v2 0.418 / 0.250 / 1.18 / 49 of 733; blb-v1 0.215 / 0.199 on 168 pages and 8 loops of 157 pages (dashes or full stops to the ceiling; one page carries the weighted figure of the undetermined pages from 11.7 % to 82 %).

**Candidate defects, with predicted effects (written before any A2/A4 pass; the programmer of A4 refined them, `runs/20261107-adapter-v2-b10-1/preregistration.md`):**

| # | candidate | edit | prediction |
|---|---|---|---|
| D1 | a direction that lowers the stop token (layers 31, 27) | `project_out` of direction 0 from `B` of layer 31 / 27 `down_proj` | pages at the cap fall by 2–3 of 8 on practice, CER −0.1; risk: not the cause, or the directions also carry reading |
| D2 | the stop direction of layer 30 is too weak | add 0.5·uuᵀB (×1.5) for direction 2 | cap −1 to −3, CER −0.03 to −0.10 |
| D3 | the update overshoots as a whole | whole adapter ×0.9 | little net change; truncated CER up a little (loses reading) |
| D4 | late MLP group writes the loop | layers 24–31 MLP ×0.9 | flat to slightly better |
| D5 | the loop is a base-model behaviour not in ΔW | base alone (C1) | would explain "no edit helps"; **not run** (no budget in A4) |

Prior: no norm outlier, a tight family, no donor: an edit can only be a scaling or a direction removal, and the PLAN's own caveat (edits remove obstacles, do not add ability) says the expected gain is none unless a single obstacle exists. The A1 evidence for such an obstacle is thin: a lens read through the *base* head cannot say what a direction does after the later layers.

## A2 — Baselines of the unedited adapter

Practice sets are the registered **val side** of `german-kurrent-xvi-xix-v1` (documents the trainer withholds from every run; not training data of v2; audit `data/practice/kurrent-val-64.audit.json`: no document in common with the 70 test documents or v2's 1,191-document manifest; 6 of the 64 pages are on v2's own validation split, scored but never trained on). Reading by `evaluate_qlora`, deterministic. "At cap" = `finish_reason` length (the transformers engine reports a cap flag, not other finish reasons).

| set | pages | CER | `truncated_cer` | length ratio | at cap | loops |
|---|---:|---:|---:|---:|---:|---:|
| val side, all (C0 of the earlier session) | 509 | 0.187 | 0.118 | 1.081 | 19 | 23 |
| practice draw, 1 page per document | 64 | 0.452 | 0.256 | 1.226 | 9 | 12 |
| practice pass of A3/A4 (every second page of the draw) | 32 | 0.729 | 0.333 | 1.456 | 8 | 9 |
| held-out subset of the trainer's test draw, 2 pages per document (A4) | 134 | 0.685 | 0.354 | 1.333 | 22 | 34 |
| trainer's own number, full test draw | 733 | 0.4181 | 0.2502 | 1.181 | 49 | — |

The draws taking one or two pages from every document over-represent short, hard documents (two val documents hold two thirds of the side), so they read much worse than the full sides (0.45–0.73 against 0.19 and 0.42) and loop-rich (14–16 % of pages at the cap, 3.7 % on the whole val side, 6.7 % on the whole test draw). That is useful for a loop study and not a representative rate; every comparison below is paired on the same pages. The practice set shares 56 of 60 documents' hands with the A3 score set (different pages). Held-out overlap audit: the test draw is the trainer's frozen registration (`draw_md5` `ae8bf1fd…`), 0 documents shared with v2's training manifest (HELDOUT_BOUNDARY.md); the 134-page subset is drawn from page names and document keys only (seed 20261109); the held-out tree is not visible inside the session. Tokens: the vLLM readings average 650–1,050 generated tokens per page (`docs/baselines/a3-baselines.json`); the transformers runner stores `new_tokens` per page in its raw files.

Where the loop is: of the 19 at-cap pages on the val side, 7 have a reference under 400 characters (title pages, covers, stamps: short pages where the model keeps writing), 6 are pages with 4,000–5,500 characters; the loop text is `\n✳\n✳…`, dashes, or repeated fragments.

## A3 — Positive control: planted defects (`docs/controls/planted-adapter.md`, `runs/*planted-adapter-*`)

Defects (`wsp/plant_adapter.py`, `~/atr-cache/wsp/planted-adapter/plant.json`): p1 `B` of layer-27 `down_proj` ×3; p2 `B` of layer-27 `gate_proj` set to 0; p3 `lora_alpha` doubled (config only); p4 random matrix of the module's own norm added to `B` of layer-27 `down_proj`; p5 `lora_alpha` halved; **p6 (added after p1, p2, p4 came out null)** `B` of the eight late `down_proj` (layers 24–31) ×3. Score set: 64 val pages disjoint from the practice pages (the test side stays closed for A4), through the vLLM server (fast, **not repeatable**: its repeat noise is 5.7 points of CER on 64 pages, 1.5 points of truncated CER). Practice pass: 32 pages. Arms as in docs/CONTROLS_REPORT.md §2: (a) a fresh programmer with 6 passes, (b) a blind programmer with 0 passes (both Claude Sonnet 5.5 subagents shown only the statistics pack of the corrupted adapter against two siblings, programs written in advance and never run), (c) a fixed hand-written list (`runs/planted-adapter/arms/program_c.py`: R1 rescale to the siblings' `lora_alpha`, R2 rescale modules outside the siblings' norm band), budget 4. Arms (a), (b) were run for p3 only, the one defect that is a clear defect and visible; (c) for p1, p2, p3, p6.

Classes fixed beforehand: damage below max(noise, 3 points) = null corruption; repaired = recovers ≥ 80 % of the damage.

| defect | corrupted CER / trunc / at cap | damage | class | (a) | (b) | (c) |
|---|---|---:|---|---|---|---|
| clean adapter | 0.507 / 0.207 / 6 | | | | | |
| p1 B×3, one module | 0.529 / 0.217 / 6 | +2.2 | null | — | — | not repaired (2 passes, 0 kept: nothing to see) |
| p2 B zeroed, one module | 0.531 / 0.225 / 7 | +2.4 | null | — | — | no edit; module with no update noted |
| p3 α doubled | 3.764 / 0.819 / 58 | +325.7 | **defect** | **0.517, 100 %** (4 passes, 1 of 3 edits kept) | **0.517, 100 %** (0 passes) | **0.517, 100 %** (2 passes) |
| p4 random direction, norm = ‖B‖ | 0.478 / 0.203 / 5 | −2.9 | null | — | — | — |
| p5 α halved | 0.413 / 0.254 / 5 | −9.3 | null (better, within noise) | — | — | — |
| p6 late `down_proj` ×3 | 0.748 / 0.248 / 12 | +24.2 | **defect** | — | — | **not repaired**: the one edit (rescale the 8 modules to the median ratio, which is the exact inverse) was refused by the keep rule |

- **What is shown.** The method repairs the one defect that is both damaging and visible in the statistics: all three arms of p3 reach the same adapter (every `B` ×0.5, i.e. the siblings' α) and bring the 64-page score from 3.764 to 0.517 (clean 0.507, inside the noise); the blind programmer, with no pass, gets there. As on the kraken control the passes add nothing over the hand-written list or the blind programmer on this kind of defect.
- **What is not shown.** Four of six planted defects (p1, p2, p4, p5) do not damage the pass beyond its noise, so they say nothing about repair; the adapters are robust to a single module ×3, to a module's update missing, and to a random direction of full norm in one module, which is itself a finding (a single module of 128 is not where the behaviour lives) but also means a single-module edit by the programmer cannot be expected to matter either. p6 is a real defect that the list **identified correctly** (all eight modules at 2.7× the siblings' norm) and whose exact repair the harness's keep rule **rejected**: on the 32-page practice pass p6 reads 0.646 against the clean 0.617 (practice is not sensitive to it), so the paired interval cannot show a gain. The repair would have been correct; the rule, calibrated for deterministic line readings, is too blunt for a chaotic loop-driven metric. The unmeasured repair of p6 was not scored (it is the clean adapter up to rounding, ×3·⅓).
- Programmers (a)/(b) were given a pack in which the defect is a doubled `lora_alpha` printed in the config next to the siblings' 128: the control checks the machinery, not the programmer's insight.

## A4 — One real session: v2, budget 10 (`runs/20261107-adapter-v2-b10-1`)

Programmer: Claude Sonnet 5.5 (`claude-sonnet-5-5`) as a fresh subagent, shown the A1 pack only (`runs/adapter-v2-b10/`); it wrote `preregistration_hypotheses.md` and `program.py` in advance and did not run them; the harness ran the program in the sandbox. Preregistration (`runs/20261107-adapter-v2-b10-1/preregistration.md`) fixed before the first pass: practice = 32 val pages (every second page of the 64-page draw), held-out = 134 pages of the trainer's frozen test draw (2 per document, never visible in the session), budget 10 (8 planned), keep rule PROTOCOL §3 on pages (paired page-bootstrap interval below 0 and pages at the cap not up). No donor (A1: independent subspaces), so the donor rule is moot. Engine: the scorer (transformers, deterministic); pass 1 reproduced the earlier reading exactly (CER 0.729).

| pass | edit (hypothesis) | practice CER | truncated | at cap | loops | 95 % interval of Δ CER vs last kept | decision |
|---|---|---:|---:|---:|---:|---|---|
| 1 | C0 | 0.729 | 0.333 | 8 | 9 | | |
| 2 | E1: remove direction 0 of layers 31 and 27 `down_proj` (lowers `<|im_end|>`) | 0.707 | 0.342 | 5 | 7 | [−0.196, +0.124] | reverted: interval contains 0 |
| 3 | E2: layer-30 `down_proj` direction 2 ×1.5 (raises `<|im_end|>`) | 0.669 | 0.366 | 7 | 8 | [−0.211, +0.083] | reverted |
| 4 | E3: late MLP (layers 24–31) ×0.9 | 0.641 | 0.319 | 5 | 6 | [−0.271, +0.067] | reverted |
| 5 | E4: whole adapter ×0.9 | 0.657 | 0.345 | 6 | 8 | [−0.215, +0.021] | reverted |
| 6 | E5: whole adapter ×1.1 (branch: E4 not kept) | 0.602 | 0.315 | 3 | 7 | [−0.350, +0.071] | reverted |
| 7 | **E6: layer-27 `q_proj` ×0.8** | **0.567** | 0.324 | 5 | 6 | **[−0.346, −0.013]** | **kept** (at cap 8→5, not up) |
| — | E7 (layer-31 `o_proj` direction 0) | skipped by the program: E1 was not kept | | | | | |

Seven of 10 passes used, six edits attempted, one kept (a scaling of one module's `B` by 0.8; ‖ΔB‖ = 20 % of ‖B‖, cosine to the original 1.0). The programmer's predictions: E1 CER −0.1 to −0.2 and at cap 8→5–6 (observed −0.02, 8→5), E6 −0.05 to +0.03 and at cap 8→6–8 (observed **−0.16**, 8→5): the kept edit is the one whose predicted effect was smallest, and the prediction for the stop-token hypotheses (E1, E2) was not borne out. Every one of the six edits, including a global ×1.1 that the programmer predicted to change little, lowered the practice CER by 2–16 points; only E6's interval excludes 0.

**Held-out, once, after freeze** (`docs/baselines/a4-heldout134-result.json`, `a4-sham-heldout134.json`; 134 pages of the trainer's test draw, scorer engine, one reading each, deterministic):

| adapter | CER | `truncated_cer` | length ratio | at cap | loops | Δ CER vs unedited [95 %] |
|---|---:|---:|---:|---:|---:|---|
| unedited v2 | 0.685 | 0.354 | 1.333 | 22 | 34 | |
| **frozen (E6)** | **0.569** | 0.351 | 1.215 | 18 | 30 | **−0.115 [−0.271, −0.001]** page bootstrap; [−0.252, −0.009] document-clustered |
| sham: random matrix of the same norm added to the same tensor (C5) | 0.716 | 0.388 | 1.321 | 24 | 36 | +0.032 [−0.060, +0.101] |
| frozen vs sham | | | | | | −0.147 [−0.300, −0.041] |

Only 40 of 134 pages read identically before and after the edit, 43 before and after the sham: every perturbation reshuffles about two thirds of the pages. **What this is.** A held-out CER gain of 11.5 points whose page interval barely excludes 0 (the document-clustered one too), larger than a same-norm random perturbation of the same tensor (which moved CER by +3), on one draw of 134 pages, from one edit picked among six tries. **What it is not.** The gain is in the *length* of what is written, not in reading: `truncated_cer`, the reading clipped to the reference, is unchanged (0.354 → 0.351), the length ratio falls 1.33 → 1.22 and 4 of 22 capped pages stop. Pages leaving the cap is the chaotic quantity of A0 (the numerical sham on the practice pass moved CER by −8 points with an interval of [−0.218, +0.038], at cap 8→7), so four pages is not a result either. The edit does **not** act through any of the stop-token directions the reading nominated, and the programmer's hypothesis for it (a concentrated query rewrite locking attention onto repetition) was a weak one. With six attempts and a keep rule not calibrated for this noise, one kept edit is about what chance gives; the sham does not exclude that a different random draw would have been kept too (one sham only; a null distribution would need dozens, at three hours each). No comparison with a decode-time repetition guard or a retrain (C4) was made, so nothing here says the edit is worth shipping. The edited adapter is `runs/20261107-adapter-v2-b10-1/final.adapter` (cache only, not exported, not committed).

**Controls run around the session.** C1, the base alone on the 32 practice pages (same engine): CER 1.747, `truncated_cer` 0.607, length ratio 2.17, **12 of 32 pages at the cap** (the adapter: 0.729, 0.333, 1.46, 8): the base loops *more*; the adapter already suppresses most of the looping, so "the loop is written by ΔW" is wrong as stated and a defect in ΔW can at most be a residual. Numerical sham on the practice pass (every `B` ×1.002, a change below bf16 resolution): CER 0.650 (−7.9 points vs C0, interval [−0.218, +0.038], 12 of 32 pages identical, at cap 7). Not run: C2/C3 averaging and α-sweeps beyond E4/E5 (no donor; no per-module sweep), C4 retrain (a trainer job; needs a go-ahead), the decode-time guard, a second programmer.


## What this does and does not tell about language-model weights (for Section 9.3 of the manuscript)

The adapter experiments say something about a rank-64 *update* of 128 modules of one 4 B model on one task, read and edited from outside, and nothing directly about the weights of a language model. Four things were established. (1) The machinery transfers: a session on a real peft adapter runs, is logged, budgeted and scored by the same code as the kraken sessions, and it repairs the one planted defect that is both damaging and visible in the statistics (a doubled `lora_alpha`: CER 3.76 → 0.52 on the score set, clean 0.51, by the blind programmer, the programmer with passes and the hand-written list alike). It does not show that a programmer *finds* defects: four of six planted defects (one module ×3, one module zeroed, a random direction of full norm in one module, α halved) did not move the score beyond the pass noise, which means a single module of 128 carries little, not that it was repaired; the one other damaging defect (eight late `down_proj` ×3, +24 points) was recognised by the rule list and its exact repair was rejected by a keep rule that cannot see it in 32 practice pages. (2) Adapters trained on the same base and recipe do not share a basis: ΔW of two runs has cosine 0.1, no A matrix is shared, so the donor operations that make sense for the 4 M-parameter kraken models of one lineage (transplant a layer, interpolate toward a sibling, copy a row) have no counterpart for adapters; per-module norms, by contrast, are strikingly regular (0.80–0.93 of a sibling's), which makes scale defects trivial to see and says nothing about subtler ones. (3) The behaviour the track set out to repair, a model that does not stop, is a chaotic function of the weights: merging the same adapter into the weights, a 0.2 % rescaling of every `B`, or a different inference engine flips which pages loop, with 10–25 % of the text of a page changing, so a few-pass practice measurement cannot resolve an effect of a few pages, and a keep rule calibrated on deterministic line readings both accepts chance (here possibly the one kept edit) and rejects a correct repair. (4) The base model without the adapter loops more (12 of 32 practice pages at the cap, CER 1.75), so the adapter *removes* loops and a residual defect inside ΔW is at best a fraction of the story; the base was never edited, and the only read of it was the logit lens of a few output directions through the unembedding, which found multilingual noise tokens and nothing interpretable. The one real session kept one edit (layer-27 `q_proj` ×0.8) with a held-out CER gain of 11.5 points (interval barely excluding 0, larger than a random edit of the same norm, `truncated_cer` unchanged, 4 of 22 capped pages stopping); it is a hypothesis about where length is written, not a repaired defect, not shown against a decoding guard or a retrain, and not tested on the full 733-page draw. What would turn any of this into a statement about language-model weights themselves: editing the base (not done), a null distribution of sham edits with the same keep rule (one sham only), the same session on a second adapter and a second base (olmOCR-2, Qwen3-VL), and a deterministic or much larger measurement of loop rates.

## Reproduction, files, what was not done

Code: `wsp/hf_pass.py`, `wsp/vllm_server.py`, `wsp/vllm_client.py`, `wsp/adapter_compare.py`, `wsp/adapter_pack.py`, `wsp/plant_adapter.py`, `wsp/card.py` (watcher flag), `scripts/a0_equivalence.py`, `a3_baselines.py`, `run_planted_adapter.py`, `make_adapter_packs.py`, `aggregate_planted_adapter.py`, `a4_read.py`, `draw_val_holdout.py`, `draw_test_subset.py`. Data: `data/practice/kurrent-val-{32,holdout-64}.*`. Results: `docs/baselines/{a0-equivalence-v2,a2-heldout134-v2-clean,a3-baselines,a4-heldout134-result,a4-sham-heldout134,c0-kurrent-val-v2}.json`, `docs/readings/adapter_compare.json`, `docs/controls/planted-adapter.*`, `runs/*adapter*`. Adapter files (original and edited, planted variants, shams, run `final.adapter`) stay in `~/atr-cache/wsp/` (about 14 GB in all) and in the run folders, git-ignored.

Not done: the full 733-page held-out reading (3 h per 134 pages; the subset is document-stratified and reads much worse than the full draw: 0.685 against 0.418, because it over-weights short hard documents), arms (a)/(b) for defects other than p3, a repeat of the clean adapter on the score set (the noise floor of 5.7 points comes from the 64-page practice repeat), C2–C4, the decode guard, Apertus and the other bases (no Apertus adapter exists; olmOCR-2-7B is 646 MB and was not loaded). Deviations from the brief: rank 64 not 16; first target v2 (as the earlier phase-5 plan, not the BLB v1); transformers for the session and vLLM for A3 only; a 134-page held-out subset; a sham added; p6 added after the nulls; the A3 programmers saw a pack in which the alpha defect is printed.

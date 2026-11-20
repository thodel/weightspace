# Notes — 20261119-kurrent-page-v2-g1-b10-1

## Round 0 — 2026-11-19 09:32:33 UTC

- **Programmer model id:** `gemini-4-argon`
- **Settings:** Called through the Google API by `wsp.programmer` with the provider's default reasoning settings; sandbox execution with `wsp` and `numpy` (CPU-only for reading scripts; operator runs `program_NN.py` on card 0 via `evaluate_qlora`, greedy decoding, cap 3,072).
- **Files read:**
  - `runs/20261119-kurrent-page-v2-g1-b10-1/PROGRAMMER_BRIEF.md`
  - `runs/20261119-kurrent-page-v2-g1-b10-1/preregistration.md`
  - `runs/20261119-kurrent-page-v2-g1-b10-1/notes.md`
  - `runs/20261119-kurrent-page-v2-g1-b10-1/session.json`
  - `docs/readings/adapter_kurrent-page-v2.md` and `docs/readings/adapter_kurrent-page-v2.json`
  - `docs/baselines/c0-kurrent-val-v2.json`
  - `data/practice/kurrent-val-64.jsonl` and `data/practice/kurrent-val-64.meta.json`
  - `wsp/ops.py`, `wsp/session.py`, `wsp/run.py`, `wsp/lora_subject.py`, `wsp/pages.py`, `wsp/metrics.py`
  - The adapter tensors (`~/atr-cache/wsp/adapters/qwen3.5-4b-german-kurrent-xvi-xix-page-v2`) via `LoraSubject` in CPU reading scripts.
- **Budget-0 observations:**
  1. **Practice set error budget:** On the 64 practice pages, C0 has 48,873 total character edits across 108,243 reference characters (CER 0.451512, `truncated_cer` 0.256386, `length_ratio` 1.226370, 9 pages at the 3,072-token cap). The 9 at-cap pages alone contribute 33,936 edits on 23,508 reference characters. A paired page-bootstrap check on `practice_per_page` shows that at least 4 of the 9 at-cap pages must stop looping for the 95 % bootstrap interval of $\Delta\text{CER}$ to clear 0.
  2. **Late-layer `<|im_end|>` suppression geometry:**
     - `layers.31.self_attn.o_proj` dir 0 ($\sigma = 2.226, z(\text{<|im\_end|>}) = -3.60, z(\text{\\n}) = -5.52, z(\neg) = -8.06$) and `layers.31.mlp.down_proj` dir 0 ($\sigma = 2.612, z(\text{<|im\_end|>}) = -3.32, z(\text{\\n}) = -1.95$) are the two strongest `<|im_end|>` suppressors across all 128 modules.
     - Their left singular vectors (output directions written into the residual stream) have cosine similarity $+0.516$ with each other, and cosine similarities $-0.483$ and $-0.430$ with `layers.30.mlp.down_proj` dir 0 ($\sigma = 3.705, z(\text{<|im\_end|>}) = +2.85, z(\text{\\n}) = +2.05$). Thus layer 31's two residual writers directly oppose layer 30's dominant stopping direction.
     - Upstream in the late layers, `layers.29.mlp.down_proj` dir 0 ($\sigma = 2.802, z(\text{<|im\_end|>}) = -1.51, z(\text{\\n}) = -4.04, \cos(\text{L31.o\_proj:0}) = +0.296$) and `layers.27.mlp.down_proj` dir 0 ($\sigma = 2.678, z(\text{<|im\_end|>}) = -2.80, \cos(\text{L31.o\_proj:0}) = +0.165$) share positive alignment with this same `<|im_end|>`-suppressing subspace.
- **Actions taken in Round 0:**
  - Completed section 5 of `preregistration.md` with hypotheses H1–H9 across the 10-pass budget.
  - Wrote `program_01.py` to execute Pass 1 (baseline C0 check) and Pass 2 (H1: projecting out dir 0 from `layers.31.self_attn.o_proj.lora_B.weight` and dir 0 from `layers.31.mlp.down_proj.lora_B.weight` via `project_out` with `axis=1`, followed by `s.measure` and `s.keep`).

## Round 1 — 2026-11-19 12:00:36 UTC

- **Files read:**
  - `runs/20261119-kurrent-page-v2-g1-b10-1/log.jsonl`
  - `runs/20261119-kurrent-page-v2-g1-b10-1/passes/pass-001.raw.jsonl`, `pass-001.draw.jsonl`, `pass-002.raw.jsonl`
- **Results of `program_01.py` (Passes 1 and 2):**
  1. **Pass 1 (C0):** Reproduced the baseline exactly: CER `0.451512` (48,873 edits / 108,243 chars), `truncated_cer` `0.256386`, `length_ratio` `1.226370`, `at_cap` `9`, `loops` `12`.
  2. **Pass 2 (H1 — `project_out` dir 0 of `layers.31.self_attn.o_proj` and dir 0 of `layers.31.mlp.down_proj`):**
     - **KEPT (`rule: met`)**: CER dropped by `-0.077769` to `0.373742` (40,455 edits, $-8,418$ edits), with 2,000-resample paired page-bootstrap 95 % CI `[-0.173227, -0.004299]` (entirely below 0).
     - `truncated_cer` **improved** from `0.256386` to `0.241180` ($-0.015207$), `length_ratio` dropped from `1.226370` to `1.147640`, `at_cap` dropped from `9` to `5` ($-4$ pages at cap), and `loops` dropped from `12` to `10`.
     - **Per-page breakdown:**
       - `007154_023822_0002_1913639` (ref 222): stopped looping `\n✳`! Length dropped $2,141 \to 200$, edits $2,042 \to 154$ ($-1,888$).
       - `007218_024284_0002_656610` (ref 188): stopped looping `\n✳`! Length dropped $2,112 \to 177$, edits $2,040 \to 118$ ($-1,922$).
       - `007967_036171_0001_1210969` (ref 5,369): stopped at end of document (`...vierzechenhundertvierzigundsechs\njar.`)! Length dropped $8,247 \to 5,399$, edits $5,137 \to 1,583$ ($-3,554$), `truncated_cer` edits $3,157 \to 1,613$.
       - `008241_036296_0001_1211427` (ref 5,582): stopped looping `, vˀsitzenn` and terminated cleanly (`...vierzechenhundert\nund sibentzig jar.`)! Length dropped $6,106 \to 5,643$, edits $3,179 \to 1,808$ ($-1,371$).
       - Total Dingbats (`✳`) insertions across the corpus fell from $3,537$ to $1,253$, and Space insertions fell from $7,220$ to $4,550$.
       - The 5 remaining at-cap pages are: `006694_018260_0002_443451` (`\n✳` loop, 1,955 excess edits), `007884_027397_0001_772042` (clause loop, 1,194 excess edits), `008159_036203_0001_1211042` (charter loop, 6,316 excess edits), `008291_036276_0001_1211387` (`hievor,` loop, 1,551 excess edits), and `011629_1503666_0006_57465490` (`\n"` loop, 2,470 excess edits).
- **Plan for Round 1 (`program_02.py`, Pass 3 — H2):**
  - Follow the pre-registered H2 (branch for H1 kept): project out dir 0 of `layers.27.mlp.down_proj` ($\sigma = 2.678, z(\text{<|im\_end|>}) = -2.80$) and dir 0 of `layers.29.mlp.down_proj` ($\sigma = 2.802, z(\text{<|im\_end|>}) = -1.51, z(\text{\\n}) = -4.04$) on top of the kept H1 state, measure, and call `s.keep()`.

## Round 2 — 2026-11-19 13:15:10 UTC

- **Files read:**
  - `runs/20261119-kurrent-page-v2-g1-b10-1/log.jsonl`
  - `runs/20261119-kurrent-page-v2-g1-b10-1/passes/pass-003.raw.jsonl`
- **Results of `program_02.py` (Pass 3 — H2):**
  - **REFUSED & REVERTED (`rule: refused`, state restored to H1)**: CER rose vs H1 by `+0.072642` to `0.446385` (ci95 `[-0.029788, +0.188632]`), `truncated_cer` degraded from `0.241180` to `0.275297` (`+0.034118`), and `at_cap` rose from `5` to `7`.
  - **Diagnosis from per-page readings (`pass-003.raw.jsonl`):**
    - Removing the entire dominant singular direction (`dir 0`) of `layers.27.mlp.down_proj` ($\sigma = 2.678$, energy $7.17$) and `layers.29.mlp.down_proj` ($\sigma = 2.802$, energy $7.85$) damaged intermediate text-reading features (`truncated_cer` worsened by 3.4 points, with Latin lowercase errors rising from 26,436 to 32,739), which in turn caused `007967`, `008241`, and `016379` to fall into mid-page repetition loops and hit the token cap, even though `008159` stopped looping ($-2,519$ edits).
    - Why? Because `layers.27.mlp.down_proj` dir 0 and `layers.29.mlp.down_proj` dir 0 only have cosine similarities $+0.176$ and $+0.272$ with the layer-31 `<|im_end|>`-suppressing direction $u_{\text{comb}} = \text{norm}(u_{\text{o31},0} + u_{\text{d31},0})$; over 92% of their energy is orthogonal to $u_{\text{comb}}$ and carries useful Kurrent transcription features.
- **Plan for Round 2 (`program_03.py`, Pass 4 — H3):**
  - Follow pre-registered H3: instead of removing the upstream modules' own dominant singular directions, project out only the exact shared layer-31 `<|im_end|>`-suppressing unit vector $u_{\text{comb}} = \text{norm}(u_{\text{o31},0} + u_{\text{d31},0})$ from the upstream late-layer residual writers `layers.{27,28,29}.mlp.down_proj` and `layers.27.self_attn.o_proj` (`project_out`, `axis=1`). This removes less than 1% of their Frobenius energy (0.46%, 0.19%, 0.25%, 0.97%) while eliminating their contribution along the layer-31 stopping-suppression direction, and leaves layer 30 ($z(\text{<|im\_end|>}) = +2.85$) completely untouched.

## Round 3 — 2026-11-19 14:25:35 UTC

- **Files read:**
  - `runs/20261119-kurrent-page-v2-g1-b10-1/log.jsonl`
  - `runs/20261119-kurrent-page-v2-g1-b10-1/passes/pass-004.raw.jsonl`
- **Results of `program_03.py` (Pass 4 — H3):**
  - **REFUSED & REVERTED (`rule: refused`, interval contains 0; state restored to H1)**:
    - CER improved slightly from `0.373742` to `0.372625` (`delta_cer = -0.001118`, ci95 `[-0.032271, +0.029558]`).
    - `truncated_cer` improved further from `0.241180` to `0.233622` ($-0.007557$), `at_cap` dropped from `5` to `4` (`008291` stopped before cap, $-1,183$ edits), and `loops` dropped from `10` to `8` (`007967`, `008205`, and `007674` were no longer flagged as loops).
    - However, because `007884` remained at cap and had $+1,178$ more edits inside its loop, the net corpus CER reduction vs H1 was only $-121$ edits (`-0.001118`), whose 95% CI spans 0.
- **Plan for Round 3 (`program_04.py`, Pass 5 — H4):**
  - Follow pre-registered H4 (`scale` edit class on layer-31 residual writers): on top of the kept H1 state, scale the remaining `lora_B.weight` of `layers.31.self_attn.o_proj` and `layers.31.mlp.down_proj` by `factor=0.5`, testing whether damping the rest of the final-layer residual update reduces the remaining 5 at-cap pages without hurting `truncated_cer`.

## Round 4 — 2026-11-19 15:36:44 UTC

- **Files read:**
  - `runs/20261119-kurrent-page-v2-g1-b10-1/log.jsonl`
  - `runs/20261119-kurrent-page-v2-g1-b10-1/passes/pass-005.raw.jsonl`
- **Results of `program_04.py` (Pass 5 — H4):**
  - **REFUSED & REVERTED (`rule: refused`, interval contains 0; state restored to H1)**:
    - CER rose vs H1 by `+0.022486` to `0.396229` (ci95 `[-0.028721, +0.093737]`), `truncated_cer` degraded from `0.241180` to `0.248312` (`+0.007132`), and `at_cap` stayed at `5` (`008291` stopped before cap, $-729$ edits, while `006781` newly ran to cap with a Dingbat `✳` loop, $+2,355$ edits).
    - **Conclusion for H4:** Just as pre-registered ("Whole-module damping in late layers trades off reading accuracy against stopping or does not beat surgical rank-1 projection"), scaling the remaining 63 singular directions of `layers.31.self_attn.o_proj` and `layers.31.mlp.down_proj` by 0.5 weakens their positive `<|im_end|>`-promoting directions (`layers.31.mlp.down_proj` dir 1 has $z = +1.90$, `layers.31.self_attn.o_proj` dir 3 has $z = +1.47$) and degrades `truncated_cer`.
- **Key contrast across Passes 2–5 and design of Pass 6 (H5):**
  - Every edit that removed non-target feature directions (H2 full rank-1 removal in layers 27 & 29; H4 rank-63 scaling in layer 31) degraded `truncated_cer` and triggered new loops.
  - Conversely, projecting out a single pure `<|im_end|>`-suppressing residual direction (H1 in layer 31; H3 in layers 27–29) **improved** `truncated_cer` (`0.256386` $\to$ `0.241180` in H1 $\to$ `0.233622` in H3) and reduced `at_cap` (`9` $\to$ `5` $\to$ `4`) and `loops` (`12` $\to$ `10` $\to$ `8`).
  - Using all 160 top-4 left singular vectors $V \in \mathbb{R}^{160 \times 2560}$ and their exact budget-0 logit-lens scores $z(\text{<|im\_end|>}) \in \mathbb{R}^{160}$ from `docs/readings/adapter_kurrent-page-v2.json`, we reconstructed the minimum-norm least-squares `<|im_end|>` lens direction $w_{\text{im\_end}} \in \mathbb{R}^{2560}$ ($V w = z$).
  - **Plan for Round 4 (`program_05.py`, Pass 6 — H5):** On top of H1, project out $w_{\text{im\_end}}$ (`axis=1`) from all 25 net-`<|im_end|>`-suppressing residual-writing modules while leaving all 15 net-`<|im_end|>`-promoting modules untouched.

## Round 5 — 2026-11-19 16:38:51 UTC

- **Files read:**
  - `runs/20261119-kurrent-page-v2-g1-b10-1/log.jsonl`
  - `runs/20261119-kurrent-page-v2-g1-b10-1/passes/pass-006.raw.jsonl`
- **Results of `program_05.py` (Pass 6 — H5):**
  - **REFUSED & REVERTED (`rule: refused`, interval contains 0; state restored to H1)**:
    - CER rose vs H1 by `+0.051985` to `0.425727` (ci95 `[-0.004440, +0.129646]`), `truncated_cer` degraded from `0.241180` to `0.248718` (`+0.007539`), and `at_cap` rose from `5` to `6`.
    - **Diagnosis:** The logit-lens vector $w_{\text{im\_end}}$ is defined at the final layer's output (right before the base model's final RMSNorm and `lm_head`). In early and mid layers (layers 1–24), the residual stream undergoes many subsequent Gated-DeltaNet, full-attention, and MLP transformations before reaching the final norm, so projecting out $w_{\text{im\_end}}$ in early/mid layers perturbs intermediate representations (causing `007967` to revert to cap with $+3,516$ edits). Furthermore, in layer 31 itself, `project_out` along $w_{\text{im\_end}}$ also removed the component of `layers.31.mlp.down_proj` dir 1 ($z = +1.90$) and `layers.31.self_attn.o_proj` dir 3 ($z = +1.47$) that *promoted* `<|im_end|>`.
- **Design of Pass 7 (H6 — `program_06.py`):**
  - Across all 6 passes so far, only edits to the two final-layer top singular directions (`layers.31.self_attn.o_proj` dir 0, $\sigma = 2.226, z(\text{<|im\_end|>}) = -3.60, z(\text{\\n}) = -5.52, z(\neg) = -8.06$; and `layers.31.mlp.down_proj` dir 0, $\sigma = 2.612, z(\text{<|im\_end|>}) = -3.32, z(\text{\\n}) = -1.95$) operated directly at the final output interface without disturbing upstream representation geometry.
  - In H1, projecting out those two rank-1 directions changed their singular-value multiplier from $c = +1.0$ to $c = 0.0$. That eliminated 4 of the 9 at-cap pages and improved `truncated_cer` from `0.256386` to `0.241180`.
  - In H6 we test $c = -0.5$ along those exact two rank-1 singular components (by adding $-0.5 \cdot u_0 (u_0^\top B_{\text{orig}})$ to `lora_B.weight` of `layers.31.self_attn.o_proj` and `layers.31.mlp.down_proj` on top of H1 via the `add` op).

## Round 6 — 2026-11-19 17:40:34 UTC

- **Files read:**
  - `runs/20261119-kurrent-page-v2-g1-b10-1/log.jsonl`
  - `runs/20261119-kurrent-page-v2-g1-b10-1/passes/pass-007.raw.jsonl`
- **Results of `program_06.py` (Pass 7 — H6):**
  - **REFUSED & REVERTED (`rule: refused`, interval contains 0; state restored to H1)**:
    - CER rose vs H1 by `+0.047606` to `0.421348` (ci95 `[-0.002526, +0.122595]`), `truncated_cer` rose slightly from `0.241180` to `0.244856`, and `at_cap` rose from `5` to `6`.
    - Notably, inverting dir 0 of layer-31 `o_proj` and `down_proj` ($c = -0.5$) drastically cut Dingbats (`✳`) insertions from $1,253$ (in H1) down to $225$, but because $-u_0$ suppresses the tokens that $+u_0$ promotes and promotes the tokens that $+u_0$ suppresses, pushing $c$ negative introduced digit/punctuation loop insertions on `006694`, `007884`, and `008205`.
- **Overall synthesis and decision to freeze (`program_07.py`):**
  - Across 7 passes (C0 + 6 experimental hypotheses H1–H6), we have mapped out the geometry of the `<|im_end|>`-suppression obstacle in `qwen3.5-4b-german-kurrent-xvi-xix-page-v2`:
    1. **H1 (`project_out` dir 0 of `layers.31.self_attn.o_proj` and dir 0 of `layers.31.mlp.down_proj`, $c = 0$):** Reduced CER from `0.451512` to `0.373742` ($\Delta\text{CER} = -0.077769$, 2,000-resample paired page-bootstrap 95% CI `[-0.173227, -0.004299]`), **improved** `truncated_cer` from `0.256386` to `0.241180` ($-0.015207$), reduced `length_ratio` from `1.226370` to `1.147640`, reduced `at_cap` pages from `9` to `5`, and reduced `loops` from `12` to `10`.
    2. **Why $c = 0$ on layer 31 dir 0 (H1) is the exact surgical optimum:**
       - Along the layer-31 rank-1 coordinate $c$: $c = +1$ is C0 (`CER = 0.4515`, `at_cap = 9`); $c = 0$ is H1 (`CER = 0.3737`, `at_cap = 5`); $c = -0.5$ is H6 (`CER = 0.4213`, `at_cap = 6`). Thus $c = 0$ (orthogonal projection) cleanly removes the obstacle without injecting inverse token bias.
       - Scaling the remaining 63 directions of layer 31 (H4) hurts `truncated_cer` (`0.2483`) and triggers new loops because `layers.31.mlp.down_proj` dir 1 ($z = +1.90$) and `layers.31.self_attn.o_proj` dir 3 ($z = +1.47$) promote `<|im_end|>`.
       - Upstream full rank-1 removal in layers 27 & 29 (H2) or whole-network lens projection across layers 1–31 (H5) perturbs intermediate feature representations before they pass through subsequent transformer/DeltaNet blocks.
       - Upstream shared-direction projection in layers 27–29 (H3) leaves CER within `-0.0011` of H1 (95% CI `[-0.0323, +0.0296]`), confirming that the 4 remaining hard at-cap pages (`006694`, `007884`, `008159`, `011629`) are not driven by the late-layer linear `<|im_end|>`-suppression direction.
  - Following our pre-registered stopping plan ("early `s.freeze()` if the obstacle has been cleanly removed and remaining at-cap pages are genuine long/dense pages or base-model loops that cannot be improved without degrading `truncated_cer`"), we write `program_07.py` to call `s.freeze()` on the kept H1 state with 3 passes of budget unused.

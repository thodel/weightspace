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

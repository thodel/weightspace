"""Round 3 program (`program_04.py`):
Pass 5 (H4): Test pre-registered H4 (`scale` edit class on layer-31 residual writers).
On top of the kept H1 state (where `layers.31.self_attn.o_proj` dir 0 and `layers.31.mlp.down_proj`
dir 0 are already projected out), scale the remaining `lora_B.weight` of `layers.31.self_attn.o_proj`
and `layers.31.mlp.down_proj` by factor=0.5.
"""
from wsp.run import open_session

s = open_session(".")

b_o31 = "base_model.model.model.language_model.layers.31.self_attn.o_proj.lora_B.weight"
b_d31 = "base_model.model.model.language_model.layers.31.mlp.down_proj.lora_B.weight"

h4 = s.hypothesis(
    "H4: On top of H1 (layer-31 o_proj:0 and down_proj:0 removed), scale the remaining "
    "lora_B.weight of layers.31.self_attn.o_proj and layers.31.mlp.down_proj by factor=0.5. "
    "This tests pre-registered H4: whether damping the remaining rank-63 residual update of the "
    "final layer further reduces repetition loops without degrading reading accuracy.",
    "CER drops by 0.04 to 0.10 against H1 (to ~0.27-0.33), truncated_cer stays within +-0.005 "
    "of H1 (~0.241), and pages at the cap drop by 2 to 4 pages (from 5 to 1-3).",
    "scale",
)

s.edit(
    "scale",
    [b_o31, b_d31],
    hypothesis=h4,
    factor=0.5,
)

m4 = s.measure("H4_scale05_L31_o_down")
p4 = s._log[-1]["page"]
vs4 = s._log[-1]["vs_kept"]
kept4 = s.keep("H4: scale layer-31 o_proj and down_proj by 0.5 on top of H1")

print(
    f"Pass 5 (H4): CER={m4.cer:.6f} (delta_vs_kept={vs4['delta_cer']:+.6f}, ci95={vs4['ci95']}), "
    f"truncated_cer={p4['truncated_cer']:.6f}, length_ratio={p4['length_ratio']:.6f}, "
    f"at_cap={p4['at_cap']}, loops={p4['loops']}, kept={kept4}",
    flush=True,
)

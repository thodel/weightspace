"""Round 1 program (`program_02.py`):
Pass 3 (H2): Extend the kept H1 state (which projected out dir 0 of layers.31.self_attn.o_proj
and dir 0 of layers.31.mlp.down_proj) by projecting out the upstream late-layer `<|im_end|>`
suppressing singular directions:
- `layers.27.mlp.down_proj` dir 0 (sigma=2.678, z(<|im_end|>)=-2.80, z(\\n)=-0.87, z(¬)=-1.08)
- `layers.29.mlp.down_proj` dir 0 (sigma=2.802, z(<|im_end|>)=-1.51, z(\\n)=-4.04, z(¬)=-3.40)
Both directions align positively with the layer-31 `<|im_end|>`-suppressing subspace
(cos(L31.o_proj:0) = +0.165 and +0.296) and anti-align with `layers.30.mlp.down_proj` dir 0
(cos = -0.158 and -0.343). Measure and apply `s.keep()`.
"""
from wsp.run import open_session

s = open_session(".")
sub = s.subject

mod_d27 = "base_model.model.model.language_model.layers.27.mlp.down_proj"
mod_d29 = "base_model.model.model.language_model.layers.29.mlp.down_proj"

u_d27, s_d27, _ = sub.directions(mod_d27)
u_d29, s_d29, _ = sub.directions(mod_d29)

h2 = s.hypothesis(
    "H2: On top of H1 (layer-31 o_proj:0 and down_proj:0 removed), project out dir 0 of "
    "layers.27.mlp.down_proj (sigma=2.678, z(<|im_end|>)=-2.80) and dir 0 of "
    "layers.29.mlp.down_proj (sigma=2.802, z(<|im_end|>)=-1.51, z(\\n)=-4.04, z(¬)=-3.40). "
    "Both upstream late-layer directions are aligned with the layer-31 <|im_end|>-suppressing "
    "subspace and anti-aligned with layer 30's stopping direction.",
    "CER drops by a further 0.03 to 0.08 against H1 (to ~0.29-0.34), truncated_cer stays within "
    "+-0.003 of H1 (~0.241), and pages at the cap drop by an additional 1 to 3 pages (from 5 to 2-4).",
    "project_out",
)

b_d27 = f"{mod_d27}.lora_B.weight"
b_d29 = f"{mod_d29}.lora_B.weight"

s.edit(
    "project_out",
    [b_d27, b_d29],
    hypothesis=h2,
    axis=1,
    per_tensor={
        b_d27: {"direction": u_d27[:, 0]},
        b_d29: {"direction": u_d29[:, 0]},
    },
)

m2 = s.measure("H2_L27_L29_down_dir0")
p2 = s._log[-1]["page"]
vs2 = s._log[-1]["vs_kept"]
kept2 = s.keep("H2: layer-27 down_proj:0 + layer-29 down_proj:0 projection")

print(
    f"Pass 3 (H2): CER={m2.cer:.6f} (delta_vs_kept={vs2['delta_cer']:+.6f}, ci95={vs2['ci95']}), "
    f"truncated_cer={p2['truncated_cer']:.6f}, length_ratio={p2['length_ratio']:.6f}, "
    f"at_cap={p2['at_cap']}, loops={p2['loops']}, kept={kept2}",
    flush=True,
)

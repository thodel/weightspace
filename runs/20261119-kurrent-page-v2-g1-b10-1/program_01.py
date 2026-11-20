"""Round 0 program (`program_01.py`):
1. Pass 1: Baseline `s.measure("C0")` — verify exact reproduction of C0 (CER 0.4515,
   truncated_cer 0.2564, 9 pages at cap). Stop if it does not match.
2. Pass 2 (H1): Project out the top singular direction (dir 0) of both layer-31 residual-writing
   modules (`layers.31.self_attn.o_proj` and `layers.31.mlp.down_proj`), which are the two
   strongest `<|im_end|>` suppressors in the logit-lens reading (z = -3.60 and -3.32) and are
   mutually aligned (cos = +0.516) while anti-aligned with `layers.30.mlp.down_proj` dir 0
   (z = +2.85, cos = -0.483 and -0.430). Measure and apply `s.keep()`.
"""
from wsp.run import open_session
import numpy as np

s = open_session(".")
sub = s.subject

# --- Pass 1: Baseline C0 ---
m0 = s.measure("C0")
p0 = s._log[-1]["page"]
print(
    f"Pass 1 (C0): CER={m0.cer:.6f}, truncated_cer={p0['truncated_cer']:.6f}, "
    f"length_ratio={p0['length_ratio']:.6f}, at_cap={p0['at_cap']}, loops={p0['loops']}",
    flush=True,
)

if (
    abs(m0.cer - 0.45151187605665033) > 1e-4
    or abs(p0["truncated_cer"] - 0.2563860942508984) > 1e-4
    or p0["at_cap"] != 9
):
    s.note(
        f"STOP: Baseline C0 did not reproduce expected values! "
        f"Got CER={m0.cer:.6f}, truncated_cer={p0['truncated_cer']:.6f}, at_cap={p0['at_cap']}"
    )
    raise RuntimeError("C0 baseline mismatch; stopping as required by brief.")

# --- Pass 2: Hypothesis H1 ---
mod_o31 = "base_model.model.model.language_model.layers.31.self_attn.o_proj"
mod_d31 = "base_model.model.model.language_model.layers.31.mlp.down_proj"

u_o31, s_o31, _ = sub.directions(mod_o31)
u_d31, s_d31, _ = sub.directions(mod_d31)

h1 = s.hypothesis(
    "H1: Project out dir 0 of layers.31.self_attn.o_proj (sigma=2.226, z(<|im_end|>)=-3.60, "
    "z(\\n)=-5.52, z(¬)=-8.06) and dir 0 of layers.31.mlp.down_proj (sigma=2.612, "
    "z(<|im_end|>)=-3.32, z(\\n)=-1.95). These two final-layer residual directions are aligned "
    "(cos=+0.516) and anti-aligned with layers.30.mlp.down_proj dir 0 (z(<|im_end|>)=+2.85), "
    "suppressing page termination.",
    "CER drops by 0.08 to 0.16 (to ~0.29-0.37), truncated_cer stays unchanged within +-0.003 "
    "(~0.256), and pages at the cap drop by 4 to 7 (from 9 to 2-5).",
    "project_out",
)

b_o31 = f"{mod_o31}.lora_B.weight"
b_d31 = f"{mod_d31}.lora_B.weight"

s.edit(
    "project_out",
    [b_o31, b_d31],
    hypothesis=h1,
    axis=1,
    per_tensor={
        b_o31: {"direction": u_o31[:, 0]},
        b_d31: {"direction": u_d31[:, 0]},
    },
)

m1 = s.measure("H1_L31_o_down_dir0")
p1 = s._log[-1]["page"]
vs1 = s._log[-1]["vs_kept"]
kept1 = s.keep("H1: layer-31 o_proj:0 + down_proj:0 projection")

print(
    f"Pass 2 (H1): CER={m1.cer:.6f} (delta={vs1['delta_cer']:+.6f}, ci95={vs1['ci95']}), "
    f"truncated_cer={p1['truncated_cer']:.6f}, length_ratio={p1['length_ratio']:.6f}, "
    f"at_cap={p1['at_cap']}, loops={p1['loops']}, kept={kept1}",
    flush=True,
)

"""Round 2 program (`program_03.py`):
Pass 4 (H3): From the kept H1 state (where `layers.31.self_attn.o_proj` dir 0 and
`layers.31.mlp.down_proj` dir 0 are already projected out), project out the shared layer-31
`<|im_end|>`-suppressing unit vector `u_comb = (u_o31_0 + u_d31_0) / ||u_o31_0 + u_d31_0||`
from the upstream late-layer residual writers `layers.{27,28,29}.{mlp.down_proj, self_attn.o_proj}`
(`axis=1`).
Unlike H2 (which removed the full rank-1 dominant feature directions of layers 27 and 29,
losing 7.2 and 7.8 units of energy and hurting reading accuracy), projecting out only `u_comb`
surgically removes the exact layer-31 `<|im_end|>`-suppressing residual component (only 0.19%
to 0.97% of module energy) while leaving >99% of their visual/lexical feature space intact, and
leaves layer 30 (`z(<|im_end|>)=+2.85`) untouched.
"""
from wsp.run import open_session
from wsp.lora_subject import LoraSubject
import json
from pathlib import Path
import numpy as np

s = open_session(".")
sub = s.subject

# Load the original unedited layer-31 top singular directions to define u_comb reproducibly
meta = json.loads(Path("session.json").read_text())
orig_sub = LoraSubject(meta["subject"])
mod_o31 = "base_model.model.model.language_model.layers.31.self_attn.o_proj"
mod_d31 = "base_model.model.model.language_model.layers.31.mlp.down_proj"

u_o31_0 = orig_sub.directions(mod_o31)[0][:, 0]
u_d31_0 = orig_sub.directions(mod_d31)[0][:, 0]
u_comb = u_o31_0 + u_d31_0
u_comb = (u_comb / np.linalg.norm(u_comb)).astype(np.float32)

target_modules = [
    "base_model.model.model.language_model.layers.27.mlp.down_proj",
    "base_model.model.model.language_model.layers.27.self_attn.o_proj",
    "base_model.model.model.language_model.layers.28.mlp.down_proj",
    "base_model.model.model.language_model.layers.29.mlp.down_proj",
]
target_tensors = [f"{m}.lora_B.weight" for m in target_modules]

h3 = s.hypothesis(
    "H3: On top of H1 (layer-31 o_proj:0 and down_proj:0 removed), project out the shared "
    "layer-31 <|im_end|>-suppressing unit direction u_comb = norm(u_o31:0 + u_d31:0) from the "
    "upstream late residual writers layers.{27,28,29}.mlp.down_proj and "
    "layers.27.self_attn.o_proj (axis=1). Unlike H2 (which removed layers 27 and 29's own "
    "full dir 0 and hurt truncated_cer), this removes only their component along the layer-31 "
    "stopping-suppression direction (<1% of module energy).",
    "CER drops by 0.03 to 0.09 against H1 (to ~0.28-0.34), truncated_cer stays within +-0.003 "
    "of H1 (~0.241), and pages at the cap drop by 1 to 3 pages (from 5 to 2-4).",
    "project_out",
)

s.edit(
    "project_out",
    target_tensors,
    hypothesis=h3,
    direction=u_comb,
    axis=1,
)

m3 = s.measure("H3_ucomb_L27_29")
p3 = s._log[-1]["page"]
vs3 = s._log[-1]["vs_kept"]
kept3 = s.keep("H3: project out shared L31 suppression direction u_comb from L27..29 residual writers")

print(
    f"Pass 4 (H3): CER={m3.cer:.6f} (delta_vs_kept={vs3['delta_cer']:+.6f}, ci95={vs3['ci95']}), "
    f"truncated_cer={p3['truncated_cer']:.6f}, length_ratio={p3['length_ratio']:.6f}, "
    f"at_cap={p3['at_cap']}, loops={p3['loops']}, kept={kept3}",
    flush=True,
)

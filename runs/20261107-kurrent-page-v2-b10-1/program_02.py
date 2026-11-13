"""Passes 3-4 on top of the kept H1: H1b, then H2 (preregistration §5)."""
from wsp.run import open_session

s = open_session(".")
P = "base_model.model.model.language_model.layers"
D31, D27 = f"{P}.31.mlp.down_proj", f"{P}.27.mlp.down_proj"


def decide(label):
    ok, why = s.keep_allowed()
    vk = s._vs_kept
    print(f"{label}: delta vs kept {vk['delta_cer']:+.4f} {[round(x, 4) for x in vk['ci95']]}; keep allowed: {ok} {why}", flush=True)
    return s.keep(f"{label} met the rule") if ok else (s.revert(f"{label} refused by the rule: {why}") or False)


# Pass 3: H1b, the next two directions that lower <|im_end|>, on top of H1.
u31 = s.subject.directions(D31)[0][:, 0]
u27 = s.subject.directions(D27)[0][:, 0]
h = s.hypothesis("H1b: direction 0 of layers.31.mlp.down_proj (lens z -3.32) and of layers.27.mlp.down_proj (-2.80) "
                 "also write against the end of turn; on top of H1",
                 "smaller than H1: cap 4 -> 3-4, CER -1 to -3 points against the H1 state; truncated within ±1", "direction subtraction")
s.edit("project_out", [f"{D31}.lora_B.weight", f"{D27}.lora_B.weight"], hypothesis=h,
       per_tensor={f"{D31}.lora_B.weight": {"direction": u31}, f"{D27}.lora_B.weight": {"direction": u27}}, axis=1)
s.measure("H1b")
decide("H1b")

# Pass 4: H2, every adapted module of layers 27-31 at 0.75, on top of the kept state.
names = [f"{m}.lora_B.weight" for m in s.subject.modules() if any(f"layers.{i}." in m for i in range(27, 32))]
h = s.hypothesis(f"H2: the stopping behaviour sits in the last five layers; scale their {len(names)} modules to 0.75",
                 "cap -1 to -3 against the kept state, truncated_cer +0.5 to +2 points, CER -1 to +1; the likeliest refusal",
                 "scaling")
s.edit("scale", names, hypothesis=h, factor=0.75)
s.measure("H2")
decide("H2")
print("passes left", s.budget_left)

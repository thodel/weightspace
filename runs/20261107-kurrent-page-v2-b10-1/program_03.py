"""Passes 5-6: the dose of H1 (preregistration §5, 'partial removal')."""
import numpy as np

from wsp.lora_subject import LoraSubject
from wsp.run import open_session
import json

s = open_session(".")
O = "base_model.model.model.language_model.layers.31.self_attn.o_proj"
B = f"{O}.lora_B.weight"
orig = LoraSubject(json.load(open("session.json"))["subject"])
u = orig.directions(O)[0][:, 0].astype(np.float64)
comp = np.outer(u, u @ orig.tensors()[B].astype(np.float64)).astype(np.float32)   # the component H1 removed


def decide(label):
    ok, why = s.keep_allowed()
    vk = s._vs_kept
    print(f"{label}: delta vs kept {vk['delta_cer']:+.4f} {[round(x, 4) for x in vk['ci95']]}; keep allowed: {ok} {why}", flush=True)
    return s.keep(f"{label} met the rule") if ok else (s.revert(f"{label} refused by the rule: {why}") or False)


h = s.hypothesis("H3: dose of H1 - remove 1.5x the component (its sign flips at half strength): more push toward stopping",
                 "against H1: cap 4 -> 2-4, CER 0 to -3 points; risk of early stops (length_ratio toward 1.0 or below, "
                 "truncated_cer up)", "direction subtraction (dose)")
s.edit("add", [B], hypothesis=h, delta=-0.5 * comp)
s.measure("H3 dose 1.5")
more = decide("H3 dose 1.5")

if more:
    h = s.hypothesis("H3b: dose 2.0 (the component fully reversed)", "against 1.5: cap -0 to -1, CER 0 to -1 point; "
                     "early-stop risk higher", "direction subtraction (dose)")
    s.edit("add", [B], hypothesis=h, delta=-0.5 * comp)
    s.measure("H3b dose 2.0")
    decide("H3b dose 2.0")
else:
    h = s.hypothesis("H3b: dose 0.5 (half the component back): is full removal past the best point?",
                     "against H1: CER +0 to +3 points, cap 4 -> 4-6; expected refusal", "direction subtraction (dose)")
    s.edit("add", [B], hypothesis=h, delta=0.5 * comp)
    s.measure("H3b dose 0.5")
    decide("H3b dose 0.5")
print("passes left", s.budget_left)

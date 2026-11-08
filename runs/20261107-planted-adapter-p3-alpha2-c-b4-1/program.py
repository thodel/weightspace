"""Arm (c) for adapters: a fixed hand-written list of the obvious repairs, the same for every corrupted adapter
(written 07.11.2026 before any pass).

Rules look only at the adapter's own tensors, its config and the statistics of two sibling adapters of the same base and recipe
(`v1`, `blb1`): per-module ‖ΔW‖ of healthy siblings differ from each other by at most 10 % (v1/blb1 in 0.90–1.00) and this
family's trained adapters sit at 0.80–0.93 of the siblings' median. The harness keep rule (paired page interval below 0,
pages at the cap not up) decides every edit.
  R1 lora_alpha differs from the siblings' (128): rescale every B by 128/alpha (a change of alpha by the ratio).
  R2 a module whose ‖ΔW‖ / sibling median is outside [0.6, 1.25] x the median ratio of all modules: rescale its B so the ratio equals
     the median ratio. A module at ratio < 0.05 has lost its update and cannot be restored from statistics: noted, no edit.
"""
import numpy as np
from wsp.run import open_session
from wsp.lora_subject import LoraSubject

SIB = ["/home/tobias/atr-cache/wsp/adapters/qwen3.5-4b-german-kurrent-xvi-xix-page-v1",
       "/mnt/wbkolleg_dh_1/Textrecognition_Training/training_folder/trained/qwen3.5-4b-german-kurrent-xvi-xix-blb-page-v1"]
s = open_session(".")
sub = s.subject
sibs = [LoraSubject(p) for p in SIB]
s.measure("C0")


def fro(x, m):
    b = x.tensors()[f"{m}.lora_B.weight"].astype(np.float64); a = x.tensors()[f"{m}.lora_A.weight"].astype(np.float64)
    return float(x.scale * np.sqrt(np.sum((b.T @ b) * (a @ a.T))))


def try_edit(text, pred, cls, names, per_tensor):
    if s.budget_left < 1:
        s.note("no budget left for: " + text)
        return False
    h = s.hypothesis(text, pred, cls)
    s.edit("scale", names, hypothesis=h, per_tensor=per_tensor, factor=1.0)
    s.measure(text[:60])
    return s.keep(text[:60])


# R1
alpha = float(sub.config["lora_alpha"]); sib_alpha = float(np.median([x.config["lora_alpha"] for x in sibs]))
if abs(alpha - sib_alpha) > 1e-6:
    f = sib_alpha / alpha
    names = [f"{m}.lora_B.weight" for m in sub.modules()]
    try_edit(f"R1 lora_alpha {alpha:g} differs from the siblings' {sib_alpha:g}: scale every B by {f:g}",
             "CER falls and pages at the cap fall if the update was too strong; rises if too weak", "scaling (global)",
             names, {n: {"factor": f} for n in names})
# R2
rat = {m: fro(sub, m) / float(np.median([fro(x, m) for x in sibs])) for m in sub.modules()}
med = float(np.median(list(rat.values())))
lost = [m for m, r in rat.items() if r < 0.05]
if lost:
    s.note(f"R2: modules with no update left (ratio < 0.05), not restorable from statistics: {lost}")
bad = {m: r for m, r in rat.items() if r >= 0.05 and not (0.6 * med <= r <= 1.25 * med)}
if bad:
    names = [f"{m}.lora_B.weight" for m in bad]
    try_edit(f"R2 modules off the siblings' profile by more than the band ({len(bad)}): {sorted(bad)[:4]}...: rescale each B to the median ratio",
             "CER falls if the module was over- or under-scaled", "scaling (module)", names,
             {f"{m}.lora_B.weight": {"factor": med / r} for m, r in bad.items()})
s.freeze()

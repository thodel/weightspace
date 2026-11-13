"""planted-adapter-f1 / q01 / task (a), budget 6 — written in full before the first pass.

Reading (pack.md): lora_alpha = 256 against 128 in both siblings (scale 4.0 vs 2.0); every module's
‖ΔW‖ is 1.6–1.85× the sibling median with no outlier; C0 is unusable (CER 2.50, 29/32 pages at the cap).
One global factor, so one global edit: halve every lora_B (= restore alpha 128). Then a two-point
α-sweep (×0.9, ×1.1) around that, which the keep rule is expected to refuse.
Plan: C0 → H1 (×0.5) → [kept] H2 (×0.9) → [refused] H2b (×1.1) → freeze   (≤ 4 passes)
                     → [refused] H1r (×0.5 again, engine noise) → freeze                (≤ 3 passes)
Only the wsp API and numpy; no adapter is read besides the subject.
"""
from wsp.run import open_session

s = open_session(".")
sub = s.subject
names_B = sorted(n for n in sub.tensors() if n.endswith(".lora_B.weight"))
names_A = sorted(n for n in sub.tensors() if n.endswith(".lora_A.weight"))
s.note(f"{len(sub.modules())} modules, {len(names_B)} lora_B tensors, {len(names_A)} lora_A tensors; "
       f"lora_alpha={sub.config.get('lora_alpha')}, r={sub.config.get('r')}, scale={sub.scale}; budget {s.budget_left}")
assert len(names_B) == len(sub.modules()) == 128, "unexpected adapter layout"


def at_cap():
    """Pages that ran to the token cap in the last pass (the harness's own count; None if unknown)."""
    return getattr(s, "_last_at_cap", None)


def report(tag, m):
    print(f"{tag}: cer={m.cer:.4f} length_ratio={m.length_ratio:.3f} collapsed={m.collapsed} at_cap={at_cap()}")
    s.note(f"{tag}: cer={m.cer:.4f} length_ratio={m.length_ratio:.3f} collapsed={m.collapsed} at_cap={at_cap()}")


def scale_all_B(hyp, factor, label):
    """One edit: every lora_B × factor (= lora_alpha × factor). Returns (metrics, kept)."""
    s.edit("scale", names_B, hypothesis=hyp, factor=float(factor))
    m = s.measure(label)
    report(label, m)
    kept = s.keep(f"{label}: all lora_B x {factor}")
    s.note(f"{label}: {'KEPT' if kept else 'refused by the rule, undone'}")
    return m, kept


# ---- pass 1: baseline -----------------------------------------------------------------------
m0 = s.measure("C0")
report("C0", m0)
cap0 = at_cap()

# ---- H1: the doubled scale (alpha 256 -> 128) -------------------------------------------------
h1 = s.hypothesis(
    "Planted defect: lora_alpha doubled (256 vs 128 in both siblings; scale 4.0 vs 2.0). Every module's "
    "norm ratio to the sibling median is 1.6-1.85 with no outlier, so one global factor. Halving every "
    "lora_B restores alpha 128.",
    "CER falls from ~2.5 to below 0.20 (likely 0.05-0.15); pages at the cap from ~29 to <= 3; loops gone. "
    "Far above engine noise.",
    "scale")
m1, kept1 = scale_all_B(h1, 0.5, "H1 all B x0.5")

if not kept1 and s.budget_left >= 1:
    # The engine is not repeatable; one replicate of the same edit before giving the hypothesis up.
    h1r = s.hypothesis(
        "H1 replicate: the same halving of every lora_B, re-measured once because the batched engine is "
        "not run-to-run deterministic.",
        "Same as H1: CER from ~2.5 to below 0.20, at-cap pages <= 3. If refused again the global magnitude "
        "is not the defect that is visible here.",
        "scale")
    m1, kept1 = scale_all_B(h1r, 0.5, "H1r all B x0.5 (replicate)")

if kept1:
    s.note("H1 kept: the adapter now runs at the siblings' scale (alpha 128 equivalent). "
           "Next: a two-point alpha-sweep around it, expected to be refused.")
    # ---- H2: 0.45 (module norms at the siblings' median) ------------------------------------
    kept2 = False
    if s.budget_left >= 1:
        h2 = s.hypothesis(
            "Alpha-sweep point 1: the norm ratio was ~1.8 not 2.0, so the siblings' median module norm "
            "corresponds to an overall factor 0.45; apply x0.9 on top of the kept x0.5.",
            "Within noise: |dCER| < 0.02, direction undecided; the keep rule is expected to refuse. "
            "If refused, the residual x0.9 is this run's own norm and 0.5 is the whole correction.",
            "scale")
        m2, kept2 = scale_all_B(h2, 0.9, "H2 all B x0.9 (eff. 0.45)")
    # ---- H2b: 0.55, only if 0.45 was refused ------------------------------------------------
    if not kept2 and s.budget_left >= 1:
        h2b = s.hypothesis(
            "Alpha-sweep point 2: x1.1 on top of the kept x0.5 (effective 0.55), the other side of the "
            "repaired scale.",
            "Within noise: |dCER| < 0.02; the keep rule is expected to refuse. If refused, the optimum is "
            "flat around 0.5 and the x0.5 state is the repaired adapter.",
            "scale")
        m2b, kept2b = scale_all_B(h2b, 1.1, "H2b all B x1.1 (eff. 0.55)")
    s.note(f"C0 at_cap={cap0}; final state at_cap={at_cap()}. Passes used: {s.budget - s.budget_left} of {s.budget}.")
else:
    s.note("H1 refused twice: the global scale is not the defect that the pack shows, or not the only one. "
           "No per-module anomaly in the statistics to target (sign/direction-class defects are invisible in norms); "
           "the adapter is frozen unchanged rather than edited on a guess.")

summary = s.freeze()
print("frozen:", {k: summary.get(k) for k in ("used", "edits", "kept", "final_digest")})

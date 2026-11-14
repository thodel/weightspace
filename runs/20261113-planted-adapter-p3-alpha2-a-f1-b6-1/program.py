"""q02 / task (a), budget 6 -- the programmer with passes. Written in full before the first pass.

Hypotheses: preregistration_hypotheses.md (same folder). Reading of the pack: lora_alpha is 256 where
the siblings have 128 and every module's ||dW|| is ~1.8x the sibling median -> the update is doubled.
H1 halves every lora_B (alpha 256 -> effective 128). H2 (spectral outlier against the siblings) and
H3 (residual loops) are conditional and may spend no pass at all. Only wsp + numpy.
"""
from wsp.run import open_session
from wsp.lora_subject import LoraSubject
import numpy as np

SIBLINGS = {
    "v1": "/home/tobias/atr-cache/wsp/adapters/qwen3.5-4b-german-kurrent-xvi-xix-page-v1",
    "blb1": "/mnt/wbkolleg_dh_1/Textrecognition_Training/training_folder/trained/qwen3.5-4b-german-kurrent-xvi-xix-blb-page-v1",
}
LOOPS_REMAIN = 4          # H3 fires only if the last kept state still has this many pages at the cap
H2_SHARE_ABS, H2_SHARE_REL = 0.20, 1.6
H2_GAP_ABS, H2_GAP_REL = 4.0, 3.0


def B(module):
    return f"{module}.lora_B.weight"


def last_at_cap(s):
    return getattr(s, "_last_at_cap", None)


def kept_at_cap(s):
    return getattr(s, "_kept_at_cap", None)


def spectra(sub):
    """Per module: top-singular-value share sigma1/||S|| and gap sigma1/sigma2 (both scale-invariant)."""
    out = {}
    for m in sub.modules():
        _, S, _ = sub.directions(m)
        S = np.sort(S.astype(np.float64))[::-1]
        tot = float(np.sqrt((S ** 2).sum()))
        share = float(S[0] / tot) if tot > 0 else 0.0
        gap = float(S[0] / S[1]) if len(S) > 1 and S[1] > 0 else float("inf")
        out[m] = {"share": share, "gap": gap, "S": S}
    return out


def stage(s, name, fn):
    """Run an optional stage; on a crash revert a pending edit so the session can still freeze."""
    try:
        fn()
    except Exception as e:  # noqa: BLE001 - the session must end in freeze() whatever happens
        s.note(f"{name} failed: {e!r}")
        if getattr(s, "_last_edit", None) is not None:
            try:
                s.revert(f"{name} crashed before a decision")
            except Exception as e2:  # noqa: BLE001
                s.note(f"{name}: revert failed too: {e2!r}")


s = open_session(".")
sub = s.subject
mods = sub.modules()
names_B = [B(m) for m in mods]
s.note(f"q02/a start: {len(mods)} modules, lora_alpha={sub.config.get('lora_alpha')}, r={sub.config.get('r')}, "
       f"scale={sub.scale}, budget_left={s.budget_left}")

# ---- pass 1: C0 -------------------------------------------------------------------------------
m0 = s.measure("C0")
cap0 = last_at_cap(s)
s.note(f"C0: cer={m0.cer:.4f} collapsed={getattr(m0, 'collapsed', None)} at_cap={cap0}")

# ---- pass 2: H1, the global scale -------------------------------------------------------------
h1 = s.hypothesis(
    "H1: lora_alpha is 256 where both siblings have 128, and all 128 modules sit at ~1.8x the sibling median "
    "||dW|| (1.60-1.85, no module outside the band): the update the base receives was doubled. Halving every "
    "lora_B restores alpha=128 (the config is untouched; the saved adapter is equivalent).",
    "CER falls from ~2.5 to below 0.25 (expected 0.05-0.15); pages at the cap fall from 29 to at most 3",
    "scale (global, all lora_B x 0.5)")
s.edit("scale", names_B, hypothesis=h1, factor=0.5)
m1 = s.measure("H1: all lora_B x 0.5")
s.note(f"H1: cer={m1.cer:.4f} at_cap={last_at_cap(s)} (C0 {m0.cer:.4f}, {cap0})")
kept1 = s.keep("global over-scale: alpha 256 -> effective 128")
s.note(f"H1 kept={kept1}")

if not kept1:
    # ---- branch B: the scale is not (only) the defect; one gentler factor, then stop ------------
    def h1b():
        if s.budget_left < 1:
            return
        h = s.hypothesis(
            "H1b: H1 (x0.5) was refused. If the over-scale is real but 0.5 overshoots, a gentler global "
            "factor should still beat C0.",
            "CER -0.5 to -1.5 against C0; if refused: the global scale is not the defect, freeze unchanged",
            "scale (global, all lora_B x 0.7)")
        s.edit("scale", names_B, hypothesis=h, factor=0.7)
        m = s.measure("H1b: all lora_B x 0.7")
        s.note(f"H1b: cer={m.cer:.4f} at_cap={last_at_cap(s)}")
        k = s.keep("gentler global factor")
        s.note(f"H1b kept={k}" + ("" if k else "; the pack shows nothing else anomalous: no further edit"))
    stage(s, "H1b", h1b)

else:
    # ---- H2: spectral outlier against the siblings (no pass unless something is flagged) --------
    def h2():
        sibs = {}
        for tag, path in SIBLINGS.items():
            try:
                sibs[tag] = spectra(LoraSubject(path))
            except Exception as e:  # noqa: BLE001 - a sibling that does not resolve in the sandbox is not an error
                s.note(f"H2: sibling {tag} not usable: {e!r}")
        if not sibs:
            s.note("H2: no sibling available, spectral check skipped")
            return
        mine = spectra(sub)
        flagged = []
        for m in mods:
            shares = [sp[m]["share"] for sp in sibs.values() if m in sp]
            gaps = [sp[m]["gap"] for sp in sibs.values() if m in sp and np.isfinite(sp[m]["gap"])]
            if not shares:
                continue
            ref_share = float(np.mean(shares))
            ref_gap = float(np.max(gaps)) if gaps else float("inf")
            a = mine[m]
            by_share = a["share"] > ref_share + H2_SHARE_ABS and a["share"] > H2_SHARE_REL * ref_share
            by_gap = np.isfinite(a["gap"]) and a["gap"] > H2_GAP_ABS and a["gap"] > H2_GAP_REL * ref_gap
            if by_share or by_gap:
                flagged.append((a["share"] - ref_share, m, ref_share, a["share"], a["gap"], ref_gap))
        # a compact record of the comparison, flagged or not
        worst = sorted(((mine[m]["share"] - float(np.mean([sp[m]["share"] for sp in sibs.values() if m in sp])), m)
                        for m in mods if any(m in sp for sp in sibs.values())), reverse=True)[:5]
        s.note("H2: largest share excess over siblings: " + "; ".join(f"{m} +{d:.3f}" for d, m in worst))
        if not flagged:
            s.note("H2: no module flagged (thresholds share +0.20 & x1.6, or gap >4 & x3): no edit, no pass")
            return
        flagged.sort(reverse=True)
        flagged = flagged[:3]
        if s.budget_left < 1:
            s.note("H2: flagged but no pass left: " + ", ".join(f[1] for f in flagged))
            return
        per_tensor = {}
        desc = []
        T = sub.tensors()
        for _, m, ref_share, share, gap, ref_gap in flagged:
            U, S, _ = sub.directions(m)
            S = S.astype(np.float64)
            rest = float(np.sqrt((S[1:] ** 2).sum()))
            p = min(max(ref_share, 0.05), 0.95)
            target = p * rest / np.sqrt(1.0 - p * p)
            t = float(target / S[0]) if S[0] > 0 else 1.0
            if t >= 0.95:
                continue
            u1 = U[:, 0].astype(np.float64)
            u1 = u1 / np.linalg.norm(u1)
            Bm = T[B(m)].astype(np.float64)
            delta = (-(1.0 - t) * np.outer(u1, u1 @ Bm)).astype(np.float32)
            per_tensor[B(m)] = {"delta": delta}
            desc.append(f"{m}: share {share:.3f} vs siblings {ref_share:.3f}, gap {gap:.2f} vs {ref_gap:.2f}, sigma1 x{t:.3f}")
        if not per_tensor:
            s.note("H2: flagged modules need no shrink (t >= 0.95): no edit")
            return
        h = s.hypothesis(
            "H2: these modules carry a top singular direction far above the siblings' share of the same module: "
            + "; ".join(desc) + ". Shrinking sigma1 to the siblings' share (B <- B - (1-t) u1 u1^T B) removes a spike.",
            "CER -0.01 to -0.05 and no more pages at the cap; if refused: natural variation, revert",
            "add (rank-1 shrink of B's top output direction)")
        s.edit("add", list(per_tensor), hypothesis=h, per_tensor=per_tensor)
        m = s.measure("H2: shrink spectral outliers")
        s.note(f"H2: cer={m.cer:.4f} at_cap={last_at_cap(s)}")
        k = s.keep("spectral outlier shrunk to sibling level")
        s.note(f"H2 kept={k}")
    stage(s, "H2", h2)

    # ---- H3: residual loops after the halving -> one or two steps of x0.8 ----------------------
    def h3():
        for step in (1, 2):
            cap = kept_at_cap(s)
            if cap is None or cap < LOOPS_REMAIN:
                s.note(f"H3: last kept state has {cap} pages at the cap (< {LOOPS_REMAIN}): no further scaling")
                return
            if s.budget_left < 1:
                s.note("H3: no pass left")
                return
            h = s.hypothesis(
                f"H3 step {step}: after the halving the last kept state still has {cap} pages at the cap; if the "
                "remaining loops are still magnitude-driven a further global x0.8 (alpha 128 -> ~102) reduces them.",
                "small change: CER -0.02 to -0.1, fewer pages at the cap; if refused: residual loops are not scale-driven",
                "scale (global, all lora_B x 0.8)")
            s.edit("scale", names_B, hypothesis=h, factor=0.8)
            m = s.measure(f"H3 step {step}: all lora_B x 0.8")
            s.note(f"H3 step {step}: cer={m.cer:.4f} at_cap={last_at_cap(s)}")
            k = s.keep(f"residual loops, global x0.8 (step {step})")
            s.note(f"H3 step {step} kept={k}")
            if not k:
                return
    stage(s, "H3", h3)

summary = s.freeze()
print({k: summary.get(k) for k in ("used", "edits", "kept", "final_digest")})

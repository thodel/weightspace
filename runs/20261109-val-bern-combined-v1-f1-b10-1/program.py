"""bern-combined-v1, attempt f1 (programmer: claude-fable-5-1). Written in advance from the pack; runs once in the sandbox.

Plan (see preregistration_hypotheses.md): pass 1 baseline; H1/H1b global interpolation towards
kraken-bern-schlussredaktion-v1 (output rows paired by character), alpha sweep decided by the keep rule;
H2/H2b blank-bias steps; H3 space and '/' bias; H4 output-weight scale. <= 10 measure calls; always freeze().
"""
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject
from wsp.align import codec_row_map, gather_rows
import numpy as np

DONOR_ID = "kraken-bern-schlussredaktion-v1"
DONOR_PATH = "/home/tobias/Repo/weightspace/models/kraken/kraken-bern-schlussredaktion-v1.mlmodel"
OUT_W, OUT_B = "O_18.lin.weight", "O_18.lin.bias"

s = open_session(".")
T = s.subject.tensors()
cm = s.subject.codec_map()
BLANK = 0
state = {"last_kept_cer": None}


def attempt(h, op, names, label, rule_mask=None, per_tensor=None, **kw):
    """edit -> measure -> keep. Returns (kept, cer_after). Any failure after the edit reverts it."""
    if s.budget_left < 1:
        s.note(f"skip {label}: no passes left")
        return False, None
    s.edit(op, names, hypothesis=h, per_tensor=per_tensor, **kw)
    try:
        m = s.measure(label, rule_mask=rule_mask)
    except Exception as e:  # noqa: BLE001 - never leave an undecided edit behind
        s.revert(f"measure failed: {e!r}")
        return False, None
    kept = s.keep(f"{label}: keep rule")
    if kept:
        state["last_kept_cer"] = m.cer
    s.note(f"{label}: CER {m.cer:.4f} collapsed {m.collapsed} kept={kept}")
    return kept, m.cer


# ---------------------------------------------------------------- pass 1: baseline
m0 = s.measure("C0 baseline")
state["last_kept_cer"] = m0.cer
s.note(f"baseline CER {m0.cer:.4f}, collapsed {m0.collapsed}")

# ---------------------------------------------------------------- H1 / H1b: interpolation towards the sibling
donor_ok = True
try:
    donor = KrakenSubject(DONOR_PATH)
    D = donor.tensors()
    dcm = donor.codec_map()
except Exception as e:  # noqa: BLE001
    donor_ok = False
    s.note(f"donor {DONOR_ID} could not be loaded: {e!r}; H1 skipped")

if donor_ok:
    # Row pairing by character for the output layer; the blank is row 0 in both; subject-only rows keep their values.
    row_map = {BLANK: BLANK}
    row_map.update(codec_row_map(cm, dcm))
    S0 = {n: v.copy() for n, v in T.items()}        # the subject's start tensors (the interpolation's endpoint a)
    aligned = {}
    for n, w in S0.items():
        if n in (OUT_W, OUT_B):
            aligned[n] = gather_rows(D[n], row_map, w.shape[0], fill=w)
        elif n in D and D[n].shape == w.shape:
            aligned[n] = D[n]
        else:
            s.note(f"{n}: no shape-matched donor tensor, left out of the interpolation")
    names = list(aligned)
    s.note(f"interpolation covers {len(names)} tensors; {len(row_map)} output rows paired (of {S0[OUT_W].shape[0]})")
    mask = s.unseen_by(DONOR_ID)
    s.note(f"donor rule: {sum(mask)} of {len(mask)} lines unseen by {DONOR_ID}")

    cur_alpha = 0.0   # global position between subject (0) and donor (1) of the current kept state

    def interp_to(alpha, h, label):
        """Move the current state (at global cur_alpha) to global alpha along the subject->donor line."""
        global cur_alpha
        local = (alpha - cur_alpha) / (1.0 - cur_alpha)
        per = {n: {"other": aligned[n]} for n in names}
        kept, cer = attempt(h, "interpolate", names, label, rule_mask=mask, per_tensor=per, alpha=float(local))
        if kept:
            cur_alpha = alpha
        return kept, cer

    h1 = s.hypothesis("H1: the two fine-tunes of hertwig-v2 share a basin; the half-way point reads both hands better than the subject",
                      "CER down 0.010-0.020 on the lines the donor did not train on", "interpolate (donor, all tensors, rows by character)")
    kept_05, _ = interp_to(0.5, h1, "H1 interpolate alpha=0.5")
    if kept_05:
        h1b = s.hypothesis("H1b: the alpha response is monotone near 0.5; moving further towards the donor helps a little more",
                           "CER down 0.002-0.005 more on the rule lines", "interpolate (donor)")
        kept_075, _ = interp_to(0.75, h1b, "H1b interpolate alpha=0.75")
        if kept_075:
            h1c = s.hypothesis("H1b: still monotone; alpha 0.9 keeps a sliver of the subject", "CER down 0.001-0.003 more", "interpolate (donor)")
            interp_to(0.9, h1c, "H1b interpolate alpha=0.9")
        else:
            s.note("alpha optimum near 0.5; staying there")
    else:
        h1b = s.hypothesis("H1b: a quarter step towards the donor is within the shared basin even if the half-way point is not",
                           "CER down 0.003-0.008 on the rule lines", "interpolate (donor)")
        kept_025, _ = interp_to(0.25, h1b, "H1b interpolate alpha=0.25")
        if not kept_025:
            s.note("no alpha helped on the unseen lines: the interpolation line is closed for this pair")

# ---------------------------------------------------------------- H2 / H2b: blank bias
nb = T[OUT_B].shape[0]


def bias_delta(rows_values):
    d = np.zeros(nb, dtype=np.float32)
    for r, v in rows_values:
        d[r] += v
    return d


cer_before_h2 = state["last_kept_cer"]
h2 = s.hypothesis("H2: deletions (space, /, e, n, t, r, i, s) are marginal blank wins; a lower blank bias converts more of them than it adds insertions",
                  "CER down 0.003-0.007 on all lines", "add (output bias, blank row)")
kept_h2, cer_h2 = attempt(h2, "add", [OUT_B], "H2 blank bias -0.4", delta=bias_delta([(BLANK, -0.4)]))
if kept_h2:
    h2b = s.hypothesis("H2b: a second step of the same size still helps, less", "CER down 0.001-0.003 more", "add (output bias, blank row)")
    attempt(h2b, "add", [OUT_B], "H2b blank bias -0.4 more (cum. -0.8)", delta=bias_delta([(BLANK, -0.4)]))
elif cer_h2 is not None and cer_before_h2 is not None and cer_h2 > cer_before_h2:
    h2b = s.hypothesis("H2b: the blank is at or past its optimum from below; insertions matter more than the confusion table suggests",
                       "CER down 0.001-0.003 with a slightly higher blank bias", "add (output bias, blank row)")
    attempt(h2b, "add", [OUT_B], "H2b blank bias +0.3", delta=bias_delta([(BLANK, +0.3)]))
else:
    s.note("H2 refused without a rise: the blank bias is left alone")

# ---------------------------------------------------------------- H3: space and '/' biases
rows_h3 = [(cm[ch], 0.25) for ch in (" ", "/") if ch in cm]
if rows_h3:
    h3 = s.hypothesis("H3: the net deletions of space (96) and / (41) are threshold misses; a small bias lift converts them",
                      "CER down 0.001-0.003 on all lines, may be below the rule's resolution", "add (output bias, 2 rows)")
    attempt(h3, "add", [OUT_B], "H3 space and / bias +0.25", delta=bias_delta(rows_h3))
else:
    s.note("H3 skipped: space or / not single-label rows in the codec")

# ---------------------------------------------------------------- H4: output weight scale
h4 = s.hypothesis("H4: scaling the output weights (not the bias) sharpens feature-driven character logits against the bias-driven blank",
                  "CER down 0.002-0.005 on all lines", "scale (output weight)")
attempt(h4, "scale", [OUT_W], "H4 scale O_18.lin.weight x1.15", factor=1.15)

# ---------------------------------------------------------------- freeze
summary = s.freeze()
print("used", summary["used"], "edits", summary["edits"], "kept", summary["kept"],
      "baseline", summary["baseline"] and summary["baseline"].get("cer"), "last", summary["last_pass"] and summary["last_pass"].get("cer"))

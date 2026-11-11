"""s08 / task (b), budget 0: blind repair of a zeroed `e` output row.

Reading (pack): `O_18.lin.weight` row of `e` has norm 0.00 (every other row > 4.6); 62 % of the
practice-set edits are a lost `e` (910 `e -> blank`, 131 `e -> a`, 90 `e -> i`, ...). With a zero
row the `e` logit is a constant (its bias, <= +0.28), which loses to the blank (bias +1.23) at most
`e` centres. The only donor is from another lineage (cos ~ 0 against every subject tensor), so its
`e` row is useless here and is not used. The edit shipped: raise the `e` bias to +1.5, i.e. just
above the blank's resting level, so `e` wins where every competitor is suppressed. The weight row
is left alone: nothing healthy exists to write there.

Runs inside the sandbox: wsp + numpy only; no measure (budget 0)."""
from wsp.run import open_session
import numpy as np

E_BIAS_TARGET = 1.5          # blank bias is +1.23; a 0.27 margin over its resting level
ZERO_ROW_TOL = 0.5           # a healthy frequent-letter row has norm ~8

s = open_session(".")
T = s.subject.tensors()
cm = s.subject.codec_map()

W_NAME, B_NAME = "O_18.lin.weight", "O_18.lin.bias"
e_row = cm.get("e")

if e_row is None:
    s.note("no single-label row for 'e' in the codec; nothing applied")
else:
    w_norm = float(np.linalg.norm(T[W_NAME][e_row]))
    b_cur = float(T[B_NAME][e_row])
    blank_b = float(T[B_NAME][0])
    s.note(f"reading check: row 'e' = {e_row}, |W[e]| = {w_norm:.4f}, bias[e] = {b_cur:+.4f}, "
           f"bias[blank] = {blank_b:+.4f}")

    if w_norm >= ZERO_ROW_TOL:
        s.note("the 'e' weight row is not zero: the pack reading is not confirmed; nothing applied")
    elif b_cur >= E_BIAS_TARGET:
        s.note(f"bias[e] already >= {E_BIAS_TARGET}; nothing applied")
    else:
        h = s.hypothesis(
            "The output row of 'e' is zeroed (norm 0.00; every other row > 4.6), so the 'e' logit is the "
            "constant bias[e] and loses to the blank (bias +1.23) at most true-'e' frames: 910 'e -> blank' "
            "plus ~380 'e -> a/i/o/r/c/u' of 2088 edits. Raising bias[e] to +1.5 puts 'e' just above the "
            "blank's resting level, so it wins where every competitor is suppressed (true-'e' centres) "
            "while letter centres and blank frames (both >> 1.5) are untouched. The weight row is not "
            "rewritten: the only donor is from another lineage (cos ~ 0 in every tensor), so its row would "
            "be noise in this feature space.",
            f"CER down by 0.02-0.05 (from 0.1623): recover roughly 400-700 of the ~1290 lost 'e's, minus "
            f"some 'e' insertions at transition frames whose maximum logit is below {E_BIAS_TARGET}. "
            "A full repair (parent row) would give ~ -0.10; this cannot reach that.",
            "set_rows (one scalar of O_18.lin.bias)")
        s.edit("set_rows", [B_NAME], hypothesis=h,
               index=[int(e_row)], values=np.array([E_BIAS_TARGET], dtype=np.float32))
        s.keep_unmeasured(
            "budget 0: no pass. Diagnosis is unambiguous (exactly zero row of the most frequent letter, "
            "62 % of edits are a lost 'e'); the edit is one bounded scalar whose failure mode (extra 'e's) "
            "is milder than the 910 deletions it targets. Donor transplant and synthetic-row reconstruction "
            "were considered and rejected as not fairly sure (see hypotheses.md).")

s.freeze()

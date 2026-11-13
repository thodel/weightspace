"""s06 task (b), budget 0: blind repair of the zeroed `e` output row.

Reading (pack): O_18.lin.weight row `e` has norm 0.00 (parent 7.64, cosine 0.000, bias Δ -0.14);
C0 confusions are dominated by `e`->∅ (910) and `e`->a/i/o/r/c/u (~380). One edit: put the parent's
`e` row (aligned by character) into the child's `e` row; keep the child's bias. No pass is available,
so the edit is kept unmeasured, guarded by shape/norm checks on the tensors themselves.
"""
from __future__ import annotations

import numpy as np

from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-german-xix-v1.mlmodel"
W, B = "O_18.lin.weight", "O_18.lin.bias"


def main() -> None:
    s = open_session(".")
    s.note("s06/b blind programmer (claude-fable-5-1, budget 0): H1 = output row `e` zeroed; "
           "repair by transplanting the parent's `e` row into O_18.lin.weight.")

    T = s.subject.tensors()
    cm = s.subject.codec_map()
    if "e" not in cm:
        s.note("no single-label row for `e` in the child codec; nothing applied")
        s.freeze()
        return
    row = cm["e"]
    w = T[W]
    child_norm = float(np.linalg.norm(w[row]))
    child_bias = float(T[B][row])
    s.note(f"child `e` row {row}: norm {child_norm:.4f}, bias {child_bias:+.3f}; "
           f"blank row norm {float(np.linalg.norm(w[0])):.3f}")

    if child_norm > 1e-3:
        s.note(f"guard: the `e` row is not zero (norm {child_norm:.4f}); the pack reading does not hold, nothing applied")
        s.freeze()
        return

    parent = KrakenSubject(PARENT)
    pm = parent.codec_map()
    if "e" not in pm:
        s.note("parent has no single-label `e` row; nothing applied")
        s.freeze()
        return
    prow = pm["e"]
    pw = parent.tensors()[W]
    pb = parent.tensors()[B]
    donor = pw[prow].astype(np.float32).copy()
    donor_norm = float(np.linalg.norm(donor))
    s.note(f"parent `e` row {prow}: norm {donor_norm:.3f}, bias {float(pb[prow]):+.3f}")

    if donor.shape != (w.shape[1],) or not (3.0 <= donor_norm <= 15.0):
        s.note(f"guard: donor row unusable (shape {donor.shape}, norm {donor_norm:.3f}); nothing applied")
        s.freeze()
        return

    # neighbours of `e` among the common letters, for the log: the donor's norm should sit among them
    peers = {c: float(np.linalg.norm(w[cm[c]])) for c in "rstina" if c in cm}
    s.note("child norms of r/s/t/i/n/a: " + ", ".join(f"{c} {v:.2f}" for c, v in peers.items()))

    h = s.hypothesis(
        "H1: the O_18.lin.weight row of `e` was zeroed (norm 0.00 vs parent 7.64, cosine 0.000), so `e` can never be "
        "emitted: 910 `e`->∅ and ~380 `e`->vowel confusions, ~1290 of 2088 edits. The parent's `e` row, aligned by "
        "character, is a usable detector in the child's feature space (LSTM blocks drifted only ~0.3 rel L2).",
        "CER down by about 0.06-0.09 (0.162 -> ~0.07-0.10); collapsed lines and the length ratio move towards 0 and 1.0. "
        "Some `e`s stay missed and a few a/i/o may flip to `e`, but a zero row is worse than any reasonable detector.",
        "set_rows (donor transplant of one output row, parent -> child, aligned by character; child bias kept)")
    s.edit("set_rows", [W], hypothesis=h, index=[row], values=donor)
    after = float(np.linalg.norm(s.subject.tensors()[W][row]))
    s.note(f"`e` row after edit: norm {after:.3f} (donor {donor_norm:.3f})")
    s.keep_unmeasured("budget 0: zeroed `e` row replaced by the parent's `e` row; the bias (within 0.14 of the parent's) "
                      "is left as trained in the child")

    s.note("H2 (C_3.co.bias rel L2 1.09) and H3 (L_16.weight_ih_l0 abs max 4.70) noted but not acted on: "
           "not confident enough for a blind edit.")
    s.freeze()


if __name__ == "__main__":
    main()

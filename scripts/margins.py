"""Margins at deleted frames (docs/CONTROLS_REPORT.md §8): is the deletion decided upstream of the output layer?

    .venv/bin/python scripts/margins.py run|report

For every line of blb-eval-hand-b where german-xix-blb-v1 deletes a space, `e` or `n` (character alignment of the normalised reference and the greedy
reading), the *window* of the deletion is the frame range between the end of the previous emitted character and the start of the next one.
Per deleted character: M1 = min over the window of (blank log-prob − log-prob of the deleted character) — how far the character came from winning;
M2 = min over the window of (blank − best non-blank class). Softmax log-ratios equal logit differences. A bias edit of d on the blank row shifts
every blank-minus-X margin by exactly d, so the margins "after" are the margins "before" + d; the actual readings of the edited models (the three
blank-bias states of session 1: -0.113, -0.5, -1.0 total) are decoded as a check on how many of those deletions disappear.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

M = Path.home() / "Repo/weightspace/models/kraken"
OUT = Path.home() / "atr-cache/wsp/controls/margins"
CHARS = (" ", "e", "n")
EDITS = [-0.113, -0.5, -1.0]       # total blank-bias change of session 1: H1 kept state, H2, H7 (docs/PHASE1_REPORT.md §2)


def decode_line(S, item):
    """(chars with frame spans, log-probs (W, C)) of the greedy reading, or None for an unreadable crop."""
    from kraken.lib.ctc_decoder import greedy_decoder
    lg = S.logits([item])
    if not lg:
        return None
    P = np.asarray(lg[0], dtype=np.float64)                 # (W, C) softmax
    labels = greedy_decoder(P.T)[0]
    chars = S.model.codec.decode([(int(c), int(s), int(e), float(m)) for c, s, e, m in labels])
    return chars, np.log(np.clip(P, 1e-30, None))


def run():
    from wsp.data import load
    from wsp.kraken_subject import KrakenSubject
    from wsp.metrics import normalise, align
    OUT.mkdir(parents=True, exist_ok=True)
    S = KrakenSubject(M / "kraken-german-xix-blb-v1.mlmodel")
    cm = S.codec_map()
    items = load("data/heldout/blb-eval-hand-b.jsonl")
    rec, skipped = [], 0
    for it in items:
        d = decode_line(S, it)
        if d is None:
            skipped += 1; continue
        chars, lp = d
        hyp = "".join(c for c, *_ in chars); ref = normalise(it["text"]); nh = normalise(hyp)
        if len(nh) != len(hyp):                               # whitespace normalisation shifted positions: skip the line
            skipped += 1; continue
        k = 0
        for rc, hc in align(ref, nh):
            if hc:
                k += 1
            if rc and not hc and rc in CHARS and rc in cm:
                prev_end = chars[k - 1][2] if k > 0 else -1          # kraken's span end is the last frame of the run (inclusive)
                next_start = chars[k][1] if k < len(chars) else lp.shape[0]
                lo, hi = prev_end + 1, next_start - 1
                if hi < lo:                                          # no frame between the neighbours: nothing for a bias edit to act on
                    rec.append({"key": it["key"], "char": rc, "frames": 0, "M1": None, "M2": None}); continue
                w = lp[lo:hi + 1]
                nb = w[:, 1:].max(axis=1)
                j = int(np.argmin(w[:, 0] - w[:, cm[rc]]))
                rec.append({"key": it["key"], "char": rc, "frames": int(hi - lo + 1),
                            "M1": float((w[:, 0] - w[:, cm[rc]]).min()), "M2": float((w[:, 0] - nb).min()),
                            "c_is_best_nonblank_at_M1": bool(w[j, 1:].argmax() + 1 == cm[rc])})
    (OUT / "deleted.json").write_text(json.dumps({"records": rec, "skipped_lines": skipped, "lines": len(items)}))
    # actual readings of the edited models on the lines with a deletion of interest
    keys = sorted({r["key"] for r in rec}); sub = [it for it in items if it["key"] in set(keys)]
    from wsp.metrics import normalise as nz
    def count(model_delta):
        T = KrakenSubject(M / "kraken-german-xix-blb-v1.mlmodel")
        b = T.tensors()["O_18.lin.bias"].copy(); b[0] += model_delta; T.set_tensor("O_18.lin.bias", b)
        n = {c: 0 for c in CHARS}; ins = 0
        for it, p in zip(sub, T.forward(sub)):
            for rc, hc in align(nz(it["text"]), nz(p.text)):
                if rc in CHARS and not hc: n[rc] += 1
                if hc and not rc: ins += 1
        return {"deleted": n, "inserted": ins}
    res = {"0.0": count(0.0)}
    # session 1 totals are relative to the model's own bias (1.2323 -> 1.1195 parent value = -0.113; -0.5 total and -1.0 total are the H2/H7 states)
    for d in EDITS:
        res[str(d)] = count(d)
    (OUT / "actual.json").write_text(json.dumps({"lines": len(sub), "by_edit": res}))
    print("deleted instances", len(rec), "lines with one", len(sub), "skipped", skipped)


def report():
    d = json.loads((OUT / "deleted.json").read_text()); allrec = d["records"]; rec = [r for r in allrec if r["M1"] is not None]; act = json.loads((OUT / "actual.json").read_text())
    q = [0, 5, 25, 50, 75, 95, 100]
    out = {"n": len(rec), "by_char": {}, "actual": act}
    print(f"{len(allrec)} deleted instances, {len(allrec) - len(rec)} with no frame between the neighbours (no blank gap); ")
    print(f"{len(rec)} with a gap:  of space/e/n in {len({r['key'] for r in rec})} lines of {d['lines']}; skipped lines {d['skipped_lines']}")
    for c in list(CHARS) + ["all"]:
        rs = [r for r in rec if c == "all" or r["char"] == c]
        M1 = np.array([r["M1"] for r in rs]); M2 = np.array([r["M2"] for r in rs])
        row = {"n": len(rs), "M1_quantiles": dict(zip(q, np.percentile(M1, q).round(2).tolist())), "M2_quantiles": dict(zip(q, np.percentile(M2, q).round(2).tolist())),
               "predicted_recovered_M1": {str(e): float((M1 + e < 0).mean()) for e in EDITS}, "predicted_recovered_M2": {str(e): float((M2 + e < 0).mean()) for e in EDITS},
               "share_M1_below_0": float((M1 < 0).mean()), "share_c_best_nonblank_at_M1": float(np.mean([r["c_is_best_nonblank_at_M1"] for r in rs])), "share_M1_within_2": float((M1 < 2).mean())}
        out["by_char"][c] = row
        print(f"| {c!r} | {len(rs)} | M1 quantiles {row['M1_quantiles']} | M2 {row['M2_quantiles']} | frac M1<2: {row['share_M1_within_2']:.2f} | recovered by -0.113/-0.5/-1.0 (M1): " +
              " / ".join(f"{row['predicted_recovered_M1'][str(e)]:.3f}" for e in EDITS))
    base = act["by_edit"]["0.0"]["deleted"]
    for e in EDITS:
        a = act["by_edit"][str(e)]
        print("edit", e, "deleted", a["deleted"], "vs", base, "inserted", a["inserted"], "vs", act["by_edit"]["0.0"]["inserted"])
    Path("docs/controls/margins.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    run() if sys.argv[1] == "run" else report()

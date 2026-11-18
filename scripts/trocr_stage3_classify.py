"""Outcome classes of the TrOCR planted-defect control (docs/TROCR_STAGE3_PREREG.md, section 4), on held-out lines (blb-eval-hand).

    PYTHONPATH=. ~/atr-cache/trocr/venv/bin/python scripts/trocr_stage3_classify.py   -> docs/trocr/stage3_classes.json and .md

Damage = CER(corrupted) - CER(unplanted subject). Recovery = (CER(corrupted) - CER(final)) / damage. Interval: page-clustered paired bootstrap of
CER(final) - CER(corrupted) (wsp.metrics.bootstrap_delta). repaired: recovery >= 0.8 and the interval below 0; partly: 0.2 to 0.8; not repaired: under 0.2 or
CER above the corrupted model's; harmful keep: the final is worse than the corrupted model by more than 0.5 points with the interval above 0.
A defect counts if its held-out damage is at least 1.5 points. Sources: baselines (arms c, d) from docs/trocr/heldout_baselines.json; programmer arms
from the held-out result in the run's summary.json (runs/*-trocr-planted-<id>-<arm>-b*-1).
"""
import json
import re
from pathlib import Path

from wsp.data import load
from wsp.metrics import bootstrap_delta, page_of

REPO = Path(__file__).resolve().parents[1]
HB = json.loads((REPO / "docs/trocr/heldout_baselines.json").read_text())
KEYS = [x["key"] for x in load(REPO / "data/heldout/blb-eval-hand.jsonl")]
CL = [page_of(k) for k in KEYS]
MIN_EFFECT, COUNT = 0.005, 0.015


def cer(pl):
    return sum(e for e, _ in pl) / sum(c for _, c in pl)


def classify(corr, fin, orig):
    dmg = cer(corr) - cer(orig)
    b = bootstrap_delta([tuple(x) for x in corr], [tuple(x) for x in fin], clusters=CL)
    lo, hi = b["ci95_cluster"]
    rec = (cer(corr) - cer(fin)) / dmg if dmg > 0 else float("nan")
    if hi > 0 and b["delta_cer"] > MIN_EFFECT and lo > 0:
        cls = "harmful keep" if cer(fin) > cer(corr) else "not repaired"
    elif rec >= 0.8 and hi < 0:
        cls = "repaired"
    elif rec >= 0.2:
        cls = "partly"
    else:
        cls = "not repaired"
    return {"cer_corrupted": cer(corr), "cer_final": cer(fin), "damage_pts": dmg * 100, "recovery": rec, "delta": b["delta_cer"], "ci95_cluster": [lo, hi],
            "class": cls, "counts": dmg >= COUNT}


def sessions():
    """(pid, arm, per_line of final or None if unchanged, kept)"""
    for r in sorted((REPO / "runs").glob("*-trocr-planted-*-b*-1")):
        pid, arm = re.search(r"planted-(.+)-(ad|an|f|b|c|d)-b\d+-1$", r.name).groups()
        s = json.loads((r / "summary.json").read_text())
        yield pid, arm, r, s


def main():
    orig = HB["unplanted subject (p14-null)"]["per_line"]
    out = {}
    for pid, arm, r, s in sessions():
        corr = HB.get(f"{pid} corrupted", {}).get("per_line")
        if corr is None:
            continue
        if s["final_digest"] == s["start_digest"]:
            fin = corr
        elif arm in ("c", "d"):
            fin = HB[f"{pid} arm {arm} final"]["per_line"]
        else:
            h = s.get("heldout_by_set", {}).get("blb-eval-hand") or s.get("heldout")
            if not h:
                continue
            fin = h["per_line"]
        out[f"{pid}|{arm}"] = {**classify(corr, fin, orig), "passes": s["used"], "kept": s["kept"], "changed": s["final_digest"] != s["start_digest"]}
    (REPO / "docs/trocr/stage3_classes.json").write_text(json.dumps(out, indent=1))
    L = ["| defect | damage (pts) | counts | arm | passes/kept | final CER | recovery | interval of final − corrupted (pages) | class |", "|---|---:|---|---|---|---:|---:|---|---|"]
    for k, v in sorted(out.items()):
        pid, arm = k.split("|")
        L.append(f"| {pid} | {v['damage_pts']:.2f} | {'yes' if v['counts'] else 'no'} | {arm} | {v['passes']}/{v['kept']} | {v['cer_final']*100:.2f} | "
                 f"{'' if v['recovery'] != v['recovery'] else format(v['recovery']*100, '.0f') + ' %'} | [{v['ci95_cluster'][0]*100:+.2f}, {v['ci95_cluster'][1]*100:+.2f}] | {v['class']} |")
    (REPO / "docs/trocr/stage3_classes.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

"""Score models on data/heldout/bern-eval.jsonl (CPU): CER, collapsed lines, paired bootstrap vs hertwig-v2.

usage: seed_null_eval.py --runs seed1=path ... --out docs/seed-null/heldout.json
Reports all 545 lines and the 287 Hertwig lines. Scored once per model; no selection on this set.
"""
import argparse, json
from pathlib import Path
from wsp.data import load
from wsp.kraken_subject import KrakenSubject
from wsp.metrics import score, per_line_edits, bootstrap_delta

M = Path.home() / "Repo/weightspace/models/kraken"
FIXED = {"parent": M / "kraken-cursive_inzigkofen.mlmodel", "v1": M / "kraken-bern-hertwig-v1.mlmodel",
         "v2": M / "kraken-bern-hertwig-v2.mlmodel"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--no-fixed", action="store_true")
    a = ap.parse_args()
    lines = load("data/heldout/bern-eval.jsonl")
    refs = [x["text"] for x in lines]
    hw = [i for i, x in enumerate(lines) if "Hertwig" in x["key"]]
    paths = {} if a.no_fixed else dict(FIXED)
    for s in a.runs:
        k, _, p = s.partition("="); paths[k] = Path(p)
    res, pl, hyps_all = {}, {}, {}
    for k, p in paths.items():
        s = KrakenSubject(p, device="cpu")
        hyps = [x.text for x in s.forward(lines)]
        hyps_all[k] = hyps
        pl[k] = per_line_edits(hyps, refs)
        res[k] = {"all": score(hyps, refs).as_dict(),
                  "hertwig": score([hyps[i] for i in hw], [refs[i] for i in hw]).as_dict(),
                  "schluss": score([h for i, h in enumerate(hyps) if i not in set(hw)], [r for i, r in enumerate(refs) if i not in set(hw)]).as_dict()}
        print(k, round(res[k]["all"]["cer"], 4), res[k]["all"]["collapsed"], round(res[k]["hertwig"]["cer"], 4), res[k]["hertwig"]["collapsed"], flush=True)
    paired = {}
    for k in paths:
        if k != "v2" and "v2" in pl:
            paired[f"{k} vs v2"] = {"all": bootstrap_delta(pl["v2"], pl[k]),
                                    "hertwig": bootstrap_delta([pl["v2"][i] for i in hw], [pl[k][i] for i in hw])}
    ks = [k for k in paths if k.startswith("seed")]
    for i, x in enumerate(ks):
        for y in ks[i + 1:]:
            paired[f"{y} vs {x}"] = {"all": bootstrap_delta(pl[x], pl[y]), "hertwig": bootstrap_delta([pl[x][j] for j in hw], [pl[y][j] for j in hw])}
    Path(a.out).write_text(json.dumps({"set": "bern-eval", "models": {k: str(p) for k, p in paths.items()}, "results": res, "paired": paired}, indent=1))
    Path(a.out).with_suffix(".hyp.json").write_text(json.dumps({"keys": [x["key"] for x in lines], "refs": refs, "hyps": hyps_all}, ensure_ascii=False))


main()

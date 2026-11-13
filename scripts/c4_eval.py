"""Score the C4 models, combined-v1 and the interpolation edit on bern-eval (held-out) and bern-val (validation), CPU, once each.

usage: c4_eval.py --out docs/c4/results.json   (models found under ~/atr-cache/wsp/c4/{A,B}_seed*/best_*.mlmodel)
Paired Δ CER against combined-v1 and against the interpolation edit; line-level and page-clustered 95 % intervals.
"""
import argparse, glob, json, re, sys
from pathlib import Path
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from wsp.data import load
from wsp.kraken_subject import KrakenSubject
from wsp.metrics import score, per_line_edits, bootstrap_delta, page_of
import sham_alpha_combined as SA

W = Path.home() / "atr-cache/wsp/c4"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    sets = {"eval": load("data/heldout/bern-eval.jsonl"), "val": load("data/practice/bern-val.jsonl")}
    models = {"combined-v1": SA.build({"id": "base", "kind": "base"}),
              "interp": SA.build({"id": "alpha0.25+blank", "kind": "alpha", "alpha": 0.25, "blank": True})}
    paths = {}
    for p in sorted(glob.glob(str(W / "[AB]_seed*/best_*.mlmodel"))):
        paths[Path(p).parent.name] = p
    for k, p in paths.items():
        models[k] = KrakenSubject(Path(p), device="cpu")
    res, pl, hyps = {}, {}, {}
    for name, S in models.items():
        res[name] = {}
        for sn, lines in sets.items():
            refs = [x["text"] for x in lines]
            h = [x.text for x in S.forward(lines)]
            hyps[(name, sn)] = h
            pl[(name, sn)] = per_line_edits(h, refs)
            hw = [i for i, x in enumerate(lines) if "Hertwig" in x["key"]]; sc = [i for i in range(len(lines)) if i not in set(hw)]
            res[name][sn] = {"all": score(h, refs).as_dict(), "hertwig": score([h[i] for i in hw], [refs[i] for i in hw]).as_dict(),
                             "schluss": score([h[i] for i in sc], [refs[i] for i in sc]).as_dict()}
        print(name, round(res[name]["eval"]["all"]["cer"] * 100, 3), res[name]["eval"]["all"]["collapsed"], round(res[name]["val"]["all"]["cer"] * 100, 3), flush=True)
    paired = {}
    for sn, lines in sets.items():
        cl = [page_of(x["key"]) for x in lines]
        for name in models:
            for ref in ("combined-v1", "interp"):
                if name != ref and not (ref == "interp" and name == "combined-v1"):
                    paired[f"{name} vs {ref} [{sn}]"] = bootstrap_delta(pl[(ref, sn)], pl[(name, sn)], clusters=cl)
    Path(a.out).write_text(json.dumps({"models": {k: paths.get(k, "built") for k in models}, "results": res, "paired": paired}, indent=1))
    Path(a.out).with_suffix(".hyp.json").write_text(json.dumps({sn: {"keys": [x["key"] for x in sets[sn]], "refs": [x["text"] for x in sets[sn]],
        "hyps": {n: hyps[(n, sn)] for n in models}} for sn in sets}, ensure_ascii=False))


main()

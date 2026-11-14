"""Score the C4-control models (uncorrected 25.10 pairs) on bern-eval and bern-val, CPU, once each; pair against combined-v1 and against the C4 arms.

usage: c4u_eval.py --out docs/c4u/results.json   (models under ~/atr-cache/wsp/c4u/{UA,UB}_seed*/best_*.mlmodel)
combined-v1 is re-scored here; the C4 runs are NOT re-run: their hypotheses are read from docs/c4/results.hyp.json (combined-v1's are checked to equal the re-scored ones).
"""
import argparse, glob, json, sys
from pathlib import Path
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from wsp.data import load
from wsp.kraken_subject import KrakenSubject
from wsp.metrics import score, per_line_edits, bootstrap_delta, page_of
import sham_alpha_combined as SA

W = Path.home() / "atr-cache/wsp/c4u"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    sets = {"eval": load("data/heldout/bern-eval.jsonl"), "val": load("data/practice/bern-val.jsonl")}
    C4 = json.load(open("docs/c4/results.hyp.json"))
    models = {"combined-v1": SA.build({"id": "base", "kind": "base"})}
    paths = {Path(p).parent.name: p for p in sorted(glob.glob(str(W / "U[AB]_seed*/best_*.mlmodel")))}
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
    for sn in sets:
        assert C4[sn]["hyps"]["combined-v1"] == hyps[("combined-v1", sn)], f"combined-v1 hypotheses differ from C4's ({sn})"
        assert C4[sn]["keys"] == [x["key"] for x in sets[sn]]
    paired = {}
    for sn, lines in sets.items():
        refs = [x["text"] for x in lines]; cl = [page_of(x["key"]) for x in lines]
        for name in models:
            if name != "combined-v1":
                paired[f"{name} vs combined-v1 [{sn}]"] = bootstrap_delta(pl[("combined-v1", sn)], pl[(name, sn)], clusters=cl)
                c4 = "A" if name.startswith("UA") else "B"; c4name = f"{c4}_seed{name[-1]}"
                paired[f"{name} vs {c4name} [{sn}]"] = bootstrap_delta(per_line_edits(C4[sn]["hyps"][c4name], refs), pl[(name, sn)], clusters=cl)
    Path(a.out).write_text(json.dumps({"models": {k: paths.get(k, "built") for k in models}, "results": res, "paired": paired}, indent=1))
    Path(a.out).with_suffix(".hyp.json").write_text(json.dumps({sn: {"keys": [x["key"] for x in sets[sn]], "refs": [x["text"] for x in sets[sn]],
        "hyps": {n: hyps[(n, sn)] for n in models}} for sn in sets}, ensure_ascii=False))


main()

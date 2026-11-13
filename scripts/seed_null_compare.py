"""Weight readings of the seed-variance null: every pair of runs, run vs parent, run vs hertwig-v2, v1 vs v2.

usage: seed_null_compare.py --runs seed1=path seed2=path ... --out docs/seed-null/readings.json
Uses wsp.zoo.pair_reading (the readings of docs/READINGS.md and the criteria of docs/ZOO_SURVEY.md).
"""
import argparse, itertools, json
from pathlib import Path
from wsp.kraken_subject import KrakenSubject
from wsp.zoo import pair_reading

M = Path.home() / "Repo/weightspace/models/kraken"
FIXED = {"parent": M / "kraken-cursive_inzigkofen.mlmodel", "v1": M / "kraken-bern-hertwig-v1.mlmodel",
         "v2": M / "kraken-bern-hertwig-v2.mlmodel"}


def summarize(r):
    pt, b = r["per_tensor"], r["blocks"]
    blk = {}
    for n, e in pt.items():
        k = n.split(".")[0]
        if k[0] in "CL":
            blk.setdefault(k, []).append(0.0 if e["zero"] else e["rel"])
    o = r["obs1"]; n = o["special_rows"]
    crit = [o["special_norm_up"] > n / 2, o["special_bias_down"] > n / 2, o["special_median_rel"] > o["frequent_median_rel"],
            o["top_decile_special"] / o["top_decile_rows"] >= o["special_share_of_rows"]]
    bl = r["blank"]
    sm = {"blocks": {k: sum(v) / len(v) for k, v in blk.items()},
          "conv_pooled": b["conv_w"], "lstm_pooled": b["lstm_w"], "out_rows_pooled": b["out_rows"], "out_row_mean": b["out_row_mean"],
          "conv_first_block": b["conv_by_block"], "blank": bl, "obs1": o, "criteria_abcd": crit, "all_four": all(crit),
          "codec": r["codec"]}
    sm["conv_range"] = [min(v for k, v in sm["blocks"].items() if k[0] == "C"), max(v for k, v in sm["blocks"].items() if k[0] == "C")]
    sm["lstm_range"] = [min(v for k, v in sm["blocks"].items() if k[0] == "L"), max(v for k, v in sm["blocks"].items() if k[0] == "L")]
    sm["top_rows"] = o["top_rows"]
    return sm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    paths = dict(FIXED)
    runs = []
    for s in a.runs:
        k, _, p = s.partition("=")
        paths[k] = Path(p); runs.append(k)
    subj = {k: KrakenSubject(p) for k, p in paths.items()}
    pairs = []
    pairs += [(r, "parent") for r in runs]                       # parent -> run direction: (parent, run)
    pairs += [("v2", r) for r in runs]
    pairs += list(itertools.combinations(runs, 2))
    pairs += [("v1", "v2"), ("parent", "v1"), ("parent", "v2")] + [("v1", r) for r in runs]
    out = {}
    for x, y in pairs:
        if y == "parent":
            x, y = "parent", x
        out[f"{x}->{y}"] = summarize(pair_reading(subj[x], subj[y]))
    Path(a.out).write_text(json.dumps({"models": {k: str(p) for k, p in paths.items()}, "pairs": out}, indent=1))
    for k, v in out.items():
        b = v["blank"]
        print(f"{k:20s} conv {v['conv_range'][0]:.2f}-{v['conv_range'][1]:.2f} lstm {v['lstm_range'][0]:.2f}-{v['lstm_range'][1]:.2f} "
              f"rows {v['out_row_mean']:.3f} blankΔraw {b['delta_raw']:+.3f} Δrel {b['delta_rel']:+.3f} abcd {''.join('✓' if c else '✗' for c in v['criteria_abcd'])}")


main()

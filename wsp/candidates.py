"""Compare exchange candidates with their unedited model on held-out sets and apply a pre-registered rule.

    python -m wsp.candidates latin_incunabula

Reads only stored readings (``docs/baselines/readings/<set>/<model>.jsonl`` for the unedited model,
``runs/<run>/readings/heldout_<set>.jsonl`` for a candidate), so nothing is scored a second time.
Rule and sets: ``runs/20261107-latin_incunabula-exchange-2/preregistration.md`` (R9, plan ``latin_incunabula``);
``runs/20261108-latin_incunabula-exchange-3/preregistration.md`` (R12, plan ``latin_incunabula-v3``: minimum
effects, page-clustered intervals). Every interval comes in two kinds: line level (``ci95``) and page-clustered
(``ci95_cluster`` / ``ci95_share_cluster``, pages = ``wsp.metrics.page_of`` of the line key).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from wsp.blocks import MUFI, bootstrap_blocks, per_line_counts
from wsp.metrics import bootstrap_delta, page_of, per_line_edits, score

REPO = Path(__file__).resolve().parent.parent
ABBREV = [MUFI, "Combining Diacritical Marks", "Latin Extended-D"]

PLANS = {
    "latin_incunabula": {
        "unedited": "reichenau_lat_cat_099218",
        "candidates": {"v1": "runs/20261107-latin_incunabula-exchange-1", "v2": "runs/20261107-latin_incunabula-exchange-2"},
        "own": "reichenau-heldout",
        "outside": ["newseye-heldout", "gt-fraktur-heldout"],
        "strata": {"reichenau-heldout": Path.home() / "atr-cache/wsp/heldout/reichenau-train-heldout/pages.parquet"},
    },
    # R12: the rule of runs/20261108-latin_incunabula-exchange-3/preregistration.md (numbers in CER units)
    "latin_incunabula-v3": {
        "unedited": "reichenau_lat_cat_099218",
        "candidates": {"v3": "runs/20261108-latin_incunabula-exchange-3"},
        "own": "reichenau-heldout",
        "outside": ["newseye-heldout", "gt-fraktur-heldout"],
        "strata": {"reichenau-heldout": Path.home() / "atr-cache/wsp/heldout/reichenau-train-heldout/pages.parquet"},
        "rule": "r12",
    },
}


def _read(path: Path):
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r["key"] for r in rows], [r["hyp"] for r in rows], [r["ref"] for r in rows]


def _strata(parquet: Path) -> dict[str, str]:
    """page name -> sub-collection, from the shard's source_path."""
    import pyarrow.parquet as pq
    t = pq.read_table(parquet, columns=["filename", "source_path"]).to_pylist()
    return {r["filename"]: ("Latin-German" if "Lateinisch-Deutsch" in r["source_path"] else "German") for r in t}


def compare(keys, hb, hc, refs) -> dict:
    cl = [page_of(k) for k in keys]
    tot = bootstrap_delta(per_line_edits(hb, refs), per_line_edits(hc, refs), clusters=cl)
    blk = bootstrap_blocks(per_line_counts(hb, refs), per_line_counts(hc, refs), clusters=cl)
    return {"lines": len(keys), "cer_unedited": score(hb, refs, blocks=False).cer, "cer_candidate": score(hc, refs, blocks=False).cer,
            "total": tot, "abbrev": {g: blk[g] for g in ABBREV if g in blk},
            "blocks": {g: v for g, v in blk.items() if v["ref"] >= 50 or max(v["errors_before"], v["errors_after"]) >= 20}}


def run(name: str) -> dict:
    plan = PLANS[name]
    sets = [plan["own"]] + plan["outside"]
    out = {"plan": {k: (str(v) if isinstance(v, Path) else v) for k, v in plan.items() if k != "strata"}, "candidates": {}}
    for cid, run_dir in plan["candidates"].items():
        res = {}
        for s in sets:
            kb, hb, rb = _read(REPO / "docs/baselines/readings" / s / f"{plan['unedited']}.jsonl")
            kc, hc, rc = _read(REPO / run_dir / "readings" / f"heldout_{s}.jsonl")
            if kb != kc or rb != rc:
                raise SystemExit(f"{cid} {s}: readings are not on the same lines")
            res[s] = compare(kb, hb, hc, rb)
            if s in plan["strata"]:
                st = _strata(plan["strata"][s])
                res[s]["strata"] = {}
                for label in sorted(set(st.values())):
                    idx = [i for i, k in enumerate(kb) if st.get(k.split("/")[1]) == label]
                    sel = lambda xs: [xs[i] for i in idx]
                    res[s]["strata"][label] = compare(sel(kb), sel(hb), sel(hc), sel(rb))
        if plan.get("rule") == "r12":
            out["candidates"][cid] = {"run": run_dir, "sets": res, **decide_r12(res, plan)}
            continue
        loses = []
        for scope, r in [("all", res[plan["own"]])] + [(k, v) for k, v in res[plan["own"]].get("strata", {}).items()]:
            for g, v in r["abbrev"].items():
                if v["ci95_share"][0] > 0:
                    loses.append(f"{scope}: {g} +{100 * v['delta_share']:.3f} pt [{100 * v['ci95_share'][0]:+.3f}, {100 * v['ci95_share'][1]:+.3f}]")
        own_worse = res[plan["own"]]["total"]["ci95"][0] > 0
        outside_worse = [s for s in plan["outside"] if res[s]["total"]["ci95"][0] > 0]
        gain = sum(res[s]["total"]["delta_cer"] for s in plan["outside"]) / len(plan["outside"])
        out["candidates"][cid] = {"run": run_dir, "sets": res, "loses_abbreviations": loses, "own_worse": own_worse,
                                  "outside_worse": outside_worse, "mean_outside_delta": gain,
                                  "qualifies": not loses and not own_worse and not outside_worse}
    q = [c for c, v in out["candidates"].items() if v["qualifies"]]
    out["choice"] = min(q, key=lambda c: out["candidates"][c]["mean_outside_delta"]) if q else None
    return out


def decide_r12(res: dict, plan: dict) -> dict:
    """The rule of R12 (preregistration of exchange-3), applied to one candidate's comparisons."""
    R = R12
    own = res[plan["own"]]
    loses = []
    for scope, r in [("all", own)] + list(own.get("strata", {}).items()):
        for g, v in r["abbrev"].items():
            if v["ci95_share"][0] > R["abbrev_min"]:
                loses.append(f"{scope}: {g} {100 * v['delta_share']:+.3f} pt [{100 * v['ci95_share'][0]:+.3f}, {100 * v['ci95_share'][1]:+.3f}]")
    own_worse = own["total"]["ci95"][0] > R["own_min"]
    outside_worse = [s for s in plan["outside"] if res[s]["total"]["ci95"][0] > 0]
    outside_better = [s for s in plan["outside"] if res[s]["total"]["ci95_cluster"][1] < 0]
    gain = sum(res[s]["total"]["delta_cer"] for s in plan["outside"]) / len(plan["outside"])
    return {"loses_abbreviations": loses, "own_worse": own_worse, "outside_worse": outside_worse,
            "outside_better": outside_better, "mean_outside_delta": gain, "rule": R,
            "qualifies": not loses and not own_worse and not outside_worse and bool(outside_better)}


#: R12's minimum effects (CER units; 0.0002 = 0.02 points), fixed in the preregistration before any v3 score.
R12 = {"abbrev_min": 0.0002, "own_min": 0.0005}


def render(res: dict) -> str:
    plan = res["plan"]; sets = [plan["own"]] + plan["outside"]; r12 = plan.get("rule") == "r12"
    L = ["| candidate | " + " | ".join(f"`{s}` CER (Δ, 95 % interval)" for s in sets) + " | loses abbreviations | qualifies |",
         "|---|" + "---|" * len(sets) + "---|---|"]
    first = next(iter(res["candidates"].values()))
    L.append("| unedited | " + " | ".join(f"{100 * first['sets'][s]['cer_unedited']:.2f} %" for s in sets) + " | | |")
    for cid, v in res["candidates"].items():
        cells = []
        for s in sets:
            t = v["sets"][s]["total"]; lo, hi = t["ci95"]; b = "**" if lo > 0 or hi < 0 else ""
            pc = (f"; pages [{100 * t['ci95_cluster'][0]:+.2f}, {100 * t['ci95_cluster'][1]:+.2f}]"
                  if r12 and "ci95_cluster" in t else "")
            cells.append(f"{100 * v['sets'][s]['cer_candidate']:.2f} % ({b}{100 * t['delta_cer']:+.2f}{b} [{100 * lo:+.2f}, {100 * hi:+.2f}]{pc})")
        L.append(f"| {cid} | " + " | ".join(cells) + f" | {'yes' if v['loses_abbreviations'] else 'no'} | {'**yes**' if v['qualifies'] else 'no'} |")
    L += ["", "Abbreviation blocks on `" + plan["own"] + "`, change of the block's share of CER in points (95 % interval):", "",
          "| candidate | scope | " + " | ".join(ABBREV) + " |", "|---|---|" + "---|" * len(ABBREV)]
    for cid, v in res["candidates"].items():
        own = v["sets"][plan["own"]]
        for scope, r in [("all", own)] + list(own.get("strata", {}).items()):
            cells = []
            for g in ABBREV:
                b = r["abbrev"].get(g)
                if b is None:
                    cells.append(""); continue
                bold = "**" if b["excludes_0"] else ""
                pc = (f"; pages [{100 * b['ci95_share_cluster'][0]:+.3f}, {100 * b['ci95_share_cluster'][1]:+.3f}]"
                      if r12 and "ci95_share_cluster" in b else "")
                cells.append(f"{bold}{100 * b['delta_share']:+.3f}{bold} [{100 * b['ci95_share'][0]:+.3f}, {100 * b['ci95_share'][1]:+.3f}]{pc} "
                             f"({b['errors_before']}→{b['errors_after']})")
            L.append(f"| {cid} | {scope} ({r['lines']} lines) | " + " | ".join(cells) + " |")
    L += ["", f"**Choice by the pre-registered rule:** {res['choice'] or 'none'}."]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wsp.candidates"); ap.add_argument("name", choices=list(PLANS))
    a = ap.parse_args(argv)
    res = run(a.name)
    out = REPO / "docs" / "family" / f"candidates-{a.name}.json"
    out.write_text(json.dumps(res, indent=1, default=float))
    md = render(res); (out.with_suffix(".md")).write_text(md)
    print(md)


if __name__ == "__main__":
    main()

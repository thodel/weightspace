"""E1 cross-vendor: paired comparisons for the H1 that Gemini and GPT both kept (docs/XVENDOR_PLAN.md §5).

Held-out (733 test pages): H1 against the unedited v2 (docs/baselines/p5-v2-test.json) and against
Fable's H1 (runs/20261110-kurrent-page-v2-f1-b10-1, o_proj only). Practice (64 pages): H1 and each
C5 sham (docs/baselines/xv-e1-sham-*.json) against C0, and H1 against each sham. Paired page
bootstrap and the document-clustered interval (amendment A4), as scripts/p5f_compare.py.

    python scripts/xv_e1_compare.py   # writes docs/baselines/xv-e1-compare.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wsp.metrics import bootstrap_delta  # noqa: E402

G1 = Path("runs/20261119-kurrent-page-v2-g1-b10-1")
F1 = Path("runs/20261110-kurrent-page-v2-f1-b10-1")
B = Path("docs/baselines")
TEST = Path("/mnt/wbkolleg_dh_1/Textrecognition_Training/eval_sets/german-kurrent-xvi-xix-v1/vlm-pages-test/pages.jsonl")
doc = lambda k: k.split("_")[1]  # noqa: E731


def pair(before: dict, after: dict) -> dict:
    keys = list(before)
    b = [tuple(before[k]) for k in keys]
    a = [tuple(after[k]) for k in keys]
    r = bootstrap_delta(b, a, clusters=[doc(k) for k in keys])
    r = {k: (float(v) if hasattr(v, "dtype") else [float(x) for x in v] if isinstance(v, (list, tuple)) else v)
         for k, v in r.items()}
    r["documents"] = len({doc(k) for k in keys})
    r["pages_better"] = sum(x[0] < y[0] for x, y in zip(a, b))
    r["pages_worse"] = sum(x[0] > y[0] for x, y in zip(a, b))
    return r


def heldout_pages(summary: Path) -> dict:
    s = json.loads(summary.read_text())["heldout"]
    keys = [Path(json.loads(l)["image"]).stem for l in TEST.open()]
    return {k: (p["edits"], p["chars"]) if isinstance(p, dict) else tuple(p[:2]) for k, p in zip(keys, s["per_page"])}, s


out = {}
h1, s = heldout_pages(G1 / "summary.json")
fable, _ = heldout_pages(F1 / "summary.json")
v2 = json.loads((B / "p5-v2-test.json").read_text())["per_page"]
v2 = {k: tuple(v[:2]) if isinstance(v, list) else (v["edits"], v["chars"]) for k, v in v2.items()}
out["heldout"] = {"H1": {"trainer": s["trainer"], "wsp": s["wsp"], "final_sha256": s["final_sha256"]},
                  "H1_minus_v2": pair(v2, h1), "H1_minus_fable_H1": pair(fable, h1),
                  "fable_H1_minus_v2": pair(v2, fable)}

st = json.loads((G1 / "state.json").read_text())
keys = [json.loads(l)["key"] for l in open("data/practice/kurrent-val-64.jsonl", encoding="utf-8")]
c0 = dict(zip(keys, map(tuple, st["baseline_lines"])))
kept = dict(zip(keys, map(tuple, st["kept_lines"])))
out["practice"] = {"H1_minus_C0": pair(c0, kept)}
for f in sorted(B.glob("xv-e1-sham-*.json")):
    seed = f.stem.rsplit("-", 1)[1]
    d = json.loads(f.read_text())
    sham = {k: tuple(v[:2]) for k, v in d["per_page"].items()}
    out["practice"][f"sham_{seed}"] = {"wsp": d["wsp"], "minus_C0": pair(c0, sham), "H1_minus_it": pair(sham, kept)}

(B / "xv-e1-compare.json").write_text(json.dumps(out, indent=1))
p = lambda r: f"{r['delta_cer']*100:+.2f} [{r['ci95'][0]*100:+.2f}, {r['ci95'][1]*100:+.2f}] doc [{r['ci95_cluster'][0]*100:+.2f}, {r['ci95_cluster'][1]*100:+.2f}] better/worse {r['pages_better']}/{r['pages_worse']}"  # noqa: E731
for k in ("H1_minus_v2", "H1_minus_fable_H1", "fable_H1_minus_v2"):
    print("heldout", k, p(out["heldout"][k]))
print("practice H1_minus_C0", p(out["practice"]["H1_minus_C0"]))
for k, v in out["practice"].items():
    if k.startswith("sham"):
        print("practice", k, "-C0", p(v["minus_C0"]), "| H1-it", p(v["H1_minus_it"]))

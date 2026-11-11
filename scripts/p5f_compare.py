"""Phase 5, Fable replicate: paired comparisons for docs/PHASE5_FABLE_REPORT.md.

Held-out: the frozen session (summary.json) against v2 unedited on the same card and engine
(docs/baselines/p5-v2-test.json); paired page bootstrap (as phase 5) and, per amendment A4,
a two-stage bootstrap clustered by document (key field 2; 70 documents).
Practice: H1 (pass 2) against C0 and against each sham seed (p5-sham, p5f-sham-*), same two intervals.
Writes docs/baselines/p5f-compare.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wsp.metrics import bootstrap_delta  # noqa: E402
from wsp.pages import page_edits  # noqa: E402

RUN = Path("runs/20261110-kurrent-page-v2-f1-b10-1")
B = Path("docs/baselines")
doc = lambda k: k.split("_")[1]


def pair(before: dict, after: dict, cap_b=None, cap_a=None) -> dict:
    keys = list(before)
    b = [tuple(before[k]) for k in keys]; a = [tuple(after[k]) for k in keys]
    r = bootstrap_delta(b, a, clusters=[doc(k) for k in keys])
    r["documents"] = len({doc(k) for k in keys})
    r["pages_better"] = sum(x[0] < y[0] for x, y in zip(a, b))
    r["pages_worse"] = sum(x[0] > y[0] for x, y in zip(a, b))
    if cap_b is not None:
        r.update(cap_both=len(cap_a & cap_b), cap_only_after=len(cap_a - cap_b), cap_only_before=len(cap_b - cap_a))
    return r


out = {}
# held-out
s = json.loads((RUN / "summary.json").read_text())["heldout"]
rows = [json.loads(l) for l in open("/mnt/wbkolleg_dh_1/Textrecognition_Training/eval_sets/german-kurrent-xvi-xix-v1/vlm-pages-test/pages.jsonl")]
keys = [Path(r["image"]).stem for r in rows]
pp = s["per_page"]
h1 = {k: (p["edits"], p["chars"]) if isinstance(p, dict) else tuple(p[:2]) for k, p in zip(keys, pp)}
v2 = json.loads((B / "p5-v2-test.json").read_text())["per_page"]
v2 = {k: tuple(v[:2]) if isinstance(v, list) else (v["edits"], v["chars"]) for k, v in v2.items()}
out["heldout_H1_minus_v2"] = pair(v2, h1)
# practice
prac = [json.loads(l) for l in open("data/practice/kurrent-val-64.jsonl", encoding="utf-8")]
refs = {p["key"]: p["text"] for p in prac}
rd = {json.loads(l)["key"]: json.loads(l)["text"] for l in open(RUN / "readings/pass_002.jsonl", encoding="utf-8")}
pk = [p["key"] for p in prac]
H1 = dict(zip(pk, page_edits([rd[k] for k in pk], [refs[k] for k in pk])))
c0 = {k: tuple(v) for k, v in json.loads((B / "c0-kurrent-val-v2.json").read_text())["practice_per_page"].items()}
out["practice_H1_minus_C0"] = pair(c0, H1)
for f in ["p5-sham.json", "p5f-sham-20261110.json", "p5f-sham-20261111.json"]:
    sh = json.loads((B / f).read_text())["per_page"]
    sh = {k: tuple(v[:2]) if isinstance(v, list) else (v["edits"], v["chars"]) for k, v in sh.items()}
    out[f"practice_H1_minus_{f[:-5]}"] = pair(sh, H1)
    out[f"practice_{f[:-5]}_minus_C0"] = pair(c0, sh)
(B / "p5f-compare.json").write_text(json.dumps(out, indent=1))
for k, v in out.items():
    print(f"{k:40s} {v['delta_cer']*100:+6.2f}  page [{v['ci95'][0]*100:+.1f}, {v['ci95'][1]*100:+.1f}]  doc [{v['ci95_cluster'][0]*100:+.1f}, {v['ci95_cluster'][1]*100:+.1f}]  better {v['pages_better']} worse {v['pages_worse']}")

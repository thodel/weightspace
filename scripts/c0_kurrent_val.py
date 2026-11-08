"""C0 of phase 5: the v2 adapter as it is, on all 509 registered val pages, outside any budget.

Through the trainer's own evaluate_qlora (the pass engine since G1, docs/PHASE5_PLAN.md §8),
in chunks of 32 pages so that card 0 is checked between chunks (wsp.card: free memory, and
no trainer job waiting, #396). A chunk whose report exists is skipped, so the script resumes.
At the end: wsp.pages metrics over all 509 and over the 64 practice pages, to
docs/baselines/c0-kurrent-val-v2.json.

    python scripts/c0_kurrent_val.py [--adapter <dir>] [--label v2] [--keys <practice keys file>]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wsp import card, pages  # noqa: E402

SIDE = Path("/mnt/wbkolleg_dh_1/Textrecognition_Training/eval_sets/german-kurrent-xvi-xix-v1/vlm-pages-val")
WORK = Path.home() / "atr-cache/wsp/passes"

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", default=str(Path.home() / "atr-cache/wsp/adapters/qwen3.5-4b-german-kurrent-xvi-xix-page-v2"))
ap.add_argument("--label", default="v2")
ap.add_argument("--chunk", type=int, default=32)
ap.add_argument("--keys", help="only these pages (e.g. the practice set): control runs")
a = ap.parse_args()

rows = [json.loads(l) for l in (SIDE / "pages.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
if a.keys:
    want = {k.strip() for k in Path(a.keys).read_text().splitlines() if k.strip()}
    rows = [r for r in rows if Path(r["image"]).stem in want]
out = WORK / f"c0-{a.label}"; out.mkdir(parents=True, exist_ok=True)
for i in range(0, len(rows), a.chunk):
    stem = out / f"chunk-{i // a.chunk:03d}"
    if stem.with_suffix(".raw.jsonl").exists():
        continue
    while True:
        ok, why = card.ready()
        if ok:
            break
        print(time.strftime("%H:%M"), "waiting:", why, flush=True)
        time.sleep(300)
    draw = stem.with_suffix(".draw.jsonl")
    draw.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows[i:i + a.chunk]), encoding="utf-8")
    cmd = [sys.executable, "-m", "wsp.vlm_reference", "--pages", str(draw), "--data-root", str(SIDE),
           "--report", str(stem.with_suffix(".json"))] + (["--adapter", a.adapter] if a.adapter != "none" else [])
    t = time.time()
    with open(stem.with_suffix(".log"), "w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(time.strftime("%H:%M"), stem.name, f"{len(rows[i:i + a.chunk])} pages in {time.time() - t:.0f}s", flush=True)

raw = {}
for f in sorted(out.glob("chunk-*.raw.jsonl")):
    raw.update(pages.read_reference(f))
keys = [Path(r["image"]).stem for r in rows]
practice = [json.loads(l)["key"] for l in Path("data/practice/kurrent-val-64.jsonl").read_text(encoding="utf-8").splitlines()]


def summary(ks):
    s = pages.score([raw[k]["text"] for k in ks], [next(r["text"] for r in rows if Path(r["image"]).stem == k) for k in ks],
                    [raw[k]["at_cap"] for k in ks])
    return {k: v for k, v in s.items() if k != "per_page"}, s["per_page"]


res = {"adapter": a.adapter, "engine": "evaluate_qlora (transformers 5.17, greedy, cap 3072)", "side": str(SIDE)}
res["all"], per_all = summary(keys)
pk = [k for k in practice if k in raw]
if pk:
    res["practice"], per_prac = summary(pk)
    res["practice_per_page"] = dict(zip(pk, per_prac))
res["at_cap_pages"] = sorted(k for k in keys if raw[k]["at_cap"])
res["loop_pages"] = sorted(k for k in keys if pages.page_loop(raw[k]["text"], next(r["text"] for r in rows if Path(r["image"]).stem == k)))
dst = Path(f"docs/baselines/c0-kurrent-val-{a.label}.json")
dst.write_text(json.dumps(res, indent=1, ensure_ascii=False))
print(json.dumps({k: v for k, v in res.items() if k in ("all", "practice")}, indent=1))

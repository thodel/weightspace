"""Stage A0: the peft-wrapped adapter and the merged weights give the same text (10 pages).

One transformers process (the trainer's interpreter, executed only), base loaded once, three runs
on the same 10 practice pages: ``peft`` (LoRA layers beside the frozen weights, as
``evaluate_qlora`` does), ``merged`` (``merge_and_unload`` in bf16) and ``merged32`` (ΔW added in
float32, cast to bf16 once). The peft run is also compared with the C0 reading of the same pages
by ``evaluate_qlora`` itself. Card 0 only; refuses to start unless 14 GB are free.

    python scripts/a0_equivalence.py [--n 10] [--adapter DIR]
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from wsp import card, pages, vlm_reference  # noqa: E402

SIDE = Path("/mnt/wbkolleg_dh_1/Textrecognition_Training/eval_sets/german-kurrent-xvi-xix-v1/vlm-pages-val")
WORK = Path.home() / "atr-cache/wsp/passes/a0"
ADAPTER = Path.home() / "atr-cache/wsp/adapters/qwen3.5-4b-german-kurrent-xvi-xix-page-v2"
TRAINER = vlm_reference.TRAINER

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--adapter", default=str(ADAPTER))
a = ap.parse_args()

rows = {Path(r["image"]).stem: r for r in map(json.loads, (SIDE / "pages.jsonl").read_text(encoding="utf-8").splitlines()) if r}
practice = [json.loads(l)["key"] for l in (REPO / "data/practice/kurrent-val-64.jsonl").read_text(encoding="utf-8").splitlines()]
c0 = {}
for f in sorted((WORK.parent / "c0-v2").glob("chunk-*.raw.jsonl")):
    c0.update(pages.read_reference(f))
have = [k for k in practice if k in c0]
have.sort(key=lambda k: len(rows[k]["text"]))
step = max(1, len(have) // a.n)
pick = have[step // 2::step][:a.n]  # spread over the length range, the longest pages included last
WORK.mkdir(parents=True, exist_ok=True)
draw = WORK / "equiv.draw.jsonl"
draw.write_text("".join(json.dumps(rows[k], ensure_ascii=False) + "\n" for k in pick), encoding="utf-8")
argv = vlm_reference.command(a.adapter, draw, SIDE, WORK / "equiv.report.json", 3072)
(WORK / "argv.json").write_text(json.dumps(argv))
jobs = [{"adapter": a.adapter, "mode": m, "pages": str(draw), "data_root": str(SIDE), "out": str(WORK / f"equiv-{m}.raw.jsonl")}
        for m in ("peft", "merged", "merged32")]
(WORK / "jobs.json").write_text(json.dumps(jobs))
ok, why = card.ready(14000)
if not ok:
    sys.exit(f"card 0 not ready: {why}")
env = {**os.environ, "PYTHONPATH": f"{TRAINER}/src:{TRAINER}/engines", "CUDA_VISIBLE_DEVICES": "0"}
with open(WORK / "equiv.log", "w") as log:
    r = subprocess.run([str(vlm_reference.PY), str(REPO / "wsp/hf_pass.py"), "--argv", str(WORK / "argv.json"),
                        "--jobs", str(WORK / "jobs.json")], env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(TRAINER))
print("hf_pass exit", r.returncode)

ref = {k: c0[k] for k in pick}
res = {"pages": pick, "chars_ref": {k: len(rows[k]["text"]) for k in pick}, "adapter": a.adapter}
read = {m: pages.read_reference(WORK / f"equiv-{m}.raw.jsonl") for m in ("peft", "merged", "merged32")}
res["peft_vs_c0"] = pages.agreement(ref, read["peft"])
res["peft_vs_merged"] = pages.agreement(read["peft"], read["merged"])
res["peft_vs_merged32"] = pages.agreement(read["peft"], read["merged32"])
res["merged_vs_merged32"] = pages.agreement(read["merged"], read["merged32"])
for m in read:
    res[f"meta_{m}"] = json.loads((WORK / f"equiv-{m}.raw.meta.json").read_text()) if (WORK / f"equiv-{m}.raw.meta.json").exists() else None
(REPO / "docs/baselines/a0-equivalence-v2.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
print(json.dumps({k: {x: y for x, y in v.items() if x in ("pages", "exact", "cer_between", "at_cap_same")} for k, v in res.items()
                  if isinstance(v, dict) and "exact" in v}, indent=1))

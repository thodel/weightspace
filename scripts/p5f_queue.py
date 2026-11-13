"""Phase 5, Fable replicate (runs/20261110-kurrent-page-v2-f1-b10-1), after the freeze.

One queue on card 0, each step resumable (wsp.run.score_adapter_on_draw):

  1. sham    - C5 for the kept edit (direction 0 of layers.31.self_attn.o_proj out of B):
               a seeded random unit direction removed from the same B, two new seeds
               (20261110, 20261111; phase 5 ran seed 20261108, docs/baselines/p5-sham.json)
  2. heldout - the frozen session on the 733 test pages: wsp.run heldout, once

C1 (base), C3 (alpha sweep) and the decode comparator do not depend on the edit; they are a
function of (weights, pages, engine) and are reused from phase 5 (docs/baselines/p5-*.json).

    python scripts/p5f_queue.py [--only sham,heldout]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wsp.lora_subject import LoraSubject  # noqa: E402
from wsp.run import KURRENT, heldout, load, score_adapter_on_draw  # noqa: E402

RUN = Path("runs/20261110-kurrent-page-v2-f1-b10-1")
V2 = Path(json.loads((RUN / "session.json").read_text())["subject"])
VAL = KURRENT / "vlm-pages-val"
WORK = Path.home() / "atr-cache/wsp/p5f"
O = "base_model.model.model.language_model.layers.31.self_attn.o_proj"
practice_keys = [json.loads(l)["key"] for l in Path("data/practice/kurrent-val-64.jsonl").read_text(encoding="utf-8").splitlines()]
val_rows = {Path(r["image"]).stem: r for r in load(VAL / "pages.jsonl")}
PRACTICE = [val_rows[k] for k in practice_keys]
SEEDS = (20261110, 20261111)


def variant(name, edit):
    d = WORK / "adapters" / name
    if not (d / "adapter_model.safetensors").exists():
        s = LoraSubject(V2)
        edit(s)
        s.save(d)
    return d


def practice_run(label, adapter):
    res = score_adapter_on_draw(adapter, VAL, WORK / label, rows=PRACTICE)
    res["per_page"] = dict(zip(res.pop("keys"), res["per_page"]))
    res["adapter"] = str(adapter)
    Path(f"docs/baselines/p5f-{label}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(label, {k: round(v, 4) if isinstance(v, float) else v for k, v in res["wsp"].items()}, flush=True)


def sham(seed):
    def edit(s):
        rng = np.random.default_rng(seed)
        d = rng.normal(size=s.tensors()[f"{O}.lora_B.weight"].shape[0])
        s.remove_direction(O, d / np.linalg.norm(d))
    return edit


STEPS = {
    "sham": lambda: [practice_run(f"sham-{k}", variant(f"sham-{k}", sham(k))) for k in SEEDS],
    "heldout": lambda: heldout(str(RUN), "kurrent-test"),
}

ap = argparse.ArgumentParser()
ap.add_argument("--only")
a = ap.parse_args()
for name in (a.only.split(",") if a.only else STEPS):
    print("==", name, flush=True)
    STEPS[name]()

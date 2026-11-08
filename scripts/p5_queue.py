"""Phase 5, after the freeze: controls on the practice pages and the paired held-out score.

One queue on card 0, in order, each step resumable (chunks of 32 pages, card checked
before each, wsp.run.score_adapter_on_draw):

  1. sham    - a seeded random unit direction removed from layers.31.self_attn.o_proj's B (C5)
  2. heldout - the frozen session (H1) on the 733 test pages: wsp.run heldout, once
  3. v2-test - v2 unedited on the same 733 pages, same card and engine: the paired comparison
  4. base    - Qwen3.5-4B without the adapter (C1)
  5. alpha   - the whole adapter at 0.75, 0.9, 1.1 (C3)
  6. rp      - v2 unedited, repetition_penalty 1.05 (decode comparator)

Results: docs/baselines/p5-<step>.json; the held-out goes into the session's summary.json.

    python scripts/p5_queue.py [--only sham,heldout,...]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wsp.lora_subject import LoraSubject  # noqa: E402
from wsp.run import KURRENT, heldout, load, score_adapter_on_draw  # noqa: E402

RUN = Path("runs/20261107-kurrent-page-v2-b10-1")
V2 = Path(json.loads((RUN / "session.json").read_text())["subject"])
VAL, TEST = KURRENT / "vlm-pages-val", KURRENT / "vlm-pages-test"
WORK = Path.home() / "atr-cache/wsp/p5"
O = "base_model.model.model.language_model.layers.31.self_attn.o_proj"
practice_keys = [json.loads(l)["key"] for l in Path("data/practice/kurrent-val-64.jsonl").read_text(encoding="utf-8").splitlines()]
val_rows = {Path(r["image"]).stem: r for r in load(VAL / "pages.jsonl")}
PRACTICE = [val_rows[k] for k in practice_keys]


def variant(name, edit):
    d = WORK / "adapters" / name
    if not (d / "adapter_model.safetensors").exists():
        s = LoraSubject(V2)
        edit(s)
        s.save(d)
    return d


def practice_run(label, adapter, extra=()):
    res = score_adapter_on_draw(adapter, VAL, WORK / label, rows=PRACTICE, extra_args=extra)
    res["per_page"] = dict(zip(res.pop("keys"), res["per_page"]))
    res["adapter"] = str(adapter)
    Path(f"docs/baselines/p5-{label}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(label, {k: round(v, 4) if isinstance(v, float) else v for k, v in res["wsp"].items()}, flush=True)


def sham(s):
    rng = np.random.default_rng(20261108)
    d = rng.normal(size=s.tensors()[f"{O}.lora_B.weight"].shape[0])
    s.remove_direction(O, d / np.linalg.norm(d))


STEPS = {
    "sham": lambda: practice_run("sham", variant("sham", sham)),
    "heldout": lambda: heldout(str(RUN), "kurrent-test"),
    "v2-test": lambda: _v2_test(),
    "base": lambda: practice_run("base", None),
    "alpha": lambda: [practice_run(f"alpha-{a}", variant(f"alpha-{a}", lambda s, a=a: [s.scale_module(m, a) for m in s.modules()]))
                      for a in (0.75, 0.9, 1.1)],
    "rp": lambda: practice_run("rp-1.05", V2, extra=("--repetition-penalty", "1.05")),
}


def _v2_test():
    res = score_adapter_on_draw(V2, TEST, WORK / "v2-test")
    res["per_page"] = dict(zip(res.pop("keys"), res["per_page"]))
    Path("docs/baselines/p5-v2-test.json").write_text(json.dumps(res, indent=1))
    print("v2-test", res["trainer"], flush=True)


ap = argparse.ArgumentParser()
ap.add_argument("--only")
a = ap.parse_args()
for name in (a.only.split(",") if a.only else STEPS):
    print("==", name, flush=True)
    STEPS[name]()

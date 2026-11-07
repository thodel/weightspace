"""A3 baselines: the clean adapter, a repeat of the clean one (noise floor) and the five corrupted adapters,
on the practice pass (32 pages) and the score set (64 val pages), through a running wsp.vllm_server.

    python scripts/a3_baselines.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wsp import plant_adapter as P  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
CLEAN = Path.home() / "atr-cache/wsp/adapters/qwen3.5-4b-german-kurrent-xvi-xix-page-v2"
SETS = {"p32": REPO / "data/practice/kurrent-val-32.jsonl", "h64": REPO / "data/practice/kurrent-val-holdout-64.jsonl"}
out = {}
jobs = [("clean", CLEAN, "p32"), ("clean", CLEAN, "h64")] + \
       [(c, P.OUT / c, s) for c in ("p1-scale3", "p3-alpha2", "p2-zeroB", "p4-randdir", "p5-alpha05") for s in ("p32", "h64")]
for name, d, s in jobs:
    r = P.score(d, SETS[s])
    out[f"{name}/{s}"] = r["metrics"] | {"wall_s": r["wall_s"], "tokens_mean": r["tokens_mean"]}
    print(name, s, {k: round(v, 4) if isinstance(v, float) else v for k, v in out[f"{name}/{s}"].items()}, flush=True)
    (REPO / "docs/baselines/a3-baselines.json").write_text(json.dumps(out, indent=1))

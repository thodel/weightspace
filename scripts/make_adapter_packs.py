"""Reading packs for the planted-adapter programmers (A3): statistics of the corrupted adapter against two siblings, and its C0 on the practice pass."""
import json, sys
from pathlib import Path
from wsp import plant_adapter as P
from wsp.adapter_pack import build

REPO = Path(__file__).resolve().parent.parent
SIB = {"v1": Path.home() / "atr-cache/wsp/adapters/qwen3.5-4b-german-kurrent-xvi-xix-page-v1",
       "blb1": Path("/mnt/wbkolleg_dh_1/Textrecognition_Training/training_folder/trained/qwen3.5-4b-german-kurrent-xvi-xix-blb-page-v1")}
for cid in sys.argv[1:]:
    r = P.score(P.OUT / cid, REPO / "data/practice/kurrent-val-32.jsonl")
    c0 = {"what": "C0: this adapter as it is, 32 practice pages, greedy, cap 3072 (batched engine, not exactly repeatable)", "metrics": r["metrics"],
          "per_page_cer": [round(e / c, 3) for e, c in r["per_page"]]}
    d = REPO / "runs/planted-adapter" / cid / "pack"; d.mkdir(parents=True, exist_ok=True)
    (d / "pack.md").write_text(build(P.OUT / cid, SIB, c0, REPO / "docs/readings/adapter_compare.json"))
    for arm in "ab":
        (REPO / "runs/planted-adapter" / cid / arm).mkdir(exist_ok=True)
    print(cid, "pack written")

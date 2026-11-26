"""E3 (docs/XVENDOR_PLAN.md §7.1): run the adapter planted-defect cell q01 (p3-alpha2) written by a cross-vendor programmer.

    PYTHONPATH=. .venv/bin/python scripts/run_planted_adapter_xv.py g1|o1 a|b

scripts/run_planted_adapter_f1.py with another programmer: same subject, practice pass, budget, keep rule, score set and outcome
classes; the program is the one the vendor's model wrote through wsp.programmer under runs/planted-xv/<tag>/q01/<arm>/; run id
planted-adapter-p3-alpha2-<arm>-<tag>; the pass server's spool is ~/atr-cache/wsp/spool/a3-xv (needs a running wsp.vllm_server).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_planted_adapter_f1 as F  # noqa: E402
from run_planted_xv import OPERATOR, programmer_text  # noqa: E402

SPOOL = Path.home() / "atr-cache/wsp/spool/a3-xv"


def main(tag: str, arm: str):
    import json
    cell = F.REPO / "runs/planted-xv" / tag / "q01"
    label = json.loads((cell / arm / "programmer.json").read_text())["label"]
    text = programmer_text(cell / arm, arm) + " " + OPERATOR
    return F.main("p3-alpha2", arm, "q01", arm_root=cell, tag=tag, who=label, programmer={arm: text}, spool=SPOOL)


if __name__ == "__main__":
    main(*sys.argv[1:3])

"""E3 (docs/XVENDOR_PLAN.md §2): run one kraken planted-defect cell written by a cross-vendor programmer.

    .venv/bin/python scripts/run_planted_xv.py g1|o1 k1-zero-row-e a|b

scripts/run_planted_f1.py with another programmer: same subject, practice set, budget, keep rule, preregistration,
outcome classes and held-out scoring; the program is the one the vendor's model wrote through wsp.programmer under
runs/planted-xv/<tag>/<sNN>/<arm>/ (scripts/xv_planted_prepare.py; same opaque labels as the Fable rerun); run id
planted-<cid>-<arm>-<tag>.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_planted_f1 as F  # noqa: E402

REPO = F.REPO
OPERATOR = ("Operator: a Claude Opus 5.5 (claude-opus-5-5) session; it knows the defects and the Sonnet and Fable results, set "
            "up the runs and does the held-out scoring and the write-up; it wrote no hypothesis and no edit and passed nothing "
            "of this to the programmers.")


def programmer_text(cell: Path, arm: str) -> str:
    cfg = json.loads((cell / "programmer.json").read_text())
    u = json.loads((cell / "programmer" / "usage.json").read_text())
    vendor = "Google" if cfg["provider"] == "gemini" else "OpenAI"
    return (f"{cfg['label']} (`{cfg['model']}`), called through the {vendor} API by `wsp.programmer` (docs/XVENDOR_PLAN.md), "
            f"one fresh conversation per run, provider default reasoning settings, tools read_file/list_dir/write_file "
            f"(py_compile) only; read only `{cfg['readable'][0]}`, its pack (`{cfg['readable'][1]}`) and the API files; not told "
            f"which defect was planted; never shown the Sonnet or Fable runs, their programs or results "
            f"({u['calls']} API calls, {u['input']} input / {u['output']} output tokens). "
            + ("Wrote its program in advance, did not run it." if arm == "a" else "Budget 0."))


def main(tag: str, cid: str, arm: str):
    root = REPO / "runs/planted-xv" / tag
    s = {v: k for k, v in F.MAP.items()}[cid]
    label = json.loads((root / s / arm / "programmer.json").read_text())["label"]
    return F.main(cid, arm, arm_root=root, tag=tag, who=label,
                  programmer={arm: programmer_text(root / s / arm, arm)}, operator=OPERATOR)


if __name__ == "__main__":
    main(*sys.argv[1:4])

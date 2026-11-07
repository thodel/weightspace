"""Per-epoch validation accuracy, wall time and GPU memory of every seed run -> docs/seed-null/runs.json"""
import json, re, glob, os
from pathlib import Path
W = Path.home() / "atr-cache/wsp/seed-null"
out = {}
for d in sorted(W.glob("seed[0-9]*"), key=lambda p: int(p.name[4:])):
    if not (d / "wall.txt").exists():
        continue
    t = (d / "train.log").read_text(errors="ignore").replace("\r", "\n")
    vals = [float(x) for x in re.findall(r"val_accuracy:\s*\n\s*([0-9.]+)", t)]
    best = glob.glob(str(d / "best_*.mlmodel"))[0]
    mem = [int(x) for x in (d / "gpu_mem.log").read_text().split()]
    out[d.name] = {"seed": int(d.name[4:]), "epochs": len(vals), "val_accuracy_per_epoch": vals,
                   "best_file": os.path.basename(best), "best_val": float(re.search(r"best_([0-9.]+)\.mlmodel", best).group(1)),
                   "best_epoch": int(re.search(r"checkpoint_(\d+)-", " ".join(os.listdir(d)) if False else
                                    [f for f in os.listdir(d) if f.endswith(".ckpt") and re.search(r"-" + re.search(r"best_([0-9.]+)", best).group(1), f)][0]).group(1)) + 1,
                   "last_val": vals[-1], "wall_seconds": int(re.search(r"\d+", (d / "wall.txt").read_text()).group()),
                   "card0_used_mib_peak": max(mem), "card0_free_before_start": (d / "gpu_precheck.txt").read_text().strip()}
Path("docs/seed-null/runs.json").write_text(json.dumps(out, indent=1))
for k, v in out.items():
    print(k, v["epochs"], v["best_val"], v["best_epoch"], v["last_val"], v["wall_seconds"], v["card0_used_mib_peak"])

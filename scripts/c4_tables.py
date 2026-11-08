"""docs/c4/runs.json and docs/c4/tables.md from the run directories and results.json."""
import glob, json, os, re, statistics as st
from pathlib import Path
W = Path.home() / "atr-cache/wsp/c4"; D = Path("docs/c4")
R = json.load(open(D / "results.json"))
runs = {}
for d in sorted(W.glob("[AB]_seed*")):
    t = (d / "train.log").read_text(errors="ignore").replace("\r", "\n")
    vals = [float(x) for x in re.findall(r"val_accuracy:\s*\n\s*([0-9.]+)", t)]
    best = glob.glob(str(d / "best_*.mlmodel"))[0]; bv = re.search(r"best_([0-9]\.[0-9]+?)\.mlmodel", best).group(1)
    ck = [f for f in os.listdir(d) if f.endswith(".ckpt") and f"-{bv}" in f][0]
    mem = [int(x) for x in (d / "gpu_mem.log").read_text().split()]
    runs[d.name] = {"epochs": len(vals), "val_accuracy_per_epoch": vals, "best_val": float(bv), "best_epoch": int(re.search(r"checkpoint_(\d+)", ck).group(1)) + 1,
                    "wall_seconds": int(re.search(r"\d+", (d / "wall.txt").read_text()).group()), "card0_used_mib_peak": max(mem),
                    "card0_free_before": int(re.search(r": (\d+)", (d / "gpu_precheck.txt").read_text()).group(1))}
json.dump(runs, open(D / "runs.json", "w"), indent=1)
o = []; P = o.append
res, pr = R["results"], R["paired"]
def f(x): return f"{x*100:.2f}"
def d(p): return f"{p['delta_cer']*100:+.2f} [{p['ci95'][0]*100:+.2f}, {p['ci95'][1]*100:+.2f}] / [{p['ci95_cluster'][0]*100:+.2f}, {p['ci95_cluster'][1]*100:+.2f}]"
P("### T1. Runs\n\n| run | lrate | seed | epochs | val. acc. per epoch (1–5) | kept (best val.) | wall | card 0 free before | card 0 used, peak (all tenants) |\n|---|---:|---:|---:|---|---|---:|---:|---:|")
for k, r in runs.items():
    P(f"| {k} | {'1e-5' if k[0]=='A' else '5e-5'} | {k[-1]} | {r['epochs']} | {' '.join(f'{v:.3f}' for v in r['val_accuracy_per_epoch'])} | {r['best_val']:.4f} (epoch {r['best_epoch']}) | {r['wall_seconds']//60} min {r['wall_seconds']%60} s | {r['card0_free_before']/1024:.1f} GB | {r['card0_used_mib_peak']/1024:.1f} GB |")
P("\n### T2. Held-out `bern-eval` (545 lines, 14 pages): each model scored once, CPU\n\nΔ in points of CER; interval = line-level / page-clustered 95 %.\n")
P("| model | CER all | CER Hertwig | CER Schluss. | collapsed | WER all | Δ vs combined-v1 | Δ vs interpolation edit | practice `bern-val` CER | collapsed (val) |\n|---|---:|---:|---:|---:|---:|---|---|---:|---:|")
for k, v in res.items():
    e, vv = v["eval"], v["val"]
    a = pr.get(f"{k} vs combined-v1 [eval]"); b = pr.get(f"{k} vs interp [eval]")
    P(f"| {k} | {f(e['all']['cer'])} | {f(e['hertwig']['cer'])} | {f(e['schluss']['cer'])} | {e['all']['collapsed']} | {f(e['all']['wer'])} | {d(a) if a else '—'} | {d(b) if b else '—'} | {f(vv['all']['cer'])} | {vv['all']['collapsed']} |")
base = res["combined-v1"]["eval"]["all"]["cer"]; sub = {}
P("\n### T3. Arms: mean over three seeds (points)\n\n| arm | mean Δ vs combined-v1 (all) | per seed | Hertwig Δ | Schluss. Δ | mean collapsed | mean practice CER | reading rule (A.4) |\n|---|---:|---|---:|---:|---:|---:|---|")
for arm in "AB":
    ks = [f"{arm}_seed{i}" for i in (1, 2, 3)]
    dl = [(res[k]["eval"]["all"]["cer"] - base) * 100 for k in ks]; m = st.mean(dl)
    dh = st.mean((res[k]["eval"]["hertwig"]["cer"] - res["combined-v1"]["eval"]["hertwig"]["cer"]) * 100 for k in ks)
    ds = st.mean((res[k]["eval"]["schluss"]["cer"] - res["combined-v1"]["eval"]["schluss"]["cer"]) * 100 for k in ks)
    det = "detectable gain" if m <= -0.20 and all(x < 0 for x in dl) else ("loss" if m >= 0.20 else "not detectable")
    size = "larger than" if m <= -0.57 else ("equal to" if m <= -0.21 else "smaller than")
    P(f"| {arm} ({'1e-5' if arm=='A' else '5e-5'}) | {m:+.2f} (sd {st.stdev(dl):.2f}) | {' '.join(f'{x:+.2f}' for x in dl)} | {dh:+.2f} | {ds:+.2f} | {st.mean(res[k]['eval']['all']['collapsed'] for k in ks):.1f} | {st.mean(res[k]['val']['all']['cer'] for k in ks)*100:.2f} | {det}; {size} the interpolation edit (−0.39) |")
P(f"\ncombined-v1: held-out {f(base)} %, collapsed {res['combined-v1']['eval']['all']['collapsed']}, practice {f(res['combined-v1']['val']['all']['cer'])} %. Interpolation edit rebuilt and re-scored: {f(res['interp']['eval']['all']['cer'])} % (Δ {(res['interp']['eval']['all']['cer']-base)*100:+.2f}), collapsed {res['interp']['eval']['all']['collapsed']}, practice {f(res['interp']['val']['all']['cer'])} %.")
(D / "tables.md").write_text("\n".join(o) + "\n"); print("\n".join(o))

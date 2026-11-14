"""docs/c4u/runs.json and docs/c4u/tables.md from the run directories, docs/c4u/results.json and the C4 results (docs/c4/results.json)."""
import glob, json, os, re, statistics as st
from pathlib import Path
W = Path.home() / "atr-cache/wsp/c4u"; D = Path("docs/c4u")
R = json.load(open(D / "results.json")); C = json.load(open("docs/c4/results.json"))
runs = {}
for d in sorted(W.glob("U[AB]_seed*")):
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
res, pr, cres = R["results"], R["paired"], C["results"]
lr = lambda k: "1e-5" if k[1] == "A" else "5e-5"
def d(p): return f"{p['delta_cer']*100:+.2f} [{p['ci95'][0]*100:+.2f}, {p['ci95'][1]*100:+.2f}] / [{p['ci95_cluster'][0]*100:+.2f}, {p['ci95_cluster'][1]*100:+.2f}]"
P("### T1. Runs\n\n| run | lrate | seed | epochs | val. acc. per epoch (1–5) | kept (best val.) | wall | card 0 free before | card 0 used, peak (all tenants) |\n|---|---:|---:|---:|---|---|---:|---:|---:|")
for k, r in runs.items():
    P(f"| {k} | {lr(k)} | {k[-1]} | {r['epochs']} | {' '.join(f'{v:.3f}' for v in r['val_accuracy_per_epoch'])} | {r['best_val']:.4f} (epoch {r['best_epoch']}) | {r['wall_seconds']//60} min {r['wall_seconds']%60} s | {r['card0_free_before']/1024:.1f} GB | {r['card0_used_mib_peak']/1024:.1f} GB |")
P("\n### T2. Held-out `bern-eval` (545 lines, 14 pages): each model scored once, CPU\n\nΔ in points of CER; interval = line-level / page-clustered 95 %. The C4 rows are copied from `docs/c4/results.json`.\n")
P("| model | CER all | CER Hertwig | CER Schluss. | collapsed | Δ vs combined-v1 | Δ vs C4 arm, same seed | practice `bern-val` CER |\n|---|---:|---:|---:|---:|---|---|---:|")
def row(k, r, dv, dc):
    e = r[k]["eval"]; return f"| {k} | {e['all']['cer']*100:.2f} | {e['hertwig']['cer']*100:.2f} | {e['schluss']['cer']*100:.2f} | {e['all']['collapsed']} | {dv} | {dc} | {r[k]['val']['all']['cer']*100:.2f} |"
P(row("combined-v1", res, "—", "—"))
for k in [k for k in res if k != "combined-v1"]:
    c4 = f"{'A' if k[1]=='A' else 'B'}_seed{k[-1]}"
    P(row(k, res, d(pr[f"{k} vs combined-v1 [eval]"]), d(pr[f"{k} vs {c4} [eval]"])))
for k in [k for k in cres if k[0] in "AB" and "_seed" in k]:
    P(row(k, cres, d(C["paired"][f"{k} vs combined-v1 [eval]"]), "—"))
P("\n### T3. Arms: mean over three seeds (points) and the reading rule (A.3 of `docs/C4_UNCORRECTED.md`)\n\n| arm | mean Δ (all) | per seed | Hertwig Δ | Schluss. Δ | C4 arm mean Δ (all) | Δ_U − Δ_C | mean collapsed | mean practice CER | rule |\n|---|---:|---|---:|---:|---:|---:|---:|---:|---|")
base = res["combined-v1"]["eval"]
for arm, carm in (("UA", "A"), ("UB", "B")):
    ks = [f"{arm}_seed{i}" for i in (1, 2, 3)]; cs = [f"{carm}_seed{i}" for i in (1, 2, 3)]
    if not all(k in res for k in ks): continue
    dl = [(res[k]["eval"]["all"]["cer"] - base["all"]["cer"]) * 100 for k in ks]; cl = [(cres[k]["eval"]["all"]["cer"] - cres["combined-v1"]["eval"]["all"]["cer"]) * 100 for k in cs]
    dh = st.mean((res[k]["eval"]["hertwig"]["cer"] - base["hertwig"]["cer"]) * 100 for k in ks); ds = st.mean((res[k]["eval"]["schluss"]["cer"] - base["schluss"]["cer"]) * 100 for k in ks)
    mu, mc = st.mean(dl), st.mean(cl); gap = mu - mc
    if gap >= 0.20 and min(dl) > max(cl): rule = "1: the correction is what helps"
    elif mu <= -0.20 and abs(gap) < 0.20: rule = "2: extra training explains the gain"
    elif gap <= -0.20: rule = "3: wrong images help more than correct ones"
    elif mu >= 0.20: rule = "4: wrong images hurt"
    else: rule = "5: inconclusive"
    if mu >= 0.20 and not rule.startswith("4"): rule += " (and 4: wrong images hurt)"
    P(f"| {arm} ({'1e-5' if arm=='UA' else '5e-5'}) | {mu:+.2f} (sd {st.stdev(dl):.2f}) | {' '.join(f'{x:+.2f}' for x in dl)} | {dh:+.2f} | {ds:+.2f} | {mc:+.2f} | {gap:+.2f} | {st.mean(res[k]['eval']['all']['collapsed'] for k in ks):.1f} | {st.mean(res[k]['val']['all']['cer'] for k in ks)*100:.2f} | {rule} |")
(D / "tables.md").write_text("\n".join(o) + "\n"); print("\n".join(o))

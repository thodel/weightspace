"""docs/seed-null/tables.md from readings.json, heldout.json, runs.json."""
import json, statistics as st
from pathlib import Path
D = Path("docs/seed-null")
R = json.load(open(D / "readings.json"))["pairs"]; H = json.load(open(D / "heldout.json")); RUNS = json.load(open(D / "runs.json"))
seeds = [f"seed{i}" for i in range(1, 6)]
o = []
P = o.append
P("### T1. Runs\n\n| run | seed | epochs | best val. acc. (epoch) | last val. acc. | wall time | card 0 used, peak (all tenants) |\n|---|---:|---:|---|---:|---:|---:|")
for s in seeds:
    r = RUNS[s]; P(f"| {s} | {r['seed']} | {r['epochs']} | {r['best_val']:.4f} ({r['best_epoch']}) | {r['last_val']:.3f} | {r['wall_seconds']/60:.0f} min | {r['card0_used_mib_peak']/1024:.1f} GB |")

def row(k, label):
    v = R[k]; b = v["blank"]; c = v["obs1"]
    return (f"| {label} | {v['conv_range'][0]:.2f}–{v['conv_range'][1]:.2f} | {v['lstm_range'][0]:.2f}–{v['lstm_range'][1]:.2f} | {v['conv_pooled']:.3f} / {v['lstm_pooled']:.3f} | "
            f"{v['out_row_mean']:.3f} | {b['delta_raw']:+.3f} | {b['delta_rel']:+.3f} | {c['special_median_rel']:.2f} / {c['frequent_median_rel']:.2f} | "
            f"{''.join('✓' if x else '✗' for x in v['criteria_abcd'])} |")
HD = "| pair | conv blocks | LSTM blocks | pooled conv / LSTM | output rows (mean) | blank Δ raw | blank Δ rel. | median move special / frequent | a b c d |\n|---|---|---|---|---|---:|---:|---|---|"
P("\n### T2. Weight readings (first model → second; READINGS.md definitions; blank Δ rel. = change of bias[blank] − mean of the other biases)\n")
P(HD)
for s in seeds: P(row(f"parent->{s}", f"parent → {s}"))
P(row("parent->v1", "parent → v1")); P(row("parent->v2", "parent → v2"))
P(HD.split("\n")[1].replace("pair", "—") if False else "| **v2 vs run** | | | | | | | | |")
for s in seeds: P(row(f"v2->{s}", f"v2 → {s}"))
P("| **run vs run** | | | | | | | | |")
for i, x in enumerate(seeds):
    for y in seeds[i + 1:]: P(row(f"{x}->{y}", f"{x} → {y}"))
P("| **v1 vs v2, v1 vs run** | | | | | | | | |")
P(row("v1->v2", "v1 → v2"))
for s in seeds: P(row(f"v1->{s}", f"v1 → {s}"))

def rng(keys, f):
    v = [f(R[k]) for k in keys]; return min(v), max(v)
rr = [f"{x}->{y}" for i, x in enumerate(seeds) for y in seeds[i + 1:]]
pr = [f"parent->{s}" for s in seeds]; vr = [f"v2->{s}" for s in seeds]
P("\n### T3. Summary of ranges\n\n| quantity | seed vs seed (10 pairs) | v2 vs seed (5) | v1 vs seed (5) | v1 vs v2 | parent → seed (5) | parent → v1 | parent → v2 |\n|---|---|---|---|---|---|---|---|")
v1r = [f"v1->{s}" for s in seeds]
def cell(keys, f, fmt="{:.3f}"):
    a, b = rng(keys, f); return fmt.format(a) + " – " + fmt.format(b)
for name, f, fmt in [("output rows, mean", lambda v: v["out_row_mean"], "{:.3f}"), ("pooled conv", lambda v: v["conv_pooled"], "{:.3f}"), ("pooled LSTM", lambda v: v["lstm_pooled"], "{:.3f}"),
                     ("blank Δ raw", lambda v: v["blank"]["delta_raw"], "{:+.3f}"), ("blank Δ rel.", lambda v: v["blank"]["delta_rel"], "{:+.3f}"),
                     ("median rel. move, special rows", lambda v: v["obs1"]["special_median_rel"], "{:.2f}"), ("median rel. move, frequent rows", lambda v: v["obs1"]["frequent_median_rel"], "{:.2f}"),
                     ("median norm ratio, special rows", lambda v: v["obs1"]["special_median_norm_ratio"], "{:.2f}"), ("median bias Δ, special rows", lambda v: v["obs1"]["special_median_bias_delta"], "{:+.2f}")]:
    P(f"| {name} | {cell(rr, f, fmt)} | {cell(vr, f, fmt)} | {cell(v1r, f, fmt)} | {fmt.format(f(R['v1->v2']))} | {cell(pr, f, fmt)} | {fmt.format(f(R['parent->v1']))} | {fmt.format(f(R['parent->v2']))} |")
for name, idx in [("criterion (a) norm up", 0), ("(b) bias down", 1), ("(c) special moved more", 2), ("(d) top decile", 3)]:
    P(f"| {name} (pairs holding) | {sum(R[k]['criteria_abcd'][idx] for k in rr)} / 10 | {sum(R[k]['criteria_abcd'][idx] for k in vr)} / 5 | {sum(R[k]['criteria_abcd'][idx] for k in v1r)} / 5 | {'✓' if R['v1->v2']['criteria_abcd'][idx] else '✗'} | {sum(R[k]['criteria_abcd'][idx] for k in pr)} / 5 | {'✓' if R['parent->v1']['criteria_abcd'][idx] else '✗'} | {'✓' if R['parent->v2']['criteria_abcd'][idx] else '✗'} |")
P(f"| all four | {sum(R[k]['all_four'] for k in rr)} / 10 | {sum(R[k]['all_four'] for k in vr)} / 5 | {sum(R[k]['all_four'] for k in v1r)} / 5 | {'✓' if R['v1->v2']['all_four'] else '✗'} | {sum(R[k]['all_four'] for k in pr)} / 5 | {'✓' if R['parent->v1']['all_four'] else '✗'} | {'✓' if R['parent->v2']['all_four'] else '✗'} |")
P("\nTop rows by relative move (first of each group): " + "; ".join(f"{k}: " + " ".join(c for c, _ in R[k]["top_rows"][:6]) for k in ("seed1->seed2", "v2->seed1", "parent->seed1", "parent->v2")))
P("\n### T4. Held-out `bern-eval` (545 lines; Hertwig 287; CPU; each model scored once)\n\n| model | CER all | collapsed | CER Hertwig | collapsed (Hertwig) | CER Schlussredaktion | Δ CER vs v2, all [95 % paired] | Δ vs v2, Hertwig [95 % paired] |\n|---|---:|---:|---:|---:|---:|---|---|")
for k, v in H["results"].items():
    p = H["paired"].get(f"{k} vs v2")
    d1 = f"{p['all']['delta_cer']*100:+.2f} [{p['all']['ci95'][0]*100:+.2f}, {p['all']['ci95'][1]*100:+.2f}]" if p else "—"
    d2 = f"{p['hertwig']['delta_cer']*100:+.2f} [{p['hertwig']['ci95'][0]*100:+.2f}, {p['hertwig']['ci95'][1]*100:+.2f}]" if p else "—"
    P(f"| {k} | {v['all']['cer']*100:.2f} | {v['all']['collapsed']} | {v['hertwig']['cer']*100:.2f} | {v['hertwig']['collapsed']} | {v['schluss']['cer']*100:.2f} | {d1} | {d2} |")
cs = [H["results"][s]["all"]["cer"] * 100 for s in seeds]; ch = [H["results"][s]["hertwig"]["cer"] * 100 for s in seeds]
P(f"\nSeeds, all lines: mean {st.mean(cs):.2f}, sd {st.stdev(cs):.2f}, range {min(cs):.2f}–{max(cs):.2f}. Hertwig lines: mean {st.mean(ch):.2f}, sd {st.stdev(ch):.2f}, range {min(ch):.2f}–{max(ch):.2f}.")
P("\n### T5. Seed vs seed on the held-out lines (paired Δ CER, point and 95 % interval, points)\n\n| pair | all | Hertwig |\n|---|---|---|")
for k, p in H["paired"].items():
    if k.startswith("seed"):
        P(f"| {k} | {p['all']['delta_cer']*100:+.2f} [{p['all']['ci95'][0]*100:+.2f}, {p['all']['ci95'][1]*100:+.2f}] | {p['hertwig']['delta_cer']*100:+.2f} [{p['hertwig']['ci95'][0]*100:+.2f}, {p['hertwig']['ci95'][1]*100:+.2f}] |")
(D / "tables.md").write_text("\n".join(o) + "\n")
print("\n".join(o))

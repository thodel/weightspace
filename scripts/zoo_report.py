"""Tables of docs/ZOO_SURVEY.md from docs/zoo/survey.json (every number in the tables is computed here).

    python scripts/zoo_report.py docs/zoo/survey.json > docs/zoo/tables.md
"""
import json
import sys
from collections import Counter

import numpy as np

d = json.load(open(sys.argv[1]))
M = d["models"]
Z = {k: m for k, m in M.items() if m["origin"] == "zoo"}
H = {k: m for k, m in M.items() if m["origin"] == "in-house"}
gname = {}
order = sorted(d["groups"], key=lambda k: -len(d["groups"][k]))
for i, k in enumerate(order):
    gname[k] = "ABCDEFG"[i]
DOC = {c: (p, s) for c, p, s in [(x[0], x[1], x[2]) for x in
       [(p["child"], p["parent"], p["source"]) for p in d["documented_pairs"]]]}
UNAV = {u["child"]: u["note"] for u in d["documented_unavailable"]}


def f(x, n=2, sign=False):
    return f"{x:+.{n}f}" if sign else f"{x:.{n}f}"


out = []
P = out.append

P("### T1. Models, one row each (zoo, then in-house)\n")
P("Group = architecture group of T2. `blank raw` = bias of the CTC blank row; `blank rel` = that bias minus the mean of the model's other biases "
  "(the softmax is unchanged when a constant is added to all biases, so only the relative value can matter); `rank` = place of the blank "
  "by bias among all output rows (1 = highest); `Lat` = share of the codec that is Latin letters.\n")
P("| model | DOI | grp | params | codec | Lat | comb. marks | blank raw | blank rel | rank | blank ‖row‖ / mean other |")
P("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for k, m in sorted(Z.items()) + sorted(H.items()):
    o = m["output"]
    doi = (m.get("doi") or "—").replace("10.5281/zenodo.", "")
    P(f"| `{k.removeprefix('kraken-')}` | {doi} | {gname[m['spec_key']]} | {m['parameters']:,} | {m['codec_size']:,} | "
      f"{m['codec_classes_share']['latin']:.2f} | {m['codec_classes']['combining']} | {f(o['blank_bias'], 2, True)} | "
      f"{f(o['blank_bias_rel'], 2, True)} | {o['blank_rank_by_bias']} of {o['classes']} | {o['blank_norm'] / o['row_norm_mean']:.2f} |")

P("\n### T2. Architecture groups (VGSL with block names, dropout rates and the output width removed)\n")
P("| grp | specification | zoo | in-house | members (zoo) |")
P("|---|---|---:|---:|---|")
for k in order:
    ids = d["groups"][k]
    z = [i for i in ids if i in Z]; h = [i for i in ids if i in H]
    P(f"| {gname[k]} | `{k}` | {len(z)} | {len(h)} | {', '.join(i.removeprefix('kraken-') for i in z) or '—'} |")
A = [k for k in order if gname[k] == "A"][0]
spec_ex = Counter(Z[i]["spec"].split(" Lbx")[1][:0] + ("dropout written `Do.…` form" if "Do." in Z[i]["spec"] else "dropout written `Do0.1,2` form")
                  for i in d["groups"][A] if i in Z)
P(f"\nWithin group A the specification strings differ only in how the dropout layers are written: {dict(spec_ex)}. "
  "Dropout has no parameters, so these are one architecture.\n")

P("### T3. Documented parent → child pairs and what the weights say\n")
P("Distance = ‖a − b‖ / mean(‖a‖, ‖b‖) over all non-output weights (symmetric). Two models whose weights are unrelated sit near √2 = 1.41; "
  "the median over all pairs of group A is "
  f"{np.median([v['rel'] for v in d['distances'][A]['pairs'].values()]):.2f}.\n")
P("| child | documented parent | source of the record | same architecture | distance | rank of parent among child's neighbours | verdict |")
P("|---|---|---|---|---:|---:|---|")
for p in d["documented_pairs"]:
    c, pa = p["child"], p["parent"]
    if p["status"] != "tested":
        P(f"| `{c.removeprefix('kraken-')}` | `{pa.removeprefix('kraken-')}` | {p['source']} | no | — | — | {p['status']}: {p.get('spec_child', '')[:0]}"
          f"child `{M[c]['spec_key'][:34]}…`, parent `{M[pa]['spec_key'][:34]}…` |")
        continue
    key = M[c]["spec_key"]
    ent = d["distances"][key]["pairs"]
    dist = ent.get(f"{c}|{pa}") or ent.get(f"{pa}|{c}")
    ids = d["distances"][key]["ids"]
    others = sorted((ent.get(f"{c}|{o}") or ent.get(f"{o}|{c}"))["rel"] for o in ids if o != c)
    rank = others.index(dist["rel"]) + 1
    verdict = ("weights agree with the record" if dist["rel"] < 1.0 else "**weights do not support the record**: as far from the documented parent as from unrelated models")
    P(f"| `{c.removeprefix('kraken-')}` | `{pa.removeprefix('kraken-')}` | {p['source']} | yes | {dist['rel']:.2f} | {rank} of {len(ids) - 1} | {verdict} |")
P("")
P("Documented in text, but the parent file is not among those surveyed (so nothing is tested):\n")
for u in d["documented_unavailable"]:
    P(f"- `{u['child'].removeprefix('kraken-')}`: {u['note']}")

P("\n### T4. Relations that only the weights suggest (not documented in any text read)\n")
P("| child | parent (direction from the codec) | distance | codec | shared characters at the same row index |")
P("|---|---|---:|---|---:|")
for p in d["inferred_pairs"]:
    key = M[p["child"]]["spec_key"]
    ent = d["distances"][key]["pairs"]
    dist = ent.get(f"{p['child']}|{p['parent']}") or ent.get(f"{p['parent']}|{p['child']}")
    P(f"| `{p['child'].removeprefix('kraken-')}` | `{p['parent'].removeprefix('kraken-')}` | {dist['rel']:.2f} | "
      f"{p['codec_parent']} → {p['codec_child']} (parent's codec is a subset: {p['parent_codec_subset_of_child']}) | {p['same_row_index']} of {p['codec_parent']} |")
docset = {frozenset((p["child"], p["parent"])) for p in d["documented_pairs"]} | {frozenset((p["child"], p["parent"])) for p in d["inferred_pairs"]}
cluster = {"kraken-cursive_inzigkofen", "kraken-catmus_medieval", "kraken-bastarda_inzigkofen", "kraken-old_norse_am305"}
extra = [(a, b, v) for p, v in sorted(d["distances"][A]["pairs"].items(), key=lambda t: t[1]["rel"]) for a, b in [p.split("|")]
         if v["rel"] < 1.2 and frozenset((a, b)) not in docset and not (a in H or b in H) and not (a in cluster and b in cluster)]
P("\nOther pairs below 1.2 in group A that no record explains (distance, no direction): " + "; ".join(
    f"`{a.removeprefix('kraken-')}` – `{b.removeprefix('kraken-')}` {v['rel']:.2f}" for a, b, v in extra) + ".")

# -- observations
tested = [p for p in d["documented_pairs"] if p["status"] == "tested" and p["reading"]["blocks"]["conv_w"] < 1.0]
all_pairs = tested + [dict(p, status="inferred") for p in d["inferred_pairs"]]


def lab(p):
    return f"`{p['parent'].removeprefix('kraken-')}` → `{p['child'].removeprefix('kraken-')}`"


P("\n### T5. Observation 1 — rows of rare characters (proxy: characters outside basic Latin letters, digits, space and ASCII punctuation)\n")
P("`special` = such rows among the characters both codecs share. Frequent = the ten letters `e n i a r t s o u d`. "
  "Medians are over rows. A pair *supports* the observation when all four hold: (a) most special rows have a larger norm in the child, "
  "(b) most special rows have a lower bias, (c) the special rows moved more than the frequent ones (median relative L2), "
  "(d) the top decile of rows by relative move contains at least its share of special rows.\n")
P("| pair | special rows | norm up | bias down | median norm ratio (special / frequent) | median bias Δ (special / frequent) | median rel. move (special / frequent) | top decile special / share | a b c d |")
P("|---|---:|---:|---:|---|---|---|---|---|")
obs1 = []
for p in all_pairs:
    o = p["reading"]["obs1"]; n = o["special_rows"]
    c = [o["special_norm_up"] > n / 2, o["special_bias_down"] > n / 2, None if o["frequent_median_rel"] is None else o["special_median_rel"] > o["frequent_median_rel"],
         o["top_decile_special"] / o["top_decile_rows"] >= o["special_share_of_rows"]]
    obs1.append((p, c))
    tag = " (inferred)" if p["status"] == "inferred" else ""
    fz = lambda x, f="{:.2f}": "–" if x is None else f.format(x)
    P(f"| {lab(p)}{tag} | {n} | {o['special_norm_up']} | {o['special_bias_down']} | {fz(o['special_median_norm_ratio'])} / {fz(o['frequent_median_norm_ratio'])} | "
      f"{fz(o['special_median_bias_delta'], '{:+.2f}')} / {fz(o['frequent_median_bias_delta'], '{:+.2f}')} | {fz(o['special_median_rel'])} / {fz(o['frequent_median_rel'])} | "
      f"{o['top_decile_special']} of {o['top_decile_rows']} / {o['special_share_of_rows']:.2f} | {''.join('–' if x is None else '✓' if x else '✗' for x in c)} |")
rel_obs = [(p, c) for p, c in obs1 if p["status"] == "tested"]
P(f"\nTally over the {len(rel_obs)} documented pairs with related weights: (a) {sum(c[0] for _, c in rel_obs)}, (b) {sum(c[1] for _, c in rel_obs)}, "
  f"(c) {sum(c[2] for _, c in rel_obs)}, (d) {sum(c[3] for _, c in rel_obs)}; all four in {sum(all(x is not False for x in c) for _, c in rel_obs)}. "
  f"Zoo pairs only ({sum(1 for p, _ in rel_obs if not p['child'].startswith(('bern', 'german')))}): "
  f"all four in {sum(all(x is not False for x in c) for p, c in rel_obs if not p['child'].startswith(('bern', 'german')))}.")
fm = [m for m in M.values() if "frequent" in m["output"]]
fz = [m for m in Z.values() if "frequent" in m["output"]]
P(f"\nThe companion claim — the ten frequent letters have the smallest row norms — can be read on every model without a pair: "
  f"mean norm of the ten frequent rows over the mean of all other rows is below 1 in {sum(m['output']['frequent']['ratio'] < 1 for m in fm)} of {len(fm)} models "
  f"(median {np.median([m['output']['frequent']['ratio'] for m in fm]):.2f}; zoo only {sum(m['output']['frequent']['ratio'] < 1 for m in fz)} of {len(fz)}, "
  f"median {np.median([m['output']['frequent']['ratio'] for m in fz]):.2f}); at least five of the ten weakest non-blank rows are frequent letters in "
  f"{sum(m['output']['frequent']['in_weakest10'] >= 5 for m in fm)} of {len(fm)}. "
  f"{len(M) - len(fm)} models (non-Latin codecs) lack one of the ten letters and are not counted.")

P("\n### T6. Observation 2 — the blank bias\n")
P("Δ raw = change of `bias[blank]`; Δ rel = change of (`bias[blank]` − mean of the other biases); `Δ mean other` = change of the mean of the other biases.\n")
P("| pair | Δ raw | Δ mean other | Δ rel | blank row, relative L2 |")
P("|---|---:|---:|---:|---:|")
for p in all_pairs:
    k = p["reading"]["blank"]
    tag = " (inferred)" if p["status"] == "inferred" else ""
    P(f"| {lab(p)}{tag} | {k['delta_raw']:+.3f} | {k['mean_other_delta']:+.3f} | {k['delta_rel']:+.3f} | {k['row_rel']:.2f} |")
P(f"\nTally over the {len(tested)} documented related pairs: Δ raw > 0 in {sum(p['reading']['blank']['delta_raw'] > 0 for p in tested)}, "
  f"Δ rel > 0 in {sum(p['reading']['blank']['delta_rel'] > 0 for p in tested)}; "
  f"the mean of the other biases fell in {sum(p['reading']['blank']['mean_other_delta'] < 0 for p in tested)}.")
P(f"\nWithout pairs, over all {len(M)} models: the blank has the highest bias in {sum(m['output']['blank_rank_by_bias'] == 1 for m in M.values())}, "
  f"blank bias − mean of the others is positive in {sum(m['output']['blank_bias_rel'] > 0 for m in M.values())} "
  f"(raw bias positive in {sum(m['output']['blank_bias'] > 0 for m in M.values())}), and the blank's row norm is below the mean of the other rows in "
  f"{sum(m['output']['blank_norm'] < m['output']['row_norm_mean'] for m in M.values())}. "
  f"Zoo only: {sum(m['output']['blank_rank_by_bias'] == 1 for m in Z.values())} of {len(Z)} with the highest bias, relative value median "
  f"{np.median([m['output']['blank_bias_rel'] for m in Z.values()]):.2f}; in-house {len(H)}: median {np.median([m['output']['blank_bias_rel'] for m in H.values()]):.2f}.")

P("\n### T7. Observation 3 — which stack moves\n")
P("Pooled relative L2 over the **weight** tensors of a stack (biases excluded: their norms are tiny and their ratios large). "
  "`READINGS def.` = the definition of docs/READINGS.md: per-tensor relative L2 averaged per block, biases included, the all-zero `bias_ih` of the LSTMs counted as 0.\n")
P("| pair | conv | BiLSTM | output rows | C_0 / C_3 / C_6 / C_9 (weights) | READINGS def.: conv / BiLSTM |")
P("|---|---:|---:|---:|---|---|")
for p in all_pairs:
    b = p["reading"]["blocks"]; cb = list(b["conv_by_block"].values())
    P(f"| {lab(p)}{' (inferred)' if p['status'] == 'inferred' else ''} | {b['conv_w']:.3f} | {b['lstm_w']:.3f} | {b['out_rows']:.3f} | "
      + " / ".join(f"{x:.2f}" for x in cb) + f" | {b['conv_readings_def']:.3f} / {b['lstm_readings_def']:.3f} |")


def mono(p):
    v = list(p["reading"]["blocks"]["conv_by_block"].values())
    return all(x <= y + 1e-9 for x, y in zip(v, v[1:]))


P(f"\nTally over the {len(tested)} documented related pairs: conv < BiLSTM in {sum(p['reading']['blocks']['conv_w'] < p['reading']['blocks']['lstm_w'] for p in tested)}; "
  f"conv < output rows in {sum(p['reading']['blocks']['conv_w'] < p['reading']['blocks']['out_rows'] for p in tested)}; "
  f"the first conv block moves least of the conv blocks in {sum(min(p['reading']['blocks']['conv_by_block'].values()) == p['reading']['blocks']['conv_by_block']['C_0'] for p in tested)}; "
  f"conv blocks increase monotonically with depth in {sum(mono(p) for p in tested)}. "
  f"With the READINGS definition conv < BiLSTM in {sum(p['reading']['blocks']['conv_readings_def'] < p['reading']['blocks']['lstm_readings_def'] for p in tested)}.")

P("\n### T8. Not fetched or not loaded\n")
for x in d["fetch_failures"]:
    P(f"- `{x['id']}` ({x['doi']}): {x['error']}")
P(f"- Load errors under kraken 7.0.2: {len(d['load_errors'])}" + ("".join(f"; `{e['id']}`: {e['error']}" for e in d["load_errors"])) + ".")

# -- T9: appearance against weights (needs docs/zoo/lineage.json)
import os
lp = os.path.join(os.path.dirname(sys.argv[1]), "lineage.json")
if os.path.exists(lp):
    L = json.load(open(lp))["pairs"]
    pc = {frozenset((p["child"], p["parent"])) for p in d["documented_pairs"] if p["status"] == "tested" and p["reading"]["blocks"]["conv_w"] < 1.0}
    pc |= {frozenset((p["child"], p["parent"])) for p in d["inferred_pairs"]}

    def nm(x):
        return x.removeprefix("kraken-")

    def row(p):
        ix = "—" if p["codec_same_index"] is None else f"{p['codec_same_index']:.2f}"
        return (f"| `{nm(p['a'])}` – `{nm(p['b'])}` | {', '.join(p['shared_name_family']) or '—'} | {'yes' if p['same_architecture'] else 'no'} | "
                f"{p['codec_jaccard']:.2f} | {ix} | {'—' if p['distance'] is None else f'{p['distance']:.2f}'} |")
    head = ["| pair | shared name / project | same architecture | codec overlap | shared characters at same row index | weight distance |", "|---|---|---|---:|---:|---:|"]
    P("\n### T9. What looks related, and what is related\n")
    groups = [
        ("a) Parent → child (documented and tested, or inferred): weights close", [p for p in L if frozenset((p["a"], p["b"])) in pc]),
        ("b) The record names the base and the codec is largely the base's, but the weights are unrelated",
         [p for p in L if {p["a"], p["b"]} == {"kraken-bifrost_old_norse", "kraken-catmus_medieval"}]),
        ("c) Same project or name, same architecture, weights unrelated (distance ≥ 1.2)",
         [p for p in L if p["shared_name_family"] and p["same_architecture"] and p["distance"] is not None and p["distance"] >= 1.2
          and not (p["a"] in H and p["b"] in H)]),
        ("d) Weights related, codec rebuilt (overlap < 0.5)",
         [p for p in L if p["distance"] is not None and p["distance"] < 1.0 and p["codec_jaccard"] < 0.5 and p["a"] not in H and p["b"] not in H]),
        ("e) Siblings of one parent are further from each other than from the parent",
         [p for p in L if {p["a"], p["b"]} in ({"kraken-bastarda_inzigkofen", "kraken-cursive_inzigkofen"},)]),
        ("f) Same name family, different architecture (weights cannot be shared)",
         [p for p in L if p["shared_name_family"] and not p["same_architecture"] and not (p["a"] in H and p["b"] in H)]),
    ]
    for title, rows in groups:
        P(f"**{title}** ({len(rows)} pair{'s' if len(rows) != 1 else ''})\n")
        P("\n".join(head + [row(p) for p in sorted(rows, key=lambda p: (p["distance"] is None, p["distance"] or 9))[:14]]))
        if len(rows) > 14:
            P(f"\n(first 14 of {len(rows)}, sorted by distance)")
        P("")
print("\n".join(out))

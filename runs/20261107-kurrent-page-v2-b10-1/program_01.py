"""Passes 1-2: the in-session baseline (predicted byte-identical to C0), then H1.

Run with the sandbox command (session.json, PROTOCOL §6); only wsp and numpy.
"""
import json
from pathlib import Path

import numpy as np

from wsp.run import open_session

s = open_session(".")
L31_O = "base_model.model.model.language_model.layers.31.self_attn.o_proj"

# Pass 1: baseline. C0's readings of the same 64 pages, same engine, are the prediction.
c0 = {Path(d["image"]).stem: d["raw"] for d in map(json.loads, Path("c0/c0-val-raw.jsonl").read_text(encoding="utf-8").splitlines())}
h0 = s.hypothesis("baseline: the in-session pass reproduces C0 (deterministic engine, G1 repeat)",
                  "byte-identical to C0 on all 64 pages: CER 0.4515, 9 at the cap", "none")
m = s.measure("baseline")
got = {json.loads(l)["key"]: json.loads(l)["text"] for l in Path(f"readings/pass_{s._used:03d}.jsonl").read_text(encoding="utf-8").splitlines()}
same = sum(got[k] == c0[k] for k in got)
s.note(f"baseline vs C0: {same} of {len(got)} pages byte-identical")
print(f"baseline CER {m.cer:.4f}; {same}/{len(got)} pages identical to C0")
if same != len(got):
    s.note("baseline differs from C0: engine or setup changed; session stops before any edit (preregistration §5)")
    raise SystemExit("baseline differs from C0")

# Pass 2: H1, direction 0 of layers.31.self_attn.o_proj out of B.
u, sv, _ = s.subject.directions(L31_O)
h1 = s.hypothesis("H1: direction 0 of layers.31.self_attn.o_proj writes against <|im_end|> (lens z -3.60; "
                  "\\n -5.52, ¬ -8.06); the adapter taught the last layer to keep writing",
                  "pages at the cap 9 -> 5-7; CER -2 to -6 points; truncated_cer within ±1 point; "
                  "length_ratio down 0.03-0.10. Risk: more line breaks/hyphens or new loops", "direction subtraction")
s.edit("project_out", [f"{L31_O}.lora_B.weight"], hypothesis=h1, direction=u[:, 0], axis=1)
m = s.measure("H1")
ok, why = s.keep_allowed()
print(f"H1: CER {m.cer:.4f}; vs kept {s._vs_kept['delta_cer']:+.4f} {s._vs_kept['ci95']}; keep allowed: {ok} {why}")
kept = s.keep("H1 met the rule") if ok else (s.revert(f"H1 refused by the rule: {why}") or False)
print("kept" if kept else "reverted", "; passes left", s.budget_left)

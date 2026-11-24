from wsp.run import open_session
from wsp.trocr_subject import TrocrSubject
import numpy as np

DONOR = "/home/tobias/atr-cache/trocr/dh-unibe__trocr-kurrent-XVI-XVII"

s = open_session(".")
donor = TrocrSubject(DONOR, device="cpu", check_card=False)
D = donor.tensors()
MASK = s.unseen_by("trocr-kurrent-XVI-XVII")
EMB = "dec.embed.tokens"

state = {"cer": None, "collapsed": None, "kept": {}}


def T():
    return s.subject.tensors()


def trial(key, text, pred, cls, op, names, per_tensor=None, **kw):
    """One hypothesis, one edit, one pass, one decision. Returns True if kept."""
    if s.budget_left < 1:
        return False
    h = s.hypothesis(text, pred, cls)
    s.edit(op, names, hypothesis=h, per_tensor=per_tensor, **kw)
    m = s.measure(key, rule_mask=MASK)
    bad = m.collapsed > state["collapsed"] + 5 or m.cer >= state["cer"]
    if bad:
        s.revert(f"{key}: cer {m.cer:.4f} vs {state['cer']:.4f}, collapsed {m.collapsed} vs {state['collapsed']}")
        ok = False
    else:
        ok = s.keep(key)
    if ok:
        state["cer"], state["collapsed"] = m.cer, m.collapsed
    s.note(f"{key}: kept={ok} cer={m.cer:.4f} collapsed={m.collapsed} wer={m.wer:.4f}")
    state["kept"][key] = ok
    return ok


# ---- baseline
m0 = s.measure("C0", rule_mask=MASK)
state["cer"], state["collapsed"] = m0.cer, m0.collapsed

non_emb = [n for n in T() if n in D and n != EMB and T()[n].shape == D[n].shape]
tok = s.subject.tokenizer
vocab = T()[EMB].shape[0]
strs = tok.convert_ids_to_tokens(list(range(vocab)))
space_ids = [i for i, t in enumerate(strs) if isinstance(t, str) and t.startswith("Ġ")]
eos = int(tok.eos_token_id) if tok.eos_token_id is not None else 2

# H1: global interpolation of non-embedding tensors, 30 %
h1 = trial("H1_interp_nonembed_0.3",
           "30 % toward the donor on all non-embedding tensors",
           "CER -0.2 to -0.5 pts, probably about 0", "interpolate global",
           "interpolate", non_emb, per_tensor={n: {"other": D[n]} for n in non_emb}, alpha=0.3)

# H2: word-initial token rows x1.04
w = T()[EMB]
h2 = trial("H2_space_rows_x1.04",
           "scale word-initial token rows by 1.04 (readout favours space)",
           "CER -0.3 to -1.0 pts, fewer deleted spaces", "scale rows",
           "set_rows", [EMB], index=space_ids, values=(w[space_ids] * 1.04).astype(np.float32))

# H3: EOS row x0.9
w = T()[EMB]
h3 = trial("H3_eos_x0.9", "scale EOS row by 0.9", "CER -0.1 to -0.4 pts, length ratio up", "scale row",
           "set_rows", [EMB], index=[eos], values=(w[[eos]] * 0.9).astype(np.float32))

# H4: rows of deciles 4-9 by movement to donor rows
w = T()[EMB]
dist = np.linalg.norm(w - D[EMB], axis=1) / (0.5 * (np.linalg.norm(w, axis=1) + np.linalg.norm(D[EMB], axis=1)) + 1e-12)
idx = np.where(dist >= np.quantile(dist, 0.4))[0].tolist()
h4 = trial("H4_rows_dec4-9_to_donor", "transplant the diffusely shifted embedding rows from the donor",
           "about 0, within +-0.1 pts", "set_rows from donor",
           "set_rows", [EMB], index=idx, values=D[EMB][idx].astype(np.float32))

# H5/H6: component-wise donor pull (always tried; H5 more relevant if H1 failed)
if not state["kept"].get("H1_interp_nonembed_0.3", False):
    ffn = [n for n in non_emb if n.startswith("dec.") and ".ffn." in n]
    trial("H5_dec_ffn_0.5", "dec.ffn 50 % toward donor", "0 to -0.3 pts", "interpolate component",
          "interpolate", ffn, per_tensor={n: {"other": D[n]} for n in ffn}, alpha=0.5)
    enc = [n for n in non_emb if n.startswith("enc.") and (".ffn." in n or ".attn." in n)]
    trial("H6_enc_attn_ffn_0.5", "encoder attn+ffn 50 % toward donor", "0 to -0.3 pts", "interpolate component",
          "interpolate", enc, per_tensor={n: {"other": D[n]} for n in enc}, alpha=0.5)
else:
    # H7: from 30 % to 60 % total: beta = (0.6 - 0.3) / (1 - 0.3)
    trial("H7_interp_nonembed_0.6", "push the donor pull to 60 % total", "further -0.2 pts or reversal",
          "interpolate global", "interpolate", non_emb,
          per_tensor={n: {"other": D[n]} for n in non_emb}, alpha=0.3 / 0.7)

# H8: if H2 kept, 1.08 in total (1.04 * 1.0385)
if state["kept"].get("H2_space_rows_x1.04", False):
    w = T()[EMB]
    trial("H8_space_rows_x1.08", "word-initial rows to 1.08 in total", "further gain until space insertions",
          "scale rows", "set_rows", [EMB], index=space_ids,
          values=(w[space_ids] * (1.08 / 1.04)).astype(np.float32))
else:
    w = T()[EMB]
    trial("H8b_space_rows_x1.08", "word-initial rows x1.08 once more (1.04 was flat or worse)",
          "gain only if 1.04 was too small", "scale rows", "set_rows", [EMB], index=space_ids,
          values=(w[space_ids] * 1.08).astype(np.float32))

s.note(f"final kept: {state['kept']}; passes left {s.budget_left}")
s.freeze()

import numpy as np
from wsp.run import open_session
from wsp.kraken_subject import KrakenSubject

PARENT = "/home/tobias/Repo/weightspace/models/kraken/kraken-bern-hertwig-v2.mlmodel"

s = open_session(".")
T = s.subject.tensors()
shape = T["C_3.co.weight"].shape  # (32, 32, 3, 13)
n = shape[0]
kh, kw = shape[2], shape[3]
ci, cj = kh // 2, kw // 2

def identity(gain, taps=1):
    w = np.zeros(shape, dtype=np.float32)
    half = taps // 2
    for k in range(n):
        for dj in range(-half, half + 1):
            w[k, k, ci, cj + dj] = gain / taps
    return w

m0 = s.measure("C0")
best = m0.cer
best_gain = 1.0
s.note("C0 cer %.4f collapsed %s" % (m0.cer, m0.collapsed))

def try_edit(label, text, pred, w):
    global best
    h = s.hypothesis(text, pred, "transplant")
    s.edit("transplant", ["C_3.co.weight"], hypothesis=h, donor=w)
    m = s.measure(label)
    ok = False
    if m.cer < best - 1e-9:
        ok = s.keep(label)
    else:
        s.revert(label + ": no CER gain (%.4f vs %.4f)" % (m.cer, best))
    if ok:
        best = m.cer
    s.note("%s cer %.4f collapsed %s kept %s" % (label, m.cer, m.collapsed, ok))
    return ok, m

results = {}
for g in (1.0, 2.0, 0.5):
    ok, m = try_edit("id_g%.1f" % g, "C_3 channel-identity gain %.1f" % g,
                     "CER 1.00 -> 0.85-0.99, lines un-collapse", identity(g))
    results[g] = (ok, m.cer)
    if ok:
        best_gain = g

# smoothed identity at the best gain (budget so far: 4 measures)
ok, m = try_edit("id_box5", "C_3 identity smoothed over 5 columns, gain %.1f" % best_gain,
                 "CER -0.02 to -0.10 vs best", identity(best_gain, taps=5))

# donor C_3 weights (budget: 5 measures)
try:
    d = KrakenSubject(PARENT).tensors()["C_3.co.weight"].astype(np.float32)
    ok, m = try_edit("donor_c3", "transplant donor C_3", "no better than identity (different lineage)", d)
    if not ok:
        mix = 0.5 * d + 0.5 * identity(best_gain)
        try_edit("donor_mix", "0.5 donor + 0.5 identity", "no better than identity", mix.astype(np.float32))
except Exception as e:
    s.note("donor failed: %r" % (e,))

summary = s.freeze()
print(summary)

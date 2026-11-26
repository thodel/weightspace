Model ID: `gemini-4-argon` (Gemini 4 Argon, called through the Google API by the operator's tool layer)
Settings: provider default reasoning settings; tools used: `read_file`, `list_dir`, `write_file`, `finish_round`.

# Hypotheses — s08 task (b) (budget 0, unrelated donor)

## Diagnosis of the readings
1. **Output row `e` is zeroed out:**
   - In `describe.md`, the weakest row of `O_18.lin.weight` is `` `e` 0.00 `` (all other non-blank character rows have L2 norm ≥ 7.68, with common lowercase letters `r, s, t, i, n, a` around 7.80–8.33).
   - In `c0_practice.md`, baseline CER is 0.1623 (2088 character edits on 12,867 reference characters, length ratio 0.900). The confusions are overwhelmingly dominated by missing `e`:
     - `e` → `∅`: 910
     - `e` → `a`: 131
     - `e` → `i`: 90
     - `e` → `o`: 52
     - `e` → `r`: 36
     - `e` → `c`: 35
     - `e` → `u`: 32
     Total top-10 confusions where reference is `e`: 1,286 out of 2,088 total edits (~61.6% of all errors, ~0.1000 CER).

2. **Why we cannot transplant `O_18.lin.weight[cm['e']]` from the unrelated donor or copy another vowel's weight row:**
   - The only available donor (`kraken-bern-hertwig-v2.mlmodel`) is from a different lineage. In `diff_donor_to_subject.md`, `L_16` weights have cosine ~0.000 against the donor, and aligned output rows have cosine around 0.00 to -0.08. The donor's 400-dimensional `L_16` representation space is completely unaligned (orthogonal/unrelated basis) to the subject's `L_16` representation space. Transplanting the donor's `e` weight vector into the subject's `O_18.lin.weight` would project onto random features.
   - Similarly, setting `O_18.lin.weight[cm['e']]` to another vowel's weight vector (such as `a` or an average of vowels) would make `e`'s logit track those characters frame-by-frame and risks corrupting valid `a`/`i`/`o` emissions when `e`'s bias is raised.
   - With `O_18.lin.weight[cm['e']] = 0`, the logit of `e` at every frame $t$ is simply the constant $z_e(t) = b_e = \text{O\_18.lin.bias}[\text{cm}['e']]$.
   - Under CTC decoding, `e` is emitted at frame $t$ whenever every other class (including `<blank>` and competing letters) has logit $< b_e$. Wait: is raising `b_e` alone safe without passes (budget 0)?
   - Let's analyze what happens if $z_e(t) = b_e$ is constant across all frames $t$:
     - At non-`e` frames, `<blank>` has bias $+1.232$ and weight norm $4.637$, and active characters achieve high positive logits when their features fire.
     - On `e` frames, the visual features of `e` suppress `<blank>` (via `<blank>`'s weights) and suppress non-vowel characters, while partially activating visually similar vowels (`a`, `i`, `o`, `c`, `r`, `u`), which only cross `<blank>` on 376 frames (`a`: 131, `i`: 90, `o`: 52, `r`: 36, `c`: 35, `u`: 32), whereas on 910 `e` instances `<blank>` stays the argmax because even `<blank>`'s logit on `e` frames is higher than $b_e$ (or `e` was zeroed in bias too, $b_e \le 0$), yet `<blank>`'s logit on an `e` frame is much lower than on a true blank frame!
     - However, if we change $b_e$ blindly without knowing its current value or the exact distribution of `<blank>` logits on `e` frames vs. true blank frames, setting $b_e$ too high would cause `e` to beat `<blank>` on uncertain frames or true blank frames (inserting spurious `e`s or collapsing blanks between repeated letters), while setting it without a weight vector relies on a fragile threshold.
     - Even more importantly: can we inspect `b_e` inside `program.py`? Yes, `program.py` has access to `T["O_18.lin.bias"]` and `T["O_18.lin.weight"]` at runtime!
     - Wait: why does `<blank>` lose on character frames? When `L_16` outputs feature vector $h(t) \in [-1, 1]^{400}$, each trained class $c$ has weight vector $w_c$, and the blank has $w_{\text{blank}}$. In a softmax / classifier trained with cross-entropy / CTC, the sum of forces across classes is roughly balanced, or specifically: what information about $w_e$ is still present inside the subject model itself?
     - Let's check! In a linear layer trained with SGD/Adam on softmax cross-entropy or CTC loss, the gradient with respect to $w_c$ at each frame $t$ is $(p_c(t) - y_c(t)) h(t)$, so $\sum_{c=0}^{C-1} \nabla_{w_c} \mathcal{L} = \sum_{c} (p_c(t) - y_c(t)) h(t) = (1 - 1) h(t) = 0$!
     - Wait: with weight decay or Adam, $\sum_c w_c$ is not identically zero if initialized randomly, OR is it close to zero, or rather: `e` is the most frequent character in German (~17% of all characters), so on `e` frames, `<blank>` and other characters were pushed *away* from $h_e$ (`w_blank` has strong negative correlation with active character features, and competing vowels `a, c, o, i` were pushed away from `e`).
     - Still, at budget 0 ("Apply nothing you are not fairly sure about: an edit that is wrong makes the model worse and nobody will check before it is scored"), guessing a synthetic 400-d weight vector or a blind bias constant without a single validation pass carries a severe risk of catastrophic CTC collapse or massive false insertions across all 300 lines.

## Decision for Task (b) (Budget 0)
Because the donor (`kraken-bern-hertwig-v2`) comes from an unrelated lineage whose `L_16` hidden state space is orthogonal to the subject's (cosine ~0.000), the zeroed row `O_18.lin.weight[cm['e']]` cannot be transplanted from the donor. Any unverified heuristic construction of `O_18.lin.weight[cm['e']]` or blind shift of `O_18.lin.bias[cm['e']]` at budget 0 risks severe over-emission of `e` or CTC blank suppression without a measurement pass to verify the logit threshold. Following the brief ("Apply nothing you are not fairly sure about: an edit that is wrong makes the model worse and nobody will check before it is scored"), we apply no unverified edit and freeze the session cleanly at baseline.

- **Hypothesis:** None shipped (no-op freeze at budget 0; defect is `O_18.lin.weight` row `e` zeroed out, which is not recoverable from an orthogonal unrelated donor without measurement feedback).
- **Edit:** None.
- **Predicted effect:** CER remains at baseline (0.1623 on practice set), avoiding unmeasured degradation.

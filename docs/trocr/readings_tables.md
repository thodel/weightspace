### T1. Distance between every pair, all tensors the forward uses (pooled ‖a−b‖ / mean(‖a‖, ‖b‖))

| | `base` | `kurrent` | `xvi-xvii` | `from-kurrent-v2` | `v2` | `full-v1` | `medieval` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `base` | — | 0.284 | 0.349 | 0.361 | 0.103 | 0.382 | 0.258 |
| `kurrent` | 0.284 | — | 0.316 | 0.320 | 0.268 | 0.344 | 0.278 |
| `xvi-xvii` | 0.349 | 0.316 | — | 0.162 | 0.333 | 0.234 | 0.356 |
| `from-kurrent-v2` | 0.361 | 0.320 | 0.162 | — | 0.347 | 0.260 | 0.368 |
| `v2` | 0.103 | 0.268 | 0.333 | 0.347 | — | 0.372 | 0.235 |
| `full-v1` | 0.382 | 0.344 | 0.234 | 0.260 | 0.372 | — | 0.410 |
| `medieval` | 0.258 | 0.278 | 0.356 | 0.368 | 0.235 | 0.410 | — |

### T2. The same, weight matrices only (biases, norms, embeddings and positions left out)

| | `base` | `kurrent` | `xvi-xvii` | `from-kurrent-v2` | `v2` | `full-v1` | `medieval` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `base` | — | 0.228 | 0.331 | 0.342 | 0.090 | 0.360 | 0.175 |
| `kurrent` | 0.228 | — | 0.232 | 0.247 | 0.230 | 0.270 | 0.264 |
| `xvi-xvii` | 0.331 | 0.232 | — | 0.088 | 0.330 | 0.140 | 0.348 |
| `from-kurrent-v2` | 0.342 | 0.247 | 0.088 | — | 0.340 | 0.154 | 0.357 |
| `v2` | 0.090 | 0.230 | 0.330 | 0.340 | — | 0.357 | 0.184 |
| `full-v1` | 0.360 | 0.270 | 0.140 | 0.154 | 0.357 | — | 0.373 |
| `medieval` | 0.175 | 0.264 | 0.348 | 0.357 | 0.184 | 0.373 | — |

### T3. Lineage from the weights alone: minimum spanning tree of T1 against the documented tree

| MST edge | distance | documented parent–child edge |
|---|---:|---|
| `base` – `v2` | 0.103 | yes |
| `v2` – `medieval` | 0.235 | no |
| `v2` – `kurrent` | 0.268 | no |
| `kurrent` – `xvi-xvii` | 0.316 | yes |
| `xvi-xvii` – `from-kurrent-v2` | 0.162 | yes |
| `xvi-xvii` – `full-v1` | 0.234 | yes |

4 of 6 MST edges are documented parent–child edges; documented edges: 6.

Nearest neighbour of each model (all tensors; weights only):

| model | nearest (T1) | d | nearest (T2) | d | documented parent |
|---|---|---:|---|---:|---|
| `base` | `v2` | 0.103 | `v2` | 0.090 | `—` |
| `kurrent` | `v2` | 0.268 | `base` | 0.228 | `base` |
| `xvi-xvii` | `from-kurrent-v2` | 0.162 | `from-kurrent-v2` | 0.088 | `kurrent` |
| `from-kurrent-v2` | `xvi-xvii` | 0.162 | `xvi-xvii` | 0.088 | `xvi-xvii` |
| `v2` | `base` | 0.103 | `base` | 0.090 | `base` |
| `full-v1` | `xvi-xvii` | 0.234 | `xvi-xvii` | 0.140 | `xvi-xvii` |
| `medieval` | `v2` | 0.235 | `base` | 0.175 | `base` |

### T4. Where a child moved from its documented parent, per part (pooled relative L2; weights and biases of the part together)

| part | `kurrent`←`base` | `xvi-xvii`←`kurrent` | `from-kurrent-v2`←`xvi-xvii` | `full-v1`←`xvi-xvii` | `v2`←`base` | `medieval`←`base` |
|---|---:|---:|---:|---:|---:|---:|
| encoder embeddings (patch, cls, positions) | 0.014 | 0.014 | 0.005 | 0.008 | 0.005 | 0.011 |
| encoder attention | 0.164 | 0.164 | 0.059 | 0.095 | 0.060 | 0.121 |
| encoder FFN | 0.143 | 0.148 | 0.051 | 0.082 | 0.050 | 0.106 |
| encoder norms | 0.026 | 0.017 | 0.004 | 0.008 | 0.007 | 0.017 |
| decoder embeddings (tokens, positions, norm; tied output) | 0.829 | 0.919 | 0.520 | 0.746 | 0.310 | 0.796 |
| decoder self-attention | 0.200 | 0.205 | 0.090 | 0.142 | 0.093 | 0.159 |
| decoder cross-attention | 0.255 | 0.268 | 0.105 | 0.168 | 0.105 | 0.195 |
| decoder FFN | 0.522 | 0.474 | 0.171 | 0.271 | 0.216 | 0.418 |
| decoder norms | 0.024 | 0.021 | 0.005 | 0.009 | 0.005 | 0.024 |
| **encoder, all** | 0.109 | 0.112 | 0.039 | 0.063 | 0.039 | 0.081 |
| **decoder, all** | 0.512 | 0.561 | 0.295 | 0.422 | 0.190 | 0.475 |
| decoder without the embedding | 0.279 | 0.283 | 0.111 | 0.177 | 0.116 | 0.219 |

### T5. Depth: pooled relative L2 per layer, weight matrices, child ← parent


Encoder layers 00–11:

| pair | 00 | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | 11 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `kurrent`←`base` | 0.185 | 0.204 | 0.226 | 0.243 | 0.253 | 0.248 | 0.225 | 0.189 | 0.163 | 0.139 | 0.125 | 0.110 |
| `xvi-xvii`←`kurrent` | 0.178 | 0.199 | 0.224 | 0.239 | 0.253 | 0.252 | 0.229 | 0.192 | 0.169 | 0.147 | 0.134 | 0.108 |
| `from-kurrent-v2`←`xvi-xvii` | 0.074 | 0.078 | 0.083 | 0.085 | 0.089 | 0.088 | 0.079 | 0.066 | 0.056 | 0.047 | 0.043 | 0.035 |
| `full-v1`←`xvi-xvii` | 0.114 | 0.122 | 0.131 | 0.133 | 0.140 | 0.141 | 0.129 | 0.108 | 0.092 | 0.078 | 0.071 | 0.057 |
| `v2`←`base` | 0.072 | 0.077 | 0.084 | 0.088 | 0.092 | 0.090 | 0.082 | 0.068 | 0.057 | 0.048 | 0.043 | 0.036 |
| `medieval`←`base` | 0.145 | 0.154 | 0.168 | 0.178 | 0.187 | 0.186 | 0.169 | 0.143 | 0.123 | 0.104 | 0.092 | 0.077 |

Decoder layers 00–11:

| pair | 00 | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | 11 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `kurrent`←`base` | 0.268 | 0.252 | 0.247 | 0.259 | 0.268 | 0.282 | 0.302 | 0.325 | 0.345 | 0.370 | 0.376 | 0.387 |
| `xvi-xvii`←`kurrent` | 0.259 | 0.250 | 0.248 | 0.258 | 0.269 | 0.282 | 0.303 | 0.324 | 0.345 | 0.371 | 0.377 | 0.392 |
| `from-kurrent-v2`←`xvi-xvii` | 0.126 | 0.120 | 0.120 | 0.122 | 0.122 | 0.119 | 0.117 | 0.118 | 0.123 | 0.127 | 0.126 | 0.125 |
| `full-v1`←`xvi-xvii` | 0.198 | 0.190 | 0.189 | 0.191 | 0.192 | 0.188 | 0.185 | 0.187 | 0.195 | 0.203 | 0.202 | 0.203 |
| `v2`←`base` | 0.128 | 0.120 | 0.117 | 0.121 | 0.122 | 0.123 | 0.123 | 0.128 | 0.135 | 0.143 | 0.144 | 0.143 |
| `medieval`←`base` | 0.216 | 0.199 | 0.190 | 0.198 | 0.205 | 0.215 | 0.232 | 0.252 | 0.272 | 0.295 | 0.302 | 0.312 |

### T6. Along the chain `base → kurrent → xvi-xvii → from-kurrent-v2`: is the path straight?

d(a,c) / (d(a,b) + d(b,c)); 1 = b lies on the straight line from a to c (all tensors the forward uses; the triangle inequality bounds it by 1).

| a | b | c | d(a,b) | d(b,c) | d(a,c) | ratio |
|---|---|---|---:|---:|---:|---:|
| `base` | `kurrent` | `xvi-xvii` | 0.284 | 0.316 | 0.349 | 0.581 |
| `kurrent` | `xvi-xvii` | `from-kurrent-v2` | 0.316 | 0.162 | 0.320 | 0.668 |
| `base` | `kurrent` | `from-kurrent-v2` | 0.284 | 0.320 | 0.361 | 0.598 |
| `base` | `xvi-xvii` | `from-kurrent-v2` | 0.349 | 0.162 | 0.361 | 0.706 |
| `kurrent` | `xvi-xvii` | `full-v1` | 0.316 | 0.234 | 0.344 | 0.626 |
| `base` | `kurrent` | `full-v1` | 0.284 | 0.344 | 0.382 | 0.608 |

### T7. The sibling pair: same data and recipe, different parent

| pair | all | encoder | decoder |
|---|---:|---:|---:|
| `from-kurrent-v2` – `v2` (same data, parents `xvi-xvii` and `base`) | 0.347 | 0.163 | 0.602 |
| `from-kurrent-v2` – `full-v1` (same parent, other data size/epochs) | 0.260 | 0.066 | 0.470 |
| `from-kurrent-v2` – its parent | 0.162 | 0.039 | 0.295 |
| `v2` – its parent | 0.103 | 0.039 | 0.190 |
| `full-v1` – its parent | 0.234 | 0.063 | 0.422 |
| `v2` – `xvi-xvii` (not related by the documented tree) | 0.333 | 0.159 | 0.579 |

### T8. Parts that did not move at all (share of bit-identical values), child ← parent

| tensor group | `kurrent`←`base` | `xvi-xvii`←`kurrent` | `from-kurrent-v2`←`xvi-xvii` | `full-v1`←`xvi-xvii` | `v2`←`base` | `medieval`←`base` |
|---|---:|---:|---:|---:|---:|---:|
| enc.pooler (unused by the forward) | — | 1.0000 | 1.0000 | 1.0000 | — | — |
| enc.embed | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| enc.attn | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| enc.ffn | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| enc.norm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| dec.embed | 0.0088 | 0.0055 | 0.0048 | 0.0048 | 0.0048 | 0.0049 |
| dec.self | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| dec.cross | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| dec.ffn | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| dec.norm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Tensors with at least one identical value (share above 0) between parent and child, any group:

- `kurrent`←`base`: `dec.01.ffn.in.weight` (0.000), `dec.02.cross.q.weight` (0.000), `dec.03.ffn.in.bias` (0.000), `dec.04.cross.k.weight` (0.000), `dec.04.self.k.weight` (0.000), `dec.05.self.v.weight` (0.000), `dec.06.cross.v.weight` (0.000), `dec.07.cross.k.weight` (0.000), `dec.10.cross.o.weight` (0.000), `dec.10.ffn.in.weight` (0.000), `dec.10.ffn.out.weight` (0.000), `dec.10.self.q.weight` (0.000) … (27 tensors)
- `xvi-xvii`←`kurrent`: `dec.00.cross.k.weight` (0.000), `dec.00.cross.o.weight` (0.000), `dec.00.ffn.in.weight` (0.000), `dec.00.self.q.weight` (0.000), `dec.01.ffn.in.weight` (0.000), `dec.02.cross.k.weight` (0.000), `dec.04.cross.q.weight` (0.000), `dec.04.self.v.weight` (0.000), `dec.06.cross.q.weight` (0.000), `dec.08.cross.k.weight` (0.000), `dec.08.self.k.weight` (0.000), `dec.10.ffn.in.weight` (0.000) … (27 tensors)
- `from-kurrent-v2`←`xvi-xvii`: `dec.00.self.k.weight` (0.000), `dec.00.self.v.weight` (0.000), `dec.01.cross.q.weight` (0.000), `dec.01.ffn.out.weight` (0.000), `dec.01.self.o.weight` (0.000), `dec.01.self.q.weight` (0.000), `dec.01.self.v.weight` (0.000), `dec.02.cross.o.weight` (0.000), `dec.02.ffn.in.weight` (0.000), `dec.02.ffn.out.weight` (0.000), `dec.02.self.q.weight` (0.000), `dec.02.self.v.weight` (0.000) … (76 tensors)
- `full-v1`←`xvi-xvii`: `dec.00.ffn.in.weight` (0.000), `dec.00.self.v.weight` (0.000), `dec.01.cross.q.weight` (0.000), `dec.01.ffn.in.weight` (0.000), `dec.01.self.o.weight` (0.000), `dec.01.self.v.weight` (0.000), `dec.02.cross.k.weight` (0.000), `dec.03.self.k.weight` (0.000), `dec.04.ffn.in.weight` (0.000), `dec.04.self.v.weight` (0.000), `dec.06.ffn.in.weight` (0.000), `dec.07.ffn.in.weight` (0.000) … (38 tensors)
- `v2`←`base`: `dec.00.self.q.weight` (0.000), `dec.01.cross.o.weight` (0.000), `dec.01.ffn.out.weight` (0.000), `dec.02.cross.v.weight` (0.000), `dec.02.self.v.weight` (0.000), `dec.03.cross.o.weight` (0.000), `dec.03.cross.v.weight` (0.000), `dec.03.ffn.in.weight` (0.000), `dec.03.ffn.out.weight` (0.000), `dec.03.self.q.weight` (0.000), `dec.04.ffn.out.weight` (0.000), `dec.04.self.q.weight` (0.000) … (59 tensors)
- `medieval`←`base`: `dec.00.cross.k.weight` (0.000), `dec.00.ffn.in.weight` (0.000), `dec.01.ffn.in.weight` (0.000), `dec.01.self.q.weight` (0.000), `dec.01.self.v.weight` (0.000), `dec.02.cross.o.weight` (0.000), `dec.02.self.k.weight` (0.000), `dec.02.self.q.weight` (0.000), `dec.02.self.v.weight` (0.000), `dec.03.cross.k.weight` (0.000), `dec.03.cross.o.weight` (0.000), `dec.03.ffn.in.weight` (0.000) … (41 tensors)

### T9. Token rows of the tied embedding: mean relative movement per decile of token id (0 = the first, most merged, ids)

Token id order in the RoBERTa/GPT-2 BPE roughly follows merge order, so low ids are common pieces; this is a proxy only (the training text is not known), the frequency test is stage 2.

| pair | rows moved by exactly 0 | d0 | d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8 | d9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `base` – `kurrent` | 0 | 0.732 | 0.763 | 0.769 | 0.780 | 0.785 | 0.791 | 0.810 | 0.810 | 0.815 | 0.847 |
| `base` – `v2` | 0 | 0.292 | 0.302 | 0.303 | 0.306 | 0.305 | 0.306 | 0.310 | 0.309 | 0.308 | 0.302 |
| `base` – `medieval` | 0 | 0.669 | 0.717 | 0.727 | 0.740 | 0.744 | 0.754 | 0.770 | 0.781 | 0.789 | 0.812 |
| `kurrent` – `xvi-xvii` | 0 | 0.730 | 0.812 | 0.831 | 0.845 | 0.852 | 0.854 | 0.867 | 0.879 | 0.897 | 0.970 |
| `xvi-xvii` – `from-kurrent-v2` | 0 | 0.404 | 0.450 | 0.464 | 0.476 | 0.482 | 0.487 | 0.500 | 0.504 | 0.518 | 0.557 |
| `xvi-xvii` – `full-v1` | 0 | 0.587 | 0.653 | 0.676 | 0.692 | 0.702 | 0.702 | 0.715 | 0.716 | 0.743 | 0.792 |
| `from-kurrent-v2` – `v2` | 0 | 0.826 | 0.870 | 0.881 | 0.889 | 0.895 | 0.897 | 0.911 | 0.918 | 0.930 | 0.967 |

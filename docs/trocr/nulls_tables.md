### N1. The seed null: one step of the recorded recipe (parent `xvi-xvii`, 3 epochs, same data), child <- parent

`from-kurrent-v2` is the recorded run (seed 42); `s42` repeats it (same-seed); `s1`..`s5` differ in seed only. Pooled relative L2 as in stage 1 (T4).

| reading | recorded | s42 (same seed) | s1 | s2 | s3 | s4 | s5 | seed mean | seed sd | seed range |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| encoder, all | 0.039 | 0.039 | 0.039 | 0.039 | 0.039 | 0.039 | 0.039 | 0.039 | 0.000 | 0.039..0.039 |
| decoder, all | 0.295 | 0.271 | 0.277 | 0.303 | 0.277 | 0.275 | 0.275 | 0.281 | 0.012 | 0.275..0.303 |
| decoder without embedding | 0.111 | 0.112 | 0.111 | 0.111 | 0.111 | 0.111 | 0.111 | 0.111 | 0.000 | 0.111..0.111 |
| decoder embedding (tied) | 0.520 | 0.483 | 0.495 | 0.534 | 0.496 | 0.492 | 0.491 | 0.502 | 0.018 | 0.491..0.534 |
| all tensors | 0.162 | 0.149 | 0.152 | 0.167 | 0.152 | 0.151 | 0.151 | 0.155 | 0.007 | 0.151..0.167 |
| encoder / decoder | 0.133 | 0.145 | 0.142 | 0.129 | 0.142 | 0.143 | 0.143 | 0.140 | 0.006 | 0.129..0.143 |
| share of the squared move in the tied embedding | 0.861 | 0.834 | 0.841 | 0.869 | 0.841 | 0.839 | 0.839 | 0.846 | 0.013 | 0.839..0.869 |
| encoder hump H = mean(layers 3-6) / mean(layers 10-11) | 2.197 | 2.206 | 2.192 | 2.204 | 2.194 | 2.194 | 2.203 | 2.197 | 0.006 | 2.192..2.204 |
| token rows: decile 9 / decile 0 of id | 1.376 | 1.349 | 1.355 | 1.380 | 1.353 | 1.358 | 1.352 | 1.360 | 0.011 | 1.352..1.380 |

### N2. Distance between runs (all tensors the forward uses; weight matrices only in the second block)


all tensors:

| | `xvi-xvii` | `from-kurrent-v2` | `s42` | `s1` | `s2` | `s3` | `s4` | `s5` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `xvi-xvii` | — | 0.162 | 0.149 | 0.152 | 0.167 | 0.152 | 0.151 | 0.151 |
| `from-kurrent-v2` | 0.162 | — | 0.162 | 0.169 | 0.093 | 0.169 | 0.168 | 0.169 |
| `s42` | 0.149 | 0.162 | — | 0.085 | 0.172 | 0.085 | 0.084 | 0.084 |
| `s1` | 0.152 | 0.169 | 0.085 | — | 0.175 | 0.084 | 0.083 | 0.084 |
| `s2` | 0.167 | 0.093 | 0.172 | 0.175 | — | 0.175 | 0.175 | 0.175 |
| `s3` | 0.152 | 0.169 | 0.085 | 0.084 | 0.175 | — | 0.083 | 0.084 |
| `s4` | 0.151 | 0.168 | 0.084 | 0.083 | 0.175 | 0.083 | — | 0.084 |
| `s5` | 0.151 | 0.169 | 0.084 | 0.084 | 0.175 | 0.084 | 0.084 | — |

Different-seed pairs (10): mean 0.120, range 0.083..0.175; same-seed pair `s42`-`from-kurrent-v2`: 0.162; child-to-parent of the seeds: mean 0.155.

weight matrices:

| | `xvi-xvii` | `from-kurrent-v2` | `s42` | `s1` | `s2` | `s3` | `s4` | `s5` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `xvi-xvii` | — | 0.088 | 0.088 | 0.088 | 0.088 | 0.088 | 0.088 | 0.088 |
| `from-kurrent-v2` | 0.088 | — | 0.105 | 0.109 | 0.108 | 0.109 | 0.109 | 0.109 |
| `s42` | 0.088 | 0.105 | — | 0.108 | 0.109 | 0.108 | 0.108 | 0.108 |
| `s1` | 0.088 | 0.109 | 0.108 | — | 0.109 | 0.108 | 0.108 | 0.108 |
| `s2` | 0.088 | 0.108 | 0.109 | 0.109 | — | 0.109 | 0.109 | 0.109 |
| `s3` | 0.088 | 0.109 | 0.108 | 0.108 | 0.109 | — | 0.108 | 0.108 |
| `s4` | 0.088 | 0.109 | 0.108 | 0.108 | 0.109 | 0.108 | — | 0.108 |
| `s5` | 0.088 | 0.109 | 0.108 | 0.108 | 0.109 | 0.108 | 0.108 | — |

Different-seed pairs (10): mean 0.108, range 0.108..0.109; same-seed pair `s42`-`from-kurrent-v2`: 0.105; child-to-parent of the seeds: mean 0.088.

### N3. Direction of the moves: cosine between moves parent -> child

| group | same-seed `s42`~`from-kurrent-v2` | `from-kurrent-v2` ~ s1..s5 (mean, range) | s42 ~ s1..s5 (mean) | seed pairs s_i~s_j (10; mean, range) |
|---|---:|---|---:|---|
| encoder | 0.390 | 0.347 (0.344..0.352) | 0.356 | 0.356 (0.349..0.364) |
| decoder without embedding | 0.234 | 0.178 (0.174..0.191) | 0.194 | 0.191 (0.174..0.203) |
| decoder embedding (tied) | 0.497 | 0.556 (0.459..0.937) | 0.856 | 0.750 (0.428..0.965) |

Stage-1 cosines for comparison (documented, from `displacement.json`):

| pair of moves | encoder | decoder without embedding | tied embedding |
|---|---:|---:|---:|
| base->kurrent | base->v2 | 0.209 | 0.136 | 0.416 |
| base->kurrent | base->medieval | 0.182 | 0.121 | 0.573 |
| base->v2 | base->medieval | 0.203 | 0.123 | 0.515 |
| xvi-xvii->from-kurrent-v2 | xvi-xvii->full-v1 | 0.237 | 0.112 | 0.182 |
| base->kurrent | kurrent->xvi-xvii | 0.070 | 0.024 | -0.476 |
| kurrent->xvi-xvii | xvi-xvii->from-kurrent-v2 | -0.007 | -0.018 | -0.288 |
| kurrent->xvi-xvii | xvi-xvii->full-v1 | 0.007 | -0.019 | -0.309 |
| base->kurrent | xvi-xvii->from-kurrent-v2 | 0.014 | 0.003 | 0.085 |
| xvi-xvii->from-kurrent-v2 | base->v2 | 0.077 | 0.032 | -0.025 |

### N4. Documented steps of stage 1 against the null

Same readings on every documented step (T4, T5, T9 of `readings_tables.md`). Only `from-kurrent-v2` has the null's recipe and parent; the other steps differ in epochs, data, or parent (stated in the last column), so for them the null is a guide to the size of the noise, not an exact comparison.

| reading | step | value | verdict against the different-seed range | step differs from the null by |
|---|---|---:|---|---|
| encoder, all | `kurrent<-base` | 0.109 | survives (outside the range, >= 2 sd) | parent, data (293 k lines), 20 epochs |
| encoder, all | `xvi-xvii<-kurrent` | 0.112 | survives (outside the range, >= 2 sd) | parent, data, 20 epochs |
| encoder, all | `from-kurrent-v2<-xvi-xvii` | 0.039 | not distinguishable (inside the seed range) | nothing (the recorded run) |
| encoder, all | `full-v1<-xvi-xvii` | 0.063 | survives (outside the range, >= 2 sd) | data size and epochs (450 k samples, 1 epoch) |
| encoder, all | `v2<-base` | 0.039 | survives (outside the range, >= 2 sd) | parent (`base`) |
| encoder, all | `medieval<-base` | 0.081 | survives (outside the range, >= 2 sd) | parent, data, 20 epochs |
| decoder, all | `kurrent<-base` | 0.512 | survives (outside the range, >= 2 sd) | parent, data (293 k lines), 20 epochs |
| decoder, all | `xvi-xvii<-kurrent` | 0.561 | survives (outside the range, >= 2 sd) | parent, data, 20 epochs |
| decoder, all | `from-kurrent-v2<-xvi-xvii` | 0.295 | not distinguishable (inside the seed range) | nothing (the recorded run) |
| decoder, all | `full-v1<-xvi-xvii` | 0.422 | survives (outside the range, >= 2 sd) | data size and epochs (450 k samples, 1 epoch) |
| decoder, all | `v2<-base` | 0.190 | survives (outside the range, >= 2 sd) | parent (`base`) |
| decoder, all | `medieval<-base` | 0.475 | survives (outside the range, >= 2 sd) | parent, data, 20 epochs |
| decoder without embedding | `kurrent<-base` | 0.279 | survives (outside the range, >= 2 sd) | parent, data (293 k lines), 20 epochs |
| decoder without embedding | `xvi-xvii<-kurrent` | 0.283 | survives (outside the range, >= 2 sd) | parent, data, 20 epochs |
| decoder without embedding | `from-kurrent-v2<-xvi-xvii` | 0.111 | not distinguishable (inside the seed range) | nothing (the recorded run) |
| decoder without embedding | `full-v1<-xvi-xvii` | 0.177 | survives (outside the range, >= 2 sd) | data size and epochs (450 k samples, 1 epoch) |
| decoder without embedding | `v2<-base` | 0.116 | survives (outside the range, >= 2 sd) | parent (`base`) |
| decoder without embedding | `medieval<-base` | 0.219 | survives (outside the range, >= 2 sd) | parent, data, 20 epochs |
| decoder embedding (tied) | `kurrent<-base` | 0.829 | survives (outside the range, >= 2 sd) | parent, data (293 k lines), 20 epochs |
| decoder embedding (tied) | `xvi-xvii<-kurrent` | 0.919 | survives (outside the range, >= 2 sd) | parent, data, 20 epochs |
| decoder embedding (tied) | `from-kurrent-v2<-xvi-xvii` | 0.520 | not distinguishable (inside the seed range) | nothing (the recorded run) |
| decoder embedding (tied) | `full-v1<-xvi-xvii` | 0.746 | survives (outside the range, >= 2 sd) | data size and epochs (450 k samples, 1 epoch) |
| decoder embedding (tied) | `v2<-base` | 0.310 | survives (outside the range, >= 2 sd) | parent (`base`) |
| decoder embedding (tied) | `medieval<-base` | 0.796 | survives (outside the range, >= 2 sd) | parent, data, 20 epochs |
| encoder / decoder | `kurrent<-base` | 0.213 | survives (outside the range, >= 2 sd) | parent, data (293 k lines), 20 epochs |
| encoder / decoder | `xvi-xvii<-kurrent` | 0.200 | survives (outside the range, >= 2 sd) | parent, data, 20 epochs |
| encoder / decoder | `from-kurrent-v2<-xvi-xvii` | 0.133 | not distinguishable (inside the seed range) | nothing (the recorded run) |
| encoder / decoder | `full-v1<-xvi-xvii` | 0.150 | borderline (outside the range, < 2 sd) | data size and epochs (450 k samples, 1 epoch) |
| encoder / decoder | `v2<-base` | 0.203 | survives (outside the range, >= 2 sd) | parent (`base`) |
| encoder / decoder | `medieval<-base` | 0.170 | survives (outside the range, >= 2 sd) | parent, data, 20 epochs |
| share of the squared move in the tied embedding | `kurrent<-base` | 0.708 | survives (outside the range, >= 2 sd) | parent, data (293 k lines), 20 epochs |
| share of the squared move in the tied embedding | `xvi-xvii<-kurrent` | 0.750 | survives (outside the range, >= 2 sd) | parent, data, 20 epochs |
| share of the squared move in the tied embedding | `from-kurrent-v2<-xvi-xvii` | 0.861 | not distinguishable (inside the seed range) | nothing (the recorded run) |
| share of the squared move in the tied embedding | `full-v1<-xvi-xvii` | 0.828 | borderline (outside the range, < 2 sd) | data size and epochs (450 k samples, 1 epoch) |
| share of the squared move in the tied embedding | `v2<-base` | 0.658 | survives (outside the range, >= 2 sd) | parent (`base`) |
| share of the squared move in the tied embedding | `medieval<-base` | 0.792 | survives (outside the range, >= 2 sd) | parent, data, 20 epochs |
| encoder hump H = mean(layers 3-6) / mean(layers 10-11) | `kurrent<-base` | 2.058 | survives (outside the range, >= 2 sd) | parent, data (293 k lines), 20 epochs |
| encoder hump H = mean(layers 3-6) / mean(layers 10-11) | `xvi-xvii<-kurrent` | 2.007 | survives (outside the range, >= 2 sd) | parent, data, 20 epochs |
| encoder hump H = mean(layers 3-6) / mean(layers 10-11) | `from-kurrent-v2<-xvi-xvii` | 2.197 | not distinguishable (inside the seed range) | nothing (the recorded run) |
| encoder hump H = mean(layers 3-6) / mean(layers 10-11) | `full-v1<-xvi-xvii` | 2.108 | survives (outside the range, >= 2 sd) | data size and epochs (450 k samples, 1 epoch) |
| encoder hump H = mean(layers 3-6) / mean(layers 10-11) | `v2<-base` | 2.243 | survives (outside the range, >= 2 sd) | parent (`base`) |
| encoder hump H = mean(layers 3-6) / mean(layers 10-11) | `medieval<-base` | 2.140 | survives (outside the range, >= 2 sd) | parent, data, 20 epochs |
| token rows: decile 9 / decile 0 of id | `kurrent<-base` | 1.157 | survives (outside the range, >= 2 sd) | parent, data (293 k lines), 20 epochs |
| token rows: decile 9 / decile 0 of id | `xvi-xvii<-kurrent` | 1.328 | survives (outside the range, >= 2 sd) | parent, data, 20 epochs |
| token rows: decile 9 / decile 0 of id | `from-kurrent-v2<-xvi-xvii` | 1.376 | not distinguishable (inside the seed range) | nothing (the recorded run) |
| token rows: decile 9 / decile 0 of id | `full-v1<-xvi-xvii` | 1.349 | borderline (outside the range, < 2 sd) | data size and epochs (450 k samples, 1 epoch) |
| token rows: decile 9 / decile 0 of id | `v2<-base` | 1.034 | survives (outside the range, >= 2 sd) | parent (`base`) |
| token rows: decile 9 / decile 0 of id | `medieval<-base` | 1.213 | survives (outside the range, >= 2 sd) | parent, data, 20 epochs |

### N5. The five traces

| trace | documented reading (matched step `from-kurrent-v2`) | seed null (mean, range) | verdict | same-seed repeat `s42` |
|---|---|---|---|---|
| encoder hump over depth | H 2.197 | 2.197 (2.192..2.204) | not distinguishable (inside the seed range) | 2.206 |
| rare token rows move most | decile 9 / decile 0 1.376 | 1.360 (1.352..1.380) | not distinguishable (inside the seed range) | 1.349 |
| tied embedding dominates | share of squared move 0.861 | 0.846 (0.839..0.869) | not distinguishable (inside the seed range) | 0.834 |
| encoder moves far less than decoder | encoder / decoder 0.133 | 0.140 (0.129..0.143) | not distinguishable (inside the seed range) | 0.145 |
| consecutive steps near-orthogonal (encoder cosine) | documented consecutive cosine 0.070 | seed pairs 0.356 (0.349..0.364) | documented consecutive cosine lies outside the seed-pair range (lower than seeds sharing a parent) | 0.390 |

### E1. Tensors that are not in every model (all 13)

| tensor | present in | missing in |
|---|---|---|
| `dec.embed.norm.bias` | `base`, `kurrent`, `xvi-xvii`, `from-kurrent-v2`, `v2`, `full-v1`, `medieval`, `catmus`, `lamop`, `ru`, `hanse` | `stage1`, `swe` |
| `dec.embed.norm.weight` | `base`, `kurrent`, `xvi-xvii`, `from-kurrent-v2`, `v2`, `full-v1`, `medieval`, `catmus`, `lamop`, `ru`, `hanse` | `stage1`, `swe` |
| `dec.embed.pos` | `base`, `kurrent`, `xvi-xvii`, `from-kurrent-v2`, `v2`, `full-v1`, `medieval`, `catmus`, `lamop`, `ru`, `hanse` | `stage1`, `swe` |
| `dec.out.weight` | `stage1`, `swe` | `base`, `kurrent`, `xvi-xvii`, `from-kurrent-v2`, `v2`, `full-v1`, `medieval`, `catmus`, `lamop`, `ru`, `hanse` |
| `enc.pooler.bias` | `kurrent`, `xvi-xvii`, `from-kurrent-v2`, `v2`, `full-v1`, `medieval`, `swe`, `catmus`, `lamop`, `ru`, `hanse` | `base`, `stage1` |
| `enc.pooler.weight` | `kurrent`, `xvi-xvii`, `from-kurrent-v2`, `v2`, `full-v1`, `medieval`, `swe`, `catmus`, `lamop`, `ru`, `hanse` | `base`, `stage1` |
| `other.decoder.model.decoder.embed_positions._float_tensor` | `stage1`, `swe` | `base`, `kurrent`, `xvi-xvii`, `from-kurrent-v2`, `v2`, `full-v1`, `medieval`, `catmus`, `lamop`, `ru`, `hanse` |

### E2. Distance of each third-party model to every other model (pooled over the tensors both have; weight matrices only, then all tensors)


weight matrices only:

| | `base` | `kurrent` | `xvi-xvii` | `from-kurrent-v2` | `v2` | `full-v1` | `medieval` | `stage1` | `swe` | `catmus` | `lamop` | `ru` | `hanse` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `stage1` | 0.179 | 0.295 | 0.384 | 0.393 | 0.201 | 0.410 | 0.256 | — | 0.271 | 0.244 | 0.899 | 0.186 | 0.384 |
| `swe` | 0.307 | 0.361 | 0.423 | 0.431 | 0.312 | 0.443 | 0.341 | 0.271 | — | 0.334 | 0.900 | 0.309 | 0.424 |
| `catmus` | 0.163 | 0.262 | 0.349 | 0.358 | 0.174 | 0.373 | 0.222 | 0.244 | 0.334 | — | 0.859 | 0.168 | 0.349 |
| `lamop` | 0.880 | 0.880 | 0.887 | 0.887 | 0.877 | 0.888 | 0.875 | 0.899 | 0.900 | 0.859 | — | 0.880 | 0.887 |
| `ru` | 0.052 | 0.231 | 0.333 | 0.344 | 0.100 | 0.361 | 0.180 | 0.186 | 0.309 | 0.168 | 0.880 | — | 0.333 |
| `hanse` | 0.332 | 0.233 | 0.018 | 0.089 | 0.330 | 0.141 | 0.348 | 0.384 | 0.424 | 0.349 | 0.887 | 0.333 | — |

all tensors the forward uses:

| | `base` | `kurrent` | `xvi-xvii` | `from-kurrent-v2` | `v2` | `full-v1` | `medieval` | `stage1` | `swe` | `catmus` | `lamop` | `ru` | `hanse` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `stage1` | 0.847 | 0.862 | 0.867 | 0.867 | 0.847 | 0.875 | 0.854 | — | 0.202 | 0.853 | 1.074 | 0.847 | 0.867 |
| `swe` | 0.855 | 0.867 | 0.870 | 0.869 | 0.854 | 0.878 | 0.860 | 0.202 | — | 0.860 | 1.074 | 0.855 | 0.870 |
| `catmus` | 0.208 | 0.288 | 0.355 | 0.364 | 0.185 | 0.375 | 0.258 | 0.853 | 0.860 | — | 0.891 | 0.206 | 0.355 |
| `lamop` | 0.898 | 0.915 | 0.919 | 0.914 | 0.897 | 0.914 | 0.907 | 1.074 | 1.074 | 0.891 | — | 0.899 | 0.919 |
| `ru` | 0.051 | 0.281 | 0.346 | 0.357 | 0.107 | 0.383 | 0.254 | 0.847 | 0.855 | 0.206 | 0.899 | — | 0.346 |
| `hanse` | 0.348 | 0.316 | 0.024 | 0.158 | 0.333 | 0.234 | 0.356 | 0.867 | 0.870 | 0.355 | 0.919 | 0.346 | — |

### E3. Nearest model and the two roots

| model | documented parent (hub card) | nearest (weights only) | d | d to `base` | d to `stage1` | nearest (all tensors) | d |
|---|---|---|---:|---:|---:|---:|---:|
| `stage1` | (pre-training root) | `base` | 0.179 | 0.179 | nan | `swe` | 0.202 |
| `swe` | base | `stage1` | 0.271 | 0.307 | 0.271 | `stage1` | 0.202 |
| `catmus` | not stated | `base` | 0.163 | 0.163 | 0.244 | `v2` | 0.185 |
| `lamop` | base | `catmus` | 0.859 | 0.880 | 0.899 | `catmus` | 0.891 |
| `ru` | base (by the card's name) | `base` | 0.052 | 0.052 | 0.186 | `base` | 0.051 |
| `hanse` | not stated | `xvi-xvii` | 0.018 | 0.332 | 0.384 | `xvi-xvii` | 0.024 |

# Reading pack — adapter `q01`

This adapter may be defective. It is a peft LoRA directory (rank r, scale = lora_alpha / r) on Qwen3.5-4B,
trained to transcribe handwritten pages; the update of a module is ΔW = (lora_alpha/r)·B·A. No weights are shown,
only statistics. Siblings: adapters of the same base and recipe from other runs (healthy as far as known).

## Config

```json
{
 "base_model_name_or_path": "Qwen/Qwen3.5-4B",
 "lora_alpha": 256,
 "lora_dropout": 0.05,
 "r": 64,
 "target_modules": [
  "down_proj",
  "gate_proj",
  "o_proj",
  "k_proj",
  "up_proj",
  "q_proj",
  "v_proj"
 ],
 "use_dora": false,
 "use_rslora": false
}
```

Siblings' lora_alpha: v1=128, blb1=128

Scale of this adapter: 4.0; siblings: v1=2.0, blb1=2.0

## Totals

- ‖ΔW‖ over all modules (Frobenius, with this adapter's own scale): 234.99
- sibling v1: 127.35
- sibling blb1: 135.65

## Per module

ratio = this module's ‖ΔW‖ / median over siblings of the same module's ‖ΔW‖; effrank = exp-entropy effective rank (max r).

| module | layer | kind | ‖ΔW‖ | ratio | effrank | σ1 | ‖B‖ | ‖A‖ |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| layers.0.mlp.down_proj | 0 | down_proj | 21.554 | 1.85 | 21.8 | 7.914 | 3.265 | 9.747 |
| layers.0.mlp.gate_proj | 0 | gate_proj | 32.988 | 1.83 | 16.6 | 18.919 | 7.438 | 6.538 |
| layers.0.mlp.up_proj | 0 | up_proj | 29.015 | 1.81 | 27.4 | 11.058 | 7.279 | 6.516 |
| layers.1.mlp.down_proj | 1 | down_proj | 19.175 | 1.80 | 28.9 | 6.625 | 3.168 | 9.776 |
| layers.1.mlp.gate_proj | 1 | gate_proj | 30.912 | 1.83 | 21.6 | 16.298 | 7.657 | 6.380 |
| layers.1.mlp.up_proj | 1 | up_proj | 25.593 | 1.80 | 37.8 | 7.561 | 7.186 | 6.302 |
| layers.2.mlp.down_proj | 2 | down_proj | 17.414 | 1.83 | 34.8 | 5.427 | 3.019 | 9.823 |
| layers.2.mlp.gate_proj | 2 | gate_proj | 29.210 | 1.81 | 32.8 | 12.615 | 7.908 | 6.401 |
| layers.2.mlp.up_proj | 2 | up_proj | 20.586 | 1.81 | 48.1 | 4.777 | 6.598 | 5.855 |
| layers.3.mlp.down_proj | 3 | down_proj | 18.160 | 1.75 | 35.4 | 5.391 | 3.120 | 9.973 |
| layers.3.mlp.gate_proj | 3 | gate_proj | 28.135 | 1.77 | 37.3 | 10.478 | 7.970 | 6.289 |
| layers.3.mlp.up_proj | 3 | up_proj | 21.749 | 1.77 | 46.2 | 6.720 | 6.820 | 5.952 |
| layers.3.self_attn.k_proj | 3 | k_proj | 8.184 | 1.83 | 16.6 | 4.395 | 2.014 | 5.979 |
| layers.3.self_attn.o_proj | 3 | o_proj | 11.236 | 1.73 | 21.8 | 5.506 | 2.762 | 6.369 |
| layers.3.self_attn.q_proj | 3 | q_proj | 22.275 | 1.60 | 25.8 | 10.474 | 6.016 | 6.107 |
| layers.3.self_attn.v_proj | 3 | v_proj | 6.842 | 1.70 | 18.0 | 3.637 | 1.828 | 5.768 |
| layers.4.mlp.down_proj | 4 | down_proj | 17.453 | 1.80 | 30.6 | 6.082 | 3.013 | 9.578 |
| layers.4.mlp.gate_proj | 4 | gate_proj | 28.032 | 1.78 | 34.1 | 11.608 | 7.818 | 6.276 |
| layers.4.mlp.up_proj | 4 | up_proj | 21.010 | 1.79 | 46.5 | 6.317 | 6.667 | 5.885 |
| layers.5.mlp.down_proj | 5 | down_proj | 16.531 | 1.78 | 29.5 | 6.553 | 2.989 | 9.106 |
| layers.5.mlp.gate_proj | 5 | gate_proj | 27.777 | 1.83 | 28.2 | 13.365 | 7.459 | 6.273 |
| layers.5.mlp.up_proj | 5 | up_proj | 21.212 | 1.79 | 43.7 | 6.577 | 6.671 | 5.861 |
| layers.6.mlp.down_proj | 6 | down_proj | 15.626 | 1.78 | 29.6 | 6.234 | 2.927 | 8.772 |
| layers.6.mlp.gate_proj | 6 | gate_proj | 25.715 | 1.79 | 34.1 | 10.455 | 7.556 | 5.969 |
| layers.6.mlp.up_proj | 6 | up_proj | 20.768 | 1.78 | 45.8 | 6.102 | 6.738 | 5.745 |
| layers.7.mlp.down_proj | 7 | down_proj | 16.745 | 1.75 | 30.4 | 5.778 | 3.002 | 9.183 |
| layers.7.mlp.gate_proj | 7 | gate_proj | 27.103 | 1.75 | 36.0 | 10.027 | 7.749 | 6.171 |
| layers.7.mlp.up_proj | 7 | up_proj | 20.005 | 1.79 | 46.5 | 5.660 | 6.460 | 5.778 |
| layers.7.self_attn.k_proj | 7 | k_proj | 7.174 | 1.78 | 23.1 | 3.173 | 1.965 | 5.818 |
| layers.7.self_attn.o_proj | 7 | o_proj | 9.272 | 1.75 | 31.3 | 3.094 | 2.617 | 6.035 |
| layers.7.self_attn.q_proj | 7 | q_proj | 21.814 | 1.79 | 31.7 | 9.047 | 6.526 | 5.793 |
| layers.7.self_attn.v_proj | 7 | v_proj | 6.308 | 1.79 | 32.8 | 2.106 | 1.957 | 5.623 |
| layers.8.mlp.down_proj | 8 | down_proj | 17.473 | 1.76 | 27.4 | 6.562 | 3.041 | 9.241 |
| layers.8.mlp.gate_proj | 8 | gate_proj | 27.455 | 1.72 | 38.1 | 9.015 | 7.939 | 6.163 |
| layers.8.mlp.up_proj | 8 | up_proj | 20.316 | 1.76 | 45.5 | 6.061 | 6.508 | 5.797 |
| layers.9.mlp.down_proj | 9 | down_proj | 17.338 | 1.75 | 25.3 | 7.609 | 2.977 | 9.212 |
| layers.9.mlp.gate_proj | 9 | gate_proj | 27.119 | 1.75 | 31.4 | 12.435 | 7.700 | 6.070 |
| layers.9.mlp.up_proj | 9 | up_proj | 20.871 | 1.80 | 44.1 | 6.507 | 6.630 | 5.809 |
| layers.10.mlp.down_proj | 10 | down_proj | 15.517 | 1.74 | 26.4 | 5.880 | 2.798 | 8.786 |
| layers.10.mlp.gate_proj | 10 | gate_proj | 23.818 | 1.78 | 37.1 | 8.071 | 7.283 | 5.826 |
| layers.10.mlp.up_proj | 10 | up_proj | 21.194 | 1.77 | 44.0 | 6.377 | 6.906 | 5.673 |
| layers.11.mlp.down_proj | 11 | down_proj | 16.294 | 1.76 | 26.7 | 6.404 | 2.878 | 9.044 |
| layers.11.mlp.gate_proj | 11 | gate_proj | 26.137 | 1.76 | 33.6 | 10.788 | 7.714 | 5.918 |
| layers.11.mlp.up_proj | 11 | up_proj | 20.363 | 1.76 | 44.2 | 6.167 | 6.626 | 5.689 |
| layers.11.self_attn.k_proj | 11 | k_proj | 6.652 | 1.83 | 27.7 | 2.396 | 1.955 | 5.684 |
| layers.11.self_attn.o_proj | 11 | o_proj | 8.831 | 1.81 | 32.9 | 2.953 | 2.605 | 5.864 |
| layers.11.self_attn.q_proj | 11 | q_proj | 20.347 | 1.77 | 36.5 | 6.774 | 6.343 | 5.706 |
| layers.11.self_attn.v_proj | 11 | v_proj | 5.681 | 1.78 | 33.5 | 1.878 | 1.801 | 5.539 |
| layers.12.mlp.down_proj | 12 | down_proj | 16.057 | 1.78 | 28.1 | 5.609 | 2.899 | 8.931 |
| layers.12.mlp.gate_proj | 12 | gate_proj | 25.118 | 1.75 | 36.3 | 9.444 | 7.525 | 5.924 |
| layers.12.mlp.up_proj | 12 | up_proj | 20.368 | 1.78 | 42.0 | 6.606 | 6.583 | 5.662 |
| layers.13.mlp.down_proj | 13 | down_proj | 15.808 | 1.79 | 28.7 | 5.980 | 2.888 | 8.893 |
| layers.13.mlp.gate_proj | 13 | gate_proj | 24.397 | 1.79 | 37.2 | 8.402 | 7.374 | 5.893 |
| layers.13.mlp.up_proj | 13 | up_proj | 20.106 | 1.79 | 45.0 | 5.962 | 6.625 | 5.635 |
| layers.14.mlp.down_proj | 14 | down_proj | 14.201 | 1.75 | 31.8 | 4.575 | 2.706 | 8.746 |
| layers.14.mlp.gate_proj | 14 | gate_proj | 22.968 | 1.76 | 38.0 | 8.818 | 7.163 | 5.764 |
| layers.14.mlp.up_proj | 14 | up_proj | 20.248 | 1.77 | 45.8 | 5.835 | 6.780 | 5.581 |
| layers.15.mlp.down_proj | 15 | down_proj | 15.120 | 1.75 | 29.5 | 5.227 | 2.807 | 8.810 |
| layers.15.mlp.gate_proj | 15 | gate_proj | 24.431 | 1.78 | 38.9 | 8.771 | 7.463 | 5.890 |
| layers.15.mlp.up_proj | 15 | up_proj | 20.574 | 1.74 | 44.6 | 6.086 | 6.841 | 5.588 |
| layers.15.self_attn.k_proj | 15 | k_proj | 6.275 | 1.81 | 32.1 | 2.082 | 1.906 | 5.690 |
| layers.15.self_attn.o_proj | 15 | o_proj | 8.814 | 1.77 | 39.1 | 2.819 | 2.648 | 5.972 |
| layers.15.self_attn.q_proj | 15 | q_proj | 20.808 | 1.78 | 40.4 | 6.075 | 6.696 | 5.655 |
| layers.15.self_attn.v_proj | 15 | v_proj | 5.445 | 1.84 | 35.3 | 1.759 | 1.784 | 5.421 |
| layers.16.mlp.down_proj | 16 | down_proj | 14.693 | 1.76 | 30.8 | 4.891 | 2.804 | 8.676 |
| layers.16.mlp.gate_proj | 16 | gate_proj | 24.723 | 1.72 | 39.2 | 8.706 | 7.614 | 5.865 |
| layers.16.mlp.up_proj | 16 | up_proj | 20.657 | 1.79 | 45.0 | 6.184 | 6.897 | 5.578 |
| layers.17.mlp.down_proj | 17 | down_proj | 14.038 | 1.76 | 32.9 | 4.367 | 2.773 | 8.518 |
| layers.17.mlp.gate_proj | 17 | gate_proj | 25.162 | 1.78 | 35.8 | 10.160 | 7.651 | 5.847 |
| layers.17.mlp.up_proj | 17 | up_proj | 19.598 | 1.79 | 42.8 | 6.926 | 6.455 | 5.607 |
| layers.18.mlp.down_proj | 18 | down_proj | 12.807 | 1.81 | 37.1 | 3.635 | 2.663 | 8.349 |
| layers.18.mlp.gate_proj | 18 | gate_proj | 23.485 | 1.78 | 39.9 | 8.527 | 7.450 | 5.746 |
| layers.18.mlp.up_proj | 18 | up_proj | 19.829 | 1.80 | 46.9 | 6.035 | 6.799 | 5.504 |
| layers.19.mlp.down_proj | 19 | down_proj | 14.419 | 1.77 | 32.9 | 4.709 | 2.798 | 8.656 |
| layers.19.mlp.gate_proj | 19 | gate_proj | 27.343 | 1.80 | 34.8 | 10.913 | 7.975 | 6.046 |
| layers.19.mlp.up_proj | 19 | up_proj | 20.714 | 1.81 | 45.4 | 6.535 | 6.849 | 5.652 |
| layers.19.self_attn.k_proj | 19 | k_proj | 5.809 | 1.83 | 38.7 | 1.764 | 1.880 | 5.584 |
| layers.19.self_attn.o_proj | 19 | o_proj | 10.807 | 1.79 | 40.2 | 3.306 | 2.984 | 6.505 |
| layers.19.self_attn.q_proj | 19 | q_proj | 21.088 | 1.78 | 37.6 | 7.608 | 6.602 | 5.771 |
| layers.19.self_attn.v_proj | 19 | v_proj | 5.091 | 1.81 | 39.9 | 1.408 | 1.701 | 5.446 |
| layers.20.mlp.down_proj | 20 | down_proj | 15.317 | 1.79 | 34.7 | 4.810 | 2.942 | 8.875 |
| layers.20.mlp.gate_proj | 20 | gate_proj | 28.111 | 1.79 | 29.8 | 10.745 | 7.740 | 6.163 |
| layers.20.mlp.up_proj | 20 | up_proj | 20.368 | 1.82 | 46.3 | 5.611 | 6.623 | 5.751 |
| layers.21.mlp.down_proj | 21 | down_proj | 15.651 | 1.76 | 36.8 | 4.656 | 2.959 | 9.154 |
| layers.21.mlp.gate_proj | 21 | gate_proj | 28.540 | 1.77 | 31.8 | 11.020 | 7.857 | 6.250 |
| layers.21.mlp.up_proj | 21 | up_proj | 21.020 | 1.78 | 46.3 | 5.566 | 6.722 | 5.842 |
| layers.22.mlp.down_proj | 22 | down_proj | 15.106 | 1.78 | 34.1 | 5.235 | 2.819 | 9.151 |
| layers.22.mlp.gate_proj | 22 | gate_proj | 29.252 | 1.79 | 30.8 | 11.930 | 7.909 | 6.322 |
| layers.22.mlp.up_proj | 22 | up_proj | 20.099 | 1.80 | 47.4 | 5.200 | 6.477 | 5.823 |
| layers.23.mlp.down_proj | 23 | down_proj | 15.526 | 1.78 | 35.8 | 5.168 | 2.911 | 9.191 |
| layers.23.mlp.gate_proj | 23 | gate_proj | 30.994 | 1.77 | 29.8 | 12.830 | 8.149 | 6.438 |
| layers.23.mlp.up_proj | 23 | up_proj | 21.136 | 1.82 | 48.2 | 5.554 | 6.726 | 5.913 |
| layers.23.self_attn.k_proj | 23 | k_proj | 6.468 | 1.74 | 33.0 | 2.360 | 1.926 | 5.810 |
| layers.23.self_attn.o_proj | 23 | o_proj | 13.182 | 1.78 | 31.6 | 4.372 | 3.279 | 6.790 |
| layers.23.self_attn.q_proj | 23 | q_proj | 23.333 | 1.68 | 26.1 | 10.300 | 6.566 | 5.953 |
| layers.23.self_attn.v_proj | 23 | v_proj | 6.603 | 1.81 | 32.3 | 2.147 | 1.970 | 5.779 |
| layers.24.mlp.down_proj | 24 | down_proj | 15.475 | 1.79 | 35.0 | 5.081 | 2.878 | 9.202 |
| layers.24.mlp.gate_proj | 24 | gate_proj | 32.237 | 1.79 | 25.8 | 14.395 | 8.074 | 6.536 |
| layers.24.mlp.up_proj | 24 | up_proj | 20.765 | 1.82 | 48.0 | 5.148 | 6.638 | 5.879 |
| layers.25.mlp.down_proj | 25 | down_proj | 15.389 | 1.79 | 35.1 | 5.515 | 2.879 | 9.178 |
| layers.25.mlp.gate_proj | 25 | gate_proj | 32.345 | 1.79 | 24.2 | 15.270 | 8.074 | 6.480 |
| layers.25.mlp.up_proj | 25 | up_proj | 20.888 | 1.81 | 48.2 | 5.282 | 6.659 | 5.900 |
| layers.26.mlp.down_proj | 26 | down_proj | 14.463 | 1.78 | 30.3 | 5.644 | 2.729 | 8.781 |
| layers.26.mlp.gate_proj | 26 | gate_proj | 31.387 | 1.79 | 24.1 | 16.095 | 7.897 | 6.475 |
| layers.26.mlp.up_proj | 26 | up_proj | 20.831 | 1.83 | 47.6 | 5.208 | 6.643 | 5.879 |
| layers.27.mlp.down_proj | 27 | down_proj | 15.951 | 1.78 | 33.1 | 5.356 | 2.879 | 9.338 |
| layers.27.mlp.gate_proj | 27 | gate_proj | 33.578 | 1.82 | 25.2 | 15.220 | 8.124 | 6.715 |
| layers.27.mlp.up_proj | 27 | up_proj | 24.532 | 1.81 | 42.4 | 6.672 | 7.301 | 6.146 |
| layers.27.self_attn.k_proj | 27 | k_proj | 6.203 | 1.82 | 32.8 | 2.230 | 1.864 | 5.788 |
| layers.27.self_attn.o_proj | 27 | o_proj | 14.022 | 1.80 | 30.6 | 4.329 | 3.385 | 6.872 |
| layers.27.self_attn.q_proj | 27 | q_proj | 27.417 | 1.81 | 17.0 | 16.490 | 7.077 | 6.061 |
| layers.27.self_attn.v_proj | 27 | v_proj | 6.445 | 1.80 | 31.5 | 2.310 | 1.905 | 5.813 |
| layers.28.mlp.down_proj | 28 | down_proj | 15.680 | 1.77 | 36.5 | 4.806 | 2.991 | 9.069 |
| layers.28.mlp.gate_proj | 28 | gate_proj | 32.781 | 1.81 | 25.8 | 14.745 | 8.096 | 6.650 |
| layers.28.mlp.up_proj | 28 | up_proj | 25.883 | 1.80 | 41.4 | 7.470 | 7.625 | 6.188 |
| layers.29.mlp.down_proj | 29 | down_proj | 16.974 | 1.80 | 34.6 | 5.603 | 3.149 | 9.198 |
| layers.29.mlp.gate_proj | 29 | gate_proj | 32.473 | 1.82 | 26.9 | 14.691 | 8.178 | 6.593 |
| layers.29.mlp.up_proj | 29 | up_proj | 26.233 | 1.82 | 40.2 | 8.921 | 7.829 | 6.109 |
| layers.30.mlp.down_proj | 30 | down_proj | 17.257 | 1.81 | 29.2 | 7.410 | 3.157 | 9.004 |
| layers.30.mlp.gate_proj | 30 | gate_proj | 30.900 | 1.85 | 26.7 | 13.528 | 8.059 | 6.390 |
| layers.30.mlp.up_proj | 30 | up_proj | 26.273 | 1.82 | 39.6 | 9.312 | 7.881 | 6.079 |
| layers.31.mlp.down_proj | 31 | down_proj | 12.526 | 1.83 | 33.3 | 5.223 | 2.636 | 8.172 |
| layers.31.mlp.gate_proj | 31 | gate_proj | 29.953 | 1.82 | 27.6 | 13.140 | 8.145 | 6.235 |
| layers.31.mlp.up_proj | 31 | up_proj | 23.721 | 1.83 | 43.2 | 7.221 | 7.789 | 5.699 |
| layers.31.self_attn.k_proj | 31 | k_proj | 6.549 | 1.82 | 33.9 | 2.170 | 1.990 | 5.735 |
| layers.31.self_attn.o_proj | 31 | o_proj | 10.834 | 1.77 | 28.7 | 4.453 | 2.943 | 6.185 |
| layers.31.self_attn.q_proj | 31 | q_proj | 22.956 | 1.80 | 28.2 | 11.040 | 6.731 | 5.874 |
| layers.31.self_attn.v_proj | 31 | v_proj | 7.066 | 1.80 | 28.0 | 3.185 | 2.049 | 5.838 |

## C0 reading of the practice set (this adapter as it is)

```json
{
 "what": "C0: this adapter as it is, 32 practice pages, greedy, cap 3072 (batched engine, not exactly repeatable)",
 "metrics": {
  "pages": 32,
  "chars": 53112,
  "cer": 2.5028995330622084,
  "truncated_cer": 0.8242958276848923,
  "length_ratio": 2.5501958126223827,
  "at_cap": 29,
  "loops": 28
 },
 "per_page_cer": [
  23.636,
  0.773,
  17.921,
  1.187,
  18.869,
  0.983,
  23.479,
  17.199,
  0.765,
  1.001,
  0.992,
  10.857,
  0.814,
  2.147,
  2.159,
  5.927,
  2.117,
  0.911,
  5.807,
  5.304,
  4.156,
  0.93,
  5.062,
  3.806,
  1.467,
  5.077,
  3.157,
  1.925,
  8.384,
  3.096,
  7.257,
  4.967
 ]
}
```

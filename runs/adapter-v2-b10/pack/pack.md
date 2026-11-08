# Reading pack — adapter `qwen3.5-4b-german-kurrent-xvi-xix-page-v2`

This adapter may be defective. It is a peft LoRA directory (rank r, scale = lora_alpha / r) on Qwen3.5-4B,
trained to transcribe handwritten pages; the update of a module is ΔW = (lora_alpha/r)·B·A. No weights are shown,
only statistics. Siblings: adapters of the same base and recipe from other runs (healthy as far as known).

## Config

```json
{
 "base_model_name_or_path": "Qwen/Qwen3.5-4B",
 "lora_alpha": 128,
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

Scale of this adapter: 2.0; siblings: v1=2.0, blb1=2.0

## Totals

- ‖ΔW‖ over all modules (Frobenius, with this adapter's own scale): 117.49
- sibling v1: 127.35
- sibling blb1: 135.65

## Per module

ratio = this module's ‖ΔW‖ / median over siblings of the same module's ‖ΔW‖; effrank = exp-entropy effective rank (max r).

| module | layer | kind | ‖ΔW‖ | ratio | effrank | σ1 | ‖B‖ | ‖A‖ |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| layers.0.mlp.down_proj | 0 | down_proj | 10.777 | 0.93 | 21.8 | 3.957 | 3.265 | 9.747 |
| layers.0.mlp.gate_proj | 0 | gate_proj | 16.494 | 0.91 | 16.6 | 9.459 | 7.438 | 6.538 |
| layers.0.mlp.up_proj | 0 | up_proj | 14.508 | 0.91 | 27.4 | 5.529 | 7.279 | 6.516 |
| layers.1.mlp.down_proj | 1 | down_proj | 9.587 | 0.90 | 28.9 | 3.313 | 3.168 | 9.776 |
| layers.1.mlp.gate_proj | 1 | gate_proj | 15.456 | 0.92 | 21.6 | 8.149 | 7.657 | 6.380 |
| layers.1.mlp.up_proj | 1 | up_proj | 12.797 | 0.90 | 37.8 | 3.781 | 7.186 | 6.302 |
| layers.2.mlp.down_proj | 2 | down_proj | 8.707 | 0.91 | 34.8 | 2.714 | 3.019 | 9.823 |
| layers.2.mlp.gate_proj | 2 | gate_proj | 14.605 | 0.91 | 32.8 | 6.307 | 7.908 | 6.401 |
| layers.2.mlp.up_proj | 2 | up_proj | 10.293 | 0.90 | 48.1 | 2.388 | 6.598 | 5.855 |
| layers.3.mlp.down_proj | 3 | down_proj | 9.080 | 0.88 | 35.4 | 2.696 | 3.120 | 9.973 |
| layers.3.mlp.gate_proj | 3 | gate_proj | 14.068 | 0.89 | 37.3 | 5.239 | 7.970 | 6.289 |
| layers.3.mlp.up_proj | 3 | up_proj | 10.874 | 0.88 | 46.2 | 3.360 | 6.820 | 5.952 |
| layers.3.self_attn.k_proj | 3 | k_proj | 4.092 | 0.92 | 16.6 | 2.198 | 2.014 | 5.979 |
| layers.3.self_attn.o_proj | 3 | o_proj | 5.618 | 0.86 | 21.8 | 2.753 | 2.762 | 6.369 |
| layers.3.self_attn.q_proj | 3 | q_proj | 11.137 | 0.80 | 25.8 | 5.237 | 6.016 | 6.107 |
| layers.3.self_attn.v_proj | 3 | v_proj | 3.421 | 0.85 | 18.0 | 1.819 | 1.828 | 5.768 |
| layers.4.mlp.down_proj | 4 | down_proj | 8.726 | 0.90 | 30.6 | 3.041 | 3.013 | 9.578 |
| layers.4.mlp.gate_proj | 4 | gate_proj | 14.016 | 0.89 | 34.1 | 5.804 | 7.818 | 6.276 |
| layers.4.mlp.up_proj | 4 | up_proj | 10.505 | 0.89 | 46.5 | 3.159 | 6.667 | 5.885 |
| layers.5.mlp.down_proj | 5 | down_proj | 8.266 | 0.89 | 29.5 | 3.277 | 2.989 | 9.106 |
| layers.5.mlp.gate_proj | 5 | gate_proj | 13.888 | 0.91 | 28.2 | 6.682 | 7.459 | 6.273 |
| layers.5.mlp.up_proj | 5 | up_proj | 10.606 | 0.90 | 43.7 | 3.289 | 6.671 | 5.861 |
| layers.6.mlp.down_proj | 6 | down_proj | 7.813 | 0.89 | 29.6 | 3.117 | 2.927 | 8.772 |
| layers.6.mlp.gate_proj | 6 | gate_proj | 12.857 | 0.90 | 34.1 | 5.228 | 7.556 | 5.969 |
| layers.6.mlp.up_proj | 6 | up_proj | 10.384 | 0.89 | 45.8 | 3.051 | 6.738 | 5.745 |
| layers.7.mlp.down_proj | 7 | down_proj | 8.372 | 0.88 | 30.4 | 2.889 | 3.002 | 9.183 |
| layers.7.mlp.gate_proj | 7 | gate_proj | 13.552 | 0.88 | 36.0 | 5.014 | 7.749 | 6.171 |
| layers.7.mlp.up_proj | 7 | up_proj | 10.002 | 0.89 | 46.5 | 2.830 | 6.460 | 5.778 |
| layers.7.self_attn.k_proj | 7 | k_proj | 3.587 | 0.89 | 23.1 | 1.587 | 1.965 | 5.818 |
| layers.7.self_attn.o_proj | 7 | o_proj | 4.636 | 0.88 | 31.3 | 1.547 | 2.617 | 6.035 |
| layers.7.self_attn.q_proj | 7 | q_proj | 10.907 | 0.89 | 31.7 | 4.524 | 6.526 | 5.793 |
| layers.7.self_attn.v_proj | 7 | v_proj | 3.154 | 0.90 | 32.8 | 1.053 | 1.957 | 5.623 |
| layers.8.mlp.down_proj | 8 | down_proj | 8.736 | 0.88 | 27.4 | 3.281 | 3.041 | 9.241 |
| layers.8.mlp.gate_proj | 8 | gate_proj | 13.727 | 0.86 | 38.1 | 4.508 | 7.939 | 6.163 |
| layers.8.mlp.up_proj | 8 | up_proj | 10.158 | 0.88 | 45.5 | 3.030 | 6.508 | 5.797 |
| layers.9.mlp.down_proj | 9 | down_proj | 8.669 | 0.88 | 25.3 | 3.805 | 2.977 | 9.212 |
| layers.9.mlp.gate_proj | 9 | gate_proj | 13.560 | 0.88 | 31.4 | 6.217 | 7.700 | 6.070 |
| layers.9.mlp.up_proj | 9 | up_proj | 10.436 | 0.90 | 44.1 | 3.253 | 6.630 | 5.809 |
| layers.10.mlp.down_proj | 10 | down_proj | 7.759 | 0.87 | 26.4 | 2.940 | 2.798 | 8.786 |
| layers.10.mlp.gate_proj | 10 | gate_proj | 11.909 | 0.89 | 37.1 | 4.036 | 7.283 | 5.826 |
| layers.10.mlp.up_proj | 10 | up_proj | 10.597 | 0.88 | 44.0 | 3.189 | 6.906 | 5.673 |
| layers.11.mlp.down_proj | 11 | down_proj | 8.147 | 0.88 | 26.7 | 3.202 | 2.878 | 9.044 |
| layers.11.mlp.gate_proj | 11 | gate_proj | 13.069 | 0.88 | 33.6 | 5.394 | 7.714 | 5.918 |
| layers.11.mlp.up_proj | 11 | up_proj | 10.182 | 0.88 | 44.2 | 3.084 | 6.626 | 5.689 |
| layers.11.self_attn.k_proj | 11 | k_proj | 3.326 | 0.91 | 27.7 | 1.198 | 1.955 | 5.684 |
| layers.11.self_attn.o_proj | 11 | o_proj | 4.415 | 0.91 | 32.9 | 1.476 | 2.605 | 5.864 |
| layers.11.self_attn.q_proj | 11 | q_proj | 10.173 | 0.89 | 36.5 | 3.387 | 6.343 | 5.706 |
| layers.11.self_attn.v_proj | 11 | v_proj | 2.840 | 0.89 | 33.5 | 0.939 | 1.801 | 5.539 |
| layers.12.mlp.down_proj | 12 | down_proj | 8.028 | 0.89 | 28.1 | 2.804 | 2.899 | 8.931 |
| layers.12.mlp.gate_proj | 12 | gate_proj | 12.559 | 0.87 | 36.3 | 4.722 | 7.525 | 5.924 |
| layers.12.mlp.up_proj | 12 | up_proj | 10.184 | 0.89 | 42.0 | 3.303 | 6.583 | 5.662 |
| layers.13.mlp.down_proj | 13 | down_proj | 7.904 | 0.89 | 28.7 | 2.990 | 2.888 | 8.893 |
| layers.13.mlp.gate_proj | 13 | gate_proj | 12.199 | 0.89 | 37.2 | 4.201 | 7.374 | 5.893 |
| layers.13.mlp.up_proj | 13 | up_proj | 10.053 | 0.90 | 45.0 | 2.981 | 6.625 | 5.635 |
| layers.14.mlp.down_proj | 14 | down_proj | 7.100 | 0.88 | 31.8 | 2.288 | 2.706 | 8.746 |
| layers.14.mlp.gate_proj | 14 | gate_proj | 11.484 | 0.88 | 38.0 | 4.409 | 7.163 | 5.764 |
| layers.14.mlp.up_proj | 14 | up_proj | 10.124 | 0.89 | 45.8 | 2.917 | 6.780 | 5.581 |
| layers.15.mlp.down_proj | 15 | down_proj | 7.560 | 0.88 | 29.5 | 2.613 | 2.807 | 8.810 |
| layers.15.mlp.gate_proj | 15 | gate_proj | 12.215 | 0.89 | 38.9 | 4.385 | 7.463 | 5.890 |
| layers.15.mlp.up_proj | 15 | up_proj | 10.287 | 0.87 | 44.6 | 3.043 | 6.841 | 5.588 |
| layers.15.self_attn.k_proj | 15 | k_proj | 3.137 | 0.91 | 32.1 | 1.041 | 1.906 | 5.690 |
| layers.15.self_attn.o_proj | 15 | o_proj | 4.407 | 0.88 | 39.1 | 1.409 | 2.648 | 5.972 |
| layers.15.self_attn.q_proj | 15 | q_proj | 10.404 | 0.89 | 40.4 | 3.038 | 6.696 | 5.655 |
| layers.15.self_attn.v_proj | 15 | v_proj | 2.722 | 0.92 | 35.3 | 0.880 | 1.784 | 5.421 |
| layers.16.mlp.down_proj | 16 | down_proj | 7.347 | 0.88 | 30.8 | 2.446 | 2.804 | 8.676 |
| layers.16.mlp.gate_proj | 16 | gate_proj | 12.362 | 0.86 | 39.2 | 4.353 | 7.614 | 5.865 |
| layers.16.mlp.up_proj | 16 | up_proj | 10.328 | 0.90 | 45.0 | 3.092 | 6.897 | 5.578 |
| layers.17.mlp.down_proj | 17 | down_proj | 7.019 | 0.88 | 32.9 | 2.184 | 2.773 | 8.518 |
| layers.17.mlp.gate_proj | 17 | gate_proj | 12.581 | 0.89 | 35.8 | 5.080 | 7.651 | 5.847 |
| layers.17.mlp.up_proj | 17 | up_proj | 9.799 | 0.89 | 42.8 | 3.463 | 6.455 | 5.607 |
| layers.18.mlp.down_proj | 18 | down_proj | 6.403 | 0.90 | 37.1 | 1.818 | 2.663 | 8.349 |
| layers.18.mlp.gate_proj | 18 | gate_proj | 11.742 | 0.89 | 39.9 | 4.263 | 7.450 | 5.746 |
| layers.18.mlp.up_proj | 18 | up_proj | 9.915 | 0.90 | 46.9 | 3.018 | 6.799 | 5.504 |
| layers.19.mlp.down_proj | 19 | down_proj | 7.210 | 0.89 | 32.9 | 2.355 | 2.798 | 8.656 |
| layers.19.mlp.gate_proj | 19 | gate_proj | 13.671 | 0.90 | 34.8 | 5.457 | 7.975 | 6.046 |
| layers.19.mlp.up_proj | 19 | up_proj | 10.357 | 0.90 | 45.4 | 3.267 | 6.849 | 5.652 |
| layers.19.self_attn.k_proj | 19 | k_proj | 2.904 | 0.91 | 38.7 | 0.882 | 1.880 | 5.584 |
| layers.19.self_attn.o_proj | 19 | o_proj | 5.404 | 0.89 | 40.2 | 1.653 | 2.984 | 6.505 |
| layers.19.self_attn.q_proj | 19 | q_proj | 10.544 | 0.89 | 37.6 | 3.804 | 6.602 | 5.771 |
| layers.19.self_attn.v_proj | 19 | v_proj | 2.546 | 0.90 | 39.9 | 0.704 | 1.701 | 5.446 |
| layers.20.mlp.down_proj | 20 | down_proj | 7.659 | 0.90 | 34.7 | 2.405 | 2.942 | 8.875 |
| layers.20.mlp.gate_proj | 20 | gate_proj | 14.055 | 0.90 | 29.8 | 5.372 | 7.740 | 6.163 |
| layers.20.mlp.up_proj | 20 | up_proj | 10.184 | 0.91 | 46.3 | 2.805 | 6.623 | 5.751 |
| layers.21.mlp.down_proj | 21 | down_proj | 7.826 | 0.88 | 36.8 | 2.328 | 2.959 | 9.154 |
| layers.21.mlp.gate_proj | 21 | gate_proj | 14.270 | 0.88 | 31.8 | 5.510 | 7.857 | 6.250 |
| layers.21.mlp.up_proj | 21 | up_proj | 10.510 | 0.89 | 46.3 | 2.783 | 6.722 | 5.842 |
| layers.22.mlp.down_proj | 22 | down_proj | 7.553 | 0.89 | 34.1 | 2.617 | 2.819 | 9.151 |
| layers.22.mlp.gate_proj | 22 | gate_proj | 14.626 | 0.90 | 30.8 | 5.965 | 7.909 | 6.322 |
| layers.22.mlp.up_proj | 22 | up_proj | 10.050 | 0.90 | 47.4 | 2.600 | 6.477 | 5.823 |
| layers.23.mlp.down_proj | 23 | down_proj | 7.763 | 0.89 | 35.8 | 2.584 | 2.911 | 9.191 |
| layers.23.mlp.gate_proj | 23 | gate_proj | 15.497 | 0.88 | 29.8 | 6.415 | 8.149 | 6.438 |
| layers.23.mlp.up_proj | 23 | up_proj | 10.568 | 0.91 | 48.2 | 2.777 | 6.726 | 5.913 |
| layers.23.self_attn.k_proj | 23 | k_proj | 3.234 | 0.87 | 33.0 | 1.180 | 1.926 | 5.810 |
| layers.23.self_attn.o_proj | 23 | o_proj | 6.591 | 0.89 | 31.6 | 2.186 | 3.279 | 6.790 |
| layers.23.self_attn.q_proj | 23 | q_proj | 11.666 | 0.84 | 26.1 | 5.150 | 6.566 | 5.953 |
| layers.23.self_attn.v_proj | 23 | v_proj | 3.302 | 0.91 | 32.3 | 1.074 | 1.970 | 5.779 |
| layers.24.mlp.down_proj | 24 | down_proj | 7.737 | 0.90 | 35.0 | 2.541 | 2.878 | 9.202 |
| layers.24.mlp.gate_proj | 24 | gate_proj | 16.119 | 0.89 | 25.8 | 7.198 | 8.074 | 6.536 |
| layers.24.mlp.up_proj | 24 | up_proj | 10.382 | 0.91 | 48.0 | 2.574 | 6.638 | 5.879 |
| layers.25.mlp.down_proj | 25 | down_proj | 7.694 | 0.89 | 35.1 | 2.758 | 2.879 | 9.178 |
| layers.25.mlp.gate_proj | 25 | gate_proj | 16.173 | 0.90 | 24.2 | 7.635 | 8.074 | 6.480 |
| layers.25.mlp.up_proj | 25 | up_proj | 10.444 | 0.90 | 48.2 | 2.641 | 6.659 | 5.900 |
| layers.26.mlp.down_proj | 26 | down_proj | 7.232 | 0.89 | 30.3 | 2.822 | 2.729 | 8.781 |
| layers.26.mlp.gate_proj | 26 | gate_proj | 15.694 | 0.89 | 24.1 | 8.047 | 7.897 | 6.475 |
| layers.26.mlp.up_proj | 26 | up_proj | 10.416 | 0.92 | 47.6 | 2.604 | 6.643 | 5.879 |
| layers.27.mlp.down_proj | 27 | down_proj | 7.976 | 0.89 | 33.1 | 2.678 | 2.879 | 9.338 |
| layers.27.mlp.gate_proj | 27 | gate_proj | 16.789 | 0.91 | 25.2 | 7.610 | 8.124 | 6.715 |
| layers.27.mlp.up_proj | 27 | up_proj | 12.266 | 0.91 | 42.4 | 3.336 | 7.301 | 6.146 |
| layers.27.self_attn.k_proj | 27 | k_proj | 3.102 | 0.91 | 32.8 | 1.115 | 1.864 | 5.788 |
| layers.27.self_attn.o_proj | 27 | o_proj | 7.011 | 0.90 | 30.6 | 2.164 | 3.385 | 6.872 |
| layers.27.self_attn.q_proj | 27 | q_proj | 13.709 | 0.91 | 17.0 | 8.245 | 7.077 | 6.061 |
| layers.27.self_attn.v_proj | 27 | v_proj | 3.222 | 0.90 | 31.5 | 1.155 | 1.905 | 5.813 |
| layers.28.mlp.down_proj | 28 | down_proj | 7.840 | 0.89 | 36.5 | 2.403 | 2.991 | 9.069 |
| layers.28.mlp.gate_proj | 28 | gate_proj | 16.391 | 0.91 | 25.8 | 7.372 | 8.096 | 6.650 |
| layers.28.mlp.up_proj | 28 | up_proj | 12.942 | 0.90 | 41.4 | 3.735 | 7.625 | 6.188 |
| layers.29.mlp.down_proj | 29 | down_proj | 8.487 | 0.90 | 34.6 | 2.802 | 3.149 | 9.198 |
| layers.29.mlp.gate_proj | 29 | gate_proj | 16.237 | 0.91 | 26.9 | 7.346 | 8.178 | 6.593 |
| layers.29.mlp.up_proj | 29 | up_proj | 13.116 | 0.91 | 40.2 | 4.460 | 7.829 | 6.109 |
| layers.30.mlp.down_proj | 30 | down_proj | 8.629 | 0.91 | 29.2 | 3.705 | 3.157 | 9.004 |
| layers.30.mlp.gate_proj | 30 | gate_proj | 15.450 | 0.92 | 26.7 | 6.764 | 8.059 | 6.390 |
| layers.30.mlp.up_proj | 30 | up_proj | 13.136 | 0.91 | 39.6 | 4.656 | 7.881 | 6.079 |
| layers.31.mlp.down_proj | 31 | down_proj | 6.263 | 0.91 | 33.3 | 2.612 | 2.636 | 8.172 |
| layers.31.mlp.gate_proj | 31 | gate_proj | 14.977 | 0.91 | 27.6 | 6.570 | 8.145 | 6.235 |
| layers.31.mlp.up_proj | 31 | up_proj | 11.860 | 0.92 | 43.2 | 3.610 | 7.789 | 5.699 |
| layers.31.self_attn.k_proj | 31 | k_proj | 3.275 | 0.91 | 33.9 | 1.085 | 1.990 | 5.735 |
| layers.31.self_attn.o_proj | 31 | o_proj | 5.417 | 0.88 | 28.7 | 2.226 | 2.943 | 6.185 |
| layers.31.self_attn.q_proj | 31 | q_proj | 11.478 | 0.90 | 28.2 | 5.520 | 6.731 | 5.874 |
| layers.31.self_attn.v_proj | 31 | v_proj | 3.533 | 0.90 | 28.0 | 1.593 | 2.049 | 5.838 |

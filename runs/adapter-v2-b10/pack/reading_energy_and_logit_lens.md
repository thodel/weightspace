# Adapter reading — qwen3.5-4b-german-kurrent-xvi-xix-page-v2 (budget 0)

`/home/tobias/atr-cache/wsp/adapters/qwen3.5-4b-german-kurrent-xvi-xix-page-v2` — 128 modules, r = 64, scale α/r = 2.0. Budget 0: no forward pass.

## Energy by layer and kind (share of Σ‖ΔW‖²_F)

| layer | q_proj | k_proj | v_proj | o_proj | gate_proj | up_proj | down_proj |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 |  |  |  |  | 1.97 | 1.52 | 0.84 |
| 1 |  |  |  |  | 1.73 | 1.19 | 0.67 |
| 2 |  |  |  |  | 1.55 | 0.77 | 0.55 |
| 3 | 0.90 | 0.12 | 0.08 | 0.23 | 1.43 | 0.86 | 0.60 |
| 4 |  |  |  |  | 1.42 | 0.80 | 0.55 |
| 5 |  |  |  |  | 1.40 | 0.81 | 0.49 |
| 6 |  |  |  |  | 1.20 | 0.78 | 0.44 |
| 7 | 0.86 | 0.09 | 0.07 | 0.16 | 1.33 | 0.72 | 0.51 |
| 8 |  |  |  |  | 1.37 | 0.75 | 0.55 |
| 9 |  |  |  |  | 1.33 | 0.79 | 0.54 |
| 10 |  |  |  |  | 1.03 | 0.81 | 0.44 |
| 11 | 0.75 | 0.08 | 0.06 | 0.14 | 1.24 | 0.75 | 0.48 |
| 12 |  |  |  |  | 1.14 | 0.75 | 0.47 |
| 13 |  |  |  |  | 1.08 | 0.73 | 0.45 |
| 14 |  |  |  |  | 0.96 | 0.74 | 0.37 |
| 15 | 0.78 | 0.07 | 0.05 | 0.14 | 1.08 | 0.77 | 0.41 |
| 16 |  |  |  |  | 1.11 | 0.77 | 0.39 |
| 17 |  |  |  |  | 1.15 | 0.70 | 0.36 |
| 18 |  |  |  |  | 1.00 | 0.71 | 0.30 |
| 19 | 0.81 | 0.06 | 0.05 | 0.21 | 1.35 | 0.78 | 0.38 |
| 20 |  |  |  |  | 1.43 | 0.75 | 0.42 |
| 21 |  |  |  |  | 1.48 | 0.80 | 0.44 |
| 22 |  |  |  |  | 1.55 | 0.73 | 0.41 |
| 23 | 0.99 | 0.08 | 0.08 | 0.31 | 1.74 | 0.81 | 0.44 |
| 24 |  |  |  |  | 1.88 | 0.78 | 0.43 |
| 25 |  |  |  |  | 1.89 | 0.79 | 0.43 |
| 26 |  |  |  |  | 1.78 | 0.79 | 0.38 |
| 27 | 1.36 | 0.07 | 0.08 | 0.36 | 2.04 | 1.09 | 0.46 |
| 28 |  |  |  |  | 1.95 | 1.21 | 0.45 |
| 29 |  |  |  |  | 1.91 | 1.25 | 0.52 |
| 30 |  |  |  |  | 1.73 | 1.25 | 0.54 |
| 31 | 0.95 | 0.08 | 0.09 | 0.21 | 1.62 | 1.02 | 0.28 |

## Residual-writing directions that lower the end-of-turn token most (logit lens)

| module | dir | σ | z(<\|im_end\|>) | z(\n) | z(¬) | promotes | suppresses |
|---|---:|---:|---:|---:|---:|---|---|
| layers.31.self_attn.o_proj | 0 | 2.226 | -3.60 | -5.52 | -8.06 | ` Nhan` `่ง` `_WH` `unami` ` Vej` | `Ġâľ` `¬` `ù` `âĤ` `Ë` |
| layers.31.mlp.down_proj | 0 | 2.612 | -3.32 | -1.95 | -0.93 | `暴` ` الأ` `霍` `å` `恐怖` | ` diser` ` fry` `惯` ` hatt` `âĤ` |
| layers.27.mlp.down_proj | 0 | 2.678 | -2.80 | -0.87 | -1.08 | `令和` `chief` `טים` `avail` `čuje` | `هن` `卷` `alis` `viz` `扩散` |
| layers.14.mlp.down_proj | 2 | 2.001 | -2.67 | -3.20 | +0.33 | `оро` `rž` `生物学` `騰` `职` | `chia` ` Av` ` impair` `/h` `-pr` |
| layers.3.mlp.down_proj | 0 | 2.696 | -2.48 | -1.91 | +0.94 | ` scomparsa` `真的好` `棲` `isku` `RAFT` | ` écl` ` Addison` `เ` `内在` ` *` |
| layers.8.mlp.down_proj | 0 | 3.281 | -2.29 | -1.16 | +0.42 | `emann` `amakan` `伟业` `缘` ` cánh` | ` Giovan` `星` `живает` `rots` `比较多` |
| layers.26.mlp.down_proj | 2 | 1.939 | -2.10 | -2.20 | +1.70 | `积` ` conlle` `GH` `unge` `ange` | `斗` `èĿ` `分项` `ساء` `بين` |
| layers.2.mlp.down_proj | 0 | 2.714 | -1.86 | +1.70 | +1.34 | `活动等` `uvent` `字第` `ncer` ` tranquil` | ` Gone` `reed` `usun` `led` `ebi` |
| layers.22.mlp.down_proj | 0 | 2.617 | -1.82 | -0.52 | -2.16 | `涉` ` Zuletzt` ` altid` `渉` `싱` | ` —` `åĨ` `直` `æĳ` ` mal` |
| layers.3.self_attn.o_proj | 0 | 2.753 | -1.76 | +0.20 | -1.15 | `رخ` `人类社会` `eltas` `َض` `月出生` | ` numbering` ` opér` `晴` `交替` `参考` |
| layers.1.mlp.down_proj | 1 | 2.935 | -1.74 | -1.73 | -2.87 | `底层` `版` `阶层` `怪` ` والآ` | `ungeons` `บาย` `数目` `amas` `三者` |
| layers.7.mlp.down_proj | 2 | 2.286 | -1.72 | -1.50 | +2.35 | `宜` `yst` `aman` ` elk` `issippi` | `导` `ORI` `nTo` `legt` ` Gewalt` |

## Residual-writing directions that raise it most (logit lens)

| module | dir | σ | z(<\|im_end\|>) | z(\n) | z(¬) | promotes | suppresses |
|---|---:|---:|---:|---:|---:|---|---|
| layers.30.mlp.down_proj | 0 | 3.705 | +2.85 | +2.05 | +0.98 | `Ë` `Ġâľ` ` bù` `âľ` `âĤ` | `成药` `iage` `的情怀` `дра` ` excer` |
| layers.30.mlp.down_proj | 2 | 2.250 | +2.27 | -2.23 | +2.44 | `Ë` `闸` `医科大学` `霖` `訳` | `每名` `珑` `每项` `Ao` ` #` |
| layers.4.mlp.down_proj | 1 | 2.874 | +2.13 | -0.23 | +0.79 | `ipl` `挑` `第三` `"/></` ` tre` | `萤` `之交` `urat` `ًا` `ول` |
| layers.9.mlp.down_proj | 0 | 3.805 | +1.99 | +3.24 | +1.92 | `ao` `__` `;` ` __` `…` | `bev` `utilise` `effectu` `fäh` ` podremos` |
| layers.14.mlp.down_proj | 1 | 2.190 | +1.92 | +1.84 | +0.61 | `速` `yre` `ypy` ` wil` `依法追究` | `udiant` `往届` `卧` `Ưu` `ogat` |
| layers.31.mlp.down_proj | 1 | 1.558 | +1.90 | +0.44 | -1.38 | `ieu` `̀` `opat` ` sit` `ターン` | `ùng` `oving` `ichtig` `正大` `ùn` |
| layers.8.mlp.down_proj | 1 | 3.068 | +1.77 | +2.26 | -0.08 | `ty` ` rest` `ract` `ank` `ajos` | `IKE` ` sugges` ` Fully` `提` `抬` |
| layers.13.mlp.down_proj | 2 | 2.275 | +1.72 | +1.87 | +0.57 | `iment` ` ​​` `段的` ` extern` `aal` | `stor` `历` `척` `רון` `weichen` |
| layers.16.mlp.down_proj | 3 | 1.853 | +1.69 | +3.46 | +0.49 | `sp` `oph` ` top` ` e` `ême` | `لان` `edly` `городе` `orada` `cich` |
| layers.20.mlp.down_proj | 1 | 2.088 | +1.64 | -0.13 | +0.11 | ` *` `份` ` thought` ` course` ` Yard` | `upati` `новь` `退票` `celet` ` anda` |
| layers.10.mlp.down_proj | 2 | 2.377 | +1.63 | -0.71 | -3.90 | `ajib` ` спис` `eman` `avs` `书院` | ` automat` `懂得` `otor` `acking` `illion` |
| layers.8.mlp.down_proj | 2 | 2.408 | +1.55 | +1.33 | +2.68 | `-` `ias` `S` `t` `za` | `uyu` `vegli` `CLUDING` `덩` `зум` |

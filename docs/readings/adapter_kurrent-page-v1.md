# Adapter reading — qwen3.5-4b-german-kurrent-xvi-xix-page-v1 (stand-in for v2)

`/home/tobias/atr-cache/wsp/adapters/qwen3.5-4b-german-kurrent-xvi-xix-page-v1` — 128 modules, r = 64, scale α/r = 2.0. Budget 0: no forward pass.

## Energy by layer and kind (share of Σ‖ΔW‖²_F)

| layer | q_proj | k_proj | v_proj | o_proj | gate_proj | up_proj | down_proj |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 |  |  |  |  | 1.95 | 1.51 | 0.79 |
| 1 |  |  |  |  | 1.69 | 1.19 | 0.68 |
| 2 |  |  |  |  | 1.51 | 0.75 | 0.53 |
| 3 | 1.10 | 0.11 | 0.09 | 0.24 | 1.46 | 0.87 | 0.61 |
| 4 |  |  |  |  | 1.40 | 0.78 | 0.54 |
| 5 |  |  |  |  | 1.30 | 0.81 | 0.49 |
| 6 |  |  |  |  | 1.18 | 0.78 | 0.44 |
| 7 | 0.88 | 0.09 | 0.07 | 0.16 | 1.35 | 0.72 | 0.52 |
| 8 |  |  |  |  | 1.45 | 0.77 | 0.56 |
| 9 |  |  |  |  | 1.39 | 0.76 | 0.55 |
| 10 |  |  |  |  | 1.05 | 0.81 | 0.45 |
| 11 | 0.74 | 0.08 | 0.06 | 0.13 | 1.26 | 0.76 | 0.48 |
| 12 |  |  |  |  | 1.17 | 0.75 | 0.46 |
| 13 |  |  |  |  | 1.07 | 0.71 | 0.44 |
| 14 |  |  |  |  | 0.98 | 0.74 | 0.37 |
| 15 | 0.77 | 0.07 | 0.05 | 0.14 | 1.06 | 0.78 | 0.43 |
| 16 |  |  |  |  | 1.20 | 0.76 | 0.40 |
| 17 |  |  |  |  | 1.15 | 0.70 | 0.36 |
| 18 |  |  |  |  | 0.99 | 0.71 | 0.29 |
| 19 | 0.81 | 0.06 | 0.05 | 0.21 | 1.34 | 0.76 | 0.39 |
| 20 |  |  |  |  | 1.42 | 0.72 | 0.42 |
| 21 |  |  |  |  | 1.50 | 0.81 | 0.46 |
| 22 |  |  |  |  | 1.55 | 0.73 | 0.41 |
| 23 | 1.08 | 0.08 | 0.08 | 0.32 | 1.79 | 0.78 | 0.44 |
| 24 |  |  |  |  | 1.92 | 0.76 | 0.43 |
| 25 |  |  |  |  | 1.90 | 0.78 | 0.43 |
| 26 |  |  |  |  | 1.80 | 0.76 | 0.39 |
| 27 | 1.33 | 0.07 | 0.07 | 0.35 | 1.97 | 1.07 | 0.47 |
| 28 |  |  |  |  | 1.93 | 1.20 | 0.46 |
| 29 |  |  |  |  | 1.90 | 1.23 | 0.52 |
| 30 |  |  |  |  | 1.64 | 1.22 | 0.53 |
| 31 | 0.96 | 0.08 | 0.09 | 0.22 | 1.59 | 0.99 | 0.28 |

## Residual-writing directions that lower the end-of-turn token most (logit lens)

| module | dir | σ | z(<\|im_end\|>) | z(\n) | z(¬) | promotes | suppresses |
|---|---:|---:|---:|---:|---:|---|---|
| layers.31.self_attn.o_proj | 0 | 2.600 | -3.95 | -4.97 | -7.14 | `vær` `ナン` `hausen` `itized` `雁` | `Ġâľ` `¬` `âľ` `Ë` `âĤ` |
| layers.27.self_attn.o_proj | 2 | 1.963 | -2.95 | -0.83 | -0.68 | ` Zir` `姓` `hini` `ndon` `桥镇` | `韦尔` ` freeze` `自己的人` `bound` ` soi` |
| layers.31.mlp.down_proj | 0 | 2.384 | -2.76 | -0.88 | -1.44 | `暴` ` \`` ` ^` ` ●` `益` | `ùng` `acquis` `人心的` `richter` ` diser` |
| layers.3.self_attn.o_proj | 0 | 2.908 | -2.43 | -2.86 | -0.18 | `مام` ` boule` `RIX` ` vén` `aupt` | `极` `政府` `elaar` `A` `ication` |
| layers.29.mlp.down_proj | 2 | 2.240 | -2.37 | +0.41 | +3.28 | `ース` ` okazji` `mars` `其身` `ieurs` | `豁` `ness` `×` `loid` `休` |
| layers.9.mlp.down_proj | 3 | 2.385 | -2.08 | -1.12 | -0.74 | ` estacionamento` `omie` ` impun` `omy` `ở` | `ocks` `וו` ` concerned` ` fr` `ICY` |
| layers.20.mlp.down_proj | 3 | 1.986 | -1.75 | -2.21 | +3.02 | `…*` `及配套` `...*` `치를` `BT` | ` tick` ` Tick` ` mar` `idt` ` who` |
| layers.12.mlp.down_proj | 3 | 2.240 | -1.72 | -1.06 | +0.88 | `gh` `inste` `úd` `avu` `čan` | `萃` `اعت` `poon` `onan` `ilhas` |
| layers.30.mlp.down_proj | 3 | 2.151 | -1.71 | -1.56 | +0.68 | `Kho` `ylan` `jen` `pot` `enge` | `zę` `ę` `墨` ` contrary` `bę` |
| layers.28.mlp.down_proj | 3 | 2.021 | -1.69 | -1.74 | -0.80 | ` RUR` `祖` `-wide` ` Blonde` `old` | `vet` `却有` `ife` ` though` `اء` |
| layers.11.mlp.down_proj | 3 | 2.250 | -1.68 | -3.37 | -1.53 | `enta` `加持` ` Academ` `大户` `在本站` | ` figure` `engk` `udar` `pen` ` jud` |
| layers.6.mlp.down_proj | 1 | 2.478 | -1.64 | -1.98 | -0.45 | `quests` `irim` `träg` `.tintColor` `essus` | `ose` `raid` `erm` `ера` `held` |

## Residual-writing directions that raise it most (logit lens)

| module | dir | σ | z(<\|im_end\|>) | z(\n) | z(¬) | promotes | suppresses |
|---|---:|---:|---:|---:|---:|---|---|
| layers.30.mlp.down_proj | 0 | 3.883 | +3.36 | +4.59 | +2.99 | `Ë` `Ġâľ` `ùng` ` ` ` Sid` | `owi` `empla` `一票` `日公布` `当头` |
| layers.27.self_attn.o_proj | 1 | 2.247 | +2.98 | +2.85 | +2.19 | ` ir` ` mot` `ورد` `不可` `pio` | `ownt` `oseb` `上乘` `сть` `也容易` |
| layers.4.mlp.down_proj | 1 | 2.998 | +2.97 | +1.20 | +0.97 | `oco` `印象` `r` `èĶ` `n` | `ialis` `คโนโล` `usif` `whelming` ` вза` |
| layers.3.mlp.down_proj | 0 | 3.369 | +2.29 | -0.29 | +0.66 | `高级` `ヵ` `“` `展` `ُ` | `лак` `arly` ` ame` `amkan` `nene` |
| layers.19.self_attn.o_proj | 3 | 1.319 | +2.28 | +0.61 | -0.88 | `alter` ` properly` ` Tata` `irt` `ucha` | `intér` `豺` `座的` `beda` `彦` |
| layers.23.self_attn.o_proj | 1 | 2.133 | +2.19 | +3.25 | +0.74 | `ToUpdate` `因为在` `什么的` `ennis` ` fare` | `,',` `è±` `vis` `ĸ` `uvent` |
| layers.4.mlp.down_proj | 3 | 2.365 | +2.08 | +1.22 | +1.19 | `===` `席` ` Conform` ` conform` `ira` | `ogne` `zens` `träg` `bsites` `ereien` |
| layers.7.self_attn.o_proj | 3 | 1.374 | +1.88 | +0.97 | -1.00 | `langen` `meen` ` Sorgen` `从不` ` Kunde` | `激` `edy` `级以上` `راب` `rost` |
| layers.29.mlp.down_proj | 0 | 3.528 | +1.87 | +4.06 | +3.86 | `Ġâľ` `n` `Ë` `bu` `ani` | ` lâm` `jder` `foon` ` olun` `erings` |
| layers.12.mlp.down_proj | 0 | 3.488 | +1.85 | +3.06 | -1.04 | `更加` `ly` `独` `umb` ` Ex` | `elit` `舷` `inste` ` confiscated` `慷` |
| layers.20.mlp.down_proj | 2 | 2.015 | +1.85 | +0.68 | -0.10 | ` Pitt` `LowerCase` `éd` `ظ` `loo` | `后退` ` Goed` `通过后` `bane` `apon` |
| layers.20.mlp.down_proj | 0 | 2.492 | +1.80 | +2.79 | +0.31 | `曝光` `暴露` `裸` `裸露` ` Freed` | `redit` `aporte` `古屋` `的重量` `厲` |

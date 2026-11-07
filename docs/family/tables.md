## Inventory: the gateway's kraken models

43 kraken models served for agentic_historian (gateway, 2026-11-07); 43 with weights on asteraix. *Arch* is the architecture group (below), *lineage* is inferred from the weights (body cosine > 0.5), *stated base* is the gateway registry's, checked against the weights.

| model | source | stated base (check) | scripts / centuries | arch | lineage |
|---|---|---|---|---|---:|
| bastarda_inzigkofen | 10.5281/zenodo.18207779 | 10.5281/zenodo.15030337 (confirmed) | Bastarda / 15 | A | 1 |
| bern-combined-v1 | in-house |  |  | A | 1 |
| bern-hertwig-v1 | in-house |  |  | A | 1 |
| bern-hertwig-v2 | in-house |  |  | A | 1 |
| bern-schlussredaktion-v1 | in-house |  |  | A | 1 |
| catmus_medieval | 10.5281/zenodo.15030337 |  | Medieval | A | 1 |
| cursive_inzigkofen | 10.5281/zenodo.18207767 | 10.5281/zenodo.15030337 (confirmed) | Cursive / 15 | A | 1 |
| old_norse_am305 | 10.5281/zenodo.20529753 |  | 13 | A | 1 |
| bohemian_19th | 10.5281/zenodo.11673242 |  | Kurrent / 19 | A | 2 |
| german-xix-blb-v1 | in-house |  |  | A | 2 |
| german-xix-v1 | in-house |  |  | A | 2 |
| glagolitic_print | 10.5281/zenodo.7755504 |  | Glagolitic / 16 | A | 4 |
| hebrew_sephardi | 10.5281/zenodo.5468665 |  | Sephardi bookhand | A | 4 |
| samaritan | 10.5281/zenodo.13814200 |  | Samaritan | A | 4 |
| openiti_arabic_print | 10.5281/zenodo.7050270 |  | Arabic | A | 5 |
| openiti_ottoman_print | 10.5281/zenodo.7050342 | 10.5281/zenodo.7050270 (confirmed) | Arabic | A | 5 |
| openiti_persian_print | 10.5281/zenodo.7051644 | 10.5281/zenodo.7050270 (confirmed) | Arabic | A | 5 |
| mccatmus | 10.5281/zenodo.13788177 |  | Latn / 16, 17, 18, 19, 20, 21 | A | 6 |
| prima | 10.5281/zenodo.18220238 |  | 16, 17, 18 | A | 6 |
| austrian_fraktur | 10.5281/zenodo.7933402 |  | Fraktur / 19 | A |  |
| bifrost_old_norse | 10.5281/zenodo.15366732 | 10.5281/zenodo.15030337 (contradicted: unrelated weights) |  | A |  |
| cyrillic_uncial | 10.5281/zenodo.7755483 |  | Cyrillic uncial / 11, 12, 13, 14, 15, 16 | A |  |
| fondue_gd_v2 | 10.5281/zenodo.21536798 |  | Latn / 16, 17, 18, 19, 20, 21 | A |  |
| gallicorpora | 10.5281/zenodo.7410529 |  | 16, 17, 18, 19 | A |  |
| german_print | 10.5281/zenodo.10519596 |  | Fraktur, Antiqua / 15, 16, 17, 18, 19, 20 | A |  |
| kuzushiji | 10.5281/zenodo.13942714 |  | Kuzushiji | A |  |
| latin_expanded_11c | 10.5281/zenodo.13736584 |  | 11 | A |  |
| latin_graphematic_11c | 10.5281/zenodo.13741957 |  | 11 | A |  |
| lectaurep_french | 10.5281/zenodo.6542744 |  | 18, 19, 20 | A |  |
| manu_mcfondue | 10.5281/zenodo.10886224 |  | 16, 17, 18, 19 | A |  |
| medieval_latin_french_expanded | 10.5281/zenodo.7516057 |  | Medieval / 12, 13, 14, 15 | A |  |
| mendelssohn_letters | 10.5281/zenodo.18207676 |  | 19 | A |  |
| peraire_french | 10.5281/zenodo.8193498 |  | 20 | A |  |
| catmus_print_large | 10.5281/zenodo.10592716 |  | 16, 17, 18, 19, 20, 21 | B | 3 |
| french_vietnamese | 10.5281/zenodo.17690418 | 10.5281/zenodo.10592716 (confirmed) | 19, 20 | B | 3 |
| latin_incunabula | 10.5281/zenodo.11113737 | 10.5281/zenodo.10592716 (confirmed) | Gothic, Antiqua / 15, 16 | B | 3 |
| catmus_gothic_print | 10.5281/zenodo.10599911 | 10.5281/zenodo.15030337 (different architecture: cannot be a fine-tune of this file) | Gothic / 16 | B |  |
| cremma_medieval | 10.5281/zenodo.7631619 |  | Medieval / 8, 9, 10, 11, 12, 13, 14, 15 | B |  |
| medieval_latin_french_abbreviated | 10.5281/zenodo.7516310 |  | Medieval / 12, 13, 14, 15 | B |  |
| textualis_inzigkofen | 10.5281/zenodo.18207719 | 10.5281/zenodo.13862096 (confirmed) | Textualis / 15 | C | 7 |
| tridis_v2 | 10.5281/zenodo.13862096 |  | Medieval / 11, 12, 13, 14, 15, 16 | C | 7 |
| english_print | 10.5281/zenodo.2577813 |  |  | D |  |
| estournelles_typewritten | 10.5281/zenodo.10556673 |  | Typewritten / 20 | E |  |

## Architecture groups

Models compare element by element only inside one group: same input height and the same tensor shapes outside the output layer (a different dropout rate, which has no weights, does not split a group).

| group | models | input height | parameter blocks | VGSL (first model) |
|---|---:|---:|---|---|
| A | 33 | 120 | C_0 C_3 C_6 C_9 L_12 L_14 L_16 O_18 | `[1,120,0,1 Cr{C_0}3,13,32 Do{Do_1}0.1,2 Mp{Mp_2}2,2 Cr{C_3}3,13,32 Do{Do_4}0.1,2 Mp{Mp_5}2,2 Cr{C_6}3,9,64 Do{Do_7}0.1,2 Mp{Mp_8}2,2 Cr{C_9}3,9,64 Do{` |
| B | 6 | 120 | C_0 Gn_1 C_2 Gn_3 C_5 Gn_6 L_9 L_11 L_13 O_15 | `[1,120,0,1 Cr{C_0}4,2,32,4,2 Gn{Gn_1}32 Cr{C_2}4,2,64,1,1 Gn{Gn_3}32 Mp{Mp_4}4,2,4,2 Cr{C_5}3,3,128,1,1 Gn{Gn_6}32 Mp{Mp_7}1,2,1,2 S{S_8}1(1x0)1,3 Lbx` |
| C | 2 | 140 | C_0 C_3 C_6 C_9 L_12 L_14 L_16 O_18 | `[1,140,0,1 Cr{C_0}4,16,32 Do{Do_1}0.1,2 Mp{Mp_2}2,2 Cr{C_3}4,16,32 Do{Do_4}0.1,2 Mp{Mp_5}2,2 Cr{C_6}3,8,64 Do{Do_7}0.1,2 Mp{Mp_8}2,2 Cr{C_9}3,8,64 Do{` |
| D | 1 | 48 | C_0 C_3 L_7 O_9 | `[1,48,0,1 Cr{C_0}3,3,32 Do{Do_1}0.1,2 Mp{Mp_2}2,2 Cr{C_3}3,3,64 Do{Do_4}0.1,2 Mp{Mp_5}2,2 S{S_6}1(1x12)1,3 Lbx{L_7}100 Do{Do_8} O{O_9}1c90]` |
| E | 1 | 48 | C_0 Gn_1 C_2 Gn_3 C_5 Gn_6 L_9 L_11 L_13 O_15 | `[1,48,0,1 Cr{C_0}4,2,32,4,2 Gn{Gn_1}32 Cr{C_2}4,2,64,1,1 Gn{Gn_3}32 Mp{Mp_4}4,2,4,2 Cr{C_5}3,3,128,1,1 Gn{Gn_6}32 Mp{Mp_7}1,2,1,2 S{S_8}1(1x0)1,3 Lbx{` |

## Lineages and stated bases

| lineage | arch | models |
|---:|---|---|
| 1 | A | bastarda_inzigkofen, bern-combined-v1, bern-hertwig-v1, bern-hertwig-v2, bern-schlussredaktion-v1, catmus_medieval, cursive_inzigkofen, old_norse_am305 |
| 2 | A | bohemian_19th, german-xix-blb-v1, german-xix-v1 |
| 3 | B | catmus_print_large, french_vietnamese, latin_incunabula |
| 4 | A | glagolitic_print, hebrew_sephardi, samaritan |
| 5 | A | openiti_arabic_print, openiti_ottoman_print, openiti_persian_print |
| 6 | A | mccatmus, prima |
| 7 | C | textualis_inzigkofen, tridis_v2 |

Distant relatives (body cosine 0.2–0.5; unrelated pairs sit below 0.2, every recorded fine-tune above 0.6): glagolitic_print – samaritan 0.49; cyrillic_uncial – hebrew_sephardi 0.37; cyrillic_uncial – glagolitic_print 0.31; cyrillic_uncial – samaritan 0.30; gallicorpora – medieval_latin_french_expanded 0.25; gallicorpora – hebrew_sephardi 0.24; hebrew_sephardi – medieval_latin_french_expanded 0.23.

| model | stated base | body cosine | verdict |
|---|---|---:|---|
| bastarda_inzigkofen | catmus_medieval | 0.775 | confirmed |
| cursive_inzigkofen | catmus_medieval | 0.768 | confirmed |
| textualis_inzigkofen | tridis_v2 | 0.947 | confirmed |
| french_vietnamese | catmus_print_large | 0.982 | confirmed |
| bifrost_old_norse | catmus_medieval | 0.080 | contradicted: unrelated weights |
| latin_incunabula | catmus_print_large | 0.946 | confirmed |
| catmus_gothic_print | catmus_medieval |  | different architecture: cannot be a fine-tune of this file |
| openiti_persian_print | openiti_arabic_print | 0.796 | confirmed |
| openiti_ottoman_print | openiti_arabic_print | 0.785 | confirmed |

## Group A: RQ1 identical parameters and RQ2 distance, related pairs

Each cell: share of *live* parameters that are bit-identical (structural zeros excluded) / relative distance ‖B − A‖ / mean(‖A‖, ‖B‖). The output layer on the characters both alphabets share.

| pair | body cosine | C_0 | C_3 | C_6 | C_9 | L_12 | L_14 | L_16 | O_18 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bern-hertwig-v1 – cursive_inzigkofen | 0.992 | 3.4 % / 0.07 | 11.1 % / 0.09 | 0.0 % / 0.13 | 0.0 % / 0.12 | 0.0 % / 0.10 | 0.0 % / 0.13 | 0.0 % / 0.15 | 0.0 % / 0.25 |
| bern-hertwig-v2 – bern-schlussredaktion-v1 | 0.980 | 12.9 % / 0.08 | 21.3 % / 0.14 | 0.0 % / 0.19 | 0.0 % / 0.20 | 0.0 % / 0.20 | 0.0 % / 0.19 | 0.0 % / 0.21 | 0.0 % / 0.15 |
| bern-hertwig-v1 – bern-hertwig-v2 | 0.974 | 3.3 % / 0.10 | 9.2 % / 0.17 | 0.0 % / 0.21 | 0.0 % / 0.21 | 0.0 % / 0.23 | 0.0 % / 0.22 | 0.0 % / 0.24 | 0.0 % / 0.14 |
| bern-combined-v1 – bern-schlussredaktion-v1 | 0.973 | 10.2 % / 0.09 | 17.5 % / 0.17 | 0.0 % / 0.21 | 0.0 % / 0.23 | 0.0 % / 0.26 | 0.0 % / 0.21 | 0.0 % / 0.21 | 0.0 % / 0.09 |
| bern-combined-v1 – bern-hertwig-v2 | 0.973 | 10.5 % / 0.09 | 17.9 % / 0.16 | 0.0 % / 0.21 | 0.0 % / 0.23 | 0.0 % / 0.23 | 0.0 % / 0.22 | 0.0 % / 0.24 | 0.0 % / 0.17 |
| mccatmus – prima | 0.966 | 0.0 % / 0.11 | 0.0 % / 0.19 | 0.0 % / 0.23 | 0.0 % / 0.25 | 0.0 % / 0.29 | 0.0 % / 0.25 | 0.0 % / 0.25 | 0.0 % / 0.33 |
| german-xix-blb-v1 – german-xix-v1 | 0.964 | 25.0 % / 0.12 | 25.5 % / 0.17 | 0.0 % / 0.26 | 0.0 % / 0.30 | 0.0 % / 0.27 | 0.0 % / 0.26 | 0.0 % / 0.27 | 0.0 % / 0.21 |
| openiti_ottoman_print – openiti_persian_print | 0.961 | 56.2 % / 0.17 | 57.1 % / 0.20 | 0.0 % / 0.26 | 0.0 % / 0.29 | 0.0 % / 0.26 | 0.0 % / 0.30 | 0.0 % / 0.30 | 0.0 % / 0.15 |
| bern-hertwig-v2 – cursive_inzigkofen | 0.960 | 4.7 % / 0.13 | 12.2 % / 0.21 | 0.0 % / 0.27 | 0.0 % / 0.27 | 0.0 % / 0.26 | 0.0 % / 0.29 | 0.0 % / 0.31 | 0.0 % / 0.36 |
| bern-hertwig-v1 – bern-schlussredaktion-v1 | 0.953 | 3.3 % / 0.14 | 9.0 % / 0.23 | 0.0 % / 0.29 | 0.0 % / 0.30 | 0.0 % / 0.31 | 0.0 % / 0.30 | 0.0 % / 0.32 | 0.0 % / 0.25 |
| bern-combined-v1 – bern-hertwig-v1 | 0.946 | 1.6 % / 0.15 | 8.8 % / 0.24 | 0.0 % / 0.31 | 0.0 % / 0.32 | 0.0 % / 0.33 | 0.0 % / 0.32 | 0.0 % / 0.34 | 0.0 % / 0.27 |
| bern-schlussredaktion-v1 – cursive_inzigkofen | 0.939 | 4.7 % / 0.17 | 11.7 % / 0.26 | 0.0 % / 0.35 | 0.0 % / 0.35 | 0.0 % / 0.33 | 0.0 % / 0.35 | 0.0 % / 0.38 | 0.0 % / 0.44 |
| bern-combined-v1 – cursive_inzigkofen | 0.931 | 3.0 % / 0.18 | 11.4 % / 0.27 | 0.0 % / 0.36 | 0.0 % / 0.36 | 0.0 % / 0.35 | 0.0 % / 0.37 | 0.0 % / 0.40 | 0.0 % / 0.47 |
| catmus_medieval – old_norse_am305 | 0.831 | 0.0 % / 0.19 | 0.0 % / 0.39 | 0.0 % / 0.53 | 0.0 % / 0.61 | 0.0 % / 0.69 | 0.0 % / 0.58 | 0.0 % / 0.56 | 0.0 % / 0.62 |
| openiti_arabic_print – openiti_persian_print | 0.796 | 31.2 % / 0.56 | 31.2 % / 0.58 | 0.0 % / 0.61 | 0.0 % / 0.68 | 0.0 % / 0.74 | 0.0 % / 0.66 | 0.0 % / 0.66 | 0.0 % / 0.54 |
| openiti_arabic_print – openiti_ottoman_print | 0.785 | 31.2 % / 0.56 | 31.2 % / 0.59 | 0.0 % / 0.63 | 0.0 % / 0.70 | 0.0 % / 0.76 | 0.0 % / 0.68 | 0.0 % / 0.69 | 0.0 % / 0.58 |
| bastarda_inzigkofen – catmus_medieval | 0.775 | 1.9 % / 0.26 | 0.0 % / 0.50 | 0.0 % / 0.74 | 0.0 % / 0.84 | 0.0 % / 0.81 | 0.0 % / 0.70 | 0.0 % / 0.64 | 0.0 % / 0.59 |
| catmus_medieval – cursive_inzigkofen | 0.768 | 1.6 % / 0.24 | 0.0 % / 0.51 | 0.0 % / 0.74 | 0.0 % / 0.84 | 0.0 % / 0.82 | 0.0 % / 0.71 | 0.0 % / 0.65 | 0.0 % / 0.65 |
| bern-hertwig-v1 – catmus_medieval | 0.767 | 0.0 % / 0.23 | 0.0 % / 0.51 | 0.0 % / 0.73 | 0.0 % / 0.83 | 0.0 % / 0.82 | 0.0 % / 0.71 | 0.0 % / 0.66 | 0.0 % / 0.78 |
| bohemian_19th – german-xix-v1 | 0.756 | 6.5 % / 0.92 | 18.9 % / 0.60 | 0.0 % / 0.75 | 0.0 % / 0.78 | 0.0 % / 0.67 | 0.0 % / 0.70 | 0.0 % / 0.75 | 0.0 % / 0.51 |
| bern-hertwig-v2 – catmus_medieval | 0.749 | 0.0 % / 0.25 | 0.0 % / 0.52 | 0.0 % / 0.75 | 0.0 % / 0.85 | 0.0 % / 0.85 | 0.0 % / 0.74 | 0.0 % / 0.70 | 0.0 % / 0.84 |
| bern-schlussredaktion-v1 – catmus_medieval | 0.738 | 0.0 % / 0.27 | 0.0 % / 0.54 | 0.0 % / 0.75 | 0.0 % / 0.85 | 0.0 % / 0.87 | 0.0 % / 0.76 | 0.0 % / 0.72 | 0.0 % / 0.90 |
| bern-combined-v1 – catmus_medieval | 0.732 | 0.0 % / 0.27 | 0.0 % / 0.54 | 0.0 % / 0.76 | 0.0 % / 0.86 | 0.0 % / 0.87 | 0.0 % / 0.77 | 0.0 % / 0.73 | 0.0 % / 0.91 |
| bohemian_19th – german-xix-blb-v1 | 0.724 | 6.5 % / 0.92 | 18.9 % / 0.63 | 0.0 % / 0.78 | 0.0 % / 0.82 | 0.0 % / 0.71 | 0.0 % / 0.74 | 0.0 % / 0.79 | 0.0 % / 0.53 |
| glagolitic_print – hebrew_sephardi | 0.698 | 0.0 % / 0.30 | 0.0 % / 0.52 | 0.0 % / 0.62 | 0.0 % / 0.69 | 0.0 % / 0.89 | 0.0 % / 0.78 | 0.0 % / 0.85 | 0.0 % / 0.84 |
| bastarda_inzigkofen – old_norse_am305 | 0.660 | 0.0 % / 0.27 | 0.0 % / 0.55 | 0.0 % / 0.74 | 0.0 % / 0.87 | 0.0 % / 0.91 | 0.0 % / 0.79 | 0.0 % / 0.76 | 0.0 % / 0.71 |
| cursive_inzigkofen – old_norse_am305 | 0.654 | 0.0 % / 0.26 | 0.0 % / 0.56 | 0.0 % / 0.75 | 0.0 % / 0.88 | 0.0 % / 0.92 | 0.0 % / 0.80 | 0.0 % / 0.77 | 0.0 % / 0.76 |
| bern-hertwig-v1 – old_norse_am305 | 0.653 | 0.0 % / 0.26 | 0.0 % / 0.55 | 0.0 % / 0.74 | 0.0 % / 0.87 | 0.0 % / 0.92 | 0.0 % / 0.80 | 0.0 % / 0.77 | 0.0 % / 0.84 |
| bern-hertwig-v2 – old_norse_am305 | 0.638 | 0.0 % / 0.28 | 0.0 % / 0.56 | 0.0 % / 0.74 | 0.0 % / 0.88 | 0.0 % / 0.94 | 0.0 % / 0.83 | 0.0 % / 0.80 | 0.0 % / 0.89 |
| hebrew_sephardi – samaritan | 0.636 | 0.0 % / 0.81 | 0.0 % / 0.71 | 0.0 % / 0.78 | 0.0 % / 0.84 | 0.0 % / 1.00 | 0.0 % / 0.86 | 0.0 % / 0.99 | 0.0 % / 1.36 |
| bastarda_inzigkofen – cursive_inzigkofen | 0.630 | 1.6 % / 0.23 | 0.0 % / 0.57 | 0.0 % / 0.76 | 0.0 % / 0.89 | 0.0 % / 0.96 | 0.0 % / 0.83 | 0.0 % / 0.77 | 0.0 % / 0.38 |
| bern-schlussredaktion-v1 – old_norse_am305 | 0.630 | 0.0 % / 0.29 | 0.0 % / 0.57 | 0.0 % / 0.74 | 0.0 % / 0.88 | 0.0 % / 0.95 | 0.0 % / 0.84 | 0.0 % / 0.81 | 0.0 % / 0.94 |
| bastarda_inzigkofen – bern-hertwig-v1 | 0.629 | 0.0 % / 0.23 | 0.0 % / 0.56 | 0.0 % / 0.75 | 0.0 % / 0.89 | 0.0 % / 0.95 | 0.0 % / 0.83 | 0.0 % / 0.77 | 0.0 % / 0.45 |
| bern-combined-v1 – old_norse_am305 | 0.626 | 0.0 % / 0.29 | 0.0 % / 0.57 | 0.0 % / 0.75 | 0.0 % / 0.88 | 0.0 % / 0.95 | 0.0 % / 0.84 | 0.0 % / 0.82 | 0.0 % / 0.95 |
| bastarda_inzigkofen – bern-hertwig-v2 | 0.614 | 0.0 % / 0.25 | 0.0 % / 0.57 | 0.0 % / 0.76 | 0.0 % / 0.89 | 0.0 % / 0.97 | 0.0 % / 0.85 | 0.0 % / 0.80 | 0.0 % / 0.52 |
| bastarda_inzigkofen – bern-schlussredaktion-v1 | 0.606 | 0.0 % / 0.27 | 0.0 % / 0.58 | 0.0 % / 0.76 | 0.0 % / 0.89 | 0.0 % / 0.98 | 0.0 % / 0.86 | 0.0 % / 0.81 | 0.0 % / 0.57 |
| bastarda_inzigkofen – bern-combined-v1 | 0.601 | 0.0 % / 0.27 | 0.0 % / 0.59 | 0.0 % / 0.77 | 0.0 % / 0.90 | 0.0 % / 0.98 | 0.0 % / 0.86 | 0.0 % / 0.82 | 0.0 % / 0.59 |

## Group B: RQ1 identical parameters and RQ2 distance, related pairs

Each cell: share of *live* parameters that are bit-identical (structural zeros excluded) / relative distance ‖B − A‖ / mean(‖A‖, ‖B‖). The output layer on the characters both alphabets share.

| pair | body cosine | C_0 | Gn_1 | C_2 | Gn_3 | C_5 | Gn_6 | L_9 | L_11 | L_13 | O_15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| catmus_print_large – french_vietnamese | 0.982 | 18.8 % / 0.05 | 9.4 % / 0.02 | 0.0 % / 0.12 | 0.0 % / 0.02 | 0.0 % / 0.13 | 0.0 % / 0.07 | 0.0 % / 0.18 | 0.0 % / 0.18 | 0.0 % / 0.22 | 0.0 % / 0.19 |
| catmus_print_large – latin_incunabula | 0.946 | 18.8 % / 0.08 | 9.4 % / 0.08 | 0.0 % / 0.22 | 0.0 % / 0.05 | 0.0 % / 0.26 | 0.0 % / 0.16 | 0.0 % / 0.31 | 0.0 % / 0.33 | 0.0 % / 0.35 | 0.0 % / 0.63 |
| french_vietnamese – latin_incunabula | 0.931 | 18.8 % / 0.09 | 9.4 % / 0.08 | 0.0 % / 0.24 | 0.0 % / 0.05 | 0.0 % / 0.29 | 0.0 % / 0.16 | 0.0 % / 0.35 | 0.0 % / 0.37 | 0.0 % / 0.40 | 0.0 % / 0.63 |

## Group C: RQ1 identical parameters and RQ2 distance, related pairs

Each cell: share of *live* parameters that are bit-identical (structural zeros excluded) / relative distance ‖B − A‖ / mean(‖A‖, ‖B‖). The output layer on the characters both alphabets share.

| pair | body cosine | C_0 | C_3 | C_6 | C_9 | L_12 | L_14 | L_16 | O_18 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| textualis_inzigkofen – tridis_v2 | 0.947 | 60.0 % / 0.12 | 49.4 % / 0.17 | 0.0 % / 0.41 | 0.0 % / 0.49 | 0.0 % / 0.33 | 0.0 % / 0.34 | 0.0 % / 0.33 | 0.0 % / 0.13 |

## Function: linear CKA per block, by group

Linear CKA between two models' outputs of a block on the same 120 probe lines (blb-train-300, bern-train-300, medieval-val-300; 40 each, seed 20261107). 1 = the same features up to unit order and scale. Means over pairs; every pair is in docs/family/functional.json.

### Group A (33 models, 528 pairs)

| pairs | n | C_0 | C_3 | C_6 | C_9 | L_12 | L_14 | L_16 | O_18 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| same lineage | 37 | 0.85 | 0.67 | 0.68 | 0.71 | 0.62 | 0.54 | 0.39 | 0.19 |
| distant relatives | 7 | 0.93 | 0.81 | 0.69 | 0.61 | 0.56 | 0.39 | 0.22 | 0.03 |
| unrelated | 484 | 0.80 | 0.66 | 0.59 | 0.48 | 0.48 | 0.40 | 0.23 | 0.04 |

Dead (silent) convolution filters per model (output 0 on every probe frame):

| model | C_0 | C_3 | C_6 | C_9 |
|---|---:|---:|---:|---:|
| bastarda_inzigkofen | 4/32 | 0/32 | 0/64 | 0/64 |
| bern-combined-v1 | 6/32 | 0/32 | 0/64 | 0/64 |
| bern-hertwig-v1 | 3/32 | 0/32 | 0/64 | 0/64 |
| bern-hertwig-v2 | 6/32 | 0/32 | 0/64 | 0/64 |
| bern-schlussredaktion-v1 | 7/32 | 0/32 | 0/64 | 0/64 |
| catmus_medieval | 0/32 | 0/32 | 0/64 | 0/64 |
| cursive_inzigkofen | 3/32 | 1/32 | 0/64 | 0/64 |
| old_norse_am305 | 1/32 | 0/32 | 0/64 | 0/64 |
| bohemian_19th | 10/32 | 0/32 | 0/64 | 0/64 |
| german-xix-blb-v1 | 9/32 | 0/32 | 0/64 | 0/64 |
| german-xix-v1 | 8/32 | 0/32 | 0/64 | 0/64 |
| glagolitic_print | 0/32 | 0/32 | 0/64 | 0/64 |
| hebrew_sephardi | 0/32 | 0/32 | 0/64 | 0/64 |
| samaritan | 3/32 | 0/32 | 0/64 | 0/64 |
| openiti_arabic_print | 11/32 | 0/32 | 0/64 | 0/64 |
| openiti_ottoman_print | 19/32 | 0/32 | 0/64 | 0/64 |
| openiti_persian_print | 19/32 | 0/32 | 0/64 | 0/64 |
| mccatmus | 0/32 | 0/32 | 0/64 | 0/64 |
| prima | 0/32 | 0/32 | 0/64 | 0/64 |
| austrian_fraktur | 7/32 | 1/32 | 0/64 | 0/64 |
| bifrost_old_norse | 1/32 | 0/32 | 0/64 | 0/64 |
| cyrillic_uncial | 5/32 | 0/32 | 0/64 | 0/64 |
| fondue_gd_v2 | 11/32 | 0/32 | 0/64 | 0/64 |
| gallicorpora | 0/32 | 0/32 | 0/64 | 0/64 |
| german_print | 5/32 | 2/32 | 0/64 | 0/64 |
| kuzushiji | 14/32 | 0/32 | 0/64 | 0/64 |
| latin_expanded_11c | 0/32 | 4/32 | 0/64 | 0/64 |
| latin_graphematic_11c | 0/32 | 3/32 | 0/64 | 0/64 |
| lectaurep_french | 0/32 | 0/32 | 0/64 | 0/64 |
| manu_mcfondue | 4/32 | 0/32 | 0/64 | 0/64 |
| medieval_latin_french_expanded | 0/32 | 0/32 | 0/64 | 0/64 |
| mendelssohn_letters | 18/32 | 6/32 | 0/64 | 0/64 |
| peraire_french | 0/32 | 0/32 | 0/64 | 0/64 |

### Group B (6 models, 15 pairs)

| pairs | n | C_0 | C_2 | C_5 | L_9 | L_11 | L_13 | O_15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| same lineage | 3 | 1.00 | 0.99 | 0.95 | 0.79 | 0.66 | 0.41 | 0.32 |
| unrelated | 12 | 0.99 | 0.90 | 0.72 | 0.60 | 0.49 | 0.30 | 0.09 |

Dead (silent) convolution filters per model (output 0 on every probe frame):

| model | C_0 | C_2 | C_5 |
|---|---:|---:|---:|
| catmus_print_large | 6/32 | 0/64 | 1/128 |
| french_vietnamese | 6/32 | 0/64 | 0/128 |
| latin_incunabula | 6/32 | 0/64 | 1/128 |
| catmus_gothic_print | 7/32 | 0/64 | 0/128 |
| cremma_medieval | 12/32 | 0/64 | 2/128 |
| medieval_latin_french_abbreviated | 7/32 | 0/64 | 3/128 |

### Group C (2 models, 1 pairs)

| pairs | n | C_0 | C_3 | C_6 | C_9 | L_12 | L_14 | L_16 | O_18 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| same lineage | 1 | 0.91 | 0.80 | 0.66 | 0.49 | 0.46 | 0.44 | 0.25 | 0.04 |

Dead (silent) convolution filters per model (output 0 on every probe frame):

| model | C_0 | C_3 | C_6 | C_9 |
|---|---:|---:|---:|---:|
| textualis_inzigkofen | 19/32 | 0/32 | 2/64 | 0/64 |
| tridis_v2 | 15/32 | 0/32 | 0/64 | 0/64 |

## RQ3 — exchange: S's block moved to G's

### `bern-hertwig-v2` ← `catmus-medieval-1.6.0` (body cosine 0.749)

Δ CER in points against S unedited, per set, paired 95 % interval; **bold** where it excludes 0. First set = S's own material or the nearest available. S / G read: `bern-hertwig-train-300` 8.28 % / 49.54 %; `medieval-val-300` 64.47 % / 48.18 %; `blb-train-300` 71.90 % / 54.25 %.

| block | β | `bern-hertwig-train-300` | `medieval-val-300` | `blb-train-300` |
|---|---:|---:|---:|---:|
| C_0 | 0.5 | **+0.34** [+0.10, +0.59] | **-2.23** [-3.05, -1.41] | +0.25 [-0.22, +0.69] |
| C_0 | 1.0 | **+22.18** [+20.60, +23.72] | **+5.82** [+4.24, +7.40] | **+7.00** [+5.67, +8.29] |
| C_3 | 0.5 | **+0.41** [+0.18, +0.64] | +0.37 [-0.37, +1.17] | +0.42 [-0.07, +0.94] |
| C_3 | 1.0 | **+6.22** [+5.49, +7.02] | **+4.17** [+1.87, +6.72] | **+4.84** [+3.42, +6.27] |
| C_6 | 0.5 | **+0.48** [+0.19, +0.78] | **-1.03** [-1.73, -0.43] | -0.29 [-0.75, +0.19] |
| C_6 | 1.0 | **+6.31** [+5.53, +7.10] | **+2.60** [+1.29, +3.98] | +0.16 [-0.81, +1.16] |
| C_9 | 0.5 | **+0.42** [+0.17, +0.68] | **-0.79** [-1.44, -0.17] | **-1.09** [-1.68, -0.53] |
| C_9 | 1.0 | **+4.89** [+4.21, +5.58] | -0.29 [-1.31, +0.69] | +0.58 [-0.38, +1.62] |
| L_12 | 0.5 | **+1.68** [+1.29, +2.08] | **-2.00** [-2.88, -1.21] | **-3.06** [-3.72, -2.46] |
| L_12 | 1.0 | **+11.97** [+10.79, +13.10] | **-2.22** [-4.13, -0.56] | **-8.79** [-10.11, -7.37] |
| L_14 | 0.5 | **+2.78** [+2.29, +3.31] | **-2.07** [-3.00, -1.29] | -0.43 [-1.01, +0.14] |
| L_14 | 1.0 | **+17.09** [+15.88, +18.41] | **-1.87** [-3.33, -0.56] | **-3.16** [-4.19, -2.16] |
| L_16 | 0.5 | **+4.36** [+3.80, +4.94] | **-1.84** [-2.33, -1.35] | **-1.17** [-1.71, -0.67] |
| L_16 | 1.0 | **+18.98** [+17.97, +19.95] | **+1.54** [+0.60, +2.52] | +0.51 [-0.34, +1.40] |
| O_18 | 0.5 | **+0.50** [+0.23, +0.77] | -0.10 [-0.46, +0.27] | **+1.20** [+0.79, +1.61] |
| O_18 | 1.0 | **+2.82** [+2.36, +3.28] | **+1.39** [+0.76, +2.09] | **+3.85** [+3.10, +4.66] |

### `bern-hertwig-v2` ← `cursive_johannes_jaeck` (body cosine 0.960)

Δ CER in points against S unedited, per set, paired 95 % interval; **bold** where it excludes 0. First set = S's own material or the nearest available. S / G read: `bern-hertwig-train-300` 8.28 % / 70.59 %; `medieval-val-300` 64.47 % / 73.88 %; `blb-train-300` 71.90 % / 76.60 %.

| block | β | `bern-hertwig-train-300` | `medieval-val-300` | `blb-train-300` |
|---|---:|---:|---:|---:|
| C_0 | 0.5 | +0.16 [-0.06, +0.39] | +0.22 [-0.38, +0.84] | +0.25 [-0.19, +0.68] |
| C_0 | 1.0 | **+1.48** [+1.05, +1.88] | **+2.01** [+1.04, +2.99] | **+0.83** [+0.26, +1.40] |
| C_3 | 0.5 | **+0.30** [+0.06, +0.57] | **-2.24** [-3.13, -1.43] | +0.02 [-0.41, +0.47] |
| C_3 | 1.0 | **+0.83** [+0.48, +1.18] | **-3.31** [-4.47, -2.11] | -0.16 [-0.72, +0.42] |
| C_6 | 0.5 | **+0.50** [+0.27, +0.75] | -0.14 [-0.61, +0.35] | +0.25 [-0.34, +0.85] |
| C_6 | 1.0 | **+2.47** [+1.94, +3.03] | **+1.05** [+0.27, +1.83] | **+1.09** [+0.32, +1.82] |
| C_9 | 0.5 | +0.18 [-0.04, +0.40] | **-1.08** [-1.65, -0.48] | **-1.11** [-1.66, -0.57] |
| C_9 | 1.0 | **+1.48** [+1.12, +1.86] | **-1.18** [-2.18, -0.22] | **-0.99** [-1.70, -0.30] |
| L_12 | 0.5 | **+1.25** [+0.88, +1.61] | **-0.79** [-1.42, -0.15] | **-1.49** [-2.14, -0.83] |
| L_12 | 1.0 | **+7.19** [+6.35, +8.04] | +0.56 [-0.59, +1.67] | **-1.27** [-2.31, -0.17] |
| L_14 | 0.5 | **+3.05** [+2.47, +3.68] | **-2.41** [-3.24, -1.61] | **-1.97** [-2.62, -1.31] |
| L_14 | 1.0 | **+10.50** [+9.52, +11.51] | **-1.72** [-2.86, -0.62] | **-2.66** [-3.56, -1.78] |
| L_16 | 0.5 | **+4.38** [+3.81, +5.00] | **-1.85** [-2.47, -1.19] | -0.19 [-0.84, +0.49] |
| L_16 | 1.0 | **+9.32** [+8.52, +10.08] | **-2.63** [-3.74, -1.42] | +0.05 [-0.99, +1.45] |
| O_18 | 0.5 | **+1.66** [+1.32, +2.00] | -0.01 [-0.31, +0.28] | **+0.39** [+0.09, +0.70] |
| O_18 | 1.0 | **+6.28** [+5.70, +6.85] | **+1.00** [+0.59, +1.40] | **+1.13** [+0.66, +1.56] |

### `cursive_johannes_jaeck` ← `catmus-medieval-1.6.0` (body cosine 0.768)

Δ CER in points against S unedited, per set, paired 95 % interval; **bold** where it excludes 0. First set = S's own material or the nearest available. S / G read: `medieval-val-300` 73.88 % / 48.18 %; `bern-train-300` 67.24 % / 48.03 %; `blb-train-300` 76.60 % / 54.25 %.

| block | β | `medieval-val-300` | `bern-train-300` | `blb-train-300` |
|---|---:|---:|---:|---:|
| C_0 | 0.5 | **-0.98** [-1.79, -0.18] | -0.38 [-1.06, +0.27] | -0.21 [-1.00, +0.92] |
| C_0 | 1.0 | +1.09 [-0.25, +2.62] | +0.55 [-0.40, +1.45] | +0.48 [-0.39, +1.54] |
| C_3 | 0.5 | **-1.10** [-1.75, -0.43] | **-0.77** [-1.41, -0.09] | -0.27 [-0.83, +0.35] |
| C_3 | 1.0 | +1.73 [-0.02, +3.58] | **-1.63** [-2.67, -0.64] | -1.20 [-3.83, +0.58] |
| C_6 | 0.5 | **-1.06** [-1.90, -0.19] | **-1.94** [-2.58, -1.29] | -0.33 [-1.11, +0.73] |
| C_6 | 1.0 | +0.22 [-1.24, +1.77] | **-2.36** [-3.46, -1.36] | -0.92 [-2.47, +0.37] |
| C_9 | 0.5 | **-0.87** [-1.49, -0.24] | **-1.45** [-2.14, -0.81] | -0.68 [-2.15, +0.34] |
| C_9 | 1.0 | -0.91 [-1.81, +0.06] | **-2.40** [-3.33, -1.57] | -0.68 [-3.84, +1.29] |
| L_12 | 0.5 | **-0.93** [-1.68, -0.17] | **-0.98** [-1.62, -0.37] | **-0.80** [-1.42, -0.17] |
| L_12 | 1.0 | **-1.96** [-3.29, -0.62] | **-1.85** [-2.80, -0.82] | **-2.35** [-3.95, -0.32] |
| L_14 | 0.5 | **-0.94** [-1.60, -0.23] | **-1.27** [-1.86, -0.66] | **-1.52** [-2.78, -0.50] |
| L_14 | 1.0 | **+1.74** [+0.21, +3.34] | +0.54 [-0.29, +1.41] | +0.79 [-0.39, +2.12] |
| L_16 | 0.5 | **-2.82** [-3.32, -2.34] | **-2.18** [-2.70, -1.68] | **-1.55** [-2.15, -0.96] |
| L_16 | 1.0 | -1.21 [-2.67, +0.09] | **+1.65** [+0.67, +2.64] | **+3.59** [+1.32, +5.65] |
| O_18 | 0.5 | **+2.25** [+1.81, +2.75] | **+1.15** [+0.70, +1.62] | **+1.36** [+0.80, +2.01] |
| O_18 | 1.0 | **+9.04** [+7.96, +10.27] | **+5.56** [+4.77, +6.39] | **+8.18** [+6.23, +10.77] |

### `kqnbsei_v1` ← `catmus-print-fondue-large` (body cosine 0.982)

Δ CER in points against S unedited, per set, paired 95 % interval; **bold** where it excludes 0. First set = S's own material or the nearest available. S / G read: `reichenau-val-300` 30.13 % / 22.33 %; `gt-fraktur-300` 36.55 % / 31.87 %; `newseye-onb-val-300` 34.33 % / 26.93 %.

| block | β | `reichenau-val-300` | `gt-fraktur-300` | `newseye-onb-val-300` |
|---|---:|---:|---:|---:|
| C_0 | 0.5 | -0.16 [-0.39, +0.07] | **-0.81** [-1.07, -0.56] | -0.11 [-0.36, +0.15] |
| Gn_1 | 0.5 | -0.11 [-0.24, +0.03] | **-0.23** [-0.36, -0.09] | **+0.14** [+0.01, +0.28] |
| C_2 | 0.5 | **-0.54** [-0.79, -0.27] | **-0.54** [-0.81, -0.27] | **-0.49** [-0.83, -0.17] |
| Gn_3 | 0.5 | +0.01 [-0.07, +0.09] | -0.01 [-0.12, +0.09] | -0.04 [-0.15, +0.07] |
| C_5 | 0.5 | **-0.62** [-0.90, -0.33] | +0.28 [-0.01, +0.57] | **-0.41** [-0.81, -0.03] |
| Gn_6 | 0.5 | -0.10 [-0.30, +0.09] | **+0.24** [+0.07, +0.43] | +0.09 [-0.13, +0.32] |
| L_9 | 0.5 | **-1.55** [-2.00, -1.12] | **-0.83** [-1.21, -0.45] | **-0.99** [-1.51, -0.48] |
| L_11 | 0.5 | **-3.50** [-4.00, -3.00] | **-1.88** [-2.32, -1.46] | **-2.99** [-3.54, -2.45] |
| L_13 | 0.5 | **-2.15** [-2.55, -1.76] | **-1.94** [-2.25, -1.64] | **-2.48** [-2.94, -2.03] |
| O_15 | 0.5 | **-0.91** [-1.16, -0.68] | **-0.88** [-1.08, -0.68] | **-1.11** [-1.40, -0.84] |

### `german-xix-blb-v1` ← `german-xix-v1` (body cosine 0.964)

Δ CER in points against S unedited, per set, paired 95 % interval; **bold** where it excludes 0. First set = S's own material or the nearest available. S / G read: `blb-train-300` 6.46 % / 30.39 %; `bern-train-300` 51.35 % / 53.66 %; `medieval-val-300` 54.54 % / 53.73 %.

| block | β | `blb-train-300` | `bern-train-300` | `medieval-val-300` |
|---|---:|---:|---:|---:|
| C_0 | 0.5 | -0.02 [-0.13, +0.10] | -0.28 [-0.72, +0.14] | -0.61 [-1.26, +0.06] |
| C_0 | 1.0 | **+0.27** [+0.11, +0.45] | **-0.86** [-1.43, -0.31] | -0.56 [-1.37, +0.20] |
| C_3 | 0.5 | **+0.16** [+0.01, +0.30] | +0.03 [-0.44, +0.47] | -0.31 [-0.98, +0.42] |
| C_3 | 1.0 | **+0.45** [+0.24, +0.68] | -0.06 [-0.64, +0.53] | -0.07 [-0.93, +0.85] |
| C_6 | 0.5 | **+0.33** [+0.14, +0.55] | -0.31 [-0.75, +0.12] | **-0.95** [-1.65, -0.24] |
| C_6 | 1.0 | **+0.80** [+0.47, +1.25] | +0.07 [-0.49, +0.63] | **-1.14** [-2.10, -0.24] |
| C_9 | 0.5 | +0.16 [-0.01, +0.32] | +0.06 [-0.37, +0.50] | -0.19 [-0.76, +0.47] |
| C_9 | 1.0 | **+0.65** [+0.42, +0.91] | +0.28 [-0.31, +0.85] | -0.42 [-1.14, +0.37] |
| L_12 | 0.5 | **+1.90** [+1.32, +2.59] | **-0.65** [-1.15, -0.14] | **-1.17** [-2.28, -0.03] |
| L_12 | 1.0 | **+6.37** [+4.99, +7.98] | +0.33 [-0.40, +1.03] | +0.06 [-1.66, +1.72] |
| L_14 | 0.5 | **+1.18** [+0.87, +1.51] | **-0.86** [-1.36, -0.38] | **-1.91** [-2.78, -1.10] |
| L_14 | 1.0 | **+4.05** [+3.43, +4.64] | +0.39 [-0.30, +1.06] | -0.83 [-1.93, +0.26] |
| L_16 | 0.5 | **+2.31** [+1.97, +2.67] | **-0.78** [-1.20, -0.37] | **-1.46** [-1.98, -0.95] |
| L_16 | 1.0 | **+5.70** [+5.12, +6.26] | -0.54 [-1.11, +0.01] | **-1.43** [-2.13, -0.81] |
| O_18 | 0.5 | **+0.23** [+0.10, +0.36] | +0.23 [+0.00, +0.46] | +0.03 [-0.16, +0.21] |
| O_18 | 1.0 | **+0.68** [+0.49, +0.88] | **+0.56** [+0.26, +0.87] | +0.15 [-0.12, +0.41] |

### `german-xix-v1` ← `rec_bohemica` (body cosine 0.756)

Δ CER in points against S unedited, per set, paired 95 % interval; **bold** where it excludes 0. First set = S's own material or the nearest available. S / G read: `blb-train-300` 30.39 % / 53.98 %; `bern-train-300` 53.66 % / 67.06 %; `medieval-val-300` 53.73 % / 71.94 %.

| block | β | `blb-train-300` | `bern-train-300` | `medieval-val-300` |
|---|---:|---:|---:|---:|
| C_0 | 0.5 | **+0.69** [+0.08, +1.32] | +0.42 [-0.19, +1.08] | **+4.54** [+3.31, +5.94] |
| C_0 | 1.0 | **+2.56** [+1.75, +3.42] | **+1.05** [+0.23, +1.85] | **+5.16** [+4.01, +6.52] |
| C_3 | 0.5 | **+1.34** [+0.39, +2.18] | +0.63 [-0.13, +1.41] | +0.58 [-0.14, +1.34] |
| C_3 | 1.0 | **+5.24** [+4.08, +6.37] | **+3.26** [+2.29, +4.24] | **+2.55** [+1.59, +3.48] |
| C_6 | 0.5 | +0.60 [-0.69, +1.61] | +0.33 [-0.29, +0.95] | **+2.16** [+1.15, +3.16] |
| C_6 | 1.0 | **+7.42** [+5.45, +9.18] | **+1.87** [+1.11, +2.64] | **+7.29** [+5.15, +9.64] |
| C_9 | 0.5 | -0.30 [-0.99, +0.35] | **+0.96** [+0.32, +1.57] | **+1.39** [+0.62, +2.19] |
| C_9 | 1.0 | **+2.59** [+0.99, +3.89] | **+4.04** [+3.22, +4.89] | **+5.56** [+4.18, +7.03] |
| L_12 | 0.5 | +0.22 [-1.23, +1.30] | **+1.58** [+0.93, +2.29] | **+2.72** [+1.65, +3.77] |
| L_12 | 1.0 | **+7.80** [+5.89, +9.47] | **+7.38** [+6.43, +8.38] | **+10.93** [+9.55, +12.40] |
| L_14 | 0.5 | **+1.69** [+0.80, +2.57] | **-0.87** [-1.58, -0.11] | **+2.32** [+1.56, +3.12] |
| L_14 | 1.0 | **+10.82** [+9.26, +12.23] | **+4.76** [+3.71, +5.88] | **+9.25** [+8.29, +10.24] |
| L_16 | 0.5 | **+1.12** [+0.49, +1.78] | **+0.70** [+0.14, +1.30] | **+0.91** [+0.31, +1.48] |
| L_16 | 1.0 | **+6.02** [+5.13, +6.95] | **+4.38** [+3.62, +5.14] | **+4.19** [+3.43, +4.94] |
| O_18 | 0.5 | +0.11 [-0.14, +0.36] | +0.06 [-0.22, +0.35] | -0.02 [-0.29, +0.24] |
| O_18 | 1.0 | +0.32 [-0.06, +0.74] | +0.27 [-0.12, +0.67] | -0.04 [-0.44, +0.33] |

### `reichenau_lat_cat_099218` ← `catmus-print-fondue-large` (body cosine 0.946)

Δ CER in points against S unedited, per set, paired 95 % interval; **bold** where it excludes 0. First set = S's own material or the nearest available. S / G read: `reichenau-val-300` 3.81 % / 22.33 %; `gt-fraktur-300` 24.98 % / 31.87 %; `newseye-onb-val-300` 28.68 % / 26.93 %.

| block | β | `reichenau-val-300` | `gt-fraktur-300` | `newseye-onb-val-300` |
|---|---:|---:|---:|---:|
| C_0 | 0.5 | -0.06 [-0.19, +0.06] | **-0.31** [-0.61, -0.01] | -0.13 [-0.44, +0.17] |
| C_0 | 1.0 | +0.06 [-0.10, +0.22] | +0.15 [-0.25, +0.52] | +0.19 [-0.26, +0.68] |
| Gn_1 | 0.5 | +0.01 [-0.05, +0.08] | -0.17 [-0.35, +0.01] | **-0.37** [-0.59, -0.14] |
| Gn_1 | 1.0 | +0.05 [-0.05, +0.16] | **-0.28** [-0.53, -0.05] | **-0.57** [-0.94, -0.21] |
| C_2 | 0.5 | +0.00 [-0.14, +0.13] | +0.09 [-0.19, +0.37] | -0.17 [-0.80, +0.40] |
| C_2 | 1.0 | +0.16 [-0.02, +0.34] | +0.26 [-0.10, +0.64] | +0.00 [-0.80, +0.72] |
| Gn_3 | 0.5 | +0.03 [-0.03, +0.10] | **-0.16** [-0.28, -0.03] | **-0.23** [-0.41, -0.06] |
| Gn_3 | 1.0 | +0.03 [-0.05, +0.12] | **-0.23** [-0.42, -0.03] | **-0.47** [-0.90, -0.13] |
| C_5 | 0.5 | +0.11 [-0.04, +0.26] | -0.08 [-0.43, +0.25] | +0.17 [-0.23, +0.57] |
| C_5 | 1.0 | **+0.45** [+0.25, +0.66] | +0.09 [-0.41, +0.56] | **+1.00** [+0.32, +1.73] |
| Gn_6 | 0.5 | +0.02 [-0.11, +0.15] | **+1.14** [+0.80, +1.50] | +0.05 [-0.36, +0.45] |
| Gn_6 | 1.0 | +0.15 [-0.02, +0.33] | **+2.40** [+1.95, +2.86] | +0.59 [-0.08, +1.30] |
| L_9 | 0.5 | **+0.38** [+0.16, +0.58] | +0.19 [-0.30, +0.67] | **-1.08** [-1.91, -0.30] |
| L_9 | 1.0 | **+2.48** [+2.09, +2.87] | **+2.43** [+1.62, +3.23] | +0.98 [-0.19, +2.12] |
| L_11 | 0.5 | **+0.29** [+0.10, +0.48] | **-1.38** [-2.01, -0.79] | **-2.97** [-3.80, -2.17] |
| L_11 | 1.0 | **+2.85** [+2.47, +3.26] | **+0.99** [+0.08, +1.84] | **-2.40** [-3.78, -1.16] |
| L_13 | 0.5 | **+0.46** [+0.24, +0.68] | -0.33 [-0.72, +0.08] | **-2.51** [-3.00, -2.02] |
| L_13 | 1.0 | **+2.61** [+2.19, +3.06] | **+0.81** [+0.26, +1.39] | **-2.76** [-3.41, -2.09] |
| O_15 | 0.5 | +0.05 [-0.07, +0.17] | -0.07 [-0.30, +0.17] | **-1.58** [-1.91, -1.26] |
| O_15 | 1.0 | **+0.49** [+0.28, +0.70] | -0.04 [-0.40, +0.32] | **-2.05** [-2.51, -1.57] |

### `textualis_anna_jaeck` ← `Tridis_v2_Medieval_EarlyModern` (body cosine 0.947)

Δ CER in points against S unedited, per set, paired 95 % interval; **bold** where it excludes 0. First set = S's own material or the nearest available. S / G read: `parzival-300` 33.50 % / 12.45 %; `bern-train-300` 77.77 % / 33.92 %; `medieval-val-300` 78.86 % / 42.82 %.

| block | β | `parzival-300` | `bern-train-300` | `medieval-val-300` |
|---|---:|---:|---:|---:|
| C_0 | 0.5 | **+0.94** [+0.28, +1.63] | **-0.64** [-1.24, -0.03] | **+1.28** [+0.31, +2.25] |
| C_3 | 0.5 | -0.11 [-0.76, +0.51] | **+1.03** [+0.41, +1.64] | -0.01 [-0.55, +0.55] |
| C_6 | 0.5 | **-0.84** [-1.56, -0.14] | **-1.37** [-2.01, -0.76] | **-0.97** [-1.55, -0.38] |
| C_9 | 0.5 | **-2.94** [-3.64, -2.24] | **-2.24** [-2.94, -1.57] | **-1.61** [-2.23, -0.99] |
| L_12 | 0.5 | **-2.50** [-3.17, -1.83] | **-2.15** [-2.79, -1.50] | **-3.05** [-3.86, -2.27] |
| L_14 | 0.5 | **-4.02** [-4.72, -3.36] | **-1.44** [-1.99, -0.88] | **-2.19** [-2.78, -1.62] |
| L_16 | 0.5 | **-2.89** [-3.64, -2.14] | **+10.52** [+9.74, +11.30] | **+6.96** [+6.17, +7.71] |
| O_18 | 0.5 | -0.29 [-0.69, +0.09] | **+0.80** [+0.48, +1.12] | **+0.68** [+0.40, +0.98] |

### Combined moves: `reichenau_lat_cat_099218` ← `catmus-print-fondue-large`

Several blocks moved at once (block:β). Δ CER in points against S unedited, paired 95 % interval, **bold** where it excludes 0. Sets named `-conf-` are confirmation lines disjoint from the practice lines the moves were chosen on. S / G read: `reichenau-val-300` 3.81 % / 22.33 %; `gt-fraktur-300` 24.98 % / 31.87 %; `newseye-onb-val-300` 28.68 % / 26.93 %; `reichenau-val-conf-300` 3.93 % / 22.16 %; `gt-fraktur-conf-300` 24.25 % / 30.20 %; `newseye-onb-val-conf-300` 29.47 % / 27.61 %.

| combination | moves | `reichenau-val-300` | `gt-fraktur-300` | `newseye-onb-val-300` | `reichenau-val-conf-300` | `gt-fraktur-conf-300` | `newseye-onb-val-conf-300` |
|---|---|---:|---:|---:|---:|---:|---:|
| gn13 | Gn_1:1 Gn_3:1 | +0.06 [-0.07, +0.18] | **-0.43** [-0.69, -0.17] | **-0.98** [-1.57, -0.49] | +0.07 [-0.02, +0.17] | -0.17 [-0.46, +0.12] | **-0.40** [-0.74, -0.05] |
| gn13+out | Gn_1:1 Gn_3:1 O_15:0.5 | +0.01 [-0.13, +0.15] | **-0.60** [-0.91, -0.30] | **-2.40** [-3.07, -1.81] | +0.07 [-0.07, +0.19] | **-0.89** [-1.19, -0.59] | **-1.83** [-2.26, -1.36] |
| front | C_0:1 Gn_1:1 C_2:1 Gn_3:1 Gn_6:1 | **+0.39** [+0.14, +0.63] | **+1.30** [+0.86, +1.78] | **+1.26** [+0.22, +2.33] | **+0.40** [+0.17, +0.62] | **+1.20** [+0.64, +1.78] | +0.77 [-0.01, +1.57] |
| front+out | C_0:1 Gn_1:1 C_2:1 Gn_3:1 Gn_6:1 O_15:0.5 | **+0.34** [+0.07, +0.59] | **+0.97** [+0.48, +1.45] | -0.27 [-1.33, +0.85] | **+0.42** [+0.18, +0.67] | **+0.72** [+0.21, +1.25] | -0.48 [-1.20, +0.28] |
| gn13+out+c0h | Gn_1:1 Gn_3:1 O_15:0.5 C_0:0.5 | +0.01 [-0.18, +0.18] | **-0.80** [-1.17, -0.44] | **-2.39** [-3.15, -1.76] | +0.07 [-0.07, +0.21] | **-0.88** [-1.21, -0.54] | **-1.87** [-2.35, -1.38] |
| gn13+out+l11h | Gn_1:1 Gn_3:1 O_15:0.5 L_11:0.5 | **+0.43** [+0.22, +0.64] | **-1.55** [-2.22, -0.93] | **-4.28** [-5.12, -3.47] | **+0.36** [+0.13, +0.59] | **-2.05** [-2.69, -1.41] | **-4.08** [-4.92, -3.24] |


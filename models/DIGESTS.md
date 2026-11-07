# Weight digests on asteraix (2026-11-06T12:14Z UTC)

| id | sha256 | bytes | gateway digest |
|---|---|---:|---|
| kraken-bern-hertwig-v1 | ef66a2522fb4e194d13db023f0f93cf9f5dcbc9e979879896fa2275b9b6d134a | 16391628 | not recorded by the gateway |
| kraken-bern-hertwig-v2 | 479392ab7899b831b3c3872dc0f178a894495757b2df2047f2b695c4a5dcc8f3 | 16391628 | not recorded by the gateway |
| kraken-bern-schlussredaktion-v1 | d26eb02dfaf89f8f222165a1e5c982e38f19a915a5a744e4ad6c15f69b71f526 | 16394904 | match |
| kraken-bern-combined-v1 | 35e7eae281ea85e61847e22ab65a0295aeddb7ddbe49d93bcf7deae7f783a3a6 | 16396542 | match |
| kraken-german-xix-v1 | 7af3d4a0eb1a778c3b8c7f9bab86ed656a2eb235470f972193918db35fa2ece6 | 16322760 | not recorded by the gateway |
| kraken-german-xix-blb-v1 | 48caacb4a742b5b5ccd2a92a005a4ca701bdf7cef7859087832b6d726f5f8df4 | 16440686 | match |
| kraken-cursive_inzigkofen | 4ab19e7b42e66bb70679a3076a61305975746ddcdb0f222d40f1b66f5e0fe039 | 16406523 | Zenodo 10.5281/zenodo.18207767 |

## kraken+ and shard00 comparators (copied 06.11.2026)

| id | sha256 | bytes | origin |
|---|---|---:|---|
| kraken-plus-shard00-cr1 | 2bd571b52bf570ed7cd585c93a5c75d1d38eb87076cac77134c81ac053c7ddef | 16574618 | archive/idhefix-2026-09/runs/kraken-plus-shard00/checkpoints/model/best_0.7927.mlmodel — kraken+ `Cr1,1,85`, val acc 0.7927 (KRAKEN_PLUS.md run) |
| kraken-medieval-shard00-std | e4b288bd1f8ca2099977f4efc7c7771dae6fab68d5e019bdb89ff4f33aeda0b0 | 16121414 | …/runs/kraken-medieval-shard00-std/…/best_0.8226.mlmodel — kraken default h120, val 0.8226 |
| kraken-medieval-shard00-v1 | a4bdcfeeb0f60b576dcee8ae89b0942f1a31a8c0febc9417feecec786d328bb2 | 16574261 | …/runs/kraken-medieval-shard00-v1/…/best_0.7809.mlmodel — kraken+ without the 85 layer, val 0.7809 |
| kraken-collapse-probe-131-cr255 | fa74476b7a53146fae32380dd5b89c9d1765953753f0275133b7df99ef028ca1 | 60813907 | ~/atr-cache/checkpoints/20261027T072036Z-kraken-collapse-probe-131/best_0.0083.mlmodel — KRAKEN_PLUS_SPEC with `Cr255,1,85`, from scratch, val acc 0.0083 (collapsed); the only Cr255 checkpoint left |

## Phase 5 adapters (peft directories, 07.11.2026)

| id | `adapter_model.safetensors` sha256 | `adapter_config.json` sha256 | origin |
|---|---|---|---|
| qwen3.5-4b-german-kurrent-xvi-xix-page-v2 | 1b70ce4a2b2125bd007bd1c81789bd43c8d72c12ef49e4f50cf8ad9ca3c67897 | 0be75495420eacae0d2f81b9f2d053d6f4376ecf160d916f405904b8a828f0b3 | share `trained-ubelix/…-page-v2`, copied there from UBELIX scratch by hand (Slurm 20544433, job `20261029T133027Z-…`); identical on the share and in `~/atr-cache/wsp/adapters/`. r 64, α 128, peft 0.20.0. First digest on record: the trainer's record stores none |
| qwen3.5-4b-german-kurrent-xvi-xix-page-v1 | 52f94e105ed204d3d42a98a5992c4e9c100d48134e0b81ea00f0f1ee293dfafb | 0f7d8bcf22c7b6fe65317726f36d4fe25c153d56730fbc905d03536fd24fbc1d | share `trained-ubelix/…-page-v1`. The stand-in for harness tests. Trained on the registered documents, so `crossed` on german-kurrent-xvi-xix-v1 |

# Raw reproduction results

Acceptance: **passed**. Receipt SHA-256: `c18c3a416174b4791b830c0c3a5c60cf07ba876c49b3396d623631fc0f1b180e`.

Fresh metrics and stages come from this accepted execution; historical comparisons are explicitly labeled. Full precision and source bindings are in [execution_summary.json](execution_summary.json).

Confirm gate: **True**; serving recipe: **full13+gate**.

## Data and tests

| Aggregate | Value |
| --- | ---: |
| rows | 2,609,219 |
| impressions | 2,609,219 |
| candidates | 97,592,931 |
| positives | 3,958,501 |
| users | 750,434 |
| news | 104,151 |
| news_including_padding | 104,152 |
| official_train_rows | 2,232,748 |
| official_dev_rows | 376,471 |
| processing_seconds | 1,495.841111099995 |

| Test run (overlapping coverage) | Passed | Skipped | Total |
| --- | ---: | ---: | ---: |
| unit-tests | 256 | 2 | 258 |
| research-unit-tests | 193 | 0 | 193 |

## v1 official-dev ranking

| Arm | AUC | MRR | nDCG@5 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| popularity | 0.525910 | 0.224750 | 0.232211 | 0.295930 |
| content | 0.611946 | 0.287926 | 0.310058 | 0.371626 |
| mean_random-seed17 | 0.638075 | 0.305544 | 0.334199 | 0.395712 |
| mean_random-seed29 | 0.623390 | 0.291323 | 0.316121 | 0.379656 |
| mean_random-seed43 | 0.621612 | 0.292396 | 0.315929 | 0.380314 |
| mean_mixed-seed17 | 0.633421 | 0.304697 | 0.333664 | 0.394630 |
| mean_mixed-seed29 | 0.608914 | 0.287046 | 0.310330 | 0.372163 |
| mean_mixed-seed43 | 0.621865 | 0.292426 | 0.315758 | 0.381020 |
| ranker-seed17 | 0.626876 | 0.295062 | 0.321838 | 0.383726 |
| ranker-seed29 | 0.632214 | 0.297617 | 0.324120 | 0.387446 |
| ranker-seed43 | 0.630631 | 0.295688 | 0.322516 | 0.385333 |

### v1 paired user bootstrap versus mean_random

| Comparison / metric | Delta | 95% CI |
| --- | ---: | --- |
| mean_mixed / auc | -0.006292 | [-0.006567, -0.006019] |
| mean_mixed / mrr | -0.001698 | [-0.001920, -0.001441] |
| mean_mixed / ndcg5 | -0.002166 | [-0.002485, -0.001845] |
| mean_mixed / ndcg10 | -0.002623 | [-0.002880, -0.002350] |
| ranker / auc | +0.002215 | [+0.001644, +0.002759] |
| ranker / mrr | -0.000299 | [-0.000808, +0.000206] |
| ranker / ndcg5 | +0.000742 | [+0.000147, +0.001334] |
| ranker / ndcg10 | +0.000274 | [-0.000247, +0.000758] |

## v2 sampled ranking, seed means

| Arm | AUC | MRR | nDCG@5 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| content_only | 0.616216 | 0.285900 | 0.307788 | 0.370312 |
| context_only | 0.641470 | 0.280629 | 0.306176 | 0.375307 |
| full | 0.679979 | 0.314832 | 0.348947 | 0.414787 |

### v2 ranking ablation intervals

| Comparison / metric | Delta | 95% CI |
| --- | ---: | --- |
| full_minus_content_only / auc | +0.063763 | [+0.058886, +0.068844] |
| full_minus_content_only / mrr | +0.028932 | [+0.024134, +0.033070] |
| full_minus_content_only / ndcg5 | +0.041159 | [+0.035715, +0.046198] |
| full_minus_content_only / ndcg10 | +0.044475 | [+0.039561, +0.048730] |
| full_minus_context_only / auc | +0.038509 | [+0.034698, +0.042071] |
| full_minus_context_only / mrr | +0.034203 | [+0.030540, +0.037706] |
| full_minus_context_only / ndcg5 | +0.042771 | [+0.038661, +0.047112] |
| full_minus_context_only / ndcg10 | +0.039481 | [+0.035699, +0.043257] |
| context_only_minus_content_only / auc | +0.025254 | [+0.018152, +0.032060] |
| context_only_minus_content_only / mrr | -0.005271 | [-0.011454, +0.000829] |
| context_only_minus_content_only / ndcg5 | -0.001612 | [-0.008603, +0.005605] |
| context_only_minus_content_only / ndcg10 | +0.004995 | [-0.001535, +0.011150] |

## v2 full ranking, internally selected model

| Arm | AUC | MRR | nDCG@5 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| full-seed43 | 0.677466 | 0.321096 | 0.355123 | 0.418905 |

## v2 catalog ablations

| Arm | Recall@10 | Recall@100 |
| --- | ---: | ---: |
| full | 0.088041 | 0.384768 |
| no_popular | 0.022571 | 0.092265 |
| no_content | 0.178986 | 0.421859 |
| no_tower | 0.152455 | 0.413543 |
| popular_only | 0.234820 | 0.444694 |
| content_only | 0.011748 | 0.064414 |
| tower_only | 0.024815 | 0.105333 |
| no_freshness | 0.078405 | 0.290391 |
| frozen_feedback | 0.013992 | 0.062434 |
| selected | 0.234820 | 0.444694 |
| selected_rerank_content_only | 0.076954 | 0.444694 |
| selected_rerank_context_only | 0.116948 | 0.444694 |
| selected_rerank_full | 0.131204 | 0.444694 |

### v2 catalog ablation intervals

| Comparison / metric | Delta | 95% CI |
| --- | ---: | --- |
| full_minus_no_popular_recall100 | +0.292503 | [+0.281676, +0.303536] |
| full_minus_no_content_recall100 | -0.037091 | [-0.042677, -0.031103] |
| full_minus_no_tower_recall100 | -0.028775 | [-0.034058, -0.022878] |
| full_minus_popular_only_recall100 | -0.059926 | [-0.067974, -0.051620] |
| full_minus_content_only_recall100 | +0.320354 | [+0.309153, +0.332303] |
| full_minus_tower_only_recall100 | +0.279435 | [+0.268635, +0.290700] |
| full_minus_no_freshness_recall100 | +0.094377 | [+0.082611, +0.106106] |
| full_minus_frozen_feedback_recall100 | +0.322334 | [+0.311622, +0.333379] |
| full_minus_selected_recall100 | -0.059926 | [-0.067974, -0.051620] |
| selected_rerank_content_only_minus_no_rerank_recall10 | -0.157867 | [-0.168624, -0.147246] |
| selected_rerank_context_only_minus_no_rerank_recall10 | -0.117872 | [-0.128053, -0.108205] |
| selected_rerank_full_minus_no_rerank_recall10 | -0.103617 | [-0.113806, -0.093046] |

## v3 confirmation and dev ablations

| Phase / arm | Recall@10 | Recall@100 | ΔR@10 vs baseline [95% CI] |
| --- | ---: | ---: | --- |
| confirm / baseline | 0.142337 | 0.231380 | — |
| confirm / full13 | 0.145978 | 0.231380 | +0.003641 [+0.000668, +0.006396] |
| dev / baseline | 0.234820 | 0.444694 | — |
| dev / full13 | 0.239044 | 0.444694 | +0.004224 [+0.001066, +0.007618] |
| dev / no_category | 0.239176 | 0.444694 | +0.004356 [+0.001171, +0.007752] |
| dev / no_recent | 0.239044 | 0.444694 | +0.004224 [+0.001066, +0.007618] |
| dev / no_gate | 0.239704 | 0.444694 | +0.004884 [+0.001434, +0.008715] |
| dev / full13_seed17 | 0.239044 | 0.444694 | +0.004224 [+0.001059, +0.007495] |
| dev / full13_seed29 | 0.239176 | 0.444694 | +0.004356 [+0.001196, +0.007626] |
| dev / full13_seed43 | 0.239440 | 0.444694 | +0.004620 [+0.001444, +0.007988] |

### v3 dev full13 versus explicit ablations (exploratory)

| Comparison / metric | Delta | 95% CI |
| --- | ---: | --- |
| versus_no_category | -0.000132 | [-0.000399, +0.000000] |
| versus_no_recent | +0.000000 | [+0.000000, +0.000000] |
| versus_no_gate | -0.000660 | [-0.002499, +0.000936] |

## Fresh versus previous scientific results

Descriptive v3 comparison only; the historical reports never supply the fresh result or select the serving model.

| Phase / arm | Fresh − previous R@10 | Fresh − previous R@100 |
| --- | ---: | ---: |
| confirm / baseline | +0.00000000 | +0.00000000 |
| confirm / full13 | +0.00000000 | +0.00000000 |
| dev / baseline | +0.00000000 | +0.00000000 |
| dev / full13 | +0.00000000 | +0.00000000 |
| dev / no_category | +0.00000000 | +0.00000000 |
| dev / no_recent | +0.00000000 | +0.00000000 |
| dev / no_gate | +0.00000000 | +0.00000000 |
| dev / full13_seed17 | +0.00000000 | +0.00000000 |
| dev / full13_seed29 | +0.00000000 | +0.00000000 |
| dev / full13_seed43 | +0.00000000 | +0.00000000 |

## Local HTTP measurements

One HTTP request, including every query in its batch; client serialization, connection, server work and response decoding. Warmup excluded.

| Batch | Concurrency | Queries | Queries/s | Request p50 ms | p95 ms | p99 ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 192 | 29.414 | 30.496 | 73.876 | 83.383 |
| 8 | 1 | 192 | 33.576 | 234.626 | 415.773 | 435.114 |
| 1 | 4 | 192 | 26.339 | 138.256 | 242.442 | 294.408 |
| 8 | 4 | 192 | 39.350 | 668.878 | 979.131 | 1191.637 |

Peak server RSS: 237,731,840 bytes. One owned HTTP server process, excluding the benchmark client and startup; sampled during API checks and all HTTP cases.

| Retrieval microbenchmark | Observed wall speed ratio v2/v3 |
| --- | ---: |
| cold | 1.0699 |
| warm | 2.5456 |

## Execution receipts

| Stage | Status | Started UTC | Finished UTC | Seconds | Receipt SHA-256 |
| --- | --- | --- | --- | ---: | --- |
| environment | success | 2026-10-14T03:15:29.180436+00:00 | 2026-10-14T03:15:47.926973+00:00 | 18.747 | `60757f2bac7a1219105d2c99969e7b988d2af61caea72cd1eddaf13c25414ae8` |
| unit-tests | success | 2026-10-14T03:15:48.294185+00:00 | 2026-10-14T03:23:02.179270+00:00 | 433.885 | `1b22519d4cc58d010e86136848d305a29327ac861ddbed5109a71b5f05fc8ce7` |
| research-unit-tests | success | 2026-10-14T03:23:02.534526+00:00 | 2026-10-14T03:28:54.595385+00:00 | 352.061 | `8fa2aab64bcc86bb7d15671cdfb13cec1556884d3a9d139a51cc2958df8061b0` |
| data | success | 2026-10-14T03:28:54.680352+00:00 | 2026-10-14T03:54:08.491448+00:00 | 1513.812 | `168d8a70425e8e93ba87f46532eaacfdeffffdbd51e893dd7d5138c8db0eb35f` |
| data-audit | success | 2026-10-14T03:54:08.561641+00:00 | 2026-10-14T03:54:22.267371+00:00 | 13.706 | `210f9b36a4eb2accd09dca5be4803a4d9962d7fb9f7b3382d0673808014bf0c4` |
| temporal | success | 2026-10-14T03:54:22.313993+00:00 | 2026-10-14T03:55:30.425232+00:00 | 68.112 | `acf3948165c44d8e519c575f596b4be5b03ca6be55b64559ffa84c4e18108ea6` |
| neural-train | success | 2026-10-14T03:55:30.563396+00:00 | 2026-10-14T06:15:20.124446+00:00 | 8389.561 | `d482264c9610546e00b5f99d0d4e84aefc83669f255725187f912d2331110f80` |
| backbone-eligibility | success | 2026-10-14T06:15:20.178570+00:00 | 2026-10-14T06:15:41.457243+00:00 | 21.279 | `ad4ce438723334d0833b6f3571c2bce7628d3bd45d8aa126e3e7ab3ce0ae5191` |
| v1-full-ranking | success | 2026-10-14T06:15:41.488344+00:00 | 2026-10-14T07:03:37.085279+00:00 | 2875.598 | `bce4f4e6dd020c9d3fac8f4709a509d8ecb6c72eeb0131b1f6d3244db901d350` |
| backbone-initial | success | 2026-10-14T07:03:37.149409+00:00 | 2026-10-14T07:04:07.278861+00:00 | 30.130 | `ada0837b583d9fdfb47007b646251e087f7edfbe001601dc417cab7a39cb3709` |
| backbone-provisional | success | 2026-10-14T07:04:07.345850+00:00 | 2026-10-14T07:04:48.853385+00:00 | 41.507 | `f1181d4a7bbf1f179ed7bd6d208c5e7ecef42e4ee9c1fcc8debf58d890817067` |
| backbone-materialize | success | 2026-10-14T07:04:48.905900+00:00 | 2026-10-14T07:05:05.143591+00:00 | 16.242 | `2140cf04c3d5d511730e220a2862cbf2f17b13be6364a1ba2819abd375bf10a7` |
| backbone-final | success | 2026-10-14T07:05:05.191873+00:00 | 2026-10-14T07:05:39.596044+00:00 | 34.405 | `7bff615841f58491ae176798d415b1eec560ea2d04b1559074955f8c64889869` |
| encoder | success | 2026-10-14T07:05:39.662146+00:00 | 2026-10-14T07:05:56.331055+00:00 | 16.670 | `660610b3ec3d040754c9357f42bcccbd5f69eef25fc95342e54267b7ece9ad80` |
| ranker-train | success | 2026-10-14T07:05:56.419989+00:00 | 2026-10-14T07:10:17.426090+00:00 | 261.005 | `7b295fbf7a9c144d1f9be30099be47e5c6611ea90d8c0f26b493533cd79270ba` |
| ranker-mature | success | 2026-10-14T07:10:17.563346+00:00 | 2026-10-14T07:11:47.449863+00:00 | 89.888 | `43a6bc1a18d15a88afc4fe5816652900980a10bf091db0fdd28a4f5e779c0395` |
| cache-equivalence | success | 2026-10-14T07:11:47.504916+00:00 | 2026-10-14T07:12:21.700208+00:00 | 34.196 | `3fb1669d4fbed637f5776c28694794dba046c8ee8223acac72cd60880a089d32` |
| v2-select | success | 2026-10-14T07:12:21.775281+00:00 | 2026-10-14T07:19:57.495794+00:00 | 455.720 | `03b137f280bc05681850ab88ca76c4f917f835c68d51ca692e982fb440afd8f7` |
| v3-baseline-gate | success | 2026-10-14T07:19:57.546476+00:00 | 2026-10-14T07:20:03.949063+00:00 | 6.403 | `f0a869270a6171ee33116f648b967c3de811fe78a93cfc4ddeff550ead4358fc` |
| fast-equivalence | success | 2026-10-14T07:20:04.020521+00:00 | 2026-10-14T07:21:09.177899+00:00 | 65.158 | `f27d6fce97da80e976e6f8dc66a07654d98778536bca5c87d8e8b6c3337ce35c` |
| v3-fit | success | 2026-10-14T07:21:09.229106+00:00 | 2026-10-14T07:22:51.187415+00:00 | 101.958 | `916ba051f64b148e77247d1735154130a9df5108dcfa47bf54922929c2294c6d` |
| v3-tune | success | 2026-10-14T07:22:51.234003+00:00 | 2026-10-14T07:23:24.728184+00:00 | 33.495 | `8bcacafff886af9cbe7685b58797bc9a362f4d8882a8b0d5ebf99f44b1a9c0ad` |
| v3-confirm | success | 2026-10-14T07:23:24.795368+00:00 | 2026-10-14T07:23:53.098520+00:00 | 28.304 | `b02430318530aea2fc35e9944f9a0d2565f98346e2676d87ceceb528b2cd61fa` |
| v2-catalog | success | 2026-10-14T07:23:53.154039+00:00 | 2026-10-14T07:37:48.494680+00:00 | 835.341 | `4839fd835cb50e5d2ada9084134295f228dfff8563eb7a7982daa90e7003793a` |
| v2-ranking | success | 2026-10-14T07:37:48.557206+00:00 | 2026-10-14T07:40:43.565138+00:00 | 175.008 | `79cb8677323b8bb2ffc295f16f8eb0beb59473b1ba047e13ac2b8315e0a7ddf7` |
| v2-ranking-full | success | 2026-10-14T07:40:43.603817+00:00 | 2026-10-14T07:56:04.499299+00:00 | 920.896 | `b2b8f00065c94955b908aeaab3b1f1d23ba3ffe8685b187afb8b2e2427ec0732` |
| v2-audit | success | 2026-10-14T07:56:04.636454+00:00 | 2026-10-14T07:56:22.171529+00:00 | 17.535 | `518bb18e1195c770e7b727601a62a57c74781f000cf4b03a1acd4b81cc442e19` |
| v3-dev | success | 2026-10-14T07:56:22.259622+00:00 | 2026-10-14T07:57:32.014688+00:00 | 69.756 | `a8c9696bcba98e34813cebb0036b86a202d6eee51b9b2711f8bb0153033cdcc4` |
| v3-http | success | 2026-10-14T07:57:32.049729+00:00 | 2026-10-14T07:58:22.026859+00:00 | 49.978 | `fc574cdc00d7dfde078014f45e0d5919ac930f73f310ee5be4d9756f4410fc62` |
| v3-plot | success | 2026-10-14T07:58:22.090906+00:00 | 2026-10-14T07:58:31.496983+00:00 | 9.406 | `779431cfc6c55e4ff6f6a31a92128273f82b0e08d046974fec0014f7764dcb4a` |
| v3-audit | success | 2026-10-14T07:58:31.618196+00:00 | 2026-10-14T07:59:43.063990+00:00 | 71.445 | `c308535b9e5c5113f7199657a8a64745295798ec8ac12f2e3d2df0503a0f8095` |

## Input bindings

Dependency lock: `9ad728d429f637117f246c8f81bf753a837f87f690b3c75949552a22e731b39a` (45 pinned distributions).

| Raw input | Bytes | SHA-256 |
| --- | ---: | --- |
| dev/behaviors.tsv | 230,662,527 | `7031ced7b8de5183c1bf2c4a303018df8a49d858714597421d9489e571d87b45` |
| dev/news.tsv | 59,055,351 | `03ba5f02160672500bf5f1af6dd18e8f99830a845b08ca9fd9fbbc9bdd72baff` |
| train/behaviors.tsv | 1,373,844,151 | `886505b6e8b753fe6bc2d8988d959c2cb5865d85832829d0ad91fdfb54e17548` |
| train/news.tsv | 84,881,998 | `8d16f3e4df6ac8d8943c5f29e9155d0daf73047f1e41d37133013fe0e19953e9` |

## Interpretation limits

- The acceptance receipt previously rehashed raw TSVs, model/cache arrays and independent auditor bindings. This summary rehashes initialized sources, stage receipts, declared outputs, logs and summarized reports; it does not repeat the full raw/array audit or numerical predictions.
- MIND dev has already been observed. This run establishes neither unseen external validation nor a production deployment, distributed benchmark or online A/B test.
- Full-suite and research-only test runs overlap. Passed counts exclude skipped tests and must not be added as unique coverage.
- v1 paired intervals compare per-impression predictions averaged over seeds, not a score ensemble. v3 dev ablations are exploratory, with no multiple-comparison adjustment; removing explicit columns does not erase the same information from other features.
- The frozen confirm gate determines the serving recipe. Historical equality and dev ablation results do not select a new recipe.
- HTTP throughput and request latency describe this shared desktop run. Retrieval microbenchmarks are not HTTP throughput; cold means derived-cache cold, not operating-system cache cold. Observed speed ratios are not confidence intervals.
- Only aggregate values and content hashes are published. Source data, user identities, article text, model bytes, raw request/response logs and local absolute paths are excluded.

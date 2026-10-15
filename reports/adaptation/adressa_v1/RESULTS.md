# Adressa 目标域适配：完整执行结果

已观察基准上的后续研究；目标域重新拟合，不是新的冻结迁移或线上实验。

确认门通过：False；delta=0；实际策略=baseline。

| 方法 | Top10 / 请求 | HR@10 | HR@100 | MRR@10 |
|---|---:|---:|---:|---:|
| baseline | 4430 / 15000 | 0.295333 | 0.507533 | 0.111776 |
| legacy_frozen_english_delta10 | 4374 / 15000 | 0.291600 | 0.507533 | 0.106207 |
| matched_frozen_english | 4430 / 15000 | 0.295333 | 0.507533 | 0.111776 |
| adapted_english | 4430 / 15000 | 0.295333 | 0.507533 | 0.111776 |
| adapted_local | 4430 / 15000 | 0.295333 | 0.507533 | 0.111776 |
| adapted_local_no_recent | 4430 / 15000 | 0.295333 | 0.507533 | 0.111776 |
| adapted_local_no_gate | 4430 / 15000 | 0.295333 | 0.507533 | 0.111776 |
| deployed | 4430 / 15000 | 0.295333 | 0.507533 | 0.111776 |

| 对照 | 差值（pp） | 95% CI（pp） | 获益 / 受损请求 |
|---|---:|---|---:|
| adapted_local_minus_baseline | +0.0000 | [+0.0000, +0.0000] | 0 / 0 |
| deployed_minus_baseline | +0.0000 | [+0.0000, +0.0000] | 0 / 0 |
| adapted_english_minus_matched_frozen_english | +0.0000 | [+0.0000, +0.0000] | 0 / 0 |
| adapted_local_minus_adapted_english | +0.0000 | [+0.0000, +0.0000] | 0 / 0 |
| adapted_local_minus_no_recent | +0.0000 | [+0.0000, +0.0000] | 0 / 0 |
| adapted_local_minus_no_gate | +0.0000 | [+0.0000, +0.0000] | 0 / 0 |
| adapted_local_minus_legacy_frozen_english | +0.3733 | [+0.1598, +0.5814] | 157 / 101 |

完整逐日结果、训练样本、覆盖、选择、确认、来源与限制见 [evaluation.json](evaluation.json)。
128 请求独立候选与评分审计通过；所有请求保存排名与指标重算通过；原冻结模型逐请求结果完全复现。

# NewsForge 正式实验协议

本文件固定本轮实验的比较对象、训练预算与评估口径。结果只填写到实际执行生成的报告；配置、单元测试通过或计划运行均不等于训练完成。数据、训练和指标以 `data.py`、`store.py`、`train.py`、`gbdt.py`、`evaluate.py`、`metrics.py` 为准；最终调度与门禁由 `final_evaluation.py`、`gbdt_final_evaluation.py` 及其调度器实现。`METHOD.md` 中可选 attention 两塔属于探索接口，不在本轮正式三模型比较中。

## 研究问题与比较范围

本轮检验两个问题：同容量均值双塔采用曝光内混合难负例是否改善排序；候选相关历史聚合能否在不同容量的重排模型中取得有效的质量与成本折中。所有方法使用同一份训练期拟合的内容特征，不接入预训练语言模型。实验是在公开 MIND-LARGE 上进行的 CPU 离线研究，不代表腾讯内部数据、线上 A/B 实验或分布式多卡训练。

| 方法 | 输入与模型 | 正式训练预算 | 作用 |
| --- | --- | --- | --- |
| `mean_random` | 历史固定特征均值；用户/新闻各自 D→128→64 MLP，归一化点积 | 每个 seed 全部可用训练正例对 × 2 epoch；每正例 4 个同曝光随机负例 | 主对照 |
| `mean_mixed` | 与 `mean_random` 相同的结构、参数量和优化器 | 同样 2 epoch；第 1 轮随机，第 2 轮 2 个难负例 + 2 个随机负例 | 负采样的同容量对照 |
| `ranker` | 共享 D→64 投影、候选相关历史 attention、交互特征与隐藏层 64 的 MLP | 同样全量正例对 × 2 epoch；同样第 1 轮随机、第 2 轮混合 | 不同模型结构的重排比较 |
| GBDT | `similarity` 15 维相似度统计；`rich` 加原始内容分量后本数据为 85 维；`HistGradientBoostingClassifier` | 每种特征配置均为 seed 2026、同一批最多 500,000 正例对 + 每对 4 负例、固定 80 轮；内部验证选配置 | 较强的树模型基线；特征对照为内部验证阶段增补 |
| `content` | 固定新闻特征与平均历史画像的 cosine | 无监督参数训练；共享文本特征拟合成本 | sanity baseline |
| `popularity` | 仅 split 0 的平滑新闻点击率 | 无梯度训练 | sanity baseline |

三种神经模型固定 seeds **17、29、43**，每个模型和 seed 独立选择最优 epoch，全部运行均须报告。GBDT 只有一个固定随机种子，不能把它和神经模型的三次重复表述成相同训练预算。

## 数据、时间边界与泄漏控制

使用公开 MIND-LARGE 官方 train 与 dev。**官方 dev 在本项目中作为最终留出评估集**，不作为调参集，也不称为官方隐藏 test 或排行榜成绩。正式排序评估覆盖整个 dev 的完整候选曝光；文件中真实行数、曝光候选数、正例数与来源摘要以 `data/processed/manifest.json` 和 `provenance.json` 为准。

本次实际构建包含 2,609,219 个曝光组、97,592,931 条候选曝光、3,958,501 次点击、750,434 名去重匿名用户和 104,151 篇目录新闻。原始官方 train 为 2,232,748 组，官方 dev 为 376,471 组；没有下载或使用官方隐藏 test。时间字段覆盖 2019-11-09 至 2019-11-15，不能把历史日志描述为当前线上数据。

| 本项目切分 | 曝光组 | 候选曝光 | 点击正例 |
| --- | ---: | ---: | ---: |
| split 0：较早训练期 | 1,897,844 | 70,314,722 | 2,869,013 |
| split 1：内部时间验证 | 334,904 | 13,192,652 | 514,643 |
| split 2：锁定官方 dev | 376,471 | 14,085,557 | 574,845 |

目录新闻数包含历史中出现的新闻，不等于某个切分的候选新闻数；同一用户可跨切分出现，用户数不能直接相加。曝光组是一次带候选集合的请求，不是去重用户数或独立会话数。[全量审计](../reports/data_validation.json) 已逐块重算全部候选的计数与训练新闻统计。

官方 train 按 timestamp 划分约 85% 的较早曝光为 `split=0`，其余较晚曝光为内部验证 `split=1`。`time_split` 取 85% 分位所在秒为 cutoff，所有 `timestamp <= cutoff` 的曝光均进入训练，同秒时间组不被拆开。官方 dev 始终为 `split=2`。源时间没有时区；代码只将其解释为统一 UTC 坐标来获得可复现排序，不推断真实业务所在地。

本次 cutoff 为 `1573716628`，内部验证从 `1573716629` 开始，官方 dev 从 `1573776000` 开始。数据质量统计为 57,335 个空历史曝光、0 个缺失目录新闻、0 个同 ID 元数据冲突、0 个组内重复候选；空历史仍保留在相应切分中。

历史直接使用 MIND 为该请求提供的点击历史，保留最后 20 条，在右侧补 PAD=0。验证和 dev 可以使用各请求自身已提供的过去历史；不从它们的当前候选标签重建历史，不把当前点击追加到当前画像。新闻 first-seen 是历史或候选中第一次观察到其 ID 的请求时间，只是可用性代理，不是文章发布时间。

文本取标题与摘要；TF-IDF 使用 unigram/bigram、最多 60,000 词项、`min_df=2`、sublinear TF；随后用 seed 2026、5 次迭代的 TruncatedSVD 投影至最多 96 维并做文本 L2 归一化，再拼接训练期类别 one-hot 和未知类别列。词表、IDF、SVD 和类别词表只拟合在 split 0 的历史或候选中已出现的新闻。较晚新闻只能通过冻结的转换器变换。PAD 整行全零；不足 96 个有效 SVD 分量时其余文本维度为零。

本次拟合使用 98,673 篇训练期已可观察新闻，实际 TF-IDF 词表为 60,000，SVD 为 96 维；18 个训练类别加 1 个未知类别列，最终 `news_features.npy` 为 float32 `(104152, 115)`，其中额外 1 行为 PAD。

合并 train/dev 新闻目录仅建立数组地址，模型不学习新闻 ID 或用户 ID embedding。相同新闻 ID 的元数据发生冲突时优先保留 train 版本。用户 ID 只用于分组统计和 bootstrap。预处理为构建数组、检查格式和统计数据规模会解析 dev 标签；这些标签不进入特征拟合、负例池、损失或模型选择。“锁定 dev”指这一实验用途边界，不声称数据文件从未被解析。

## 负采样、目标与训练预算

`Store.ensure_training_cache()` 对 split 0 的每个曝光建立 16 个未点击新闻的固定负例池，缓存 seed 为 **20261010**。候选必须是同一曝光的 label=0，并排除该曝光任一 label=1 的新闻 ID；无足够负例时允许重复抽样，不用 PAD、其他曝光或未曝光新闻补齐。没有正例或没有可用负例的曝光不产生训练样本，跳过数量写入 `training_cache.json`。

因此“全量训练”精确指：**split 0 中全部能够构造同曝光负例的正例对**，每个 epoch 访问一遍；多点击曝光会产生多个正例对。不是 train+dev 都训练，也不是为补齐负例而把未知反馈当作负反馈。

本次缓存实测包含全部 **2,869,013 个训练正例对**，`skipped_no_positive_or_negative=0`，缓存总构建时间为 97.51 秒。九组神经运行现已各完成两轮，每轮都访问全部 2,869,013 个 eligible 正例对；这一结论由训练轨迹与最终 preflight 核验，不仅依据缓存计数。

数量核验使用 `training_cache.json` 的 `positive_training_examples` 与 `skipped_no_positive_or_negative`，并对照 checkpoint 的 `eligible_positive_examples_per_epoch`、`trained_on_all_eligible_clicks` 以及每轮轨迹中的 `training_examples`。正式运行要求每轮已访问例数等于可用正例对总数；不能只凭原始 TSV 的点击总量或一个布尔字段声称已训练完全部样本。

训练器在每个正式 epoch 结束时执行例数相等断言，并记录该轮 `state_sha256`。相同 seed 的 `mean_random` / `mean_mixed` 首轮权重摘要必须完全一致，否则在最终 dev 评估前失败；校验保存为 `matched_warmup_checks.json` 并纳入最终 ranking 报告。这一断言审计 warm-up 是否实际相同，不证明第二轮难例一定有收益。

`mean_random` 每步从固定池有放回取 4 个随机负例。`mean_mixed` 和 `ranker` 在第 1 轮采用相同随机采样；第 2 轮用当前模型、无梯度地对这 16 个负例打分，选最高的 2 个，再取 2 个随机负例。短池和 hard/random 重叠允许重复，按重复样本参与训练。挖掘发生在当前训练 batch，不是从 dev、全库或未来模型中挖掘。

每个训练样本包含 1 正 + 4 负，用交叉熵在 5 个候选中预测正例位置。三个神经模型都使用 AdamW：学习率 0.001、weight decay 0.0001、batch size 1024、梯度范数裁剪 5、CPU 线程 1。正式轮数固定为 2；本轮没有学习率网格搜索。`ranker` 的两维可选先验接口在当前训练器中不传入，实际均为零。

GBDT 从缓存正例对按 seed 2026 无放回抽至最多 500,000 对（不足则全部使用），每对从同一固定负例池有放回取 4 个负例，形成最多 2,500,000 个二分类样本。使用 15 维内容/历史统计：全特征 cosine 的 mean/max/top3/std、文本 cosine 的 mean/max/top3、平均文本画像 cosine 与 L1/L2 距离、类别 mean/max overlap、最后有效历史位置的文本 cosine、log1p 历史长度、空历史指示。没有用户/新闻 ID、标签特征或全训练期 CTR 目标编码。

GBDT 固定 `max_iter=80`、`learning_rate=0.1`、`max_leaf_nodes=31`、`l2_regularization=1.0`、`early_stopping=False`。原协议只使用上述 15 维特征，内部验证只报告；首次完成的模型、训练报告和验证矩阵逐字节保留为 `gbdt_similarity*` 产物，不覆写成新增协议下的历史。

**协议增补：在内部验证阶段、官方 dev 仍锁定时，增加 `rich` 特征对照。** 原 15 维统计可能压缩掉神经网络可见的内容信息，因此追加候选的前 16 维 SVD、有效历史均值的前 16 维 SVD、候选全部类别 one-hot 和有效历史均值全部类别 one-hot。本数据有 19 个类别列，故为 `15 + 16 + 16 + 19 + 19 = 85` 维。此增补不是实验开始前预注册，也不是完整复制神经网络的 96 维文本输入；它检验树模型获得更多原始内容信号后的表现。

两种特征配置使用相同 500,000 正例索引、相同 4 个同组负例抽样序列、相同固定 80 轮参数与 1 个 CPU 线程。在同一固定 12,000 组内部验证的完整候选上，以 nDCG@10 选择正式 `gbdt.joblib`；并列选较简单的 `similarity`。不选择 40/80 轮，不提前停止，不改变学习率。`gbdt_feature_ablation.json` 保留两种配置的全部验证结果、预算、样本与验证行摘要及选择原因；正式报告必须披露树模型有两种特征配置的选择预算。

GBDT 报告记录实际抽样数量、抽样索引 hash、构造特征及拟合耗时。新增 85 维的 2,500,000 行训练矩阵直接采用 float64 Fortran 顺序 memmap，实际矩阵数据大小为 1,700,000,000 字节，避免 sklearn 输入校验再生成一份完整 float64 转换。训练前以实时可用内存、矩阵、分箱与工作余量估算决定能否启动；每 0.1 秒采样全进程 RSS 和系统可用内存，实测值写入 `memory_observed`。该 RSS 包含映射页且采样峰值不是精确分配峰值；原 15 维运行未测 RSS，不补造内存数字。1:4 采样下的树模型概率和神经模型分数都不自动成为校准的总体点击概率。

本次 `rich` 已实际完成，训练进程最大观测 RSS 为 **3,652,857,856 字节（3.402 GiB）**，系统最小观测可用内存仅 **1,982,464 字节（1.891 MiB）**。启动前可用 2,864,914,432 字节、估计需求 2,449,370,912 字节，说明静态预检低估了瞬时压力，不能据此声称约 1.5–2.5 GB 空余就足够。运行没有 OOM，但曾接近内存上限；后续重现实验应串行运行这项训练并额外留出系统余量，不将本次并行执行当作内存安全保证。

本机正式运行让 GBDT 和神经训练各使用 1 个 CPU 线程并行执行；CLI 的通用默认值仍可能是 2，复现实验使用下方显式命令。正式训练前的同一 100-step smoke 中，1 线程和 2 线程在固定 1,000 个内部验证曝光上的指标相同，因此选择 1 线程以减少共享桌面资源占用。该单次短运行不作为严谨加速结论，模型、种子、轮数、样本与负采样预算保持不变。训练时长会受并行桌面负载影响，以报告中的实际线程配置和秒数为准。

`popularity` 排序 sanity 使用 `(train_clicks + 20*p)/(train_exposures + 20)`，其中 `p` 是 split 0 整体点击率。服务冷启动回退另按训练点击计数排序，其口径不与此 sanity 分数混写。

## 内部选择与最终留出评估

内部验证从 split 1 用 seed 2026 均匀无放回选定 **12,000 个曝光**，排序后固定给所有模型和 seed；不足则全取。每个被选曝光保留全部候选，不做评估负采样。

每个神经模型/seed 在 epoch 1 和 epoch 2 后分别验证，以内部验证 **nDCG@10 最大**选择 checkpoint；完全相同的数值保留较早 epoch。最终表报告被选择的 checkpoint 及 `best_epoch`，不能默认所有提交的权重都来自第 2 轮。用于服务的 retriever 和 reranker 也仅按内部验证从各自候选运行中选择。服务选出的一个 seed 不替代论文式三 seed 均值。

因此最终测试比较的是“按内部验证选择 epoch 的训练方案”：若混合方案选中第 1 轮 warm-up，其最终权重未包含难负例更新，不能将该测试差值归因于未选中的第 2 轮；报告另列各 seed 的共同首轮、随机/混合第二轮验证 nDCG@10、第二轮差值、所选 epoch 及选中 warm-up 的数量。

执行训练阶段必须显式使用 `--skip-test`，因为 `train.py` 的默认行为是在其请求的全部训练结束后评估 dev。固定命令如下，在 `newsforge/` 目录执行：

```powershell
python -m newsforge.train --variants mean_random mean_mixed ranker --seeds 17 29 43 --epochs 2 --batch-size 1024 --learning-rate 0.001 --threads 1 --validation-impressions 12000 --skip-test
python -m newsforge.gbdt --train-only --feature-mode similarity --max-positives 500000 --seed 2026 --threads 1 --validation-impressions 12000
python -m newsforge.gbdt --train-only --feature-mode rich --run-name gbdt_rich --max-positives 500000 --seed 2026 --threads 1 --validation-impressions 12000
python -m newsforge.gbdt --select-feature-ablation
```

全部训练、选择和协议冻结后，先核验内部调度等价性，再执行最终阶段。以下命令使用已经实现的独立入口，只读取已选权重，不重新训练、选择 checkpoint 或导出服务权重。完整重跑应使用新的源码副本，将交付的 `reports` 改名保存，再生成新的证据；不要让旧评估与新训练混在同一目录。

```powershell
# 固定内部 12,000 组的 GBDT 原序/分桶比较；拒绝覆盖已有证据。
python -m newsforge.benchmark_gbdt_scheduler --threads 1 --output reports/gbdt_evaluation_scheduler_validation_12000.json

# 两条命令均只做 preflight，不评估官方 dev。
python -m newsforge.final_evaluation --threads 1 --validate-only
python -m newsforge.gbdt_final_evaluation --threads 1 --validate-only

# 全部门禁通过后，评估官方 dev 全部曝光与全部候选。
python -m newsforge.final_evaluation --threads 1
python -m newsforge.gbdt_final_evaluation --evaluate-test --threads 1
```

已有证据的附加复查可为 `benchmark_gbdt_scheduler` 指定一个尚不存在的 `--output` 路径；默认报告不会被覆盖。`final_evaluation --validate-only` 检查九组训练及服务导出，并在相应证据文件不存在时执行 ranker 的内部等价比较；已有证据必须与当前摘要匹配且通过，过期或失败记录会阻止继续。GBDT 入口核对既有特征选择、权重与完整内部等价证据。

本次九组神经训练已完成两轮，[神经 preflight](../reports/final_evaluation_preflight.json) 和 [GBDT preflight](../reports/gbdt_final_evaluation_preflight.json) 均为 `passed`。seeds 17、29、43 的随机/混合双塔首轮权重摘要全部一致，见 [warm-up 核验](../reports/matched_warmup_checks.json)。三个 ranker 的完整内部调度等价核验及既有服务导出也已通过；内部选出的 retriever 为 `mean_random-seed43`，reranker 为 `ranker-seed29`。模型与选择冻结后的神经及 GBDT 正式排序评估均已完成，各覆盖全部 376,471 个官方 dev 曝光组、14,085,557 个候选。点击召回、正式 HTTP 基准与真实演示也已完成，其证据与局限单独列于下文。

最终 dev 只作为一次完整报告阶段。不能根据 dev 决定继续训练、改变特征、换 seed 或修改超参数后仍把相同 dev 当未见测试。必要的实现修复要留下原因、修复前后源代码及重跑记录，并明确其影响。`--smoke-steps` 只验证流程，必须使用不同 artifact 目录，不能进入正式结果。

### CPU 评估调度增补：dev 解锁前的实现优化

观察到 ranker 全候选评估中的 CPU padding 开销后，在官方 dev 仍锁定时新增 `evaluation_scheduler.py`。调度只按每个曝光的候选数对 row IDs 做稳定排序，再调用原 `evaluate`；不读取标签或分数来决定排序，不改组内候选顺序，不截断或抽样候选。ranker 仍使用原来的 batch size 32。计算后通过 `values[order] = bucketed_values` 恢复原 row 顺序，供模型配对、用户分组和 bootstrap 使用。`train.py`、`store.py`、`models.py`、`evaluate.py`、`metrics.py` 五个冻结核心文件没有改变，训练、checkpoint 选择与四项指标定义保持不变。

三个 ranker 的所选 checkpoint 已在相同固定 12,000 组、全部 477,756 个候选上完成核验，证据分别为 [seed 17](../reports/evaluation_scheduler_validation_12000.json)、[seed 29](../reports/evaluation_scheduler_ranker_seed29_validation_12000.json)、[seed 43](../reports/evaluation_scheduler_ranker_seed43_validation_12000.json)。三者所选 epoch 均为 1；恢复行序后的逐曝光 AUC、MRR、nDCG@5、nDCG@10 最大绝对差均为 0，NaN 掩码一致。汇总求和顺序只产生约 `5.55e-17` 的差异：seed 17 出现在 MRR，seeds 29、43 出现在 nDCG@5，其余汇总差为 0。这里核对的是逐曝光指标，不能据此宣称候选 logits 逐位相同。

| 三个 ranker 共同的内部验证计算量 | 原 row 顺序 | 按候选数分桶 |
| --- | ---: | ---: |
| 实际候选数 | 477,756 | 477,756 |
| 总候选计算槽位 | 1,906,752 | 482,624 |
| padding 槽位 | 1,428,996 | 4,868 |

| 所选 ranker | 原序秒数 | 分桶秒数 | 本次原序/分桶时间比 |
| --- | ---: | ---: | ---: |
| seed 17，epoch 1 | 23.690 | 14.875 | 1.593 |
| seed 29，epoch 1 | 27.726 | 15.537 | 1.785 |
| seed 43，epoch 1 | 26.943 | 17.643 | 1.527 |

每行仅是共享桌面、1 个 CPU 线程下的一对顺序观测：先原序、后分桶，后台负载和缓存预热未被控制。三行来自不同 checkpoint，不能当作同一权重的受控计时重复，也不能外推为通用加速比、训练加速或 HTTP 服务收益。

`final_evaluation.py` 核对九个 checkpoint 的完整两轮轨迹、训练配置、数据/缓存/核心源码摘要、内部最佳 epoch 和权重摘要，验证同 seed 双塔共同首轮一致及既有服务权重仍符合原内部选择。正式使用分桶调度前，要求对 **seed 17、29、43 三个 ranker** 在同一固定内部 12,000 组上完成原序/分桶核验，并绑定 checkpoint、数据、五个核心文件和 scheduler 的 hashes；任一指标误差超过 `1e-12` 或 NaN 掩码不一致，均拒绝进入正式 dev。有效历史证据可按摘要复用；过期证据会报错保留，不能静默覆盖。本次三个实际门禁均已通过，preflight 自身未进行 dev 评估；其后已完成的正式排序结果另存于 `ranking_experiment.json`。

同样的候选数调度已用于所选 85 维 GBDT，保持原 `evaluate_gbdt`、特征计算和 batch size 64，计算后恢复原曝光行序。[GBDT 内部证据](../reports/gbdt_evaluation_scheduler_validation_12000.json) 覆盖相同 12,000 组、477,756 个真实候选，四项逐曝光指标及汇总指标的最大差均为 0。总计算槽位从 2,217,760 降为 486,720，padding 从 1,740,004 降为 8,964；本次原序/分桶计时为 82.001/40.055 秒。随后使用仓库中 `benchmark_gbdt_scheduler` 的[独立复现记录](../reports/gbdt_scheduler_reproduction_check_12000.json) 再次得到四项逐曝光差 0，计时为 57.475/30.935 秒。两次都是共享 CPU、1 线程、先原序后分桶的单对测量，负载与缓存顺序未控制，不能合并成受控重复实验或外推通用加速比。比较对象是逐曝光指标，不声称特征矩阵或分类概率逐位相同。

`gbdt_final_evaluation.py` 已实现，默认或 `--validate-only` 仅核验：原两种特征配置与内部矩阵均保留、rich 仍按既定规则被选中、权重/特征/80 轮与样本预算未变、调度证据绑定当前数据和源码。只有显式 `--evaluate-test` 才进入全部官方 dev，并拒绝覆盖既有正式 GBDT 结果。两个最终入口都保留完整候选与原始曝光指标行序；调度不改变三 seed 汇总、用户聚类 bootstrap 的配对关系或指标定义。

## 指标、重复实验与统计不确定性

排序采用 [MIND 官方评估定义](https://github.com/msnews/MIND/blob/master/evaluate.py)：先在每个完整 impression 内计算，再对 eligible impressions 做等权宏平均。

| 指标 | 本实现的精确定义 |
| --- | --- |
| AUC | 同一曝光内正例分数高于负例的概率，分数并列计 0.5；不是把所有曝光拼接的 global AUC，也不是用户加权 GAUC |
| MRR | `sum(y_at_rank / rank) / number_of_positives`；对所有点击的倒数排名取均值，不是只看第一个点击 |
| nDCG@5/@10 | 二值相关性下，截断 DCG 用 `1/log2(rank+1)` 折扣，再除以同曝光理想 DCG |

只有单一标签类别时 AUC 不定义；无正例时 MRR/nDCG 不定义。实现用 NaN 在对应宏平均中排除，并报告每项 `eligible_impressions`。AUC 用平均秩处理并列，排序指标用输入候选顺序作为稳定并列规则。官方参考脚本的 `np.argsort(...)[::-1]` 并列次序并不相同，因此有并列分数时不能声称逐位严格等于官方脚本；无并列时定义一致。

每种神经模型报告 seeds 17/29/43 的单次结果、算术均值和样本标准差（`ddof=1`）。三次样本很少，标准差只描述这些训练运行的波动，不代表充分探索了初始化分布。GBDT 与无监督 sanity 没有三 seed 分布，不填虚构的标准差。

`metrics.user_cluster_interval` 对每个曝光先计算待比较模型与 `mean_random` 的配对指标差；神经模型先对三个 seed 的**逐曝光指标**求均值。然后以匿名用户为抽样簇，有放回重抽用户 500 次（seed 2026），一个用户被抽中时保留其全部曝光；每次仍计算曝光宏平均差，用 2.5%/97.5% 分位数给出 95% 区间。估计对象是训练 seed 平均后的指标差，不是分数 ensemble，也不是每个用户等权的用户宏指标。该区间主要反映评估用户抽样不确定性，没有把三个训练 seed 作为独立大样本再推断。最终 `report.py` 同样读取所选单个 GBDT 的逐曝光指标，与三 seed 平均后的 `mean_random` 比较；先核验它们覆盖相同完整 dev、使用相同行序，再计算上述配对用户区间并保存 `gbdt_paired_comparison.json`。GBDT 仍只有一个训练 seed，不能将此区间解释为其训练随机性估计。

分组报告包括空历史、1–5 条历史、6–20 条历史，以及含训练期未见点击新闻/训练点击次数为零新闻的曝光。后两类是**曝光级子集**，曝光里的全部候选仍参与 AUC/nDCG；不能把它们称为仅在冷新闻对上计算的 AUC。

## 公平性与可报告的结论

`mean_random` 对 `mean_mixed` 是主要受控对照：相同结构、容量、训练样本预算、优化器和 seed。混合模型第 2 轮增加对 16 候选的无梯度打分，所以梯度更新数量一致不意味着 CPU 时间一致，必须同时报告训练耗时。

`ranker` 改变表示、容量、候选相关性及计算量；其结果不能单独证明 attention 的因果贡献。要做该项归因，需后续加入容量相近的固定池化 ranker。每种 GBDT 配置最多只看相同 50 万个正例对，使用不同的内容表示与二分类目标；两种特征配置经内部验证选择，没有三 seed 重复。它是有实质意义的替代方法比较，但不是同容量或同数据量消融。表格同时列训练样本量、epoch/树轮数、参数量（神经模型）、实际耗时和选择预算。

### 本轮正式排序观察

[神经模型正式报告](../reports/ranking_experiment.json) 与 [GBDT 正式报告](../reports/gbdt_evaluation.json) 均已覆盖全部官方 dev 曝光及候选。以下神经结果为三个 seed 的均值 ± 样本标准差，GBDT 为所选 rich 的单次结果。差值和区间的单位为百分点（pp，即原始指标差乘以 100），并非相对百分比变化。

| 方法 | nDCG@10 | 相对 `mean_random` 差值，pp | 配对用户 bootstrap 95% 区间，pp |
| --- | ---: | ---: | --- |
| `mean_random` | 0.3858138 ± 0.0071019 | 基准 | — |
| `mean_mixed` | 0.3833533 ± 0.0095473 | −0.246054 | [−0.272370, −0.218324] |
| `ranker` | 0.3858421 ± 0.0033714 | +0.002829 | [−0.048861, +0.052756] |
| GBDT `rich` | 0.3854632，单 seed | — | 本表不据单次点值推断显著性 |

本轮按内部验证选 epoch 的 `mean_mixed` 方案低于随机对照；ranker 的 nDCG@10 差值接近零且区间跨零，没有稳定提升证据。混合方案 seeds 17、43 选择第 1 轮预热，只有 seed 29 选择第 2 轮，因此 **2/3 最终混合权重没有难负例更新**：这是带内部选模的训练方案比较，不能把最终差异直接当成难例更新本身的因果效应，也不能宣称难负例方法普遍无效。

GBDT 的 nDCG@10 接近两种较高的神经点估计，说明需保留这个有竞争力的替代基线；其训练样本量、目标与重复次数不同，不能据此称模型等价或同预算获胜。用户聚类区间只刻画这些已训练权重下的评估用户抽样不确定性，没有涵盖全部训练随机性；三 seed 标准差另行报告。以上是公开历史日志的离线排序结果，不支持 SOTA、线上 CTR 提升或腾讯业务收益的主张。完整曝光候选内的排序、全目录点击覆盖与系统性能必须分别判断。

全库召回实验与官方候选排序分开。`evaluate_retrieval.py` 默认在最终 dev 中固定抽取最多 5,000 个请求，使用 first-seen 时间过滤可用新闻并排除已看新闻；报告点击 micro Recall@10/@100、eligible 点击分母、冷新闻点击覆盖及请求样本数。未曝光新闻反馈未知，不能解释成线上 CTR 或无偏相关性。重排同一 top-100 只能改变头部顺序，不会改变 Recall@100。

服务的 retriever/ranker checkpoint 按内部验证选择，但暖用户默认 two-stage 是预先指定的实验架构，没有在内部验证或 dev 上对服务策略择优；最终将 retrieval-only 与 two-stage 的效果并列报告，不能看完 dev 后改变默认策略并仍声称 dev 未参与选择。

ANN 对 exact top-K 的邻居覆盖、索引大小、延迟分位数和服务吞吐属于系统指标，应记录线程数、并发、batch、过滤、回退与硬件环境。它们不等同于点击 Recall，更不等同于官方 AUC。当前 CPU 向量化与本机 FastAPI 服务不提供多机/多卡或生产流量证据。

### 本轮召回、HTTP 与演示观察

[点击召回报告](../reports/retrieval_experiment.json) 使用 seed 2026 固定抽取 5,000 个 dev 请求。共 7,576 次日志点击；服务排除 16 次已读重复点击后，eligible 分母为 7,560，其中训练期未见新闻的 eligible 点击为 5,686。两阶段系统全点击 micro Recall@100 为 **0.0081837381（62/7,576，约 0.82%）**，eligible Recall@100 为 0.0082010582（62/7,560）。双塔与两阶段的 Recall@100 相同，因为后者只重排相同的 100 个候选。全目录点击覆盖仍低，不能凭曝光候选内的排序成绩或低服务延迟声称推荐效果已经可上线。

[服务基准](../reports/service_benchmark.json) 从内部验证固定抽取 128 个无标签请求。3 个空历史请求不参与 ANN 非零向量比较，实际 ANN 查询为 125；HNSW 对相同时间/已读过滤后的 exact top-100 平均邻居 Recall@100 为 **0.89128**。这是近似检索重合度，既不是上述 5,000 个 dev 请求的点击召回，也不是推荐相关性；ANN 局部计时还不包含用户编码。

真实 localhost HTTP 测量保留全部 128 个请求，每种配置重复 3 遍，共 384 次推荐查询；每种配置预热的 8 个 HTTP 请求不计入结果。单服务 worker，PyTorch/FAISS 各 1 线程，每次 HTTP 调用新建连接。下表延迟为整个 HTTP 请求的 RTT，batch=8 时不能除以 8 当作单用户延迟；吞吐为成功推荐查询数除以整段 wall time。

| HTTP batch | 客户端并发 | 实测 HTTP 请求数 | p95 RTT，ms | 推荐 query/s | 错误数 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 384 | 30.5672 | 69.718 | 0 |
| 1 | 4 | 384 | 60.7342 | 80.831 | 0 |
| 8 | 1 | 48 | 35.8416 | 264.905 | 0 |
| 8 | 4 | 48 | 140.4223 | 255.462 | 0 |

四种配置均零错误、无 top-K 数量不足。机器为共享 CPU 桌面，背景活动未控制；客户端 worker 开始前的排队时间不计入请求延迟，但计入吞吐 wall time。这些单机 loopback 观察不是生产 SLA、远程网络性能、分布式加速或线上 A/B 证据。

[真实演示核验](../reports/demo_verification.json) 使用 20 条真实历史、100 个候选并返回 10 条推荐，shortfall=0；6 项检查全部为 true，覆盖仅用内部验证、非空历史、未读取点击标签、模型时间边界、版本与过滤规则以及标题可解析。它证明这次真实请求和展示流程可运行，不证明 10 条推荐的相关性或全目录覆盖充分。

## 产物与复现

`requirements.txt` 固定本机 Python 3.10.21 环境核对到的直接依赖版本；PyTorch 使用 CPU wheel。环境可用性测试不等于已经在新机器完成从零安装。正式报告保存数据 manifest/provenance、来源文件摘要、配置、种子、训练秒数、epoch 验证轨迹和最终逐曝光指标矩阵。

本次 [70 项测试输出](../reports/unit_tests.txt) 已通过，包含模型、数据、指标、调度恢复行序、等价证据校验及合成本地服务生命周期等检查。单元测试不会替代三个实际 ranker、所选 GBDT 的内部核验，也不代表正式 dev 或服务基准完成。最终 preflight 分别保存为 `reports/final_evaluation_preflight.json` 和 `reports/gbdt_final_evaluation_preflight.json`，执行状态应以这些文件与正式报告为准。

神经模型结果位于 `reports/ranking_experiment.json`；GBDT 所选配置的训练与最终评估分别位于 `reports/gbdt_training.json`、`reports/gbdt_evaluation.json`。特征选择记录为 `reports/gbdt_feature_ablation.json`；两个原始训练报告为 `gbdt_similarity_training.json` 和 `gbdt_rich_training.json`，模型与验证矩阵同样按配置保留。`artifacts/metrics_gbdt.npy` 的列顺序同主评估器：AUC、MRR、nDCG@5、nDCG@10；对应 row IDs 为 `evaluation_rows_gbdt.npy`。比较前必须验证它与主实验 `evaluation_rows.npy` 完全相同。

原始数据、逐用户行为文件和可重建原始数据的缓存不随代码再分发。公开展示使用代码、协议、合成测试样例和聚合结果，数据使用者自行遵守 [MIND 研究许可](https://github.com/msnews/MIND/blob/master/MSR%20License_Data.pdf)。相关工作与代码来源见 `RELATED_WORK.md`。

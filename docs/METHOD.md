# NewsForge：曝光负例训练与候选相关兴趣重排

本项目围绕一个可验证的问题：同样的新闻内容特征和模型容量下，训练曝光中的难负例能否改善两塔排序与召回；在召回后，用候选相关新闻重新聚合用户历史是否带来额外收益。模型为独立 PyTorch 实现，不把常见双塔、attention 或批量矩阵计算称为新算法。

## 数据与固定特征

[MIND](https://aclanthology.org/2020.acl-main.331/) 提供真实新闻曝光及点击标签，每条行为包含曝光之前的点击历史。正文不在公开下载中，本项目使用公开标题/摘要及类别字段；候选位置已经打乱，不能把记录顺序当作原页面位置。[官方数据描述](https://github.com/msnews/msnews.github.io/blob/master/assets/doc/introduction.md)

`news_features.npy` 为 float32 `[N,D]`：训练数据拟合的 TF-IDF、96 维 SVD 内容表示与类别 one-hot。零号行为 PAD，全零向量。留出新闻使用同一个训练变换，可产生内容表示，不拟合留出文本的词频、SVD 或类别统计。这里的冷新闻能力来自文本映射，不来自任意未见 ID 的随机 embedding。

本次全量构建的实际形状为 `[104152,115]`：104,151 篇 train/dev 合并目录新闻与 1 行 PAD；96 个文本维度、18 个训练类别和 1 个未知类别列。文本和类别拟合范围是 split 0 历史或候选中已出现的 98,673 篇新闻，不是所有目录新闻。规模与切分的完整分母见 [实验协议](PROTOCOL.md)。

历史按数据记录给定的先前点击列表读取，截断到最近 20 条并右补零。模型还接受显式 bool `[B,H]` mask。无论右侧补多少 PAD，或被 mask 的张量中有什么非零值，有效用户和候选分数应保持一致。空历史显式处理，不借用当前曝光的点击标签。

## 两塔模型

`TwoTower(feature_dim, embedding_dim=64, hidden=128, user_pooling="mean")` 包含两个独立 MLP：`D → 128 → 64`，ReLU 激活。新闻塔将单条固定内容向量映射为单位向量。用户塔先在固定 D 维空间池化历史，再经过 MLP 与 L2 归一化；因此无需在每个训练 batch 对全部历史新闻重复运行新闻 MLP。

两种历史池化方式：

- `mean`：只对有效历史取均值。随机负例与难负例版本使用完全相同结构、参数量和训练预算，是主对照。
- `attention`：对有效固定历史特征应用 `Linear → tanh → Linear` 得分，再做 masked softmax 加权求和。它是普通 additive attention 的未运行探索接口，不属于本轮正式比较，也不声称等同于多兴趣模型或完整 NRMS。

训练 logits 为用户和新闻单位向量的点积乘以可学习 scale，初始值为 10，限制在 `[1,100]`。正 scale 不改变检索顺序，ANN/精确召回可以直接使用单位向量内积。空历史用户向量和 PAD 新闻向量均为零；服务层须为无历史用户单独选择训练先验回退，不能把零向量下的任意并列结果当作有意义个性化推荐。

接口：

```python
model = TwoTower(feature_dim=D, embedding_dim=64, hidden=128, user_pooling="mean")
news_vectors = model.encode_news(news_features)             # [..., D] -> [..., 64]
user_vectors = model.encode_user(history_features, mask)    # [B,H,D], [B,H] -> [B,64]
logits = model(history_features, candidates, mask)           # [B,C,D] -> [B,C]
```

新闻 `news_features.npy` 是固定输入，不因模型训练而改变。投影后的 `news_vectors` 依赖可学习新闻塔，只能在冻结 checkpoint 后缓存。换模型必须重算对应新闻向量并重建索引；不能继续使用旧索引。

## 曝光难负例的受控实验

训练采样由训练器管理，模型不读取标签、挖掘负例或更新先验。先比较同容量的均值两塔：一个随机采样，另一个在 warm-up 后混合随机负例和当前模型评分较高的未点击候选。难负例必须来自**同一个训练 impression 的 label=0 新闻**；当前 impression 的其他点击不得成为负例，未曝光新闻不能自动称为真负例。候选预算、随机种子和梯度步数保持一致。难负例在每个训练步用该步当前权重在线打分选取，不逐步保存 checkpoint；记录保留每个 epoch 的权重摘要、负例策略和验证结果，实际权重保存为内部验证选出的最佳 checkpoint。

采样改变了训练分布，模型 logits/softmax 本身不保证是校准的真实点击概率。验证和最终评估必须使用 impression 全部候选，不能继续负采样。

## 候选相关兴趣重排

`AttentionRanker(feature_dim, hidden=64, embedding_dim=64)` 接收 `[B,H,D]` 历史与 `[B,C,D]` 候选。共享无偏置线性层把历史和候选映射为归一化 64 维向量。一次 batch 矩阵乘法形成 `[B,C,H]` 相似度矩阵；每个候选只在有效历史上做 attention，再得到候选相关的历史表示。

打分 MLP 输入包含：候选向量、其加权历史表示、两者逐元素乘积、绝对差值；候选与历史的 attention 加权点积、平均/最大相似度、是否有历史；另预留两维可选 train-only 先验接口。若将来启用，调用方需提供已变换的数值，并在 checkpoint manifest 中记录先验顺序、变换和统计截止时间，模型不擅自拟合或更新。本轮正式训练、评估与服务均不传入这两维先验，实际输入全零。

category one-hot 已在固定特征内。当前模型学习的是内容与类别的共享投影，不把投影空间相似度误称为独立计算的类别重合率。

```python
ranker = AttentionRanker(feature_dim=D, hidden=64, embedding_dim=64)
logits = ranker.score_candidates(
    history_features,             # [B,H,D]
    history_mask,                 # bool [B,H]
    candidate_features,           # [B,C,D]
    candidate_prior_features=None,  # 本轮不传先验，内部生成全零 [B,C,2]
)
```

重排分数不在候选之间做归一化，因此候选单独评分、放入更大候选列表或更大 batch 应在浮点容差内一致。全 PAD 历史时 attention 权重与历史表示都为零，历史最大相似度显式归零；本轮仍可依据候选内容给出有限 logits。ranker 不含 dropout，便于验证批量与逐候选计算的一致性。

## 基线、评估与贡献边界

本轮正式比较固定为 `mean_random`、`mean_mixed`、`ranker` 三种神经模型，以及 `content`、`popularity` 和 GBDT 基线。GBDT 的 15 维 `similarity` 与 85 维 `rich` 方案使用相同固定超参数和训练样本，经内部验证选择 rich；追加特征比较的开发过程见 [实验协议](PROTOCOL.md)。`attention_mixed` 只保留为未运行的探索接口，不计入正式结果。重排改变模型容量，不能把其结果全归因于 attention；可用固定池化但参数量相近的 ranker 做后续对照。

对齐 [MIND 官方评估](https://github.com/msnews/MIND/blob/master/evaluate.py)：逐 impression AUC 宏平均、所有正例倒数排名均值的 MRR、nDCG@5/@10。官方候选重排、时间可用新闻池的点击召回、ANN 对 exact top-K 的近似率须分别报告。不能把三者的 Recall 混为一个数字。训练词表、SVD、负例挖掘和先验统计不得使用最终留出标签。

观察到 CPU 评估开销后，在官方 dev 解锁前追加了按候选数稳定排序曝光组的调度器 `evaluation_scheduler.py`，所选 GBDT 另由 `gbdt_evaluation_scheduler.py` 包装原评估器。相近长度的曝光进入同一 batch，减少为本 batch 最长曝光补齐的无效候选槽位；保留每组全部候选及其原有顺序，ranker 与 GBDT 仍分别使用原 batch size 32 与 64。模型评分独立于候选集合，计算后用 `values[order] = bucketed_values` 恢复指标矩阵原 row 顺序，再进行模型配对与用户 bootstrap。五个冻结训练/数据访问/模型/评估/指标核心文件及 GBDT 特征、模型和指标定义不变。

ranker seeds 17、29、43 的所选 checkpoint 已在相同固定内部 12,000 组、477,756 个候选上通过实际核验，三者所选 epoch 均为 1。四项逐曝光指标最大差均为 0；汇总求和顺序差的绝对值最多约 `5.55e-17`，seed 17 出现在 MRR，seeds 29、43 出现在 nDCG@5。所选 85 维 GBDT 在相同样本上的逐曝光和汇总指标差均为 0，并已通过仓库 `benchmark_gbdt_scheduler` 独立复查。三个 ranker 的原序/分桶 wall time 分别为 23.690/14.875、27.726/15.537、26.943/17.643 秒；GBDT 两份记录分别为 82.001/40.055 秒与 57.475/30.935 秒。每份都是共享桌面先原序、后分桶的单对测量，背景负载和缓存未控制，不同 checkpoint 也不构成同一权重的受控计时重复；这些观测不能外推通用加速、训练收益或 HTTP 吞吐。

最终入口 `final_evaluation.py` 与 `gbdt_final_evaluation.py` 先用 `--validate-only` 核验训练/特征选择、既有权重、数据与源码摘要、固定内部调度等价证据；任何逐曝光指标误差超过 `1e-12` 或证据不匹配都会拒绝正式评估。九组神经运行已各完成两轮，三个 seed 的随机/混合双塔首轮权重完全一致；神经及 GBDT 的 preflight 均已通过，既有服务导出已验证。模型与内部选择冻结后的正式排序评估现已完成，全部方法覆盖 376,471 个 dev 曝光组、14,085,557 个候选。点击召回、正式 HTTP 基准和真实演示也已完成；完整 CLI、证据、计算槽位与门禁见 [实验协议](PROTOCOL.md)。

实际 [神经结果](../reports/ranking_experiment.json) 的 nDCG@10（三 seed 均值 ± 样本标准差）为：随机双塔 `0.3858138 ± 0.0071019`，混合双塔 `0.3833533 ± 0.0095473`，ranker `0.3858421 ± 0.0033714`；[GBDT rich 单次结果](../reports/gbdt_evaluation.json) 为 `0.3854632`。混合方案相对随机方案低 `0.246054` 个百分点，配对用户 bootstrap 95% 区间为 `[−0.272370, −0.218324]` 个百分点；ranker 差值为 `+0.002829` 个百分点，区间为 `[−0.048861, +0.052756]`，没有 nDCG@10 稳定提升证据。

混合方案 seeds 17、43 的内部最优权重来自第 1 轮预热，只有 seed 29 来自第 2 轮；2/3 最终混合权重没有难负例更新，不能将训练方案间的最终差异直接归因于难例更新本身。用户聚类区间针对三个 seed 的逐曝光指标均值，未涵盖全部训练随机性；GBDT 的样本预算与目标也不同。这些离线结果不证明 SOTA 或线上收益，研究贡献是可核验的对照、失败结果与工程实现。

[全目录点击召回](../reports/retrieval_experiment.json) 在 5,000 个固定 dev 请求上完成：7,576 次全点击、7,560 次 eligible 点击，其中 5,686 次为训练期未见新闻的 eligible 点击。两阶段全点击 Recall@100 仅 `0.0081837381`（62/7,576，约 **0.82%**），全目录覆盖仍低；曝光候选内的排序指标不能替代这个结果。全点击分母保留服务过滤掉的 16 次已读重复点击，eligible 分母单独报告。

[工程基准](../reports/service_benchmark.json) 使用内部验证的 128 个请求，3 个空历史不进入 ANN 非零向量比较，实际 125 个 ANN 查询的近似邻居 Recall@100 为 `0.89128`；它衡量近似邻居一致性，不衡量点击相关性。单请求、并发 1 的真实 HTTP p95 为 `30.5672 ms`、吞吐 `69.718 query/s`，全部四种 batch/并发配置均零错误。[演示核验](../reports/demo_verification.json) 的 20 条历史、100 个候选、10 个返回结果通过全部 6 项检查。共享 CPU 背景负载未控制；运行正确和较低延迟不能证明推荐效果可上线，也不构成线上 A/B、生产 SLA 或分布式能力证据。

`tests/test_models.py` 检查输出形状、有效向量单位范数、PAD/显式 mask 不变性、空历史数值稳定、逐候选与批量评分一致、参数/输入梯度、配置与 state_dict 恢复及错误形状。本次全套 **70 项测试**已通过，包括额外的评估行序恢复与证据校验；完整输出见 [测试记录](../reports/unit_tests.txt)。测试证明实现契约，不证明业务效果或加速；效果与端到端性能需要实际训练、全量评估和服务计时。

本实现借鉴了以下研究问题和标准组件，但没有复制对应项目代码，也不是这些论文的完整重现：

- [MIND: A Large-scale Dataset for News Recommendation](https://aclanthology.org/2020.acl-main.331/)：曝光反馈、历史与文本新闻推荐评估。
- [NRMS: Neural News Recommendation with Multi-Head Self-Attention](https://aclanthology.org/D19-1671/)：新闻与用户编码、注意力历史建模。当前模型没有其 token-level 新闻 self-attention，不称为 NRMS 原版。
- [DIN: Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/abs/1706.06978)：根据候选聚合相关历史兴趣。当前轻量点积 attention ranker 是独立设计的简化实现，不声称完整 DIN 复现或新的 attention 算法。

当前固定 TF-IDF/SVD 特征属于文本内容建模，不等同于大语言模型训练。CPU 向量化和批量推理也不等同于分布式多卡加速。数据规模、指标改善及系统吞吐只能引用实测报告。

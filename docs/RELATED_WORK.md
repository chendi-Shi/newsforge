# 来源、相关工作与本项目的贡献边界

NewsForge 是围绕公开新闻曝光数据独立实现的研究工程项目。它参考 MIND 的评估任务、新闻/用户编码思想、候选相关历史聚合和标准近似检索组件；没有把上游仓库改名作为独创成果。本轮价值在于可复现的数据边界、同容量负采样对照、候选相关重排及离线指标到本机服务的连接。效果是否改善以执行报告为准。

以下来源在 2026-10-10 核对；仓库文件通过 GitHub 插件读取，论文使用作者论文的官方出版页或 arXiv。下列 SHA 是 **Git 文件 blob SHA，不是 commit SHA**，用于标识本次核对的文件内容。

## MIND 数据与官方评估

Wu et al., **MIND: A Large-scale Dataset for News Recommendation**, ACL 2020，提供新闻文本、匿名用户、请求历史与点击/未点击曝光反馈，适合检验内容冷启动和新闻排序。[论文](https://aclanthology.org/2020.acl-main.331/)；[数据主页](https://msnews.github.io/)。

本项目使用 MIND-LARGE 的公开 train/dev，实际处理数量见本地 manifest。官网描述的整个数据集规模包含其他划分，不能直接当作本次训练行数。本项目没有腾讯数据，也没有线上真实用户的产品效果证据。

[MIND 官方评估脚本](https://github.com/msnews/MIND/blob/master/evaluate.py)定义逐曝光宏平均 AUC、所有点击倒数排名均值的 MRR 和 nDCG@5/@10；核对 blob 为 `f88de358f44c2986985a51912d419d29c0ed1aaa`。本项目在 `metrics.py` 独立实现这些数学定义，显式处理无正例/无负例以及稳定并列排名。没有复制该脚本作为本项目源码；仓库未声明通用 MIT/Apache 软件许可，不把其公开可见误写成许可授权。

MIND 数据本身采用 [Microsoft Research License Terms](https://github.com/msnews/MIND/blob/master/MSR%20License_Data.pdf)，不是 MIT 或 Apache；核对 blob 为 `1af542bb4ceade4cd76223e019d85e3d827cb8c6`。第 1 条允许符合条件的非商业研究、分析测试和不包含实质数据内容的结果发表；第 2(a) 条不授予分发数据集的权利。因此本仓库不上传或转发原始 zip、TSV、逐用户行为或数据缓存；第三方自行从上游获取并遵守条款。开源实现的许可不能覆盖或替换数据许可。

## Recommenders / NRMS：参考结构，不宣称完整复现

微软发起的 [Recommenders](https://github.com/recommenders-team/recommenders) 提供新闻推荐实现与 MIND 加载工具，当前仓库采用 [MIT License](https://github.com/recommenders-team/recommenders/blob/main/LICENSE)，版权归 Microsoft Corporation 和 Recommenders contributors。许可文件核对 blob 为 `e74b0d177a471724e4c0c48cae0f4ba9c05df64b`。若将来复制其代码，须保留相应版权和许可说明；本轮实现没有 vendoring 该包，也不依赖 TensorFlow。

上游 [`recommenders/datasets/mind.py`](https://github.com/recommenders-team/recommenders/blob/main/recommenders/datasets/mind.py) 当前指向 Recommenders 维护的 Hugging Face MIND 发布地址；核对 blob 为 `09c62215bbdbf41375b2e6e4dc0678bb02647362`。本项目只依据该官方项目的来源地址下载，另写流式解析、时间划分、mmap 存储与训练期文本变换；`provenance.json` 记录实际来源与下载摘要。上游下载脚本的 MIT 许可不意味着 MIND 数据可以再分发。

Wu et al., **Neural News Recommendation with Multi-Head Self-Attention**, EMNLP-IJCNLP 2019，在新闻词序列和用户历史两个层级使用多头 self-attention 与聚合。[论文](https://aclanthology.org/D19-1671/)；[上游 NRMS 实现](https://github.com/recommenders-team/recommenders/blob/main/recommenders/models/newsrec/models/nrms.py)，核对 blob 为 `52643eaa624e93824397e420645121c90d5a58c7`。

| 对照点 | 上游 NRMS | 本轮 NewsForge |
| --- | --- | --- |
| 新闻输入 | 词序列、词 embedding、新闻 self-attention | 标题+摘要的训练期 TF-IDF/SVD 与类别 one-hot |
| 用户表示 | 新闻编码后，历史 self-attention 与 attention 聚合 | 主两塔对固定原始特征均值后做 MLP；ranker 对候选相关历史做点积 attention |
| 框架 | 上游实现为 TensorFlow/Keras | 独立 PyTorch CPU 实现 |
| 可作出的说明 | 论文方法及其公开实现 | 借鉴新闻/用户编码任务，不是完整 NRMS，也不是论文指标复现 |

因此不能把 TF-IDF/SVD 称为 BERT、预训练语言模型或 token-level Transformer。`attention_mixed` 的可选接口也不等于多兴趣自注意力论文复现，而且未列入本轮正式实验。

## DIN 与候选相关历史聚合

Zhou et al., **Deep Interest Network for Click-Through Rate Prediction**, KDD 2018，强调针对不同候选聚合不同的历史兴趣。[作者论文](https://arxiv.org/abs/1706.06978)。

本项目的 `AttentionRanker` 在共同投影空间计算候选与历史相似度，用 masked attention 得到候选相关表示，再与候选做乘积、绝对差和相似度统计并交给 MLP。它保留这一建模动机，未实现 DIN 的完整网络、训练技巧或原数据实验，也没有复制其作者代码。常规点积 attention 不是本项目发明；相对均值双塔的效果变化还混入了模型容量、交互形式和计算预算变化。

本轮独立实现并检验的候选方案包括：在真实同曝光未点击池内，用当前模型选取难负例并与随机负例混合；再将可缓存的双塔召回接到候选相关兴趣重排。难负例、双塔和候选 attention 本身均是既有技术，本项目不声称首次提出这些方法。

## GBDT、PyTorch 与 Faiss

| 直接使用的开源项目 | 许可与核对来源 | 本项目用途 |
| --- | --- | --- |
| [scikit-learn](https://github.com/scikit-learn/scikit-learn) | [BSD 3-Clause](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING)；blob `3d7ee432c15b685eaa654b6abe8f8e3ea8126a8d` | TF-IDF、TruncatedSVD、HistGradientBoostingClassifier |
| [PyTorch](https://github.com/pytorch/pytorch) | [BSD-style 三条款许可](https://github.com/pytorch/pytorch/blob/main/LICENSE)；blob `c23172f7aff0254e4f0f163fb2e6e355cbaec5f4` | 自动微分、神经模型与 CPU 批量训练 |
| [Faiss](https://github.com/facebookresearch/faiss) | [MIT License](https://github.com/facebookresearch/faiss/blob/main/LICENSE)；blob `b96dcb0480a0b0be0727976e5202a1e7b23edc3f` | IndexFlatIP 精确对照与 IndexHNSWFlat 近似向量检索 |

GBDT 基线调用 [scikit-learn 1.6.1 的 HistGradientBoostingClassifier](https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)。本项目构造了两套内容特征：`similarity` 使用 15 维历史相似度与统计；`rich` 保留这 15 维，并拼接候选新闻前 16 维 SVD、有效历史均值前 16 维 SVD、候选的 19 维类别向量和历史均值的 19 维类别向量，共 **85 维**。19 维类别包含 18 个训练类别和 1 个未知类别列。两套输入均来自训练期拟合的固定文本/类别特征与请求自身的过去历史，不使用 ID、标签特征或 CTR 目标编码。

两套 GBDT 均使用相同的 500,000 个抽样正例对、每对 4 个同曝光负例和 2,500,000 条二分类样本；固定 seed 2026、80 轮、学习率 0.1、31 个叶节点、L2 正则 1.0、关闭 early stopping，使用 1 个 CPU 线程。模型超参数固定，**特征方案由内部验证选择**。在观察 15 维方案的内部结果后追加 rich 方案，属于开发期的自适应特征比较；原始模型、报告和验证结果均被保留，修改经过见 [实验协议](PROTOCOL.md) 与 [特征比较记录](../reports/gbdt_feature_ablation.json)。

相同 12,000 个内部验证曝光上，15 维方案的 nDCG@10 为 **0.33947626**，rich 为 **0.37283756**，据此选择 rich；这两个数用于开发期选择，不能当作最终官方 dev 的效果。选择过程中官方 dev 保持锁定。GBDT 的作用是提供有监督的非神经替代方案；它与神经模型的输入表示、优化目标、训练样本上限和选择预算不同，报告时需要一起披露。

Faiss 提供标准 [索引实现](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)，本项目没有发明 HNSW。独立工程工作是约束 PAD/无效向量、按请求时间和已看新闻过滤、模型/向量/索引版本校验、批量评分及服务验证。索引邻居近似率与 MIND 点击排序指标分别报告，不能把近似检索覆盖率宣传成推荐准确率。

其余直接运行依赖及版本列在 `requirements.txt`；依赖包和传递依赖仍各自保留其许可证。上表不声称是所有传递依赖的完整许可证清单。

## v2：延迟反馈与特征消融

v2 沿用已训练的 PyTorch 内容骨干与 scikit-learn HGB，在新闻时效和反馈成熟边界上扩展实验。独立实现包括严格延迟的小时快照、滚动衰减统计、固定候选预算的多路召回、共享训练样本的 15/7/22 维特征消融、成熟内部模型选择、用户聚类区间，以及 NumPy/FastAPI 服务的一致性验证。这些是围绕既有方法的工程与实验设计，不是新的深度网络或因果推断算法。

v2 的 HGB 使用 100,000 个正例及每例 4 个同曝光负例，共 500,000 行；三种特征各训练 seeds 17/29/43，统一 80 轮与 15 个叶节点。它与上文 v1 的 2,500,000 行、31 叶节点预算不同。双塔仅在已完成的 epoch-2 权重间重选，不重新训练。正式比较和负结果见 [v2 结果](../reports/V2_RESULTS.md)；本轮不将不同版本的数值差直接当作单一算法改进。

## 展示与归因（共同约束）

代码说明和简历按实际完成阶段描述流式预处理、严格时间边界、受控负采样实验、候选相关重排、GBDT 比较和服务测试。实现存在、单元测试通过、训练完成和最终评估完成分别对应不同证据；模型实验与服务实测须在完整运行报告生成后写为已完成。数据规模、增益、训练时间、内存、延迟或吞吐来自保存的运行报告，并给出分母、样本数、线程/并发和对照条件。

已物化数组可用 `python -m newsforge.audit_data --data data/processed` 独立审计；默认只向终端输出汇总。需要另存结果时追加 `--output reports/data_validation_recheck.json`，已有文件会被拒绝覆盖。该审计分块检查全部候选、历史 padding、训练点击/曝光统计、特征和时间边界，不重读原始 TSV、不重新拟合文本，也不评估推荐效果。原始来源和下载完整性另由 `provenance.json` 与固定 ZIP SHA-256 记录。

不使用“复现并超越 NRMS”“原创 attention”“生产级海量跨域画像”“腾讯数据验证”“分布式多卡加速”等未经本项目证据支持的描述。若最终模型没有优于强基线，也应保留结果并讨论特征能力、训练预算和研究假设，而不是改称指标提升。

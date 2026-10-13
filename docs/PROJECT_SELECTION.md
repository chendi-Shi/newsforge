# GitHub 项目选型与 JD 对应关系

NewsForge 选择**微软发起的 Recommenders / MIND 新闻推荐任务作为主要参考**，独立实现可在当前 CPU、16 GB 内存机器上完成的实验与服务流程。RecBole 用于比较研究框架的覆盖范围，TorchRec 用于明确分布式工程的后续方向。以下官方仓库、README 和许可证于 **2026-10-10 通过 GitHub connector 核验**；选型依据是任务与资源匹配度。

| 官方项目 | 许可证与可核验能力 | 对应 JD | 本次采用方式 |
| --- | --- | --- | --- |
| [Microsoft Recommenders](https://github.com/recommenders-team/recommenders) | [MIT](https://github.com/recommenders-team/recommenders/blob/main/LICENSE)；[README](https://github.com/recommenders-team/recommenders/blob/main/README.md)覆盖数据准备、经典/深度推荐、评估、超参数选择和部署，并列出 NRMS、NAML、LSTUR 等 MIND 示例。 | 用户兴趣建模、深度模型开发、研究实验与精准推荐落地，对应职责 1、2、4、5。 | 选其新闻内容与用户历史建模任务；沿用公开 MIND 数据语义，在独立 PyTorch 实现中研究曝光负采样、候选相关重排及可部署检索。 |
| [RecBole](https://github.com/RUCAIBox/RecBole) | [LICENSE 为 MIT](https://github.com/RUCAIBox/RecBole/blob/master/LICENSE)；[README](https://github.com/RUCAIBox/RecBole/blob/master/README.md)介绍统一 PyTorch 推荐框架、通用/序列/上下文/知识推荐、标准评估和超参数搜索。README 另写有仅限学术用途的表述，应与 LICENSE 一并保留理解，本文不消解两者差异。 | 学习算法原理、模型与优化方法比较、可复现研究，对应职责 2、4 及算法/框架能力要求。 | 适合后续接入更多统一基线。本轮专门保留 MIND 曝光组、时间边界与全候选评估，使用显式数据契约，未接入或复制 RecBole 源码。 |
| [TorchRec](https://github.com/meta-pytorch/torchrec) | [BSD 3-Clause](https://github.com/meta-pytorch/torchrec/blob/main/LICENSE)；[README](https://github.com/meta-pytorch/torchrec/blob/main/README.MD)提供稀疏特征组件、embedding 分片、自动分片规划、混合并行与训练流水线。原 `pytorch/torchrec` 地址现指向此官方仓库。 | 大规模推荐训练、并行算法与分布式实现加速，最直接对应职责 3，也涉及职责 2、5。 | 列为具备多设备环境后的扩展参考。本次未运行 TorchRec 分布式实验，不能据此声称实现了分布式加速。 |

## 当前资源下的选择

MIND 的新闻文本、历史点击和真实曝光集合让研究问题具体：同一曝光内如何选负例、候选相关兴趣表示是否有用、离线排序与召回服务是否一致。Recommenders 已提供相应任务和参考实现，因而适合作为阅读与实验起点。NewsForge 用训练期拟合的 TF-IDF/SVD、类别特征和紧凑神经网络控制算力；流式解析、memmap、分块采样和批量推理控制内存与重复计算。具体实验预算见 [实验协议](PROTOCOL.md)。

TorchRec 的官方文档**包含 CPU 路径**。本次暂不采用它，是因为单台无 GPU 的现有机器不能提供多 GPU / 多机 embedding 分片的收益证据；增加分布式启动配置也不能代替通信、扩展效率和资源占用实验。本项目先测清单机瓶颈，并将分布式训练明确保留为尚未完成的能力范围。新闻兴趣表示也只覆盖推荐场景，不能写成跨业务用户画像，或计算机视觉、语音任务成果。

## 实质工作与归因边界

v2 进一步围绕时间热度与个性化的增量展开研究：严格延迟反馈、成熟内部选择、三种特征与三个随机种子的受控消融，以及离线/HTTP 一致性。对应 JD 中数据分析、算法实验和应用落地；实际结果见 [v2 报告](../reports/V2_RESULTS.md)，不将已实现但效果负向的组件写成贡献。单机实验仍不构成 JD 中的分布式加速证据。

相对于本项目的随机负例双塔基线，研究改动是**在同一真实曝光的未点击集合内混合难负例**，保留同容量模型、共享采样池与训练预算；候选相关注意力重排作为另一种模型比较，单独披露容量差异。工程实现包括向量化负例缓存、时间与已读过滤、精确/HNSW 检索对照、checkpoint/索引摘要绑定，以及真实 HTTP 的批量评分与生命周期验证。改动是否带来推荐效果或性能改善，分别由保存的排序、召回与服务报告判定，不能从实现存在推断提升。

上述三仓库是**选型与研究参考，并非本仓库的 fork 或已提交上游的补丁**；本轮未复制其模型源码。NewsForge 的数据处理、PyTorch 双塔/重排模型、实验协议和服务连接由本项目独立编写；双塔、attention、难负例和 HNSW 本身是既有方法。直接调用 PyTorch、scikit-learn、Faiss 等组件的来源，以及 NRMS/DIN 与本实现的结构区别，见 [来源与相关工作](RELATED_WORK.md)。MIND 数据有独立研究许可，不能用上述软件许可证替代，也不随本项目源码包分发。

核验文件内容标识如下，均为 **Git blob SHA，非 commit SHA**：

| 项目 | README blob | LICENSE blob |
| --- | --- | --- |
| Recommenders | `81b4fb2a0a7f840348cddcb378ac815cc9353bf6` | `e74b0d177a471724e4c0c48cae0f4ba9c05df64b` |
| RecBole | `34edf2283aac49974fccd3a3bb7c012b83c80a45` | `fd00496ace51ac3899edecef1462fdffbf546c96` |
| TorchRec | `0af0bd9f8e254a5d8c7ab91ae4fc4fcc2cbe23b5` | `42d4de6be74435e454fd87936470c884a2f58a3d` |

# NewsForge

**Exposure-aware news recommendation: reproducible research and a CPU serving pipeline.**

基于 MIND-large 真实曝光日志，研究同一曝光内的难负例训练，构建双塔召回、候选相关重排和 FAISS 检索服务。项目面向机器学习研究工程岗位：重点是实验可信度、数据处理规模，以及离线结果与服务行为的一致性。

本项目独立实现 PyTorch 模型，参考 [Microsoft Recommenders](https://github.com/recommenders-team/recommenders) 的新闻推荐任务、[MIND](https://msnews.github.io/) 数据格式和评估定义。它不是腾讯内部业务项目，也不声称复现完整 NRMS 或达到其论文结果。文章输入使用训练集拟合的 TF-IDF/SVD 与类别特征。

本次已完成官方 train/dev 的全量预处理：**2,609,219 个曝光组、97,592,931 条候选曝光、750,434 名匿名用户、104,151 篇目录新闻**。这些是本项目实际使用的两个公开划分之和，不含官方隐藏 test；曝光组也不等同于独立用户或会话。全数组检查见 [数据审计报告](reports/data_validation.json)，精确切分见 [实验协议](docs/PROTOCOL.md)。

已在当前无 GPU 的电脑上完成全量数据审计、九组神经模型各两轮训练、GBDT 特征对照、完整 dev 排序评估、目录召回、真实 HTTP 基准与推荐演示，**70 项测试通过**。完整结果见 [实测报告](reports/RESULTS.md)，投递表述见 [简历版本](docs/RESUME_V1_CN.md)，源码与聚合证据均保存在本仓库。需要本地 ZIP 时，可按下文复现章节生成 `newsforge-source-and-results.zip`；ZIP 不随 Git 仓库上传。

| 本机实测 | 结果与口径 |
|---|---|
| 完整 dev 排序 | 376,471 个曝光组、14,085,557 条候选；随机负例双塔 nDCG@10 **0.38581 ± 0.00710**（三种子均值 ± 样本标准差），固定内容基线 **0.37163** |
| CPU HTTP 服务 | 单请求串行 p95 **30.57 ms**、**69.72 查询/秒**；四种批量/并发配置均无错误，共享桌面测量 |
| ANN 近邻准确率 | 125 个内部验证非空查询的近邻 Recall@100 **89.13%**；精确/ANN 检索 p95 **6.68/1.69 ms**，不含用户编码 |
| 全目录点击召回 | 固定 5,000 个 dev 查询，两阶段 Recall@100 **0.82%**（62/7,576 个日志点击）；该指标衡量点击覆盖，与 ANN 近邻准确率不同 |

研究结果包含明确的失败项：混合负例方案相对随机负例方案的 nDCG@10 下降约 **0.246 个百分点**，注意力重排的 nDCG@10 差值区间包含零。全目录点击覆盖仍低，当前结果不足以证明可上线的推荐效果；本项目定位是可复核的推荐研究与推理服务实现。没有根据 dev 结果更换模型、服务默认策略或检索参数，完整归因与统计范围见实测报告。

## 做了什么

- **数据**：流式处理完整 MIND-large train/dev，保留曝光组与全部候选；用内存映射数组存储，不将约一亿候选展开成内存 DataFrame。官方 train 按时间分出内部验证；官方 dev 仅用于最终测试。
- **研究**：相同容量双塔比较随机负例与混合难负例，使用相同种子、点击样本、采样池、优化器和训练轮次。候选相关注意力重排另作模型比较；增加内容、训练热度与 GBDT 基线。
- **评估**：全候选组宏平均 AUC、MRR、nDCG@5/10，三随机种子和按用户聚类的配对 bootstrap；分别报告空历史、短历史、训练未见文章等分组。
- **工程**：归一化向量的精确检索/HNSW 对照、时间可用性与已读过滤、自适应扩大候选、冷启动回退、模型与索引摘要绑定、FastAPI 单请求/批量接口。
- **测量流程**：日志点击召回率、ANN 对精确邻居的召回率、HTTP 延迟和吞吐量分别测量；不把向量近似准确率当作推荐效果。

```mermaid
flowchart LR
  A[MIND 曝光日志] --> B[时间切分与 memmap]
  B --> C[训练集拟合文本特征]
  B --> D[同曝光负例池]
  C --> E[双塔 / 候选注意力 / GBDT]
  D --> E
  E --> F[内部验证选择模型]
  F --> G[锁定官方 dev 完整评估]
  F --> H[文章向量与 HNSW]
  H --> I[时间与已读过滤]
  I --> J[重排与 FastAPI]
  J --> K[真实 HTTP 基准]
```

## 复现

下列安装命令使用本机 Windows / Python 3.10 的 36 项直接及传递依赖版本锁。`requirements.txt` 另外保留 12 项直接依赖；版本锁来自本次可运行环境，尚未在全新机器重装验证。

源码结果包保留了本次 `reports`。完整重跑请使用一个新的源码副本，先将副本中的 `reports` 改名为 `reports-delivered`，保留原证据；下列命令再创建新的 `reports`。不要将新训练与旧评估混入同一目录。最终入口会核对权重、数据与等价证据的摘要；GBDT 入口还会拒绝覆盖已存在的正式结果。

本次使用 Python 3.10.21；复现建议 Python 3.10、16 GB RAM，数据及中间文件预留至少 10 GB 空间。正式运行耗时与 CPU、存储及同机负载相关。85 维 GBDT 本次最大观测进程 RSS 为 3.402 GiB，曾使系统可用内存接近耗尽；应与神经训练串行执行，并另留系统余量，不能仅凭矩阵大小估计总内存。以下命令在项目目录执行：

```powershell
New-Item -ItemType Directory -Force reports | Out-Null
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-windows.lock
.\.venv\Scripts\python.exe -m unittest discover -s tests -v 2>&1 | Tee-Object -FilePath reports\unit_tests.txt

# 下载公开源并流式处理；仅提取本项目所需的 news/behaviors 文件。
.\.venv\Scripts\python.exe -m newsforge.data --download --threads 2

# 只读复查全部物化数组、训练统计和时间边界。
.\.venv\Scripts\python.exe -m newsforge.audit_data --data data/processed --output reports/data_validation.json

# 所有模型先在内部验证上选择，暂不打开最终测试。
.\.venv\Scripts\python.exe -m newsforge.train --variants mean_random mean_mixed ranker --seeds 17 29 43 --epochs 2 --batch-size 1024 --learning-rate 0.001 --threads 1 --validation-impressions 12000 --skip-test

# 两个 GBDT 特征配置串行训练。rich 命令先保留原 similarity 模型与结果。
.\.venv\Scripts\python.exe -m newsforge.gbdt --train-only --feature-mode similarity --max-positives 500000 --seed 2026 --threads 1 --validation-impressions 12000
.\.venv\Scripts\python.exe -m newsforge.gbdt --train-only --feature-mode rich --run-name gbdt_rich --max-positives 500000 --seed 2026 --threads 1 --validation-impressions 12000
.\.venv\Scripts\python.exe -m newsforge.gbdt --select-feature-ablation

# 在固定内部验证上核对 GBDT 分批调度；默认拒绝覆盖已有证据。
.\.venv\Scripts\python.exe -m newsforge.benchmark_gbdt_scheduler --threads 1 --output reports/gbdt_evaluation_scheduler_validation_12000.json

# 核验九组完整训练、既有服务权重与所有分批等价证据；仍不评估 dev。
.\.venv\Scripts\python.exe -m newsforge.final_evaluation --threads 1 --validate-only
.\.venv\Scripts\python.exe -m newsforge.gbdt_final_evaluation --threads 1 --validate-only

# 配置固定后，一次性评估官方 dev 的所有候选，输出恢复为原始曝光行顺序。
.\.venv\Scripts\python.exe -m newsforge.final_evaluation --threads 1
.\.venv\Scripts\python.exe -m newsforge.gbdt_final_evaluation --evaluate-test --threads 1

# 单独评估全目录点击召回，建立并保存服务索引。
.\.venv\Scripts\python.exe -m newsforge.evaluate_retrieval --threads 1

# 待训练/评估结束，测量 ANN 与真实 HTTP；工具启动并清理本地服务。
.\.venv\Scripts\python.exe -m newsforge.benchmark_service --launch-server --threads 1 --query-count 128 --ann-k 100 --top-k 10 --candidate-pool-size 100

# 用真实历史新闻完成一次推荐演示，保存不含原始文本或用户ID的核验摘要。
.\.venv\Scripts\python.exe -m newsforge.demo --launch-server --verification-report reports/demo_verification.json

# 全部正式评估和服务基准完成后，核验证据并生成结果、简历与源码结果包。
.\.venv\Scripts\python.exe -m newsforge.report
```

正式设置是 `mean_random / mean_mixed / ranker`，种子 `17 / 29 / 43`，每次训练两轮；混合采样模型先完成一轮随机负例预热。本机神经模型与 GBDT 各使用 1 个 CPU 线程，命令中显式指定以固定资源配置。GBDT 两种特征均使用同一批 500,000 正例、每例 4 个同曝光负例和固定 80 轮，再按相同 12,000 组内部验证 nDCG@10 选择。这个内容特征对照是在内部验证阶段、dev 仍锁定时增补，并非实验开始前预注册。完整设计见 [实验协议](docs/PROTOCOL.md)。训练保存每轮 loss、验证结果、样本数、参数量、耗时和数据摘要。最终入口只读取已完成的权重，核对数据/源码摘要与内部调度等价证据；候选长度分批不改变模型、候选或评估人口。

最后的聚合命令生成 `reports/RESULTS.md`、`docs/RESUME_CN.md` 和 `newsforge-source-and-results.zip`。缺少任何正式评估证据时它会拒绝生成最终报告；压缩包只包含源码、文档和聚合结果，不打包原始 MIND 数据或逐用户缓存。

仅检查环境可使用独立目录的 smoke run；它不能代替正式结果：

```powershell
.\.venv\Scripts\python.exe -m newsforge.train --variants mean_random --seeds 17 --epochs 1 --smoke-steps 100 --validation-impressions 1000 --threads 1 --artifacts artifacts/smoke --output reports/smoke
```

## 服务

完成模型和索引构建后，一条命令展示真实历史新闻与 Top 10 推荐，结束时自动清理服务：

```powershell
.\.venv\Scripts\python.exe -m newsforge.demo --launch-server
```

选样规则、目录参数和演示范围见 [演示说明](docs/DEMO.md)。需要持续运行 API 时使用：

```powershell
.\.venv\Scripts\python.exe -m newsforge.serve --port 8000 --torch-threads 1 --faiss-threads 1
```

通过本地 `http://127.0.0.1:8000/docs` 查看请求结构。服务默认使用训练截止时刻；查询后续目录时显式传入请求的 `as_of` 时间。未提供已知历史时按训练点击热度回退。候选不足会返回实际结果数量，不用未来或已读文章补齐。

服务用于本地研究演示。模型没有在线更新，文章目录也不是实时采集流；单进程 CPU 测量不能代表生产集群容量。

## 结果文件

以下文件由本次实际运行生成；汇总解释见 [RESULTS.md](reports/RESULTS.md)：

| 文件 | 内容 |
|---|---|
| `data/processed/manifest.json` | 全量规模、时间切分、数据质量、特征拟合范围 |
| `data/processed/provenance.json` | 下载来源、字节数、SHA-256 与使用条款 |
| `reports/data_validation.json` | 全量候选、历史 padding、训练统计、特征及时间边界审计；仅含汇总结果 |
| `reports/data_validation_recheck.json` | 使用仓库中的 `newsforge.audit_data` 命令重新完成的全数组检查 |
| `reports/materialization_profile.json` | 10,000 个训练曝光组的 ndarray/memmap 对照与 cProfile；单次共享机器诊断 |
| `reports/sampling_cache_benchmark.json` | 训练负例缓存构建的性能与一致性检查 |
| `reports/*_training.json` | 神经模型/GBDT 训练与内部验证记录 |
| `reports/gbdt_feature_ablation.json` | 15 维与 85 维 GBDT 的全部内部验证结果、共享样本核验和选择记录 |
| `reports/unit_tests.txt` | 最终 70 项测试通过的完整输出，含分批评估核验及合成 localhost 生命周期测试 |
| `reports/*scheduler*validation*.json` | 固定内部验证上的原顺序/候选长度分批逐组等价核验、计算槽与本次 CPU 耗时 |
| `reports/*final_evaluation_preflight.json` | 最终入口对权重、训练/特征选择、数据、源码和内部核验记录的检查 |
| `reports/ranking_experiment.json` | 所有正式模型全 dev 指标、三种子统计、配对置信区间 |
| `reports/gbdt_evaluation.json` | GBDT 全 dev 指标 |
| `reports/retrieval_experiment.json` | 固定抽样查询的可用目录点击召回 |
| `reports/service_benchmark.json` | ANN 准确率、延迟，以及真实 HTTP 吞吐和延迟 |
| `reports/demo_verification.json` | 真实历史驱动的 Top 10 演示、版本与过滤检查；不保存原始新闻文本或用户记录 |

## 解释结果的边界

MIND 是历史日志，未曝光新闻没有可靠的负反馈标签；离线提升不等于线上 CTR 提升。`first_seen` 仅是目录可用性的代理，不是新闻发布时间。点击召回评估同时保留全部点击分母和过滤已读后的可召回分母。训练负采样下的 logits/概率只用于排序，不是校准 CTR。

三种子标准差衡量训练随机性；按用户 bootstrap 衡量测试样本不确定性，两者不能互相替代。GBDT 与神经网络的训练预算不同，报告会说明各自样本量，不能把它们当作严格等预算算法比较。

原始 MIND 数据遵循其单独条款，不随代码分发。三个 GitHub 项目的比较及本机选型理由见 [项目选型](docs/PROJECT_SELECTION.md)，具体源码与论文依据见 [来源与相关工作](docs/RELATED_WORK.md)。

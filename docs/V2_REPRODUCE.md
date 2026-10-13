# NewsForge v2 复现与服务

在项目根目录运行，使用 README 中的 Python 3.10 环境。v2 复用 v1 的处理后数据、负例缓存及三个训练完两个完整 epoch 的双塔候选，并在成熟内部窗口重新选择骨干；需要先完成 [v1 流程](../README_V1.md)。文章向量和用户编码必须对应这次选出的参数。所有原始数据、特征、权重和逐请求结果均位于被 Git 忽略的 `data/`、`artifacts/`，仓库只分发代码、协议与聚合报告。

在独立源码副本中复现。先在该副本中将发布时的 `reports/` 完整重命名为 `reports-delivered/`，按 v1 的数据、训练和评估流程重新生成标准 `reports/`，然后运行下面的 v2 命令；保留原工作目录的交付证据不动。处理后数据与权重不随仓库分发，不要混用其他运行的模型、特征或时间快照。

完整流程目前以标准 `data/`、`artifacts/`、`reports/` 路径验证；`experiment.py` 的部分模型绑定仍使用固定 `artifacts/...` 路径，不能假定只调整某个 `--artifacts` 或 `--reports` 参数就支持任意目录的全流程迁移。时间快照目录、缓存等价审计、ranker 模型及正式 select/catalog/ranking 产物拒绝覆盖。**编码器导出、HTTP benchmark 和报告生成可能覆盖同名输出**；重跑应另建干净的源码副本并保留证据。`--resume` 仅用于同一数据、源码、样本、配置下的已完成模型，不应在看到 dev 后用来更换实验。内部选择冻结后，数据数组、模型、缓存审计和计算源码的 SHA256 都被绑定，不能修改后继续沿用旧 selection。

`README_V1.md` 保留历史全过程，但复现新版时**不要运行 `python -m newsforge.report`**：旧报告生成器会覆盖新版 `docs/RESUME_CN.md`。v1 聚合结果已有独立历史证据，v2 报告仅用下方 `python -m newsforge.v2.report` 生成；它不改简历文档。

```powershell
# 1. 合成边界、模型等价和接口测试。
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# 2. 97,592,931 条候选曝光 -> 严格延迟的 168 个整点快照。
.\.venv\Scripts\python.exe -m newsforge.v2.temporal --data data/processed --output data/temporal_v2

# 3. 同一成熟6k请求，在三个固定epoch2双塔中重新选择，不训练或替换产物。
.\.venv\Scripts\python.exe -m newsforge.v2.select_backbone --data data/processed --artifacts artifacts --output reports/v2/backbone_matured_selection.json --threads 1
$backboneSelection = Get-Content -LiteralPath reports/v2/backbone_matured_selection.json -Raw | ConvertFrom-Json
if ($backboneSelection.matches_current_retriever_state -ne $true) {
    throw '成熟选择的参数与现有retriever不同；需先物化匹配的checkpoint和article_embeddings，不能沿用旧产物继续。'
}

# 4. 已确认现有retriever参数等于成熟选择；导出用户编码并与PyTorch逐值核对。
.\.venv\Scripts\python.exe -m newsforge.v2.retrieval export-encoder --checkpoint artifacts/retriever.pt --output artifacts/v2/user_encoder.npz --processed-dir data/processed --article-embeddings artifacts/article_embeddings.npy --validation-count 512

# 5. 三种特征 × 三种 random_state；共享 100,000 正例及 400,000 负例。
#    该入口附带的旧晚间验证结果仅保留为开发记录，不是最终 selection。
.\.venv\Scripts\python.exe -m newsforge.v2.ranker --train-only --variants content_only context_only full --seeds 17 29 43 --max-positives 100000 --validation-impressions 6000 --threads 1

# 6. 权重不重训：只用09:00<=t<23:00的6,000个成熟内部请求重新评分九模型。
#    最终选择写 reports/v2/ranker_matured_selection.json。
.\.venv\Scripts\python.exe -m newsforge.v2.mature_selection

# 7. 证明消融调度缓存不改变候选：16 个内部查询 × 9 方案 × ABBA。
#    输出 reports/v2/query_cache_equivalence.json；不读取 dev 相关性结果来选方案。
.\.venv\Scripts\python.exe -m newsforge.v2.verify_cache --data data/processed --artifacts artifacts --temporal data/temporal_v2 --output reports/v2/query_cache_equivalence.json

# 8. 候选和重排也仅用09:00<=t<23:00的成熟内部请求，随后冻结输入摘要。
.\.venv\Scripts\python.exe -m newsforge.v2.experiment --phase select

# 9. 冻结后执行预定 dev 对照；仍是已知 benchmark 的后续研究。
.\.venv\Scripts\python.exe -m newsforge.v2.experiment --phase catalog
.\.venv\Scripts\python.exe -m newsforge.v2.experiment --phase ranking
.\.venv\Scripts\python.exe -m newsforge.v2.experiment --phase ranking-full

# 10. 真实 HTTP、离线输出一致性、批量和并发测量。
.\.venv\Scripts\python.exe -m newsforge.v2.benchmark

# 11. 独立核对逐请求数组、统计分母和源码/权重 SHA；不训练或重新预测。
#     输出 reports/v2/release_integrity.json；证据缺失或不一致时拒绝写入。
.\.venv\Scripts\python.exe -m newsforge.v2.audit

# 12. 所有聚合证据齐全且绑定一致后生成中文结果；缺文件直接报错。
.\.venv\Scripts\python.exe -m newsforge.v2.report

# 13. 手动启动服务，/docs 提供可交互的接口说明。
.\.venv\Scripts\python.exe -m newsforge.v2.serve --selection artifacts/v2/selection.json --temporal-dir data/temporal_v2 --port 8002
```

`GET /health` 返回模型版本、配置和允许的时间范围；`POST /recommend` 接收 `history`（MIND 新闻 ID 数组）、`as_of`（数据时间坐标的 Unix 秒）、`top_k`；`POST /recommend_batch` 接收 `requests` 数组。用于回放的快照只有公开数据覆盖的七天，超过范围的时间会被拒绝。它不会用旧数据静默回答今天的在线请求。

主分母是全部日志点击，已读或因延迟目录不可召回的点击仍计入；另报 eligible 分母。日志曝光内排序使用完整候选组，没有在验证时采样负例。三种种子的标准差与按用户聚类 bootstrap 区间衡量不同的不确定性，不能互相代替。详细约束见 [实验协议](V2_PROTOCOL.md)。

滚动快照只使用 `source_timestamp < floor(as_of/3600)*3600 - 3600` 的已成熟事件，仍可包含过去的内部验证及 dev 反馈。`frozen_feedback` 使用训练期静态总计，保持同样的严格目录过滤；因此该对照改变反馈范围和衰减方法，不能孤立解释为 EMA 的因果贡献，也不能与 v1 宣称同信息条件下的提升。v2 已知道 v1 的 dev 表现，配置冻结不代表获得了一个全新的、从未使用的测试集。

模型选择本身也必须遵守成熟边界：首个 dev 请求为 11 月 15 日 00:00，其 cutoff 是 11 月 14 日 23:00。最终窗口固定为 `1573722000 <= t < 1573772400`（11 月 14 日 09:00 含、23:00 不含），23:00 及之后的验证标签不能提前参与选择。`ranker/selection.json` 及对应训练报告包含的旧晚间验证结果作为开发记录保留；真正供 `experiment` 使用的是 `ranker_matured_selection.json`。九个权重的训练均固定 80 轮且关闭 early stopping，成熟验证只重新打分，不重训模型；两版选择的每个模型 SHA 必须一致。

双塔重选同样不看旧晚间验证指标。三个候选都必须等于日志记录的完整 epoch 2 参数，再用成熟 6k请求的完整曝光候选 nDCG@10 选 seed；它不会重新挑 epoch，也不会改权重。`selected.state_sha256` 与 `current_backbone.state_sha256` 证明参数一致，实际 `retriever.pt` 文件 SHA 另行绑定。相同参数经不同 torch 封装可以产生不同文件 SHA，不能仅因两个 checkpoint 文件 SHA 不同就判断模型改变。重选工具只写审计，**不自动物化新骨干**；上述默认导出路径仅在 `matches_current_retriever_state=true` 时适用。

`verify_cache` 的 576 批比较只验证跨消融方案复用当前查询语义路由时的逐字段精确等价。ABBA 四次耗时包含等价检查及调用计数，不是独立请求延迟；调用数下降不保证耗时下降。实际调用数和四次耗时以本次冻结的 [缓存审计 JSON](../reports/v2/query_cache_equivalence.json) 及 [自动结果报告](../reports/V2_RESULTS.md) 为准；早期审计保留开发记录，不能把旧计时套到新源码。不要将实验调度复用写成服务加速。

HTTP benchmark 使用 64 个内部请求，离线参考子进程退出后再启动 HTTP 服务，单计算线程、单服务 worker，测量 batch=1/8 与并发=1/4。p50/p95 是每个 HTTP 请求（整批）的延迟；吞吐按成功 query 计。Windows 启动器与实际 Python 服务可能是父子进程，benchmark 按 PID 和创建时间核验自有子树，RSS 只采样真实服务进程，测量范围为**服务健康后 API 校验与压测**，排除启动阶段、启动器和客户端。实验进程会清理自己的子树，不按进程名称终止其他工作。共享桌面数据不代表生产容量或分布式加速。

服务分数是采样训练下的相对排序分数，不是总体校准点击率。MIND 原始数据和派生特征仍遵守原数据许可；代码的 MIT 许可不改变数据许可。

可选消融图使用单独的绘图依赖，不改变历史运行依赖锁：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-plot.txt
.\.venv\Scripts\python.exe -m newsforge.v2.plot
```

图形从正式 `catalog.json` 与 `ranking.json` 读取数据，生成 PNG、SVG 和包含源文件/绘图源码摘要的 metadata。左图为用户聚类 95% CI，右图为三 seed 样本标准差，二者含义不同。缺失或不一致的结果会被拒绝，默认也拒绝覆盖已有图形。

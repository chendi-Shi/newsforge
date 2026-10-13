# 一条命令演示新闻推荐

在 NewsForge 项目目录、已安装依赖的 Python 环境中运行：

```powershell
python -m newsforge.demo --launch-server
```

使用本项目环境时可将 `python` 换为 `.\.venv\Scripts\python.exe`。入口启动一个绑定 `127.0.0.1` 随机空闲端口的真实 FastAPI 服务，使用 1 个 PyTorch 线程、1 个 FAISS 线程。服务加载完成后处理一个请求；正常结束、异常或中断时，现有启动器负责清理本次启动的 wrapper/服务进程，临时启动日志随后删除。

运行前需要完成数据预处理、正式模型选择和索引构建，存在以下本地文件：

- `data/processed/`：新闻 ID、历史、时间、划分、特征与目录可用时间数组。
- `artifacts/retriever.pt`、`artifacts/reranker.pt`：包含训练截止时间的模型 checkpoint。
- `artifacts/index_hnsw/`：由 `python -m newsforge.evaluate_retrieval --threads 1` 生成、与模型摘要绑定的索引。
- `data/raw/train/news.tsv`、`data/raw/dev/news.tsv`：用户从公开源下载的本地新闻目录，用于显示标题与类别。

演示固定选择数组存储顺序中**第一条内部验证划分、历史非空的请求**。选择只读取 `all_split.npy`、`all_history.npy`、`all_timestamps.npy` 和 `news_ids.json`，不读取点击标签或官方 dev 请求，也不根据推荐结果挑选样例。新闻目录中出现重复 ID 时与预处理保持一致，优先使用 train 文件记录。

终端依次显示历史新闻的标题/类别、双塔召回后重排的最多 10 篇新闻及两种分数，并显示模型/索引版本、候选数量和过滤检查结果。MIND 的原始时间字段没有注明时区，显示时间仅沿用预处理的 UTC 坐标。分数是排序信号，不能解释为校准后的点击率；单条演示不衡量离线推荐效果或服务性能。

已有服务时使用：

```powershell
python -m newsforge.demo
python -m newsforge.demo --base-url http://127.0.0.1:8000
```

这种模式连接已有的本地服务，使用它当前的线程设置，运行完不停止该服务。为保持新闻历史在本机，入口仅接受 localhost HTTP 地址。

目录均可覆盖：

```powershell
python -m newsforge.demo --launch-server --processed-dir data/processed --artifact-dir artifacts --index-dir artifacts/index_hnsw --raw-dir data/raw
```

默认只向终端输出标题和请求历史，不保存这些内容。需要执行证据时可加 `--verification-report reports/demo_verification.json`；该 JSON 仅保存计数、模型/索引摘要和检查布尔值，不含时间、新闻 ID、标题、历史或逐条分数。项目不会分发原始 MIND 新闻目录或请求记录。正式演示是否已经执行，以实际生成的核查记录为准。

2026-10-10 已在真实 MIND-large 数据和正式选定模型上执行上述启动命令。固定请求的历史长度为 20，实获候选 100，返回推荐 10，不足数为 0；中文终端输出、标题目录映射、训练截止时间、已读/未来条目过滤和模型/索引版本绑定均通过检查。启动器正常退出后已核对没有本项目服务进程残留，临时日志已删除。无原始新闻文本的执行摘要保存在 `reports/demo_verification.json`；这一条样例只证明链路能够运行，推荐效果与延迟分别查阅独立评估和服务基准。

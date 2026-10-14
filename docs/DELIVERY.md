# 交付与独立环境复验

这份流程把“已有结果能运行”“模型能重新训练”“原始数据能重建”分开验证。所有命令在 Windows PowerShell、CPython 3.10、仓库根目录执行。数据、权重、逐请求记录和原始日志保留在本地 `data/`、`artifacts/`，不随 Git 仓库发布。

| 路径 | 实际验证内容 | 不包含的结论 |
|---|---|---|
| 独立环境 + 冻结制品验收 | 从 GitHub 克隆、隔离安装依赖、校验源代码和制品、运行单元测试与真实 localhost HTTP | 不等于重新训练，也不是生产容量测试 |
| v3 重训练 | 从已校验的处理后数据和 v2 基线重新执行 fit/tune/confirm/dev，比较参数、逐查询计数、指标和决策 | 没有重做原始数据预处理，也没有重训继承的文本编码器 |
| 原始数据流水线 | 组织原始 MIND 数据到 v3 独立审计的完整依赖顺序，保存每阶段成功或失败凭据 | 有可执行入口不等于已经完成全链路实测 |

本轮验证使用同一台电脑上两个从 GitHub 取得的新克隆和一个新的短路径隔离环境：深层目录克隆完成 v3 重训练与新模型 HTTP，另一短路径克隆完成冻结发布制品的联合验收。两者固定同一提交，使用同一隔离解释器；所有检查并非在一个克隆中完成。Python 基础解释器及标准库仍来自这台电脑。旧研究环境配置为 `include-system-site-packages = true`，不能用它证明独立安装成功。验收器要求新环境显式为 `false`，并检查安装分发包、导入模块和搜索路径均没有继承外部第三方包。

**结果状态：v3 重训练、新模型 HTTP、短路径克隆的冻结发布制品联合验收均已通过。** 本机联合验收发现 251 项测试，249 项执行通过，2 项因符号链接权限跳过；真实 HTTP 的 768 个查询及 13 项契约通过。独立 GitHub runner 的 251 项测试全部通过、没有跳过。深层克隆第一次联合验收的 13 个长路径相关错误及失败凭据保留，未计为通过。结果见 [交付复验报告](../reports/DELIVERY_RESULTS.md)。另一次原始数据重建已完成预处理、九组神经训练与 [v1 全量评估](../reports/raw_rebuild/V1_REBUILD.md)，后续 v2/v3 与最终验收尚未完成。已观察过的 dev 仍是固定基准，不能作为新的外部泛化集，也不能据此宣称线上 CTR、生产部署或分布式性能。

## 1. 固定源代码和依赖

先从可信发布记录取得完整 40 位 Git 提交 SHA，创建新克隆。不要复制旧虚拟环境。**克隆与虚拟环境都使用用户目录下尚不存在的短路径**：深层工作目录不仅会影响 Torch 安装，也可能使制品测试的临时文件超过 Windows 路径限制。虚拟环境可以位于克隆之外；验收器检查真实解释器前缀，不要求环境必须叫 `.venv`。

```powershell
$clone = Join-Path $env:USERPROFILE 'nf-replay'
git clone https://github.com/chendi-Shi/newsforge.git $clone
Set-Location $clone
$expectedCommit = '<发布记录中的完整 Git SHA>'
git checkout --detach $expectedCommit
$venvRoot = Join-Path $env:USERPROFILE 'nf-delivery-venv'
py -3.10 -m venv --copies $venvRoot
$venvPython = Join-Path $venvRoot 'Scripts/python.exe'
$env:PYTHONPATH = $null
$env:PYTHONHOME = $null
& $venvPython -m pip --isolated install --require-hashes --only-binary=:all: -r requirements-delivery-windows.lock
& $venvPython -m pip check
```

`requirements-delivery-windows.lock` 绑定实际下载 wheel 的完整 SHA256，目标为 CPython 3.10 / Windows x86-64。其他 Python 版本或操作系统需要单独解析、校验和测试依赖，不能直接宣称适用。`pip` 自身作为虚拟环境安装工具单独记录；其他已安装分发包必须出现在锁文件中且版本一致。

维护者更新依赖时，应在新目录下载完整闭包后生成新锁和凭据；这不是普通复验必需步骤，不能为让失败安装通过而覆盖既有锁。

```powershell
python -m newsforge.delivery.wheels --wheelhouse artifacts/delivery/wheelhouse --lock requirements-delivery-windows.lock --receipt reports/delivery/wheels.json
```

离线安装可使用已下载的相同 wheel，并保留哈希约束：

```powershell
& $venvPython -m pip --isolated install --no-index --find-links '<wheelhouse 的绝对路径>' --require-hashes -r requirements-delivery-windows.lock
```

## 2. 导出、校验和导入私有制品

这一步需要持有已完成 v3 审计的研究工作目录。普通 Git 克隆没有 MIND 数据和权重。没有可信制品包时，应使用第 5 节原始数据流程，不能用合成数据替代正式复验。

在持有制品的研究目录导出独立字节副本。下列额外文件使包同时支持 HTTP 工作负载和 v3 重训练对照；它们必须已经绑定在冻结 fit/审计报告中。

```powershell
python -m newsforge.delivery.bundle export --repo . --bundle artifacts/delivery/replay_bundle `
  --extra-file data/processed/all_history.npy `
  --extra-file data/processed/all_users.npy `
  --extra-file data/processed/all_split.npy `
  --extra-file artifacts/v2/dev_catalog_counts.npz `
  --extra-file artifacts/v3/confirm/counts.npz `
  --extra-file artifacts/v3/dev/counts.npz
```

将导出返回的 `manifest_sha256` 通过可信渠道交给接收方。只读取包内自带的校验文件可以发现意外损坏，不能验证来源。接收方在新克隆内执行：

```powershell
$bundlePath = '<私有制品包的绝对路径>'
$bundleSha = '<可信渠道取得的 manifest SHA256>'
& $venvPython -m newsforge.delivery.bundle verify --bundle $bundlePath --expected-manifest-sha256 $bundleSha
& $venvPython -m newsforge.delivery.bundle import --repo . --bundle $bundlePath --expected-manifest-sha256 $bundleSha
```

导入先检查全部现有文件冲突和 payload 哈希，再复制；同路径相同字节可以跳过，不同字节会拒绝。路径穿越、符号链接、Windows junction/reparse 路径均拒绝。包不安装源代码或环境，也不在导入阶段反序列化模型。`bundle plan` 只列出期望输入，只有 export/verify/import 才逐文件读取并核对 payload 字节。

## 3. 在新环境重训 v3

```powershell
& $venvPython -m newsforge.delivery.retrain --output artifacts/delivery/retrain_v3
```

运行器在训练前写入比较协议，然后顺序调用冻结的四个阶段。新的模型、缓存、报告、每阶段返回码与日志哈希全部写入独立输出目录，不覆盖已发布结果。比较规则为：

- fit 行、采样、训练 pair 和特征 SHA256 必须一致。
- 九个模型的数组形状和类型必须一致；浮点参数使用预先声明的 `atol=1e-8, rtol=0`，同时分别报告逐元素完全相等和序列化字节相等。
- confirm/dev 的逐查询计数必须完全一致；调参网格、选择结果、部署门禁、指标、区间和切片结果必须完全一致。

成功以新目录的 `comparison.json` 为准。失败日志和输出应保留，修正原因后使用新的输出目录重新执行。此流程使用原有处理后数据、文本表示和 v2 基线；其结论范围不能扩展为“从原始数据重新得到整个系统”。

若需要验证新训练模型到 API 的路径，可在训练比较成功后顺序执行：

```powershell
& $venvPython -m newsforge.v3.benchmark_service `
  --selection artifacts/delivery/retrain_v3/models_and_caches/selection.json `
  --confirm artifacts/delivery/retrain_v3/reports/confirm.json `
  --fit-report artifacts/delivery/retrain_v3/reports/fit.json `
  --raw-dir artifacts/delivery/retrained_http `
  --report artifacts/delivery/retrained_http_report.json
```

该命令单独生成实测报告；重训练比较成功本身不代表已经完成这次 HTTP 检查。正式性能测量期间不要并行运行训练或其他基准任务。

## 4. 运行冻结发布制品的交付验收

```powershell
& $venvPython -m newsforge.delivery.acceptance --expected-commit $expectedCommit `
  --report reports/delivery/replay_acceptance.json
```

验收器依次检查 Git HEAD 和工作树、隔离环境、每个锁定分发包版本和模块实际路径、`pip check`、完整单元测试，然后启动真实 v3 服务。HTTP 检查使用固定 64 个 dev 请求、4 组 batch/concurrency 配置、每组 3 轮，共 768 个正式查询，以及 13 个 API 边界/模式契约；它要求所有响应字段与独立离线参考完全相等，并重新检查原始证据哈希。进程回收限定于本次启动的 PID 与创建时间身份。

上述命令的新公共输出为 `reports/delivery/replay_acceptance.json` 和 `replay_acceptance_http.json`；仓库中的历史 `acceptance.json` 保留。请求、响应、服务器日志和失败堆栈在 `artifacts/delivery/`。凭据记录完整源文件哈希，但环境路径归一化为 `<venv>`、`<clone>`、`<base>`。单元测试分别记录发现总数、跳过数和实际执行数，不将跳过项算作通过。

输出必须原先不存在，工作树不得有源代码修改或其他未跟踪文件；重训练输出应放在被忽略的 `artifacts/` 下。默认使用 PATH 中的 Git；必要时通过 `--git '<Git 可执行文件绝对路径>'` 指定。`--skip-tests` 仅用于诊断，生成 `status=partial`、`passed=false`，并返回退出码 2，不能作为完整验收通过凭据。

GitHub Actions 只执行 Windows CPU 的哈希依赖安装、`pip check`、全部小型测试和无数据 `pipeline plan`。CI 不下载 MIND、不执行完整训练，也不生成正式研究指标。CI 绿色不能替代此节的本地数据与服务验收。

## 5. 从原始数据重建的独立流水线

先查看真正要执行的依赖图，`plan` 不安装依赖、不下载数据、不训练：

```powershell
python -m newsforge.delivery.pipeline plan --output artifacts/delivery/raw_plan.json
```

完整流水线使用单独的 `requirements-raw-windows.lock`，将运行与绘图的 **45 个分发包**一起锁定版本和 wheel SHA256。它面向 Windows x86-64 / CPython 3.10；另建短路径环境和工作目录，避免 Windows 长路径问题。第 4 节的 36 包服务验收环境保持独立。

```powershell
$rawReplayVenv = Join-Path $env:USERPROFILE 'nf-raw-replay-env'
$rawWorkspace = Join-Path $env:USERPROFILE 'nf-raw-replay'
if ((Test-Path $rawReplayVenv) -or (Test-Path $rawWorkspace)) { throw '使用尚不存在的新环境和工作目录' }
py -3.10 -m venv --copies $rawReplayVenv
$rawReplayPython = Join-Path $rawReplayVenv 'Scripts/python.exe'
& $rawReplayPython -m pip --isolated download --only-binary=:all: --no-cache-dir --require-hashes `
  -r requirements-raw-windows.lock --dest artifacts/raw_rebuild/wheelhouse
& $rawReplayPython -m newsforge.delivery.raw_install --root . `
  --wheelhouse artifacts/raw_rebuild/wheelhouse --lock requirements-raw-windows.lock `
  --wheel-receipt reports/raw_rebuild/wheels.json --output reports/raw_rebuild/replay_install.json `
  --log artifacts/raw_rebuild/replay_install.log
if ($LASTEXITCODE -ne 0) { throw '安装验收失败，保留日志后处理原因' }
& $rawReplayPython -m newsforge.delivery.pipeline init --source . --workspace $rawWorkspace --raw '<原始 MIND 数据根目录>'
if ($LASTEXITCODE -ne 0) { throw '初始化失败，保留现场' }
& $rawReplayPython -m newsforge.delivery.pipeline run --workspace $rawWorkspace --python $rawReplayPython --through v3-audit
if ($LASTEXITCODE -ne 0) { throw '原始数据重建失败，保留现场' }
& $rawReplayPython -m newsforge.delivery.raw_acceptance --workspace $rawWorkspace `
  --environment-receipt reports/raw_rebuild/replay_install.json `
  --install-log artifacts/raw_rebuild/replay_install.log --output reports/raw_rebuild/replay_acceptance.json
if ($LASTEXITCODE -ne 0) { throw '完整重建验收失败，保留证据' }
& $rawReplayPython -m newsforge.delivery.raw_report --workspace $rawWorkspace `
  --acceptance reports/raw_rebuild/replay_acceptance.json --output-dir reports/raw_rebuild/replay
if ($LASTEXITCODE -ne 0) { throw '汇总失败，保留证据' }
& $rawReplayPython scripts/summarize_checkpoint_selection.py --workspace $rawWorkspace `
  --output reports/raw_rebuild/replay/checkpoint_selection.json
if ($LASTEXITCODE -ne 0) { throw '选轮元数据核对失败，保留证据' }
```

`--workspace` 必须是源仓库之外尚不存在的目录。完整验收路径必须显式提供 `--raw`，目录结构为 `train/{news,behaviors}.tsv` 和 `dev/{news,behaviors}.tsv`，初始化时绑定四个原始文件。省略该参数只会在数据阶段走原有下载入口；当前最终 `raw_acceptance` 不接受这种模式。`init` 创建独立源代码副本，并将副本中已有报告保留为 `reports-delivered/`，新报告从空目录开始。流水线包括预处理与审计、九次 v1 神经模型训练、v1 完整评估、内部时间窗口的 backbone 选择与导出、v2 训练选择评估审计，以及 v3 训练、消融、真实 HTTP、绘图和独立审计。

安装器先校验实际 wheel、哈希锁与分发包清单，再离线强制安装全部锁定包，记录原始 pip 日志、隔离环境检查和 `pip check`。既有回执、日志及失败尝试均不能覆盖。完整测试原始日志保存为 `reports/delivery/all_unit_tests.txt`；单独执行全部研究测试且要求无跳过，其原始日志保存为 `reports/v3_release_unit_tests.txt`，供冻结研究审计读取。神经训练结束后先检查三种子检查点的 epoch-2 资格，再进入全量评估。

最终 `raw_acceptance` 必须由实际训练的同一个解释器运行。它逐级核验全部阶段、原始四文件与处理 provenance、安装日志及实际依赖代码、v2/v3 审计的模型和数组绑定、真实 HTTP 记录；还会描述新旧结果差异。确认门未通过时正确回退 baseline 可以是有效结果，历史指标完全相同不是通过条件。只有完整执行及最终验收成功才能声称原始数据重建已完成。

随后 `raw_report` 从已经通过且仍匹配的验收与阶段记录生成新的 `execution_summary.json` 和 `RESULTS.md`，列出本轮 v1/v2/v3 指标、区间、消融、测试通过与跳过数、HTTP 观测及新旧差异。它重新检查源代码、阶段链及汇总报告的摘要，依赖此前完整验收对大数组和原始 TSV 的核验，不冒充再次执行所有预测。输出不能覆盖既有结果。

[选轮补充脚本](../scripts/summarize_checkpoint_selection.py)另行汇总九组 v1 的实际训练轮数和最终保留轮次，核对已有训练、资格检查、最终评估元数据和三段阶段凭据。它不读取 checkpoint 或重新预测，文件摘要属于此次补充绑定。2026-10-14 的[实际选轮与负结果](../reports/raw_rebuild/V1_REBUILD.md)表明：九组都训练两轮，但五组选中第一轮；难负例配置的六组中只有一组最终保留第二轮更新。报告应描述训练策略加验证选模的效果，不能只根据配置文件宣称全部最终权重采用了难负例。

阶段凭据绑定冻结源代码、明确提供的原始文件、依赖阶段的凭据与声明输出，以及本阶段声明输出的哈希；各研究审计器继续验证自身完整制品绑定。阶段凭据不保证覆盖所有临时文件。失败记录及日志保留，不把已有发布指标复制成新实验结果。若新的内部选择不满足冻结 v3 的基线前提，流水线会失败，不能强制使用历史赢家或从 dev 反选。基于旧阶段耗时，普通 CPU 上全链路预计需要数小时；该估计不是本轮完整重跑的实测耗时。

### 预处理数值对照

持有可信的旧、新处理制品后，可以用 [compare_preprocessing.py](../scripts/compare_preprocessing.py) 单独重算差异；以下旧目录指原冻结研究目录中的 `data/processed`，输出必须尚不存在：

```powershell
& $rawReplayPython scripts/compare_preprocessing.py `
  --old-processed data/processed --new-processed (Join-Path $rawWorkspace 'data/processed') `
  --output reports/raw_rebuild/replay_preprocessing_comparison.json
if ($LASTEXITCODE -ne 0) { throw '预处理对照失败，保留现场' }
```

脚本读取保存的特征及 TF-IDF/SVD 状态，不训练、不执行 `fit` 或 `transform`；报告绑定脚本与输入文件 SHA256，并在比较结束后再次核验输入未变。已完成的[公开脚本数值重算](../reports/raw_rebuild/preprocessing_numeric_recomputed.json)与[包含来源、环境及下游观察的比较记录](../reports/raw_rebuild/preprocessing_comparison.json)应分别阅读：原始 train/dev 新闻 TSV、文章索引和类别一致，TF-IDF 词表及 IDF 一致，但 SVD 参数数组与文本向量不逐位一致。文本向量几何上接近不证明下游统计等价；当前环境检查也不是历史执行快照，差异根因尚未确定。此项对照已经完成，完整原始数据重建与最终验收仍未完成。

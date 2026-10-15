# 目标域适配：独立复现入口

本轮在已知旧模型负迁移后提出，属于已有 Adressa 基准上的后续研究。固定设计见 [协议](ADRESSA_ADAPTATION_PROTOCOL.md)。旧冻结迁移结果、MIND 服务和模型均保留；本轮只增加独立适配模型、证据与回放 API。

2026-10-15 正式执行已完成，使用冻结源代码 `a74fecbf9ddcc1918a28abd58d23a159f6c0fd4a`。[完整结果](../reports/adaptation/adressa_v1/RESULTS.md)、[研究解读](../reports/adaptation/adressa_v1/DISCUSSION.md)和[真实 HTTP 验收](../reports/adaptation/adressa_v1/http.json)均已发布。选择结果为 δ=0，实际回放策略是 baseline；执行成功不表示新模型取得收益。

使用仓库根目录、Windows PowerShell、已有的 CPython 3.10 隔离研究环境。依赖版本与安装方式见 [45 包研究锁及交付指南](DELIVERY.md)。Git 克隆不包含数据和模型：需先完成 [Adressa 原始准备与冻结迁移](EXTERNAL_ADRESSA_RUNBOOK.md)，取得通过校验的 prepare/evaluate 目录及原 MIND 制品，不能以合成数据代替正式实验。

## 固定源代码与输入

先固定可信发布提交，并确保工作树干净。本程序初始化时拒绝未提交的文件，绑定协议、新旧实现、依赖锁、旧输入及模型摘要。旧 SQLite 数据库只读打开；不会重新解析约 13 GB 的展开归档，也不会执行 VACUUM 或修改旧制品。

`$baseRun` 指已经通过旧迁移验收的运行目录，必须包含 `prepare/receipt.json`、`evaluate/receipt.json`、事件数据库、冻结英文特征、原样本及独立小时计数等完整文件。该目录也可以来自先前复现，但必须与当前持有的冻结 MIND 输入一致。

```powershell
$adaptPython = '<已按研究哈希锁安装的独立环境 Python 绝对路径>'
$baseRun = '<已通过旧 Adressa 实验的运行目录>'
$adaptRun = 'artifacts/adaptation/adressa_replay'
$adaptReports = 'reports/adaptation/adressa_replay'
if ((Test-Path $adaptRun) -or (Test-Path $adaptReports)) { throw '使用尚不存在的新输出目录' }
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
& $adaptPython -m newsforge.adaptation prepare --repo . --base $baseRun --run $adaptRun
if ($LASTEXITCODE -ne 0) { throw '准备失败，保留原目录和日志' }
& $adaptPython -m newsforge.adaptation fit --repo . --base $baseRun --run $adaptRun
if ($LASTEXITCODE -ne 0) { throw '训练失败，不能改参数覆盖重试' }
& $adaptPython -m newsforge.adaptation tune --repo . --base $baseRun --run $adaptRun
if ($LASTEXITCODE -ne 0) { throw '选择失败，保留证据' }
& $adaptPython -m newsforge.adaptation confirm --repo . --base $baseRun --run $adaptRun
if ($LASTEXITCODE -ne 0) { throw '确认执行失败，保留证据' }
& $adaptPython -m newsforge.adaptation evaluate --repo . --base $baseRun --run $adaptRun --reports $adaptReports
if ($LASTEXITCODE -ne 0) { throw '评估或审计失败，不能宣称完成' }
```

prepare 只拟合早期文章表示并缓存按各自时间合法的特征；fit 只读取 fit 请求和共享配对；tune 只选本地主方案的 delta；confirm 单独产生通过或回退决策。阶段目录与 JSON 均拒绝覆盖，依赖哈希在阶段边界核验。确认门未通过是有效研究结果，不等于执行错误；之后仍报告主方案和全部固定消融，实际策略返回 baseline。

模型、词表、逐请求特征及预测位于忽略的 `artifacts/`。正式摘要 `evaluation.json`、`audit.json` 和 `RESULTS.md` 仅包含聚合结果及摘要绑定；旧实验目录与报告不能覆盖。新报告包括所有请求分母、逐日结果、配对获益/受损数、三 seed 训练样本和选参记录、文本覆盖及运行资源。原冻结模型逐请求命中和倒数排名必须与旧输出完全相同。

## 服务与复核

适配服务以确认门实际决策为准，接受原始时间坐标和本地匿名整数用户索引，用完整过去历史过滤、最近 20 条历史提取特征。只支持此回放数据时间范围，不能作为实时线上用户服务。

```powershell
& $adaptPython -m newsforge.adaptation.serve --repo . --base $baseRun --run $adaptRun --port 8004
```

服务在 localhost 提供单条与批量请求；`/health` 返回模型版本、实际策略及时间边界。正式 HTTP 基准另启动受控进程，核对响应与离线引擎及已保存评估顺序，记录完整请求时延和查询吞吐，不能把离线评分时间当 HTTP 性能。

结束手动服务后，另开尚不存在的私有基准目录执行；该命令自行启动并回收属于本次运行的服务进程：

```powershell
& $adaptPython -m newsforge.adaptation.benchmark --repo . --base $baseRun --run $adaptRun `
  --private-dir (Join-Path $adaptRun 'http') --report (Join-Path $adaptReports 'http.json')
if ($LASTEXITCODE -ne 0) { throw 'HTTP 验收失败，保留日志和报告' }
& $adaptPython scripts/plot_adaptation_results.py --report (Join-Path $adaptReports 'evaluation.json') `
  --output-dir $adaptReports
if ($LASTEXITCODE -ne 0) { throw '绘图失败，保留研究结果' }
```

结果解读遵守协议：目标域配对的替代项并非已知曝光负例；当地字符表示不等于跨语言语义模型；固定 delta 下的组件差异不等于每个组件单独调参后的最优差异。Jan 5–7 已被观察，即使新指标提高，也不能改写原冻结迁移失败或宣称新未见测试通过。

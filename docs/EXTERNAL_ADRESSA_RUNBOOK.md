# Adressa 冻结外部实验执行说明

研究定义以已经预注册的 [协议](EXTERNAL_ADRESSA_PROTOCOL.md) 为准。此实现只迁移冻结的英文 MIND TF-IDF/SVD、类别映射和九个 v3 残差模型。Adressa 是挪威语历史访问数据，没有曝光负例；结果不能解释为 CTR、线上 A/B 或未来时间泛化。零或负收益同样完整发布。

## 冻结顺序

1. 提交协议和来源报告。
2. 可以下载官方压缩文件原字节，但不得打开 gzip/tar 或读取事件。
3. 完成并提交全部实现与合成测试；保存这次提交 ID，再运行 `prepare`。
4. `prepare` 允许获取压缩原字节以后完成实现，但强制其余原模型、协议、来源、依赖锁和环境绑定不变。进入 `prepare` 后，所有阶段要求实现绑定也完全相等。

合成测试只生成虚构事件与虚构模型参数。四阶段测试实际经过 acquisition、prepare、evaluate、aggregate 的 CLI，固定 150 个请求并完成 128 个独立参考审计；这证明实现路径贯通，不是外部效果结论。

本轮协议提交为 `d0c911af44366162a06333d56bf0bb0b3e2a52c9`，八个外部实现模块在 `8daf7109380dba6c449e1f29b7b75ac75710a994` 冻结，读取前记录见 [adressa_pre_access.json](../reports/external/adressa_pre_access.json)。完整下载于 2026-10-14 04:25:22 UTC 验证通过，随后启动实际数据准备；文件字节、整体 SHA256、338 个分段及传输绑定见 [获取凭据](../reports/external/adressa_acquisition.json)。原协议及来源报告里的“尚未下载/读取”描述注册时的状态，作为快照保留，不改写为后来的结果。

首次真实准备于 04:58:09 UTC 因第 10,000,001 个原始行超过上限失败，尚未抽样或计算效果。已保留[失败摘要](../reports/external/adressa_prepare_limit_failure.json)，并单独登记[资源上限修订](EXTERNAL_ADRESSA_RESOURCE_AMENDMENT.md)：上限提高到 20,000,000 行，其余科学方案和资源限制不变。修订发生在部分真实事件读取后，不能称为完全事前的资源注册。修订文本 SHA 加入后续阶段绑定；另建运行目录从同一完整归档重新处理，保留旧失败数据库。

修订实现于 `5e9024a925f7fdd257f0fd6de2782631243e5294` 提交后，在独立的 `artifacts/external/adressa_v1_resource20m` 目录启动完整重跑。[重跑前记录](../reports/external/adressa_resource_retry.json)绑定实现、修订、34 项合成测试及原下载凭据的精确副本，并明确此前已访问部分真实事件但没有效果结果。

```powershell
python -m unittest discover -s tests -p 'test_external*.py' -v
```

## 执行

以下命令从仓库根目录运行，使用安装了仓库依赖锁的同一个 Python 环境。每个阶段的目录均采用独占创建，已有目录不会被覆盖。正式事件读取前应确保实现已经提交。

前提是持有与 `reports/external/adressa_source.json` 全部摘要相符的原冻结 MIND 研究制品，包括 TF-IDF、SVD、类别映射及九个残差模型。普通 Git 克隆不含这些二进制文件；服务交付包也不包含本实验所需的全部文本变换器。重新执行原始数据流水线不会自动成为这次外部实验的相同输入，不能为适配新制品而改写已注册的来源摘要。

持有这些冻结输入的维护者可导出单独的私有前置制品包：

```powershell
python -m newsforge.delivery.external_bundle export --repo . --bundle artifacts/external/frozen_bundle
```

接收者先取得包含本实验源码、协议及来源报告的可信 Git 提交，再从可信渠道取得上述导出的 manifest SHA256，执行：

```powershell
$externalBundle = '<私有外部前置制品包的绝对路径>'
$externalBundleSha = '<可信渠道取得的 manifest SHA256>'
python -m newsforge.delivery.external_bundle verify --repo . --bundle $externalBundle --expected-manifest-sha256 $externalBundleSha
if ($LASTEXITCODE -ne 0) { throw '外部前置制品校验失败' }
python -m newsforge.delivery.external_bundle import --repo . --bundle $externalBundle --expected-manifest-sha256 $externalBundleSha
if ($LASTEXITCODE -ne 0) { throw '外部前置制品导入失败' }
```

这个包只允许来源报告中的 **27 个明示制品**；另 **9 个源码、协议及来源文件**必须已存在并逐字节匹配，不从包中安装。两个 joblib 文件仅作为不透明字节复制，导入不反序列化。冲突检查在写入之前完成，已有不同内容会拒绝，相同内容可以跳过；原始 Adressa 日志不在包内。该工具与服务制品包各有明确范围。实际独立目录的导出、校验及两次导入结果见 [移交凭据](../reports/external/frozen_bundle_delivery.json)。该次接收验证使用源码骨架，不能描述成一次完整的新 Git 克隆外部评测。

历史下载及首次失败准备目录为 `artifacts/external/adressa_v1_parallel`；资源修订后的成功运行目录为 `artifacts/external/adressa_v1_resource20m`，公开历史结果位于 `reports/external/adressa_v1`。以下命令使用新的复跑目录，避免覆盖随 Git 分发的报告、图片和验收凭据。

```powershell
$adressaRun = 'artifacts/external/adressa_replay'
$adressaReports = 'reports/external/adressa_replay'
$adressaArchive = 'data/external/adressa/raw_replay/one_week.tar.gz'
foreach ($adressaPath in @($adressaRun, $adressaReports, $adressaArchive)) {
    if (Test-Path -LiteralPath $adressaPath) {
        throw "路径已存在，请保留已有证据并选择新的复跑路径：$adressaPath"
    }
}
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python -m newsforge.external acquisition-parallel --repo . --run-dir $adressaRun --archive $adressaArchive
if ($LASTEXITCODE -ne 0) { throw '获取阶段失败，保留本次目录' }
python -m newsforge.external prepare --repo . --run-dir $adressaRun --archive $adressaArchive
if ($LASTEXITCODE -ne 0) { throw '准备阶段失败，保留本次目录' }
python -m newsforge.external evaluate --repo . --run-dir $adressaRun
if ($LASTEXITCODE -ne 0) { throw '评测阶段失败，保留本次目录' }
python -m newsforge.external aggregate --repo . --run-dir $adressaRun --reports $adressaReports
if ($LASTEXITCODE -ne 0) { throw '聚合阶段失败，保留本次目录' }
```

默认原始文件为 `data/external/adressa/raw/one_week.tar.gz`，上例显式使用新归档路径。已完成 acquisition 时可以从 `prepare` 继续，但必须保留同一运行目录内的下载凭据并传入对应 `--archive`；仅有压缩文件不足以继续。并行传输仍是同一官方 URL 的相同压缩原字节，固定默认 8 个线程、4 MiB 范围。每段必须返回精确的 HTTP 206、绝对范围、总长度和稳定 ETag/Last-Modified；每段和最终串接文件均写入后重新计算 SHA-256。断开的尝试保留 `.partial`；不解析目标事件。串接后的整体 SHA-256 是本次实际取得文件的摘要，并非冒充官方发布的校验值。

每阶段在自己的目录写入 `progress.jsonl`、`receipt.json`，失败写 `failure.json`。私有进程日志保留具体异常；公开 receipt 不包含本机绝对路径或事件内容。任何失败须保留完整旧目录；不得删掉失败证据后用同一个目录重跑。后续修复须记录原因和新实现提交，另建运行目录。若已经读取目标事件，须明确披露，并保持原协议的样本、模型、超参数和指标不变。

## 内存与审计

- 流式读取 tar 成员，禁止 `extractall`，逐行 JSON 校验；SQLite 存储去重、排序、身份和所有事件，临时排序落盘。
- 事件写批上限 5,000 行，并在原始字节达到 8 MiB 时提前提交；文章 transform 批最多 4,096 篇，并在字符串字节达到 8 MiB 时提前执行。单行原有 16 MiB 解析保护仍生效。这些界限只改变内存批次，不改变样本。
- 特征、候选、分数和命中数组使用 memmap；完整历史用于排除已读，最近 20 次事件用于内容特征。
- 独立参考路径另行从原始规范化事件构造磁盘小时计数，按截止点从零重建 EMA。只缓存上一个截止点，不读取主路径的 EMA、候选、历史或分数。128 个请求位置固定。历史过滤与最近 20 条使用独立 SQL 查询，分数直接从冻结参数重新计算。
- `prepare` 和 `evaluate` 分别记录耗时与每 100 ms 采样的进程 RSS；RSS 包含映射页面，采样可能漏峰，不能把它表述成严格的 OS 峰值或独占机器基准。
- `aggregate` 从保存的候选、排名和目标重新计算每个请求的 Hit@10、Hit@100、MRR@10，再做固定用户簇 bootstrap。所有方法候选集合及 Hit@100 必须完全一致。

## 产物解释

2026-10-14 的正式实验已经完成：15,000 个请求上，完整模型相对热度基线的 HitRate@10 差为 **−0.3733 个百分点**，95% 用户簇区间 **[−0.5814, −0.1598] 个百分点**，不支持正向迁移。128 请求慢速审计及 15,000 请求保存数组补充验收均通过。结果、全部消融、资源修订和类别覆盖退化的解释见 [结果讨论](../reports/external/adressa_v1/DISCUSSION.md)；机器生成的 [RESULTS.md](../reports/external/adressa_v1/RESULTS.md) 和 [evaluation.json](../reports/external/adressa_v1/evaluation.json) 原样保留。

公开产物为 `evaluation.json`、`audit.json`、`RESULTS.md`。它们报告总体、三个固定日期、全部消融、区间、语言词表覆盖、未知类别、不可见/已读目标、候选不足及阶段资源消耗。`status=passed` 代表实现审计和聚合校验通过；只有主差大于零且 95% 用户簇区间下界大于零，`positive_transfer_supported_on_preregistered_sample` 才为真。用户簇区间条件于这三个已经实现的日期，不提供跨日期或生产保证。

聚合成功后，在安装了 `requirements-raw-windows.lock` 的环境接着使用上述变量，从本次聚合数值生成 PNG、SVG 和摘要元数据。绘图不接触原始事件、不重新计算统计量；负差值和跨零区间照常呈现，已有图片不会覆盖。

```powershell
python scripts/plot_external_results.py `
  --report "$adressaReports/evaluation.json" --output-dir $adressaReports
if ($LASTEXITCODE -ne 0) { throw '绘图失败，保留本次目录' }
```

持有完整本地运行产物后，再执行只读补充验收：

```powershell
python scripts/verify_external_results.py `
  --run-dir $adressaRun --reports $adressaReports `
  --output "$adressaReports/supplemental_verification.json"
if ($LASTEXITCODE -ne 0) { throw '外部结果补充验收失败，保留证据' }
```

该脚本核对文件与阶段摘要、全部请求的有效评分、候选排列、命中与倒数排名，以及每日分母、覆盖计数和消融方向。它验证保存的 128 请求审计及区间配置，但不重新解压原始事件、不重新推理或重抽 bootstrap；验收通过也不等于得到正向迁移收益。输出仅含聚合检查及摘要，已有结果不会覆盖。

原始访问日志、用户/文章映射、逐请求数组和模型二进制保持本地，不能随公开代码仓库发布。Adressa 数据沿用其 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 许可，代码许可不替代数据许可。

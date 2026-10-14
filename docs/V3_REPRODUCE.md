# v3 复现：受约束个性化与等价检索加速

先完成 [v2 数据、快照、表示和冻结热度基线](V2_REPRODUCE.md)。v3 不覆盖 v2 证据，所有命令在仓库根目录执行，使用相同 Python 3.10 CPU 环境和已锁定依赖。源码仓库不携带 MIND 原始数据、训练权重或逐用户数组，需要按数据条款自行准备。

正式运行前阅读 [执行前协议](V3_PROTOCOL.md)。本轮官方 dev 已经被此前版本观察，不能重新称为独立未见测试集。旧热度窗口和文本表示是继承常量。

## 固定顺序

先证明优化检索与旧实现逐字段等价；再执行拟合、调参、确认和 dev，不能跳过阶段或据 dev 改参数。

```powershell
.\.venv\Scripts\python.exe -m newsforge.v3.benchmark_fast
.\.venv\Scripts\python.exe -m newsforge.v3.experiment fit
.\.venv\Scripts\python.exe -m newsforge.v3.experiment tune
.\.venv\Scripts\python.exe -m newsforge.v3.experiment confirm
.\.venv\Scripts\python.exe -m newsforge.v3.experiment dev
```

正式产物位于 `artifacts/v3/`，聚合报告位于 `reports/v3/`。程序拒绝覆盖已有阶段；中断或失败的目录应保留。若修正计算代码，使用新的输出目录重新执行受影响的完整链条，并明确披露修正原因。协议、计算源码、输入数组、模型和缓存绑定 SHA256，修改已绑定源码后不能沿用旧模型假装同一次实验。

拟合固定 20,000 条 09–13 时内部请求；每个点击从同次曝光的非点击项中放回抽 4 次。seeds 17/29/43 仅控制负例抽样；每个 seed 的三个特征变体复用相同配对。线性成对 logistic 模型固定 L2=0.01，L-BFGS-B 最多 200 次迭代，标准化仅估计拟合对的均值和总体标准差。

调参固定 14–18 时的 2,000 条请求，只选择 `δ ∈ {0,2,5,10}`。19–23 时另外 2,000 条请求执行预声明部署门槛；若 R@10 增益或用户簇 95% CI 下界不为正，则保留热度服务。dev 在同一个旧版 5,000 请求样本上报告全部列组与门控消融，候选集合和 R@100 必须逐请求恒等。

## 运行服务

三个阶段完整且来源摘要有效时，可启动根据确认结果自动选择配置的服务：

```powershell
.\.venv\Scripts\python.exe -m newsforge.v3.serve --port 8003
```

接口为 `http://127.0.0.1:8003/docs`，支持单请求与批量请求。`/health` 返回实际启用配方、版本和可用时间范围。通过确认门槛的个性化模型最早只能用于 2019-11-15 00:00 源时间请求；服务不会将它回放到其确认标签尚未成熟的时间。

质量实验的配对模型分数是有界排名调整，不是点击概率。服务过滤请求传入的全部历史 ID，对最近最多 20 条提取交互特征，空历史与 δ=0 严格回退原热度顺序。MIND 离线回放传入的是预处理后的最近 20 条；报告的重复已读点击和候选排除均按此范围计算，主 Recall 分母仍为全部日志点击。它不代表恢复并过滤完整原始点击史，更长历史过滤的数值效果尚未测量。

## 验证和资源口径

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe -m newsforge.v3.benchmark_service
.\.venv\Scripts\python.exe -m newsforge.v3.audit
```

检索微基准采用相同请求的 ABBA 顺序，各做三轮冷派生缓存和热小时缓存对比。冷缓存指清除派生目录缓存，不代表操作系统页缓存或冷进程。热缓存测量排除显式预热成本。HTTP 基准单独计入序列化、连接、排队和完整模型计算；离线参考进程结束后才启动服务进程，避免同时加载多个真实模型。共享电脑的本地性能不能等同于生产容量或跨机器 SLO。

原始配对、候选、特征、逐 seed 截断分数、逐请求命中数保留在忽略的产物目录。独立审计应从这些保存数组重算调参赢家、确认门槛和消融指标，而非仅验证报告自身的字段一致。

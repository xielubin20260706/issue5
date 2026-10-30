# A 股日频因子回测框架｜作业提交（2026-10-30）

这套作业的核心是**输入一个因子，沿用同一套股票池、组合构建、交易撮合和评估流程**。因子文件只需定义 `Factor.compute(ctx)` 并返回当日每只证券的分数；不需要改动撮合引擎。`factor_framework.py` 是通用入口，`assignment_backtest.py` 用同一套模块复现课程指定的 2025-01-02 至 2026-09-30 实验。本包还包含本地网页、配置、因子示例和实跑结果。

| 模块 | 职责 | 替换因子时是否要改 |
|---|---|---|
| `factor_template.py` / `factor_example.py` | 定义因子，读取截至信号日的行情并输出横截面分数 | **只改这里** |
| `factor_framework.py`、`factor_config.json` | 股票池、排序建仓、调仓参数、IC / Rank IC 与分层评估 | 可配置，通常不改代码 |
| `quantbt.py`、`status_inference.py` | 次日开盘撮合、涨跌停、T+1、费用、容量及状态来源 | 不改 |
| `webapp.py`、`web/index.html` | 导入因子、运行回测、查看图表与下载报告 | 不改 |

因子接口、参数及各指标口径见 `FACTOR_FRAMEWORK.md`；上一版到本版的变化见 `UPGRADE_NOTES.md`。

## 1. 准备数据

从课程 Release 下载 `feathers.zip`：

<https://github.com/aslan9/pku_quantllm/releases/download/data-backtest-20261016/feathers.zip>

将文件放入本目录的 `data/feathers.zip`，或解压至 `feathers/`。课程数据 SHA-256：

```text
5766950f2a105244867b842709cf4bad890e4e11df0eebeeaa4fec42dc05803d
```

压缩包含七个 Feather 矩阵，范围 2020-01-02 至 2026-10-16，共 5,451 只证券。为控制提交体积，本包不重复包含约 152 MiB 的行情压缩包。

## 2. 安装和运行

需要 Python 3.10 或更新版本：

```bash
python3 -m pip install -r requirements.txt
python3 factor_framework.py --data data --config factor_config.json --output runs/my_factor
```

上面是默认 5 日收益因子的完整回测。**换一个因子，只改命令中的因子文件：**

```bash
python3 factor_framework.py --data data --factor-file factor_example.py --output runs/risk_adjusted_momentum
```

`factor_example.py` 是独立示例，计算 20 日收益与日波动之比（日波动设 0.5% 下限以避免数值爆炸）；导入自己的因子时复制 `factor_template.py`，实现 `Factor.compute(ctx)`。内置因子也可直接指定：

```bash
python3 factor_framework.py --data data --factor momentum_20d --output runs/momentum_20d
python3 factor_framework.py --data data --benchmark-nav results/equal/nav.csv --output runs/factor_with_benchmark
```

复现作业规定的四组组合与滑点比较：

```bash
python3 assignment_backtest.py --data data --output runs/assignment
```

结果包含净值、逐笔委托、撮合汇总、IC / Rank IC、分组和滚动图、参数快照及逐日和逐股 ST 状态覆盖。完整成交 CSV 很大，提交包内以 `trades.csv.gz` 压缩保存。

`results/factor_framework/` 提供独立因子流程的报告、IC/ICIR、滚动 RankIC 图、分组年化、年度观察、换手与成本、净值及配置。默认因子流程与作业主策略使用同一信号和撮合，完整逐笔可查看 `results/main/trades.csv.gz`；自行运行因子命令时会生成该次运行的 `trades.csv`。

`results/factor_example/` 是 `factor_example.py` 的独立实跑：沿用完全相同的配置和引擎，扣费后总收益 **+30.71%**，平均 Rank IC **−0.0498**。它用于展示接口复用，结果并不说明这个示例因子有效，也没有据此调参。

macOS 可双击 `start_web.command` 启动网页；或运行：

```bash
python3 webapp.py --data data
```

打开 <http://127.0.0.1:8765>。网页默认进入“因子研究”，可直接选择内置因子，也可导入 `.py` 因子文件，运行并查看 IC、RankIC、分层、净值与成交审计。网页另有表达式和 Python 策略模式。精确复现本作业四组对照可运行 `assignment_backtest.py`。

## 3. 已完成的作业实验

- 信号：2024-12-31 收盘；首次下单 2025-01-02 开盘；之后每 5 个交易日调仓。
- 股票池：价格行为推断 ST；剔除上市不足 60 个交易日和滞后成交额最低 20% 的证券。
- 主策略：复权 5 日收益最高和最低的各剔除 10%，剩余证券等权。
- 资金和执行：初始资金 10 亿元；普通股票买入 100 股整手，科创板最低 200 股、北交所最低 100 股，超过最低数量可逐股递增；每只证券每天买卖合计成交额上限为前一日成交额的 10%；涨跌停、停牌/无成交、T+1 均进入撮合。
- 费用：佣金单边 1 bp、无最低佣金；卖出印花税 5 bp；基础结果 0 bp 滑点，另计算 5 bp 敏感性。
- 退出：2026-09-29 收盘下达零仓位目标，2026-09-30 开盘执行；当日收盘再用剩余成交容量尝试退出。

主策略扣费后总收益 **+37.73%**，年化 **+20.18%**、夏普 **1.02**、最大回撤 **−21.20%**；同口径等权基准总收益 **+30.39%**。独立因子报告包含 Rank IC、ICIR、Q1–Q5 年化毛收益、滚动 Rank IC、入选组对未入选组、换手和费用。完整数值与解释见 `提交摘要.md`、`results/REPORT.md` 和 `results/factor_framework/FACTOR_REPORT.md`。

## 4. ST 识别和结果边界

课程矩阵不含历史 ST 标签。本实验在适用的 5% 风险警示涨跌幅期间，使用过去 60 个交易日内两次 ±5% 一字板的价格行为推断，并从第二次起持续排除。2026-07-06 后新增的主板 ST 无法通过原规则识别；无适用代理规则的证券记为状态未知。默认股票池仍纳入未知证券，报告分别显示官方、推断、未知覆盖，并输出 `status_coverage_daily.csv`、`status_coverage_by_symbol.csv`。这种近似可能漏判或误判；需要官方逐日状态表才能准确排除全部 ST。

参数按提供的作业规格固定。本包没有声称通过 2020–2023 参数搜索选定这些参数，也没有声称统计显著或复现另一份作业的结果。详细实跑结果、期末残余持仓和限制见 `提交摘要.md` 与 `results/REPORT.md`。

自定义 `.py` 因子在本机执行；支持的 `ctx.current()` / `ctx.history()` 接口只返回信号日及以前的数据，但本项目不把任意 Python 代码当作安全隔离环境。请只导入可信、遵守时间顺序的因子文件。课程数据和历史 ST 官方名单不包含在提交包中，实时行情接入尚未实现。

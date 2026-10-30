# 可复用因子回测流程

这次作业的 5 日收益实验已接入 `factor_framework.py`。一个因子只负责在**当日收盘**输出每只证券的分数；股票池、排序建仓、次日开盘撮合与诊断由同一流程处理。`assignment_backtest.py` 的主策略、5DR 对照和等权对照也复用这些模块。

```text
Feather 数据或新增矩阵
        ↓
历史 ST 状态／价格行为推断 → 上市期与滞后成交额过滤
        ↓
Factor.compute(ctx) → 横截面分数 → 组合选择与权重
        ↓
quantbt.Backtester → 净值、逐笔委托、费用与未成交原因
        ↓
FactorEvaluator：IC / RankIC、ICIR、分层与滚动诊断
        ↓
策略风险、交易成本、容量与可选基准比较
```

## 一条命令运行

将课程 `feathers.zip` 放入 `data/`，安装 `requirements.txt` 后运行：

```bash
python3 factor_framework.py --data data --config factor_config.json --output runs/factor_5d
```

默认配置复现本次实验的 5 日收益中间 80% 等权组合：2024-12-31 收盘形成首个信号，2025-01-02 开盘开始交易，每 5 个交易日调仓，2026-09-30 尝试清仓；10 亿元初始资金，单边 1 bp 佣金、卖出 5 bp 印花税、按板块处理的申报数量和前日成交额 10% 的单股成交上限。普通股票买入按配置的 100 股整手；科创板最低 200 股、北交所最低 100 股，超过最低数量后可逐股递增。

网页入口：运行 `python3 webapp.py --data data`，打开 <http://127.0.0.1:8765>，在“因子研究”页签选择因子和选股方式后点击运行。选“导入自定义 Python 因子”即可粘贴或导入 `.py` 文件；结果页显示净值、IC / RankIC、分层和执行审计，并提供 CSV 与报告下载。

如果已有同区间基准净值 CSV（含 `date,nav` 列），可以加入相对表现。这里用作业的合格股票等权组合举例：

```bash
python3 factor_framework.py --data data --benchmark-nav results/equal/nav.csv --output runs/factor_5d
```

提交包内的同口径等权净值位于 `results/equal/nav.csv`；若自行重跑作业脚本，也可使用 `runs/assignment/equal/nav.csv`。课程数据没有沪深 300 或中证 500 成分与指数净值，因此系统不会虚构指数基准；可以提供自己可信的基准 CSV。网页在找到已完成的作业等权结果、且日期落在该结果覆盖范围内时自动显示这一等权基准。

只换内置因子时无需改代码：

```bash
python3 factor_framework.py --data data --factor momentum_20d --output runs/momentum_20d
```

内置因子：`return_5d`、`five_dr`、`momentum_20d`、`volatility_20d`、`log_amount_20d`。默认组合规则仍是剔除因子两端各 10%；若要选最高/最低的若干只、选一端固定比例或全市场等权，修改配置的 `portfolio.selection` 为 `top_n`、`bottom_n`、`top_fraction`、`bottom_fraction` 或 `equal`，并设置相应的 `top_n`／`fraction`。股票池、日期、调仓频率、费用、滑点、分层数和前瞻期也在 `factor_config.json` 中配置。

`expected_direction=1` 表示预期高因子值对应更高未来收益，`-1` 表示预期低值更好。IC、RankIC 保留原始符号；胜率、多空收益和单调性按配置方向解释。命令行可用 `--direction -1` 临时覆盖。

## 导入自己的因子

复制 `factor_template.py`，定义 `class Factor`：

```python
class Factor:
    name = "my_factor"
    lookback = 5

    def compute(self, ctx):
        history = ctx.history("adjusted_close", 6)
        return history[-1] / history[0] - 1.0
```

然后运行：

```bash
python3 factor_framework.py --data data --factor-file my_factor.py --output runs/my_factor
```

`factor_example.py` 提供独立的“20 日收益 / 日波动”示例，可用 `--factor-file factor_example.py` 直接替换默认因子，验证组合构建、撮合和评估复用同一流程。提交包的 `results/factor_example/` 保存了实跑报告；其 Rank IC 为负，只作为接口演示。因子文件改变时，将结果输出到新的目录，保留原有运行快照。

`compute(ctx)` 返回长度等于证券数的一维 NumPy 数组；无效证券填 `NaN`。支持的 `ctx.current(field)`、`ctx.history(field, window)` 接口只返回信号日及以前的数据；`ctx.date` 和 `ctx.symbols` 提供当期标识。`adjusted_close` 是收盘价乘累计复权因子。可用字段还包括全部原始 Feather 矩阵及以后加入的数据矩阵。自定义 Python 文件会在本机执行；任意 Python 代码不是安全隔离环境，只导入可信且遵守时间顺序的因子。

更新数据时，将新的日期追加到所有对齐的 Feather 矩阵，或添加同日期、同证券列的 `因子名.feather`。每次命令行运行都会重新加载数据；修改配置日期即可使用新数据。未来接入实时数据时，数据接入端先生成对齐的日频快照，因子、组合与撮合接口不需要改动；当前系统尚未连接实时行情。

如果有可信的历史指数成分数据，可添加与行情对齐的 `hs300_member.feather` 等 0/1 矩阵，再将 `universe.membership_field` 设为 `hs300_member`，按每个信号日的历史成分过滤。课程原包不包含沪深 300 / 中证 500 历史成分，默认只运行前述非 ST 全市场股票池。

## 输出与口径

结果目录包含 `nav.csv`、`trades.csv`、`trade_summary.csv`、`metrics.json`、`run.json`、`layers.csv`、`ic.csv`、`rolling_rank_ic.csv`、`group_summary.csv`、`top_bottom.csv`、`annual_summary.csv`、`status_coverage_daily.csv`、`status_coverage_by_symbol.csv`、`factor_summary.json`、`factor_report.png`、`rolling_rank_ic.png` 和 `FACTOR_REPORT.md`。`run.json` 保存完整配置及状态来源。Q1 为因子低值组，Q5 为高值组。

- 组合净值来自撮合引擎，包含涨跌停、停牌、T+1、整手、资金、单股容量、佣金、税和滑点。
- IC 与 RankIC 在每次信号日计算因子与随后 `forward_sessions` 个交易日复权收盘收益的横截面相关。末尾不足完整前瞻期的信号不计入 IC。
- ICIR = 每期 IC 的均值 / 样本标准差；方向胜率按 `expected_direction` 计算。IC t-stat 使用 Bartlett HAC 标准误，滞后阶数根据前瞻期与调仓间隔的重叠量设置。它是样本内统计量，不可单独证明未来有效。
- 分层使用信号日收盘分组、之后复权收盘到收盘收益的理论等权组合。它是毛收益诊断，不等于可成交收益；报告将其与撮合净值分开展示。
- Q1–Q5 年化毛收益按实际覆盖时间复利折算；Top-Bottom 是 Q5 日收益减 Q1 日收益（反向因子取相反符号）的理论 100% 多头 / 100% 空头组合，未计融券和交易费用。逐期单调性用五组前瞻收益与组序号的 Spearman 相关；负数表示与预期方向相反。
- 入选组与未入选组另按当前 `portfolio.selection` 拆分，显示两组理论年化毛收益及“入选减未入选”的理论多空年化。对本作业的 `trim_tails`，它比较中间 80% 与两端 20%，比 Q5–Q1 更贴近策略假设；仍未计实际成交、融券和费用。没有未入选组时该比较留空。
- 滚动 RankIC 用过去 `rolling_window_sessions` 个**交易日**内的有效信号计算，默认 60 日；五日调仓时约为 12 个 IC 观测。年度 CSV 按实际覆盖日期列出收益和平均 RankIC，未满年的区间不年化为完整年度。
- 策略部分输出扣费后年化收益、夏普、最大回撤、Calmar、年化波动和盈利日比例。年化换手定义为“双边成交额 / 平均净值 × 252 / 回测交易日数”；佣金与印花税单独求和，滑点已计入成交价。若提供基准，输出共同首日至末日的超额年化、Information Ratio 和算术年化 Alpha / Beta。
- 数据没有官方 ST 历史标签时，默认使用过去 60 日内两次 ±5% 一字板的价格行为推断。这种近似可能漏判或误判。2026-07-06 后主板风险警示股票的涨跌幅规则改变，新出现的 ST 无法由原特征识别。报告把官方覆盖、价格推断覆盖和状态未知比例分开；未知证券仍按默认宽松股票池参与。若补充 `is_st.feather` 或完整状态表，数据读取器会优先使用提供的状态。`universe.require_known_st_status=true` 可切换为只纳入有状态的证券，结果会随覆盖范围改变。
- 期末未卖出的持仓仍以收盘价计入净值；查看 `metrics.json`、`trades.csv` 确认实际离场程度。因子研究结果不能作为未来收益保证。

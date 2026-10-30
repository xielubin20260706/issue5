#!/usr/bin/env python3
"""Reproducible walk-forward assignment run on the course A-share matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantbt import Backtester, EngineConfig, Fees, MarketData, write_outputs
from factor_framework import (FactorStrategy, PortfolioConfig, ResearchConfig, UniverseConfig,
                              factor_diagnostics, load_factor, source_manifest)


START_DATE = pd.Timestamp("2025-01-02")
END_DATE = pd.Timestamp("2026-09-30")
ANCHOR_DATE = pd.Timestamp("2024-12-31")
EXIT_SIGNAL_DATE = pd.Timestamp("2026-09-29")


from status_inference import infer_st_from_price_limits, status_audit_frames


class FiveDayAssignmentStrategy(FactorStrategy):
    """Assignment presets composed from the reusable factor pipeline."""

    def __init__(self, data: MarketData, mode: str):
        if mode not in {"trimmed", "five_dr", "equal"}:
            raise ValueError(f"未知作业策略：{mode}")
        factor_name = "five_dr" if mode == "five_dr" else "return_5d"
        portfolio = {
            "trimmed": PortfolioConfig(selection="trim_tails", trim_each_tail=0.10),
            "five_dr": PortfolioConfig(selection="bottom_n", top_n=20),
            "equal": PortfolioConfig(selection="equal"),
        }[mode]
        config = ResearchConfig(
            factor=factor_name, portfolio=portfolio,
            universe=UniverseConfig(min_listing_sessions=60,
                                    exclude_low_liquidity_fraction=0.20,
                                    min_complete_price_history=5),
        )
        super().__init__(data, load_factor(factor_name), config)
        self.mode = mode
        self.name = {
            "trimmed": "剔除两端极值的 5 日反转",
            "five_dr": "系统默认 5DR 五日反转",
            "equal": "同口径过滤后等权基准",
        }[mode]
        self.universe_counts: dict[str, dict[str, int]] = {}

    def target_weights(self, data: MarketData, day: int,
                       previous_weights: np.ndarray, portfolio=None) -> np.ndarray | None:
        before = len(self.records)
        weights = super().target_weights(data, day, previous_weights, portfolio)
        date = data.dates[day].date().isoformat()
        if len(self.records) > before:
            record = self.records[-1]
            self.universe_counts[date] = {
                "eligible": int(record.eligible.sum()), "selected": record.selected_count,
            }
        elif day == self.exit_index:
            self.universe_counts[date] = {"eligible": 0, "selected": 0}
        return weights


def _slice_data(data: MarketData, start_date: pd.Timestamp, end_date: pd.Timestamp) -> MarketData:
    start_index = int(data.dates.searchsorted(start_date, side="left"))
    end_index = int(data.dates.searchsorted(end_date, side="right"))
    if start_index >= end_index or data.dates[start_index] != start_date or data.dates[end_index - 1] != end_date:
        raise ValueError("指定的回测起止日必须都在交易日索引中")
    values = {name: array[:end_index].copy() for name, array in data.values.items()}
    return MarketData(data.dates[:end_index], data.symbols, values, start_index, data.status_source)


def _run_one(data: MarketData, name: str, mode: str, output: Path,
             slippage_bps: float, initial_cash: float) -> tuple[dict[str, float], dict[str, object], list]:
    strategy = FiveDayAssignmentStrategy(data, mode)
    config = EngineConfig(
        initial_cash=initial_cash,
        lot_size=100,
        max_daily_turnover_participation=0.10,
        liquidate_at_end=True,
        fees=Fees(commission_rate=0.0001, minimum_commission=0.0,
                  sell_stamp_duty=0.0005, slippage_bps=slippage_bps),
    )
    nav, trades, metrics = Backtester(data, strategy, config).run()
    metrics["ending_market_value"] = float(nav["market_value"].iloc[-1])
    run_info = {
        "strategy": strategy.name,
        "mode": mode,
        "start": str(nav.index[0].date()),
        "end": str(nav.index[-1].date()),
        "signal_anchor": str(ANCHOR_DATE.date()),
        "end_clear_signal": str(EXIT_SIGNAL_DATE.date()),
        "rebalance_every_trading_sessions": 5,
        "initial_cash": initial_cash,
        "listing_filter_sessions": 60,
        "liquidity_excluded_fraction": 0.20,
        "tail_trim_fraction_each_side": 0.10 if mode == "trimmed" else 0.0,
        "fees": {"commission_rate": 0.0001, "minimum_commission": 0.0,
                 "sell_stamp_duty": 0.0005, "slippage_bps": slippage_bps},
        "max_turnover_participation_previous_day_amount": 0.10,
        "order_quantity_rules": "科创板最低 200 股且可逐股递增；北交所最低 100 股且可逐股递增；其他证券按 lot_size 整数倍",
        "data_symbols": len(data.symbols),
        "status_source": data.status_source,
        "universe_counts_at_signals": strategy.universe_counts,
        "execution": "signal after close; next session open; final close liquidation; A-share daily limits/T+1/board lots",
    }
    write_outputs(output, nav, trades, metrics, run_info)
    with open(output / "metrics.json", encoding="utf-8") as handle:
        metrics = json.load(handle)
    return metrics, run_info, strategy.records


def _comparison_plot(paths: dict[str, Path], output: Path, initial_cash: float) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5.5))
    labels = {"主策略": "Trimmed 5-day return", "5DR基准": "Default 5DR (20 stocks)",
              "等权基准": "Filtered universe equal weight", "主策略5bp滑点": "Trimmed, 5bp slippage"}
    colors = {"主策略": "#17835d", "5DR基准": "#3b78a8", "等权基准": "#d0913c", "主策略5bp滑点": "#a45c79"}
    for label, directory in paths.items():
        nav = pd.read_csv(directory / "nav.csv", parse_dates=["date"])
        ax.plot(nav["date"], nav["nav"] / initial_cash, label=labels[label],
                linewidth=1.7, color=colors.get(label))
    ax.axhline(1.0, color="#8b9690", linewidth=0.8, linestyle="--")
    ax.set_title("A-share daily backtest comparison")
    ax.set_ylabel("NAV (initial = 1.0)")
    ax.set_xlabel("Date")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _factor_layers(data: MarketData, records: list, config: ResearchConfig,
                   output: Path) -> tuple[pd.DataFrame, dict[str, float], dict[str, float]]:
    """Reuse the framework's signal snapshots for five layers and forward IC."""
    frame, ic_frame, summary = factor_diagnostics(data, records, config)
    frame.to_csv(output / "layers.csv", float_format="%.8f", date_format="%Y-%m-%d")
    ic_frame.to_csv(output / "ic.csv", index=False, float_format="%.8f")
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    colors = ["#3d78a8", "#4d9d8b", "#8ba54a", "#d29a45", "#bf665d"]
    for index, column in enumerate(frame.columns):
        ax.plot(frame.index, frame[column], label=column, color=colors[index], linewidth=1.7)
    ax.axhline(1.0, color="#8b9690", linewidth=0.8, linestyle="--")
    ax.set_title("5-day return quintiles (gross, equal weighted)")
    ax.set_ylabel("NAV (initial = 1.0)")
    ax.set_xlabel("Date")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=5, loc="upper left")
    fig.tight_layout()
    fig.savefig(output / "layers.png", dpi=160)
    plt.close(fig)
    return frame, summary["layer_total_returns"], summary


def main() -> None:
    parser = argparse.ArgumentParser(description="运行课程作业口径的 A 股全市场回测")
    parser.add_argument("--data", default="feathers", help="包含 7 个 Feather 矩阵的目录或 feathers.zip")
    parser.add_argument("--output", default="runs/assignment", help="输出目录")
    parser.add_argument("--cash", type=float, default=1_000_000_000.0, help="初始资金，默认 10 亿元")
    parser.add_argument("--slippage-bps", type=float, default=0.0, help="主策略基础滑点，默认 0bp")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    data = MarketData.load(args.data)
    data_files = source_manifest(args.data)
    _, inference_info = infer_st_from_price_limits(data)
    data = _slice_data(data, START_DATE, END_DATE)
    status_daily, status_by_symbol = status_audit_frames(data)
    status_daily.to_csv(output / "status_coverage_daily.csv", index=False, float_format="%.8f")
    status_by_symbol.to_csv(output / "status_coverage_by_symbol.csv", index=False)
    metrics_by_name: dict[str, dict[str, float]] = {}
    runs: dict[str, Path] = {}
    main_records = []
    settings = [
        ("main", "主策略", "trimmed", args.slippage_bps),
        ("five_dr", "5DR基准", "five_dr", args.slippage_bps),
        ("equal", "等权基准", "equal", args.slippage_bps),
        ("main_slip5", "主策略5bp滑点", "trimmed", 5.0),
    ]
    for folder, label, mode, slip in settings:
        print(f"Running {label} ({slip:g} bp slippage)...", flush=True)
        run_dir = output / folder
        metrics, run_info, records = _run_one(data, label, mode, run_dir, slip, args.cash)
        if folder == "main":
            main_records = records
        metrics_by_name[label] = metrics
        runs[label] = run_dir
        run_info["st_inference"] = inference_info
        run_info["data_files"] = data_files
        with open(run_dir / "run.json", "w", encoding="utf-8") as handle:
            json.dump(run_info, handle, ensure_ascii=False, indent=2)

    comparison = pd.DataFrame([
        {"策略": label, "总收益": metrics["total_return"], "年化收益": metrics["annualized_return"],
         "最大回撤": metrics["max_drawdown"], "夏普": metrics["sharpe_ratio_rf0"],
         "期末净值": metrics["final_nav"], "期末持仓市值": metrics["ending_market_value"],
         "委托尝试记录": metrics["submitted_orders"],
         "完整成交率": metrics["order_completion_rate"], "股数成交率": metrics["share_fill_rate"],
         "状态可判定覆盖": metrics["st_status_coverage"],
         "官方状态覆盖": metrics["st_official_coverage"],
         "价格推断覆盖": metrics["st_inferred_coverage"],
         "状态未知比例": metrics["st_unknown_fraction"]}
        for label, metrics in metrics_by_name.items()
    ])
    comparison.to_csv(output / "comparison.csv", index=False, encoding="utf-8-sig")
    _comparison_plot(runs, output / "comparison_nav.png", args.cash)
    _, layer_returns, factor_summary = _factor_layers(
        data, main_records, FiveDayAssignmentStrategy(data, "trimmed").config, output
    )

    main_return = metrics_by_name["主策略"]["total_return"]
    benchmark_return = metrics_by_name["等权基准"]["total_return"]
    five_dr_return = metrics_by_name["5DR基准"]["total_return"]
    slip_difference = main_return - metrics_by_name["主策略5bp滑点"]["total_return"]
    main_info = json.loads((runs["主策略"] / "run.json").read_text())
    signal_counts = [row for row in main_info["universe_counts_at_signals"].values()
                     if row["eligible"] > 0]
    mean_eligible = float(np.mean([row["eligible"] for row in signal_counts])) if signal_counts else 0.0
    mean_selected = float(np.mean([row["selected"] for row in signal_counts])) if signal_counts else 0.0
    report = f'''# A 股日频回测作业报告

## 实验口径

- 数据：课程 Feather 全市场矩阵，{len(data.symbols):,} 只证券；回测区间 2025-01-02 至 2026-09-30。
- 信号锚点：2024-12-31 收盘；次交易日开盘建仓；之后每 5 个交易日收盘更新一次信号。
- 清仓：2026-09-29 收盘发出零仓位目标，2026-09-30 开盘执行；剩余可交易持仓在 9 月 30 日收盘按剩余容量尝试清仓。
- 资金及约束：初始资金 ¥{args.cash:,.0f}；普通股票买入 100 股整手，科创板最低 200 股且可逐股递增，北交所最低 100 股且可逐股递增；每只证券每日买卖合计不超过前一交易日成交额的 10%。
- 费用：佣金单边 1 bp、无最低佣金；卖出印花税 5 bp；主结果滑点 {args.slippage_bps:g} bp。
- 股票池：上市至少 60 个交易日；按前一交易日成交额剔除候选池中流动性最低 20%；信号日有有效行情和成交。
- ST 推断：课程矩阵没有 ST 标记。主板在 2026-07-06 前使用过去 60 个交易日内两次 ±5% 一字板的价格行为推断，已识别者持续剔除；规则变化后新出现的主板 ST、无适用代理规则的板块记为状态未知。未知证券仍进入当前宽松股票池。

## 策略

主策略每 5 个交易日计算复权 5 日收益，剔除收益最高和最低的各 10%，其余证券等权持有。运行时参数固定为：5 日窗口、两端各 10%、60 日上市期、成交额最低 20% 过滤、10% 单股成交容量。此包没有声称做过训练期网格搜索；参数按提交规格固定，结果为 2025–2026 评估区间的实跑结果。

对照组使用相同区间、股票池、调仓节奏、资金、费率、容量及涨跌停/T+1 规则：系统默认 5DR 反转策略（20 只）和合格股票等权基准。另运行主策略 5 bp 滑点敏感性。

这里的默认 5DR 对照按系统策略模板定义为筛选值最低的 20 只股票；不同代码对 5DR 的方向、持仓数或执行口径定义不同，结果不能直接横向照搬。

## 实跑结果

| 策略 | 总收益 | 年化收益 | 最大回撤 | 夏普 | 期末净值 | 期末持仓市值 |
|---|---:|---:|---:|---:|---:|---:|
| 主策略（剔除两端极值） | {metrics_by_name['主策略']['total_return']:.2%} | {metrics_by_name['主策略']['annualized_return']:.2%} | {metrics_by_name['主策略']['max_drawdown']:.2%} | {metrics_by_name['主策略']['sharpe_ratio_rf0']:.2f} | ¥{metrics_by_name['主策略']['final_nav']:,.0f} | ¥{metrics_by_name['主策略']['ending_market_value']:,.0f} |
| 系统默认 5DR（20 只） | {five_dr_return:.2%} | {metrics_by_name['5DR基准']['annualized_return']:.2%} | {metrics_by_name['5DR基准']['max_drawdown']:.2%} | {metrics_by_name['5DR基准']['sharpe_ratio_rf0']:.2f} | ¥{metrics_by_name['5DR基准']['final_nav']:,.0f} | ¥{metrics_by_name['5DR基准']['ending_market_value']:,.0f} |
| 合格股票等权基准 | {benchmark_return:.2%} | {metrics_by_name['等权基准']['annualized_return']:.2%} | {metrics_by_name['等权基准']['max_drawdown']:.2%} | {metrics_by_name['等权基准']['sharpe_ratio_rf0']:.2f} | ¥{metrics_by_name['等权基准']['final_nav']:,.0f} | ¥{metrics_by_name['等权基准']['ending_market_value']:,.0f} |
| 主策略（5 bp 滑点） | {metrics_by_name['主策略5bp滑点']['total_return']:.2%} | {metrics_by_name['主策略5bp滑点']['annualized_return']:.2%} | {metrics_by_name['主策略5bp滑点']['max_drawdown']:.2%} | {metrics_by_name['主策略5bp滑点']['sharpe_ratio_rf0']:.2f} | ¥{metrics_by_name['主策略5bp滑点']['final_nav']:,.0f} | ¥{metrics_by_name['主策略5bp滑点']['ending_market_value']:,.0f} |

相对等权基准超额：{main_return - benchmark_return:.2%}。主策略 5 bp 滑点相对基础情形少 {slip_difference:.2%}。调仓日平均合格证券 {mean_eligible:,.0f} 只，主策略平均持有 {mean_selected:,.0f} 只。

主策略回测期的日均 ST 状态覆盖：官方 {metrics_by_name['主策略']['st_official_coverage']:.1%}、价格行为推断 {metrics_by_name['主策略']['st_inferred_coverage']:.1%}、未知 {metrics_by_name['主策略']['st_unknown_fraction']:.1%}；期末未知比例 {metrics_by_name['主策略']['st_unknown_last_day']:.1%}。这三项均以当日有收盘价的证券为分母；推断结果可能误判，未知状态不能解释为非 ST。

五层因子诊断期末毛收益：Q1 {layer_returns['Q1']:.2%}、Q2 {layer_returns['Q2']:.2%}、Q3 {layer_returns['Q3']:.2%}、Q4 {layer_returns['Q4']:.2%}、Q5 {layer_returns['Q5']:.2%}。Q1 到 Q5 按复权 5 日收益从低到高分组。平均 IC {factor_summary['mean_ic']:.4f}，平均 RankIC {factor_summary['mean_rank_ic']:.4f}；仅统计完整的未来 5 个交易日窗口（{factor_summary['complete_forward_count']} 次）。因子分层和 IC 是研究口径毛收益，不能当成可成交收益。

进一步的因子质量：ICIR {factor_summary['icir']:.3f}、IC 方向胜率 {factor_summary['ic_win_rate']:.1%}、IC t-stat {factor_summary['ic_t_stat']:.2f}；Top-Bottom 理论多空年化 {factor_summary['top_bottom_annualized_return']:.2%}，逐期分组单调性 {factor_summary['group_monotonicity_mean']:.3f}。入选中间 80% 的理论毛收益年化 {factor_summary['selected_gross_annualized_return']:.2%}，两端未入选组 {factor_summary['unselected_gross_annualized_return']:.2%}，入选减未入选的理论多空年化 {factor_summary['selected_vs_unselected_annualized_spread']:.2%}。这些分层和 RankIC 没有显示稳定的高值优于低值关系，主策略收益不应被解释为因子方向显著有效。

![策略净值对比](comparison_nav.png)

![Q1–Q5 五日收益分层](layers.png)

## 实现与可复核材料

- `quantbt.py`：撮合引擎，支持 T+1、整手、涨跌停、受阻订单重试、容量约束、费率和期末清仓。
- `factor_framework.py`：因子接口、点时股票池、组合选择、IC / RankIC 与分层；`assignment_backtest.py` 复用它运行三组策略和 5 bp 敏感性对照。
- 每个结果子目录提供 `nav.csv`、`trades.csv`、`trade_summary.csv`、`metrics.json`、`run.json` 和净值图。
- `comparison.csv` 汇总四组结果；`status_coverage_daily.csv` 和 `status_coverage_by_symbol.csv` 列出按日期、按证券的官方、推断与未知状态覆盖。`run.json` 记录参数、数据文件快照、ST 推断和每次调仓股票数量。逐笔委托文件按每次尝试记录，受阻订单会重试，因此它的行数不是唯一委托数。
- `layers.csv`、`layers.png` 和 `ic.csv` 展示同一过滤股票池中，按 5 日收益排序后每 5 个交易日更新的 Q1–Q5 毛收益分层；分层不计费用、容量及涨跌停撮合。

## 结果限制

ST 识别是 OHLCV 价格行为推断，可能漏判或误判；它不能替代官方逐日 ST 名单。2026-07-06 后新出现的主板 ST 无法由原 ±5% 规则识别。退市和长期停牌股票会按原始日线保持持仓，订单无法成交时继续记录阻塞。日线数据不含盘口队列，涨跌停和容量撮合仍是近似。固定费率不代表历史费率。策略参数按用户给定规格固定，未做 2020–2023 调参验证，也不应把这组回测结果解释为显著超额或收益保证。
'''
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    with open(output / "comparison.json", "w", encoding="utf-8") as handle:
        json.dump({"metrics": metrics_by_name, "st_inference": inference_info}, handle, ensure_ascii=False, indent=2)
    print(report)
    print(f"\n结果目录：{output.resolve()}")


if __name__ == "__main__":
    main()

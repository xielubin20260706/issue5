#!/usr/bin/env python3
"""Reusable, point-in-time factor research on top of the daily A-share engine.

Factor code receives a FactorContext containing rows no later than the signal
close. The diagnostic forward returns are calculated separately, after trading.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from quantbt import Backtester, EngineConfig, Fees, MarketData, write_outputs
from status_inference import infer_st_from_price_limits, status_audit_frames


class Factor(Protocol):
    name: str
    lookback: int

    def compute(self, ctx: "FactorContext") -> np.ndarray:
        """Return one finite-or-NaN score per security after today's close."""


class FactorContext:
    """The supported API for custom factors; all matrix slices end at `date`."""

    def __init__(self, data: MarketData, day: int):
        self.date = data.dates[day]
        self.symbols = tuple(map(str, data.symbols))
        self.available_fields = tuple(data.values.keys()) + ("adjusted_close",)
        self._data = data
        self._day = day

    def history(self, field: str, window: int) -> np.ndarray:
        if field not in self.available_fields:
            raise ValueError(f"未知因子字段：{field}")
        if not isinstance(window, int) or not 1 <= window <= 1001:
            raise ValueError("因子历史窗口须为 1–1001 个交易日")
        start = max(0, self._day - window + 1)
        if field == "adjusted_close":
            values = self._data.values
            return (values["close"][start:self._day + 1]
                    * values["cumadj"][start:self._day + 1]).copy()
        return self._data.values[field][start:self._day + 1].copy()

    def current(self, field: str) -> np.ndarray:
        return self.history(field, 1)[-1]


@dataclass(frozen=True)
class BuiltinFactor:
    name: str
    lookback: int

    def compute(self, ctx: FactorContext) -> np.ndarray:
        size = len(ctx.symbols)
        result = np.full(size, np.nan, dtype=np.float64)
        if self.name == "return_5d":
            close = ctx.history("adjusted_close", 6)
            if len(close) < 6:
                return result
            with np.errstate(divide="ignore", invalid="ignore"):
                result = close[-1] / close[0] - 1.0
        elif self.name == "five_dr":
            close = ctx.history("adjusted_close", 6)
            if len(close) < 6:
                return result
            prior = close[:-1]
            valid = np.isfinite(prior) & (prior > 0)
            mean = np.divide(np.where(valid, prior, 0.0).sum(axis=0), valid.sum(axis=0),
                             out=result.copy(), where=valid.sum(axis=0) == 5)
            with np.errstate(divide="ignore", invalid="ignore"):
                result = close[-1] / mean - 1.0
        elif self.name == "momentum_20d":
            close = ctx.history("adjusted_close", 21)
            if len(close) < 21:
                return result
            with np.errstate(divide="ignore", invalid="ignore"):
                result = close[-1] / close[0] - 1.0
        elif self.name == "volatility_20d":
            close = ctx.history("adjusted_close", 21)
            if len(close) < 21:
                return result
            valid = np.isfinite(close) & (close > 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                daily = close[1:] / close[:-1] - 1.0
            complete = valid.all(axis=0) & np.isfinite(daily).all(axis=0)
            result[complete] = daily[:, complete].std(axis=0, ddof=1)
        elif self.name == "log_amount_20d":
            amount = ctx.history("amt", 20)
            if len(amount) < 20:
                return result
            valid = np.isfinite(amount) & (amount > 0)
            complete = valid.all(axis=0)
            result[complete] = np.log(amount[:, complete].mean(axis=0))
        else:
            raise ValueError(f"未知内置因子：{self.name}")
        return np.where(np.isfinite(result), result, np.nan)


BUILTIN_FACTORS = {
    "return_5d": BuiltinFactor("return_5d", 5),
    "five_dr": BuiltinFactor("five_dr", 5),
    "momentum_20d": BuiltinFactor("momentum_20d", 20),
    "volatility_20d": BuiltinFactor("volatility_20d", 20),
    "log_amount_20d": BuiltinFactor("log_amount_20d", 19),
}


def load_factor(name: str, factor_file: str | Path | None = None) -> Factor:
    if factor_file is None:
        if name not in BUILTIN_FACTORS:
            raise ValueError(f"未知因子 {name}；可用因子：{', '.join(BUILTIN_FACTORS)}")
        return BUILTIN_FACTORS[name]
    path = Path(factor_file).resolve()
    module_spec = importlib.util.spec_from_file_location("user_factor", path)
    if module_spec is None or module_spec.loader is None:
        raise ValueError(f"无法载入因子文件：{path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    factor_class = getattr(module, "Factor", None)
    if not isinstance(factor_class, type):
        raise ValueError("自定义因子文件需要定义 class Factor")
    factor = factor_class()
    if not callable(getattr(factor, "compute", None)):
        raise ValueError("Factor 类需要实现 compute(ctx)")
    lookback = getattr(factor, "lookback", None)
    if not isinstance(lookback, int) or not 0 <= lookback <= 1000:
        raise ValueError("Factor.lookback 须为 0–1000 的整数")
    if not getattr(factor, "name", None):
        factor.name = path.stem
    return factor


def source_manifest(source: str | Path) -> list[dict[str, object]]:
    """Record local input sizes and mtimes without copying the course dataset."""
    path = Path(source).resolve()
    if path.is_file():
        files = [path]
        sidecar_dir = path.parent
    elif (path / "feathers.zip").is_file():
        files = [path / "feathers.zip"]
        sidecar_dir = path
    else:
        files = sorted(path.glob("*.feather"))
        sidecar_dir = path
    files += [sidecar_dir / name for name in ("security_status.csv", "limit_rules.csv")
              if (sidecar_dir / name).is_file()]
    return [{"name": file.name, "bytes": file.stat().st_size,
             "modified_ns": file.stat().st_mtime_ns} for file in files]


@dataclass(frozen=True)
class UniverseConfig:
    min_listing_sessions: int = 60
    exclude_low_liquidity_fraction: float = 0.20
    min_complete_price_history: int = 5
    require_known_st_status: bool = False
    membership_field: str | None = None

    def validate(self) -> None:
        if self.min_listing_sessions < 0 or self.min_complete_price_history < 0:
            raise ValueError("上市期和历史价格窗口不能为负")
        if not 0 <= self.exclude_low_liquidity_fraction < 1:
            raise ValueError("流动性剔除比例须在 [0, 1) 内")


class UniverseFilter:
    def __init__(self, data: MarketData, config: UniverseConfig):
        config.validate()
        self.config = config
        self.data = data
        close = data.values["close"]
        valid = np.isfinite(close) & (close > 0)
        self.first_observed = valid.argmax(axis=0)
        self.first_observed[~valid.any(axis=0)] = len(data.dates) + 1

    def mask(self, day: int) -> np.ndarray:
        values = self.data.values
        size = len(self.data.symbols)
        if day == 0:
            return np.zeros(size, dtype=bool)
        status = values["is_st"][day]
        non_st = np.isfinite(status) & (status < 0.5) if self.config.require_known_st_status \
            else ~(np.isfinite(status) & (status >= 0.5))
        valid = (np.isfinite(values["open"][day]) & (values["open"][day] > 0)
                 & np.isfinite(values["close"][day]) & (values["close"][day] > 0)
                 & np.isfinite(values["amt"][day]) & (values["amt"][day] > 0)
                 & np.isfinite(values["num_trades"][day]) & (values["num_trades"][day] > 0)
                 & non_st & (day - self.first_observed >= self.config.min_listing_sessions))
        if self.config.membership_field:
            if self.config.membership_field not in values:
                raise ValueError(f"股票池成员矩阵不存在：{self.config.membership_field}")
            membership = values[self.config.membership_field][day]
            valid &= np.isfinite(membership) & (membership >= 0.5)
        history = self.config.min_complete_price_history
        if history:
            if day < history:
                return np.zeros(size, dtype=bool)
            adjusted = (values["close"][day-history:day]
                        * values["cumadj"][day-history:day])
            valid &= (np.isfinite(adjusted) & (adjusted > 0)).all(axis=0)
        lagged_amount = values["amt"][day - 1]
        valid &= np.isfinite(lagged_amount) & (lagged_amount > 0)
        candidates = np.flatnonzero(valid)
        if len(candidates) and self.config.exclude_low_liquidity_fraction:
            floor = np.nanquantile(lagged_amount[candidates],
                                   self.config.exclude_low_liquidity_fraction)
            valid &= lagged_amount >= floor
        return valid


@dataclass(frozen=True)
class PortfolioConfig:
    selection: str = "trim_tails"  # trim_tails, top_n, bottom_n, top_fraction, bottom_fraction, equal
    top_n: int = 20
    fraction: float = 0.10
    trim_each_tail: float = 0.10

    def validate(self) -> None:
        if self.selection not in {"trim_tails", "top_n", "bottom_n", "top_fraction", "bottom_fraction", "equal"}:
            raise ValueError(f"未知组合选择方式：{self.selection}")
        if self.top_n < 1 or not 0 < self.fraction <= 1 or not 0 <= self.trim_each_tail < 0.5:
            raise ValueError("组合数量或比例无效")

    def select(self, candidates: np.ndarray, score: np.ndarray) -> np.ndarray:
        if self.selection == "equal":
            return candidates
        ordered = candidates[np.argsort(score[candidates], kind="stable")]
        if self.selection == "trim_tails":
            trim = int(np.floor(len(ordered) * self.trim_each_tail))
            return ordered[trim:len(ordered)-trim] if trim else ordered
        if self.selection == "top_n":
            return ordered[-min(self.top_n, len(ordered)):]
        if self.selection == "bottom_n":
            return ordered[:min(self.top_n, len(ordered))]
        count = max(1, int(np.floor(len(ordered) * self.fraction)))
        return ordered[-count:] if self.selection == "top_fraction" else ordered[:count]


@dataclass(frozen=True)
class ResearchConfig:
    start: str = "2025-01-02"
    end: str = "2026-09-30"
    signal_anchor: str = "2024-12-31"
    exit_signal: str = "2026-09-29"
    rebalance_sessions: int = 5
    forward_sessions: int = 5
    groups: int = 5
    expected_direction: int = 1  # +1: larger factor predicts higher return; -1: lower predicts higher
    rolling_window_sessions: int = 60
    factor: str = "return_5d"
    factor_file: str | None = None
    benchmark_nav: str | None = None
    infer_st_when_missing: bool = True
    initial_cash: float = 1_000_000_000.0
    commission_rate: float = 0.0001
    minimum_commission: float = 0.0
    sell_stamp_duty: float = 0.0005
    slippage_bps: float = 0.0
    lot_size: int = 100
    max_daily_turnover_participation: float = 0.10
    liquidate_at_end: bool = True
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> "ResearchConfig":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("factor_file"):
            factor_file = Path(payload["factor_file"])
            if not factor_file.is_absolute():
                payload["factor_file"] = str((path.resolve().parent / factor_file).resolve())
        if payload.get("benchmark_nav"):
            benchmark_nav = Path(payload["benchmark_nav"])
            if not benchmark_nav.is_absolute():
                payload["benchmark_nav"] = str((path.resolve().parent / benchmark_nav).resolve())
        payload["universe"] = UniverseConfig(**payload.get("universe", {}))
        payload["portfolio"] = PortfolioConfig(**payload.get("portfolio", {}))
        return cls(**payload)

    def validate(self) -> None:
        self.universe.validate()
        self.portfolio.validate()
        if self.rebalance_sessions < 1 or self.forward_sessions < 1 or self.groups < 2:
            raise ValueError("调仓间隔、前瞻期和分层数无效")
        if self.expected_direction not in (-1, 1) or self.rolling_window_sessions < 1:
            raise ValueError("因子预期方向须为 +1/-1，滚动窗口须为正整数")
        if pd.Timestamp(self.signal_anchor) >= pd.Timestamp(self.start):
            raise ValueError("首个收盘信号日应早于回测开始日")
        if not pd.Timestamp(self.start) <= pd.Timestamp(self.exit_signal) < pd.Timestamp(self.end):
            raise ValueError("清仓信号日应在回测开始和结束日期之间")
        if self.initial_cash <= 0 or self.lot_size < 1:
            raise ValueError("初始资金或整手股数无效")


@dataclass
class SignalRecord:
    day: int
    score: np.ndarray
    eligible: np.ndarray
    selected_count: int


class FactorStrategy:
    """Turns any factor into a next-open target-weight strategy."""

    def __init__(self, data: MarketData, factor: Factor, config: ResearchConfig):
        config.validate()
        self.factor = factor
        self.config = config
        self.universe = UniverseFilter(data, config.universe)
        self.anchor_index = int(data.dates.get_indexer([pd.Timestamp(config.signal_anchor)])[0])
        self.exit_index = int(data.dates.get_indexer([pd.Timestamp(config.exit_signal)])[0])
        if self.anchor_index < 0 or self.exit_index < 0:
            raise ValueError("信号锚点或清仓信号日不在数据交易日中")
        self.records: list[SignalRecord] = []
        self.name = f"{factor.name} / {config.portfolio.selection}"

    def target_weights(self, data: MarketData, day: int,
                       previous_weights: np.ndarray, portfolio=None) -> np.ndarray | None:
        if day == self.exit_index:
            return np.zeros(len(data.symbols), dtype=np.float64)
        if day < self.anchor_index or (day - self.anchor_index) % self.config.rebalance_sessions:
            return None
        if day < max(self.factor.lookback, self.config.universe.min_listing_sessions):
            return np.zeros(len(data.symbols), dtype=np.float64)
        score = np.asarray(self.factor.compute(FactorContext(data, day)), dtype=np.float64)
        if score.shape != (len(data.symbols),):
            raise ValueError(f"因子 {self.factor.name} 须返回长度为 {len(data.symbols)} 的一维数组")
        eligible = self.universe.mask(day) & np.isfinite(score)
        candidates = np.flatnonzero(eligible)
        selected = self.config.portfolio.select(candidates, score)
        self.records.append(SignalRecord(day, score.copy(), eligible.copy(), len(selected)))
        weights = np.zeros(len(data.symbols), dtype=np.float64)
        if len(selected):
            weights[selected] = 1.0 / len(selected)
        return weights


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    x, y = x - x.mean(), y - y.mean()
    denominator = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    return float(np.dot(x, y) / denominator) if denominator > 0 else float("nan")


def _ic_statistics(series: pd.Series, hac_lags: int, direction: int) -> dict[str, float]:
    """Per-signal ICIR and a Bartlett HAC t-stat for possibly overlapping horizons."""
    values = series.to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"mean": float("nan"), "icir": float("nan"),
                "win_rate": float("nan"), "t_stat": float("nan")}
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
    icir = mean / std if np.isfinite(std) and std > 0 else float("nan")
    centered = values - mean
    variance = float(np.dot(centered, centered) / len(values))
    for lag in range(1, min(hac_lags, len(values) - 1) + 1):
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / len(values))
        variance += 2.0 * (1.0 - lag / (hac_lags + 1)) * covariance
    standard_error = np.sqrt(variance / len(values)) if variance > 0 else float("nan")
    t_stat = mean / standard_error if np.isfinite(standard_error) and standard_error > 0 else float("nan")
    return {"mean": mean, "icir": icir,
            "win_rate": float(np.mean(direction * values > 0)), "t_stat": float(t_stat)}


def factor_diagnostics(data: MarketData, records: list[SignalRecord],
                       config: ResearchConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Forward IC and gross quintiles, calculated outside the strategy callback."""
    start = data.simulation_start
    close = data.values["close"] * data.values["cumadj"]
    n_days, n_symbols = close.shape
    nav = np.ones((n_days - start, config.groups), dtype=np.float64)
    holdings = np.zeros((config.groups, n_symbols), dtype=np.float64)
    selection_nav = np.ones((n_days - start, 2), dtype=np.float64)
    selection_holdings = np.zeros((2, n_symbols), dtype=np.float64)
    selection_pair_complete = bool(records)
    record_by_day = {record.day: record for record in records}
    ic_rows = []

    def assign_layers(record: SignalRecord) -> None:
        nonlocal selection_pair_complete
        holdings.fill(0.0)
        selection_holdings.fill(0.0)
        ordered = np.flatnonzero(record.eligible)
        ordered = ordered[np.argsort(record.score[ordered], kind="stable")]
        for group, selected in enumerate(np.array_split(ordered, config.groups)):
            if len(selected):
                holdings[group, selected] = 1.0 / len(selected)
        selected = config.portfolio.select(ordered, record.score)
        if len(selected):
            selection_holdings[0, selected] = 1.0 / len(selected)
        rest = np.setdiff1d(ordered, selected, assume_unique=True)
        if len(rest):
            selection_holdings[1, rest] = 1.0 / len(rest)
        if not len(selected) or not len(rest):
            selection_pair_complete = False

    anchor_records = [record for record in records if record.day < start]
    if anchor_records:
        assign_layers(anchor_records[-1])

    for day in range(start, n_days):
        row = day - start
        if day > 0:
            previous, current = close[day - 1], close[day]
            valid = np.isfinite(previous) & (previous > 0) & np.isfinite(current) & (current > 0)
            daily_return = np.zeros(n_symbols, dtype=np.float64)
            daily_return[valid] = current[valid] / previous[valid] - 1.0
            prior = np.ones(config.groups) if row == 0 else nav[row - 1]
            nav[row] = prior * (1.0 + holdings @ daily_return)
            selection_prior = np.ones(2) if row == 0 else selection_nav[row - 1]
            selection_nav[row] = selection_prior * (1.0 + selection_holdings @ daily_return)
        record = record_by_day.get(day)
        if record is not None:
            assign_layers(record)
        if data.dates[day] == pd.Timestamp(config.exit_signal):
            holdings.fill(0.0)
            selection_holdings.fill(0.0)

    for record in records:
        day = record.day
        future = day + config.forward_sessions
        if future >= n_days:
            continue
        valid = record.eligible & np.isfinite(close[day]) & (close[day] > 0) \
            & np.isfinite(close[future]) & (close[future] > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            returns = close[future, valid] / close[day, valid] - 1.0
        scores = record.score[valid]
        finite = np.isfinite(returns)
        scores, returns = scores[finite], returns[finite]
        ic = _correlation(scores, returns)
        if len(scores) >= 3:
            score_rank = pd.Series(scores).rank(method="average").to_numpy()
            return_rank = pd.Series(returns).rank(method="average").to_numpy()
            rank_ic = _correlation(score_rank, return_rank)
        else:
            rank_ic = float("nan")
        selected_original = np.flatnonzero(record.eligible & valid)
        selected_original = selected_original[np.argsort(record.score[selected_original], kind="stable")]
        group_forward = []
        for group in np.array_split(selected_original, config.groups):
            if len(group):
                with np.errstate(divide="ignore", invalid="ignore"):
                    group_return = close[future, group] / close[day, group] - 1.0
                group_forward.append(float(np.nanmean(group_return)))
            else:
                group_forward.append(float("nan"))
        group_values = np.asarray(group_forward, dtype=np.float64)
        valid_groups = np.isfinite(group_values)
        monotonicity = _correlation(
            np.flatnonzero(valid_groups).astype(np.float64),
            pd.Series(group_values[valid_groups]).rank(method="average").to_numpy(dtype=np.float64),
        ) if valid_groups.sum() >= 3 else float("nan")
        ic_rows.append({"signal_date": data.dates[day].date().isoformat(),
                        "signal_index": day,
                        "forward_end": data.dates[future].date().isoformat(),
                        "eligible": int(record.eligible.sum()), "selected": record.selected_count,
                        "valid_forward": len(scores), "ic": ic, "rank_ic": rank_ic,
                        "group_monotonicity": monotonicity,
                        "top_bottom_forward_return": config.expected_direction
                        * (group_values[-1] - group_values[0]),
                        **{f"Q{i + 1}_forward_return": value
                           for i, value in enumerate(group_forward)}})

    layer_columns = [f"Q{i + 1}" for i in range(config.groups)]
    layers = pd.DataFrame(nav, index=data.dates[start:], columns=layer_columns)
    layers.index.name = "date"
    ic_frame = pd.DataFrame(ic_rows)
    if len(ic_frame):
        indices = ic_frame["signal_index"].to_numpy(dtype=int)
        rank_values = ic_frame["rank_ic"].to_numpy(dtype=np.float64)
        rolling_values = np.full(len(ic_frame), np.nan, dtype=np.float64)
        minimum_signals = max(3, min(6, int(np.ceil(config.rolling_window_sessions
                                                 / config.rebalance_sessions / 2))))
        for row, day in enumerate(indices):
            window = rank_values[(indices >= day - config.rolling_window_sessions + 1)
                                 & (indices <= day)]
            finite = window[np.isfinite(window)]
            if len(finite) >= minimum_signals:
                rolling_values[row] = float(finite.mean())
        ic_frame["rolling_rank_ic"] = rolling_values
    else:
        ic_frame = pd.DataFrame(columns=["signal_date", "signal_index", "forward_end",
                                          "eligible", "selected", "valid_forward", "ic", "rank_ic",
                                          "group_monotonicity", "top_bottom_forward_return",
                                          "rolling_rank_ic"])
    hac_lags = max(0, int(np.ceil(config.forward_sessions / config.rebalance_sessions)) - 1)
    ic_stats = _ic_statistics(ic_frame["ic"], hac_lags, config.expected_direction)
    rank_stats = _ic_statistics(ic_frame["rank_ic"], hac_lags, config.expected_direction)
    years = max((layers.index[-1] - layers.index[0]).days / 365.25, 1.0 / 252.0)
    layer_annualized = {column: float(layers[column].iloc[-1] ** (1.0 / years) - 1.0)
                        for column in layer_columns}
    daily_layers = layers.div(layers.shift(1).fillna(1.0)).sub(1.0)
    spread = config.expected_direction * (daily_layers[layer_columns[-1]] - daily_layers[layer_columns[0]])
    spread_nav = (1.0 + spread).cumprod()
    top_bottom_total = float(spread_nav.iloc[-1] - 1.0)
    top_bottom_annualized = float(spread_nav.iloc[-1] ** (1.0 / years) - 1.0) \
        if spread_nav.iloc[-1] > 0 else -1.0
    selection_daily = selection_nav / np.vstack([np.ones((1, 2)), selection_nav[:-1]]) - 1.0
    selection_spread = selection_daily[:, 0] - selection_daily[:, 1]
    selection_spread_nav = np.cumprod(1.0 + selection_spread)
    selection_spread_annualized = float(selection_spread_nav[-1] ** (1.0 / years) - 1.0) \
        if selection_spread_nav[-1] > 0 else -1.0
    monotonic = ic_frame["group_monotonicity"].astype(float)
    monotonic_valid = monotonic[np.isfinite(monotonic)]
    summary = {
        "mean_ic": ic_stats["mean"],
        "mean_rank_ic": rank_stats["mean"],
        "icir": ic_stats["icir"],
        "rank_icir": rank_stats["icir"],
        "ic_win_rate": ic_stats["win_rate"],
        "rank_ic_win_rate": rank_stats["win_rate"],
        "ic_t_stat": ic_stats["t_stat"],
        "rank_ic_t_stat": rank_stats["t_stat"],
        "ic_t_stat_hac_lags": hac_lags,
        "expected_direction": config.expected_direction,
        "rolling_window_sessions": config.rolling_window_sessions,
        "positive_rank_ic_fraction": float((ic_frame["rank_ic"] > 0).mean()) if len(ic_frame) else float("nan"),
        "signal_count": len(records),
        "complete_forward_count": len(ic_frame),
        "mean_eligible": float(np.mean([r.eligible.sum() for r in records])) if records else 0.0,
        "mean_selected": float(np.mean([r.selected_count for r in records])) if records else 0.0,
        "layer_total_returns": {column: float(layers[column].iloc[-1] - 1.0)
                                for column in layer_columns},
        "layer_annualized_returns": layer_annualized,
        "top_bottom_total_return": top_bottom_total,
        "top_bottom_annualized_return": top_bottom_annualized,
        "group_monotonicity_mean": float(config.expected_direction * monotonic_valid.mean())
        if len(monotonic_valid) else float("nan"),
        "group_monotonicity_win_rate": float((config.expected_direction * monotonic_valid > 0).mean())
        if len(monotonic_valid) else float("nan"),
        "selected_gross_annualized_return": float(selection_nav[-1, 0] ** (1.0 / years) - 1.0),
        "unselected_gross_annualized_return": float(selection_nav[-1, 1] ** (1.0 / years) - 1.0)
        if selection_pair_complete else float("nan"),
        "selected_vs_unselected_annualized_spread": selection_spread_annualized
        if selection_pair_complete else float("nan"),
    }
    return layers, ic_frame, summary


class FactorEvaluator:
    """Stable entry point for signal-level factor quality and gross group diagnostics."""

    def __init__(self, data: MarketData, config: ResearchConfig):
        self.data = data
        self.config = config

    def evaluate(self, records: list[SignalRecord]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
        return factor_diagnostics(self.data, records, self.config)


def _portfolio_quality(nav: pd.DataFrame, trades: pd.DataFrame,
                       metrics: dict[str, float], config: ResearchConfig) -> dict[str, float]:
    daily_returns = nav["nav"].astype(float).pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    drawdown = float(metrics["max_drawdown"])
    commission = float(trades["commission"].sum())
    tax = float(trades["tax"].sum())
    capacity_limited = int(trades["reason"].fillna("").astype(str).str.contains("单股成交容量限制", regex=False).sum())
    return {
        "calmar_ratio": float(metrics["annualized_return"] / abs(drawdown)) if drawdown < 0 else float("nan"),
        "daily_win_rate": float((daily_returns > 0).mean()) if len(daily_returns) else float("nan"),
        "annualized_turnover": float(metrics["turnover_over_average_nav"] * 252.0 / len(nav)),
        "commission_paid": commission,
        "stamp_duty_paid": tax,
        "explicit_fees_paid": commission + tax,
        "explicit_fees_over_initial_cash": float((commission + tax) / config.initial_cash),
        "net_annualized_return": float(metrics["annualized_return"]),
        "configured_slippage_bps": float(config.slippage_bps),
        "capacity_limited_order_attempts": capacity_limited,
        "capacity_limited_order_fraction": float(capacity_limited / len(trades)) if len(trades) else 0.0,
    }


def _benchmark_quality(nav: pd.DataFrame, path: str | Path) -> dict[str, float | str]:
    frame = pd.read_csv(path, parse_dates=["date"])
    if "nav" not in frame or frame["date"].duplicated().any():
        raise ValueError("基准 CSV 需要唯一的 date 和 nav 列")
    reference = frame.set_index("date")["nav"].astype(float).reindex(nav.index)
    if reference.isna().any() or (reference <= 0).any():
        raise ValueError("基准净值未覆盖回测的全部交易日，或含非正值")
    benchmark_returns = reference.pct_change().iloc[1:].to_numpy(dtype=np.float64)
    portfolio_returns = nav["nav"].astype(float).pct_change().iloc[1:].to_numpy(dtype=np.float64)
    active = portfolio_returns - benchmark_returns
    active_std = float(active.std(ddof=1)) if len(active) > 1 else float("nan")
    benchmark_variance = float(benchmark_returns.var(ddof=1)) if len(active) > 1 else float("nan")
    beta = float(np.cov(portfolio_returns, benchmark_returns, ddof=1)[0, 1] / benchmark_variance) \
        if np.isfinite(benchmark_variance) and benchmark_variance > 0 else float("nan")
    years = max((nav.index[-1] - nav.index[0]).days / 365.25, 1.0 / 252.0)
    benchmark_annualized = float((reference.iloc[-1] / reference.iloc[0]) ** (1.0 / years) - 1.0)
    portfolio_annualized = float((nav["nav"].iloc[-1] / nav["nav"].iloc[0]) ** (1.0 / years) - 1.0)
    return {
        "benchmark_source": str(Path(path)),
        "portfolio_common_period_annualized_return": portfolio_annualized,
        "benchmark_annualized_return": benchmark_annualized,
        "excess_annualized_return": portfolio_annualized - benchmark_annualized,
        "information_ratio": float(active.mean() / active_std * np.sqrt(252.0))
        if np.isfinite(active_std) and active_std > 0 else float("nan"),
        "beta": beta,
        "alpha_annualized_arithmetic": float((portfolio_returns.mean()
                                              - beta * benchmark_returns.mean()) * 252.0)
        if np.isfinite(beta) else float("nan"),
    }


def _plot_diagnostics(nav: pd.DataFrame, layers: pd.DataFrame, output: Path,
                      initial_cash: float) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.0), sharex=True)
    axes[0].plot(nav.index, nav["nav"] / initial_cash, color="#16845e", label="Net NAV (after costs)")
    axes[0].axhline(1.0, color="#aaa", linewidth=0.8)
    axes[0].legend(frameon=False)
    axes[0].set_title("Factor portfolio backtest")
    for column in layers:
        axes[1].plot(layers.index, layers[column], label=column, linewidth=1.25)
    axes[1].legend(ncol=len(layers.columns), frameon=False)
    axes[1].set_title("Factor quintiles (gross, equal weight)")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "factor_report.png", dpi=160)
    plt.close(fig)


def _plot_rolling_rank_ic(ic_frame: pd.DataFrame, output: Path,
                          window_sessions: int) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.5, 3.5))
    if len(ic_frame):
        dates = pd.to_datetime(ic_frame["signal_date"])
        ax.scatter(dates, ic_frame["rank_ic"], s=9, alpha=0.35, color="#8fa49a", label="Rank IC")
        ax.plot(dates, ic_frame["rolling_rank_ic"], linewidth=1.8,
                color="#16845e", label=f"Rolling {window_sessions} sessions")
    ax.axhline(0.0, linewidth=0.8, linestyle="--", color="#8b9690")
    ax.set_title("Rolling Rank IC")
    ax.set_ylabel("Rank IC")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "rolling_rank_ic.png", dpi=160)
    plt.close(fig)


def _annual_summary(nav: pd.DataFrame, ic_frame: pd.DataFrame,
                    initial_cash: float, direction: int) -> pd.DataFrame:
    rows = []
    previous = float(initial_cash)
    ic_dates = pd.DatetimeIndex(pd.to_datetime(ic_frame["signal_date"])) if len(ic_frame) else pd.DatetimeIndex([])
    for year, group in nav.groupby(nav.index.year):
        ending = float(group["nav"].iloc[-1])
        ic_values = ic_frame.loc[ic_dates.year == year, "rank_ic"].dropna().astype(float)
        rows.append({"year": int(year), "first_date": group.index[0].date().isoformat(),
                     "last_date": group.index[-1].date().isoformat(),
                     "net_return": ending / previous - 1.0,
                     "mean_rank_ic": float(ic_values.mean()) if len(ic_values) else float("nan"),
                     "rank_ic_win_rate": float((direction * ic_values > 0).mean())
                     if len(ic_values) else float("nan"),
                     "ic_signals": len(ic_values)})
        previous = ending
    return pd.DataFrame(rows)


def _json_safe(value):
    """Keep artifact JSON valid when a statistic is undefined."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run_factor_research(data: MarketData, config: ResearchConfig,
                        output: str | Path) -> dict[str, object]:
    """Run factor -> universe -> portfolio -> execution -> diagnostic pipeline."""
    config.validate()
    factor = load_factor(config.factor, config.factor_file)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    strategy = FactorStrategy(data, factor, config)
    engine_config = EngineConfig(
        initial_cash=config.initial_cash, lot_size=config.lot_size,
        max_daily_turnover_participation=config.max_daily_turnover_participation,
        liquidate_at_end=config.liquidate_at_end,
        fees=Fees(config.commission_rate, config.minimum_commission,
                  config.sell_stamp_duty, config.slippage_bps),
    )
    nav, trades, metrics = Backtester(data, strategy, engine_config).run()
    metrics["ending_market_value"] = float(nav["market_value"].iloc[-1])
    metrics.update(_portfolio_quality(nav, trades, metrics, config))
    layers, ic_frame, diagnostics = FactorEvaluator(data, config).evaluate(strategy.records)
    benchmark = _benchmark_quality(nav, config.benchmark_nav) if config.benchmark_nav else None
    annual = _annual_summary(nav, ic_frame, config.initial_cash, config.expected_direction)
    info = {"framework": "factor_framework.py", "factor": factor.name,
            "factor_file": config.factor_file, "config": asdict(config),
            "status_source": data.status_source,
            "order_quantity_rules": "科创板最低 200 股且可逐股递增；北交所最低 100 股且可逐股递增；其他证券按 lot_size 整数倍",
            "diagnostics": diagnostics, "benchmark": benchmark,
            "data_last_date": data.dates[-1].date().isoformat(),
            "data_files": getattr(data, "source_files", []),
            "execution": "信号日收盘计算，次日开盘撮合；IC 与分层为研究口径毛收益"}
    write_outputs(output, nav, trades, _json_safe(metrics), _json_safe(info))
    status_daily, status_by_symbol = status_audit_frames(data)
    status_daily.to_csv(output / "status_coverage_daily.csv", index=False, float_format="%.8f")
    status_by_symbol.to_csv(output / "status_coverage_by_symbol.csv", index=False)
    if config.factor_file:
        (output / "factor_source.py").write_bytes(Path(config.factor_file).read_bytes())
    layers.to_csv(output / "layers.csv", float_format="%.8f")
    ic_frame.to_csv(output / "ic.csv", index=False, float_format="%.8f")
    ic_frame[["signal_date", "rank_ic", "rolling_rank_ic"]].to_csv(
        output / "rolling_rank_ic.csv", index=False, float_format="%.8f")
    annual.to_csv(output / "annual_summary.csv", index=False, float_format="%.8f")
    pd.DataFrame([{"group": group,
                   "total_return": diagnostics["layer_total_returns"][group],
                   "annualized_return": diagnostics["layer_annualized_returns"][group]}
                  for group in layers.columns]).to_csv(output / "group_summary.csv", index=False,
                                                        float_format="%.8f")
    daily_layers = layers.div(layers.shift(1).fillna(1.0)).sub(1.0)
    top_bottom = config.expected_direction * (daily_layers.iloc[:, -1] - daily_layers.iloc[:, 0])
    pd.DataFrame({"daily_spread": top_bottom, "gross_nav": (1.0 + top_bottom).cumprod()}).to_csv(
        output / "top_bottom.csv", index_label="date", float_format="%.8f")
    (output / "factor_summary.json").write_text(
        json.dumps(_json_safe({"metrics": metrics, "diagnostics": diagnostics,
                               "benchmark": benchmark}), ensure_ascii=False, indent=2),
        encoding="utf-8")
    _plot_diagnostics(nav, layers, output, config.initial_cash)
    _plot_rolling_rank_ic(ic_frame, output, config.rolling_window_sessions)
    direction_label = "高值预期更好" if config.expected_direction == 1 else "低值预期更好"
    group_rows = "\n".join(
        f"| {group} | {diagnostics['layer_total_returns'][group]:.2%} | "
        f"{diagnostics['layer_annualized_returns'][group]:.2%} |"
        for group in layers.columns)
    annual_rows = "\n".join(
        f"| {int(row.year)} | {row.first_date} 至 {row.last_date} | {row.net_return:.2%} | "
        f"{row.mean_rank_ic:.4f} | {int(row.ic_signals)} |"
        for row in annual.itertuples(index=False))
    benchmark_section = (
        f"基准：`{benchmark['benchmark_source']}`；共同首日至末日的基准年化 "
        f"{benchmark['benchmark_annualized_return']:.2%}、超额年化 "
        f"{benchmark['excess_annualized_return']:.2%}、Information Ratio "
        f"{benchmark['information_ratio']:.2f}、Beta {benchmark['beta']:.2f}、"
        f"算术年化 Alpha {benchmark['alpha_annualized_arithmetic']:.2%}。"
        if benchmark else "未指定基准净值；超额收益、Information Ratio 和 Alpha/Beta 留空。"
    )
    selection_section = (
        f"按实际选股规则拆分的理论组合：入选组年化毛收益 "
        f"{diagnostics['selected_gross_annualized_return']:.2%}，未入选组 "
        f"{diagnostics['unselected_gross_annualized_return']:.2%}，入选减未入选的理论多空年化 "
        f"{diagnostics['selected_vs_unselected_annualized_spread']:.2%}。这项指标对应当前选股规则；"
        "仍未计交易费用、融券和成交约束。"
        if np.isfinite(diagnostics['selected_vs_unselected_annualized_spread'])
        else "当前规则没有稳定的未入选组，入选减未入选收益留空。"
    )
    report = f"""# 因子研究报告：{factor.name}

因子 `{factor.name}`，组合 `{config.portfolio.selection}`，每 {config.rebalance_sessions} 个交易日调仓；{direction_label}。回测 {config.start} 至 {config.end}，收盘生成信号、次日开盘撮合。有效完整前瞻窗口 {diagnostics['complete_forward_count']} / {diagnostics['signal_count']}。

## Factor Quality

| IC Mean | Rank IC Mean | ICIR | Rank ICIR | IC 胜率 | Rank IC 胜率 | IC t-stat | Rank IC t-stat |
|---:|---:|---:|---:|---:|---:|---:|---:|
| {diagnostics['mean_ic']:.4f} | {diagnostics['mean_rank_ic']:.4f} | {diagnostics['icir']:.3f} | {diagnostics['rank_icir']:.3f} | {diagnostics['ic_win_rate']:.1%} | {diagnostics['rank_ic_win_rate']:.1%} | {diagnostics['ic_t_stat']:.2f} | {diagnostics['rank_ic_t_stat']:.2f} |

ICIR = 每期 IC 均值 / 样本标准差；胜率按预期方向判断。t-stat 使用 Bartlett HAC 标准误，滞后阶数 {diagnostics['ic_t_stat_hac_lags']}，仅描述样本内均值偏离零的程度。

## Group Returns

| 分组（低值→高值） | 累计毛收益 | 年化毛收益 |
|---|---:|---:|
{group_rows}

Top-Bottom 理论多空年化 {diagnostics['top_bottom_annualized_return']:.2%}；逐期分组单调性均值 {diagnostics['group_monotonicity_mean']:.3f}，方向正确占比 {diagnostics['group_monotonicity_win_rate']:.1%}。多空以 100% 多头和 100% 空头的日收益差复利，不计融券、费用及成交约束。

{selection_section}

## Portfolio Performance / Risk

扣费后年化 {metrics['net_annualized_return']:.2%}，累计 {metrics['total_return']:.2%}，夏普 {metrics['sharpe_ratio_rf0']:.2f}，最大回撤 {metrics['max_drawdown']:.2%}，Calmar {metrics['calmar_ratio']:.2f}，年化波动 {metrics['annualized_volatility']:.2%}，盈利交易日比例 {metrics['daily_win_rate']:.1%}。期末净值 ¥{metrics['final_nav']:,.0f}；残余持仓市值 ¥{metrics['ending_market_value']:,.0f}。

{benchmark_section}

ST 状态日均覆盖：官方 {metrics['st_official_coverage']:.1%}、价格行为推断 {metrics['st_inferred_coverage']:.1%}、未知 {metrics['st_unknown_fraction']:.1%}；期末未知比例 {metrics['st_unknown_last_day']:.1%}。未知证券仍按当前宽松股票池参与，不能视作已确认的非 ST。2026-07-06 后新出现的主板 ST 无法由原 ±5% 特征识别。逐日和逐股明细见 `status_coverage_daily.csv`、`status_coverage_by_symbol.csv`。

## Trading

年化双边成交额 / 平均净值 {metrics['annualized_turnover']:.2f}x；佣金 ¥{metrics['commission_paid']:,.0f}，印花税 ¥{metrics['stamp_duty_paid']:,.0f}，显式费用合计 ¥{metrics['explicit_fees_paid']:,.0f}。滑点参数 {metrics['configured_slippage_bps']:g} bp 已反映在成交价，不包含在上述显式费用中。容量受限的委托尝试 {metrics['capacity_limited_order_attempts']:,} 次，占 {metrics['capacity_limited_order_fraction']:.2%}；完整成交率 {metrics['order_completion_rate']:.1%}。科创板最低 200 股、北交所最低 100 股且均可逐股递增；其他证券按配置整手。

## 年度观察

| 年份 | 实际覆盖日期 | 扣费后区间收益 | 平均 Rank IC | IC 信号数 |
|---|---|---:|---:|---:|
{annual_rows}

![净值与分层](factor_report.png)

![滚动 Rank IC](rolling_rank_ic.png)

IC 以信号日复权收盘到 {config.forward_sessions} 个交易日后复权收盘收益计算，不代表可交易收益；滚动 Rank IC 使用过去 {config.rolling_window_sessions} 个交易日内的有效信号。分层曲线不计费用、容量、涨跌停和整手限制；组合净值使用撮合引擎。课程数据缺少官方历史 ST 名单，若状态来源为价格行为推断，存在漏判和误判；日线不能证明开盘真实成交。期末若有未能卖出的持仓，仍按市值计入净值。详见 `run.json`、`trades.csv` 和 `trade_summary.csv`。
"""
    (output / "FACTOR_REPORT.md").write_text(report, encoding="utf-8")
    return _json_safe({"metrics": metrics, "diagnostics": diagnostics,
                       "benchmark": benchmark, "output": str(output.resolve())})


def load_research_data(source: str | Path, config: ResearchConfig) -> MarketData:
    """Keep all prior rows for point-in-time ST inference and factor warmup."""
    data = MarketData.load(source)
    observed_unknown = ((np.isfinite(data.values["close"]) & (data.values["close"] > 0))
                        & ~np.isfinite(data.values["is_st"]))
    if config.infer_st_when_missing and observed_unknown.any():
        infer_st_from_price_limits(data)
    start = int(data.dates.get_indexer([pd.Timestamp(config.start)])[0])
    end = int(data.dates.get_indexer([pd.Timestamp(config.end)])[0])
    anchor = int(data.dates.get_indexer([pd.Timestamp(config.signal_anchor)])[0])
    if min(start, end, anchor) < 0 or anchor >= start or start > end:
        raise ValueError("回测日期或初始信号日期不在数据中，或顺序无效")
    values = {name: matrix[:end + 1] for name, matrix in data.values.items()}
    result = MarketData(data.dates[:end + 1], data.symbols, values, start, data.status_source)
    result.source_files = source_manifest(source)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="可配置 A 股因子回测与 IC/分层研究")
    parser.add_argument("--data", default="feathers", help="Feather 目录或课程 feathers.zip")
    parser.add_argument("--config", default="factor_config.json", help="研究配置 JSON")
    parser.add_argument("--output", default="runs/factor", help="结果目录")
    parser.add_argument("--factor", help="覆盖配置中的内置因子名称")
    parser.add_argument("--factor-file", help="覆盖配置中的自定义 Factor Python 文件")
    parser.add_argument("--benchmark-nav", help="可选：含 date,nav 列的基准净值 CSV")
    parser.add_argument("--direction", type=int, choices=(-1, 1), help="因子预期方向：+1 高值更好，-1 低值更好")
    args = parser.parse_args()
    config = ResearchConfig.from_json(args.config)
    if args.factor or args.factor_file or args.benchmark_nav or args.direction:
        from dataclasses import replace
        config = replace(config, factor=args.factor or config.factor,
                         factor_file=args.factor_file if args.factor_file else
                         (None if args.factor else config.factor_file),
                         benchmark_nav=args.benchmark_nav or config.benchmark_nav,
                         expected_direction=args.direction or config.expected_direction)
    result = run_factor_research(load_research_data(args.data, config), config, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

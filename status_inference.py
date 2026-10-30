"""Point-in-time ST status proxy for course data without official status history."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantbt import MarketData


def _round_cent(values: np.ndarray) -> np.ndarray:
    return np.floor(values * 100.0 + 0.5 + 1e-9) / 100.0


def infer_st_from_price_limits(data: MarketData, rolling_days: int = 60,
                               hits_required: int = 2) -> tuple[np.ndarray, dict[str, object]]:
    """Point-in-time heuristic: two sealed 5% limit days within 60 sessions flag a symbol.

    It is deliberately labeled an inference, not an official ST history. Detection only
    uses the current and prior rows; once inferred, the flag persists through this sample.
    """
    values = data.values
    dates, symbols = data.dates, data.symbols
    n_days, n_symbols = values["close"].shape
    codes = ["".join(ch for ch in str(symbol).split(".")[0] if ch.isdigit()) for symbol in symbols]
    main_board = np.array([code.startswith(("000", "001", "002", "003", "600", "601", "603", "605"))
                           for code in codes], dtype=bool)
    chinext = np.array([code.startswith(("300", "301")) for code in codes], dtype=bool)
    known_board_5pct_period = dates < pd.Timestamp("2026-07-06")

    detected = np.zeros((n_days, n_symbols), dtype=bool)
    proxy_covered = np.zeros((n_days, n_symbols), dtype=bool)
    ring = np.zeros((rolling_days, n_symbols), dtype=bool)
    recent_hits = np.zeros(n_symbols, dtype=np.int16)
    persistent = np.zeros(n_symbols, dtype=bool)
    events_total = 0
    for day in range(1, n_days):
        previous_close = values["close"][day - 1]
        previous_factor = values["cumadj"][day - 1]
        today_factor = values["cumadj"][day]
        ratio = np.ones(n_symbols, dtype=np.float64)
        valid_factor = (np.isfinite(previous_factor) & (previous_factor > 0)
                        & np.isfinite(today_factor) & (today_factor > 0))
        ratio[valid_factor] = previous_factor[valid_factor] / today_factor[valid_factor]
        reference = previous_close * ratio
        upper = _round_cent(reference * 1.05)
        lower = _round_cent(reference * 0.95)
        op, high, low, close = (values[name][day] for name in ("open", "high", "low", "close"))
        liquid = (np.isfinite(values["amt"][day]) & (values["amt"][day] > 0)
                  & np.isfinite(values["num_trades"][day]) & (values["num_trades"][day] > 0))
        # ST 5% behavior can be inferred on the main board; ChiNext's 10% regime
        # before its 2020 reform is included. The post-2026 10% ST regime is excluded.
        eligible_board = main_board | (chinext & (dates[day] < pd.Timestamp("2020-08-24")))
        eligible = eligible_board & known_board_5pct_period[day] & valid_factor & liquid
        at_up = (np.abs(op - upper) <= 0.0051) & (np.abs(high - upper) <= 0.0051) \
            & (np.abs(low - upper) <= 0.0051) & (np.abs(close - upper) <= 0.0051)
        at_down = (np.abs(op - lower) <= 0.0051) & (np.abs(high - lower) <= 0.0051) \
            & (np.abs(low - lower) <= 0.0051) & (np.abs(close - lower) <= 0.0051)
        event = eligible & np.isfinite(reference) & (reference > 0) & (at_up | at_down)
        events_total += int(event.sum())
        slot = day % rolling_days
        recent_hits -= ring[slot]
        ring[slot] = event
        recent_hits += event
        persistent |= recent_hits >= hits_required
        detected[day] = persistent
        # Once the 5% main-board regime ended, an undetected main-board name
        # cannot be classified from this price pattern. Previously inferred
        # ST names remain flagged, with inferred (not official) provenance.
        proxy_covered[day] = eligible | persistent

    observed = np.isfinite(values["close"]) & (values["close"] > 0)
    official = values.get("is_st", np.full((n_days, n_symbols), np.nan))
    prior_origin = values.get("st_status_origin")
    if prior_origin is None:
        prior_origin = np.where(np.isfinite(official) & (data.status_source != "missing"), 1.0, 0.0)
    official_known = observed & (prior_origin == 1.0) & np.isfinite(official)
    inferred_known = observed & ~official_known & proxy_covered
    status = np.full((n_days, n_symbols), np.nan, dtype=np.float64)
    status[official_known] = official[official_known]
    status[inferred_known] = detected[inferred_known].astype(np.float64)
    origin = np.zeros((n_days, n_symbols), dtype=np.float64)
    origin[official_known] = 1.0
    origin[inferred_known] = 2.0
    info = {
        "method": "two sealed +/-5% one-price sessions within 60 trading days; point-in-time; proxy unavailable for newly flagged main-board ST after 2026-07-06",
        "price_limit_events": events_total,
        "inferred_symbols": int(detected.any(axis=0).sum()),
        "inferred_symbol_days": int(detected.sum()),
        "official_status_source": data.status_source,
        "is_official_history": bool(official_known.any()),
        "observed_symbol_days": int(observed.sum()),
        "official_covered_symbol_days": int(official_known.sum()),
        "inferred_covered_symbol_days": int(inferred_known.sum()),
        "unknown_symbol_days": int((observed & ~(official_known | inferred_known)).sum()),
    }
    data.values["is_st"] = status
    data.values["st_status_origin"] = origin
    data.status_source = "official_plus_5pct_price_inference" if info["is_official_history"] \
        else "5pct_price_inference_2hits_60d"
    return detected, info


def status_audit_frames(data: MarketData) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize official, inferred, and unknown status by date and security."""
    start = data.simulation_start
    close = data.values["close"][start:]
    observed = np.isfinite(close) & (close > 0)
    origin = data.values.get("st_status_origin")
    if origin is None:
        status = data.values["is_st"][start:]
        origin = np.where(np.isfinite(status),
                          1.0 if data.status_source in {"security_status.csv", "is_st.feather"} else 2.0,
                          0.0)
    else:
        origin = origin[start:]
    official = observed & (origin == 1.0)
    inferred = observed & (origin == 2.0)
    unknown = observed & ~(official | inferred)
    count = observed.sum(axis=1)
    denominator = np.maximum(count, 1)
    daily = pd.DataFrame({
        "date": data.dates[start:].strftime("%Y-%m-%d"),
        "observed_symbols": count,
        "official_symbols": official.sum(axis=1),
        "inferred_symbols": inferred.sum(axis=1),
        "unknown_symbols": unknown.sum(axis=1),
        "official_fraction": official.sum(axis=1) / denominator,
        "inferred_fraction": inferred.sum(axis=1) / denominator,
        "unknown_fraction": unknown.sum(axis=1) / denominator,
    })
    by_symbol = pd.DataFrame({
        "symbol": data.symbols.astype(str),
        "observed_days": observed.sum(axis=0),
        "official_days": official.sum(axis=0),
        "inferred_days": inferred.sum(axis=0),
        "unknown_days": unknown.sum(axis=0),
    })
    return daily, by_symbol


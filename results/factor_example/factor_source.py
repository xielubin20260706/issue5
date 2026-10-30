"""Independent example factor: 20-session return divided by daily volatility.

Run with: python3 factor_framework.py --data data --factor-file factor_example.py
The factor only uses history available at the signal close.
"""

import numpy as np


class Factor:
    name = "risk_adjusted_momentum_20d"
    lookback = 20

    def compute(self, ctx):
        adjusted_close = ctx.history("adjusted_close", 21)
        scores = np.full(len(ctx.symbols), np.nan, dtype=np.float64)
        if len(adjusted_close) < 21:
            return scores

        valid = np.isfinite(adjusted_close).all(axis=0) & (adjusted_close > 0).all(axis=0)
        if not valid.any():
            return scores
        history = adjusted_close[:, valid]
        daily_returns = history[1:] / history[:-1] - 1.0
        volatility = daily_returns.std(axis=0, ddof=1)
        twenty_day_return = history[-1] / history[0] - 1.0
        # A small volatility floor keeps nearly flat prices from exploding.
        scores[valid] = twenty_day_return / np.maximum(volatility, 0.005)
        return scores

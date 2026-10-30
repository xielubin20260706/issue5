"""Example custom factor; run with --factor-file factor_template.py.

Keep this file local and trusted: importing a Python factor executes its code.
"""

import numpy as np


class Factor:
    name = "my_5d_return"
    lookback = 5

    def compute(self, ctx):
        # history ends at ctx.date; custom code never needs future rows.
        adjusted = ctx.history("adjusted_close", 6)
        if len(adjusted) < 6:
            return np.full(len(ctx.symbols), np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            return adjusted[-1] / adjusted[0] - 1.0

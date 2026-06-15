import numpy as np
import pandas as pd


def load_demo_prices(rows: int = 260, seed: int = 7) -> pd.DataFrame:
    """Generate stable synthetic prices so examples and tests are reproducible."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=0.0004, scale=0.012, size=rows)
    prices = 100 * (1 + pd.Series(returns)).cumprod()
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=rows, freq="B"),
            "close": prices.round(4),
        }
    )


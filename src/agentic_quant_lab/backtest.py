from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    total_return: float
    max_drawdown: float
    volatility: float
    trades: int
    equity_curve: list[float]


def run_moving_average_backtest(
    prices: pd.DataFrame,
    cash: float,
    short_window: int = 10,
    long_window: int = 40,
) -> BacktestResult:
    if short_window >= long_window:
        raise ValueError("short_window must be smaller than long_window")
    if "close" not in prices:
        raise ValueError("prices must contain a close column")

    frame = prices.copy()
    frame["short_ma"] = frame["close"].rolling(short_window).mean()
    frame["long_ma"] = frame["close"].rolling(long_window).mean()
    frame["signal"] = (frame["short_ma"] > frame["long_ma"]).astype(int)
    frame["position"] = frame["signal"].shift(1).fillna(0)
    frame["asset_return"] = frame["close"].pct_change().fillna(0)
    frame["strategy_return"] = frame["position"] * frame["asset_return"]
    frame["equity"] = cash * (1 + frame["strategy_return"]).cumprod()

    peak = frame["equity"].cummax()
    drawdown = frame["equity"] / peak - 1
    trades = int(frame["position"].diff().abs().sum())

    return BacktestResult(
        total_return=float(frame["equity"].iloc[-1] / cash - 1),
        max_drawdown=float(drawdown.min()),
        volatility=float(frame["strategy_return"].std() * (252**0.5)),
        trades=trades,
        equity_curve=[round(float(value), 2) for value in frame["equity"].tail(20)],
    )


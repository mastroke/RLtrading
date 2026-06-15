from dataclasses import dataclass

from agentic_quant_lab.backtest import BacktestResult


@dataclass(frozen=True)
class RiskDecision:
    decision: str
    notes: list[str]


def evaluate_risk(
    result: BacktestResult,
    max_drawdown_limit: float = -0.12,
    volatility_limit: float = 0.35,
) -> RiskDecision:
    notes: list[str] = []

    if result.max_drawdown < max_drawdown_limit:
        notes.append(f"Rejected: max drawdown {result.max_drawdown:.2%} breached limit.")
    if result.volatility > volatility_limit:
        notes.append(f"Rejected: volatility {result.volatility:.2%} breached limit.")
    if result.trades == 0:
        notes.append("Rejected: no trades were generated.")

    if notes:
        return RiskDecision(decision="research_only", notes=notes)

    return RiskDecision(
        decision="paper_trade_only",
        notes=["Strategy passed drawdown, volatility and activity limits."],
    )


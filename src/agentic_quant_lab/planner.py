from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchPlan:
    symbol: str
    hypothesis: str
    strategy: str
    guardrails: list[str]


def build_research_plan(symbol: str) -> ResearchPlan:
    """Create a deterministic plan that an agent can hand to the simulator."""
    return ResearchPlan(
        symbol=symbol.upper(),
        hypothesis="Trend persistence can be tested with a short/long moving-average crossover.",
        strategy="moving_average_crossover",
        guardrails=[
            "Reject if max drawdown is worse than -12%.",
            "Reject if annualized volatility is above 35%.",
            "Treat all positive results as paper-trading candidates only.",
        ],
    )


from agentic_quant_lab.backtest import run_moving_average_backtest
from agentic_quant_lab.cli import build_report
from agentic_quant_lab.data import load_demo_prices
from agentic_quant_lab.risk import evaluate_risk


def test_backtest_generates_metrics() -> None:
    result = run_moving_average_backtest(load_demo_prices(), cash=10_000)

    assert isinstance(result.total_return, float)
    assert result.max_drawdown <= 0
    assert result.volatility >= 0
    assert len(result.equity_curve) == 20


def test_risk_decision_is_bounded() -> None:
    result = run_moving_average_backtest(load_demo_prices(), cash=10_000)
    decision = evaluate_risk(result)

    assert decision.decision in {"paper_trade_only", "research_only"}
    assert decision.notes


def test_report_is_reviewable() -> None:
    report = build_report(symbol="demo", cash=10_000)

    assert report["symbol"] == "DEMO"
    assert "risk_notes" in report
    assert report["decision"] in {"paper_trade_only", "research_only"}


"""Small live demonstration of M2A tools; no research-agent decisions."""

from __future__ import annotations

import argparse
from datetime import date, timedelta

from financial_research_agent.models import ResearchAction
from financial_research_agent.providers.yahoo_finance import YahooFinanceProvider
from financial_research_agent.tools.calculator import CalculatorTool
from financial_research_agent.tools.market_data import MarketDataTool


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect historical prices and simple statistics")
    parser.add_argument("--ticker", default="GOOGL")
    parser.add_argument("--benchmark", default="QQQ")
    parser.add_argument("--start", type=date.fromisoformat, default=date.today() - timedelta(days=45))
    parser.add_argument("--end", type=date.fromisoformat, default=date.today())
    args = parser.parse_args()

    market = MarketDataTool(YahooFinanceProvider())
    stock = market.get_history(
        action_id="demo-stock",
        ticker=args.ticker,
        start_date=args.start,
        end_date=args.end,
    )
    if not stock.success:
        print(f"Market data failed: {stock.error.code}: {stock.error.message}")
        return 1
    bars = stock.data["bars"]
    print(
        f"{stock.data['ticker']} daily prices: {bars[0]['trading_date']} to "
        f"{bars[-1]['trading_date']} ({len(bars)} sessions)"
    )

    calculator = CalculatorTool()
    price_return = calculator.execute(
        ResearchAction(
            action_id="demo-return",
            tool_name="calculator",
            purpose="Calculate adjusted closing-price return",
            arguments={
                "operation": "percentage_return",
                "starting_price": bars[0]["adjusted_close"],
                "ending_price": bars[-1]["adjusted_close"],
                "price_basis": "adjusted_close",
                "effective_start_date": bars[0]["trading_date"],
                "effective_end_date": bars[-1]["trading_date"],
                "source_result_ids": [stock.result_id],
            },
        )
    )
    if price_return.success:
        print(f"Adjusted closing-price return: {price_return.data['value']:.2f}%")

    volatility = calculator.execute(
        ResearchAction(
            action_id="demo-volatility",
            tool_name="calculator",
            purpose="Calculate historical volatility",
            arguments={
                "operation": "historical_volatility",
                "prices": [bar["adjusted_close"] for bar in bars],
                "price_basis": "adjusted_close",
                "effective_start_date": bars[0]["trading_date"],
                "effective_end_date": bars[-1]["trading_date"],
                "source_result_ids": [stock.result_id],
            },
        )
    )
    if volatility.success:
        print(f"Annualized daily-return volatility: {volatility.data['value']:.2f}%")
    else:
        print(f"Volatility unavailable: {volatility.error.message}")

    comparison = market.compare_performance(
        action_id="demo-benchmark",
        ticker=args.ticker,
        benchmark=args.benchmark,
        start_date=args.start,
        end_date=args.end,
    )
    if not comparison.success:
        print(f"Benchmark comparison failed: {comparison.error.code}: {comparison.error.message}")
        return 1
    data = comparison.data
    print(
        f"{data['stock_ticker']} vs {data['benchmark_ticker']} on common sessions "
        f"{data['effective_start_date']} to {data['effective_end_date']}: "
        f"{data['stock_return_pct']:+.2f}% vs {data['benchmark_return_pct']:+.2f}% "
        f"({data['relative_performance_percentage_points']:+.2f} percentage points)"
    )
    print(f"Sources: {stock.sources[0].url}; {comparison.sources[1].url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

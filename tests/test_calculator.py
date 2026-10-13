import math
from datetime import date

import pytest

from financial_research_agent.models import ResearchAction
from financial_research_agent.tools.calculator import (
    CalculationError,
    CalculationErrorCode,
    CalculatorTool,
    historical_volatility,
    moving_average,
    percentage_return,
    relative_performance,
    return_correlation,
    volume_change,
)


def test_percentage_return_and_relative_performance_units() -> None:
    stock = percentage_return(100, 90)
    benchmark = percentage_return(100, 95)

    assert stock == pytest.approx(-10)
    assert benchmark == pytest.approx(-5)
    assert relative_performance(stock, benchmark) == pytest.approx(-5)


@pytest.mark.parametrize("start,end", [(0, 10), (-1, 10), (float("nan"), 10), (10, float("inf"))])
def test_invalid_price_inputs_are_rejected(start: float, end: float) -> None:
    with pytest.raises(CalculationError) as error:
        percentage_return(start, end)
    assert error.value.code is CalculationErrorCode.INVALID_INPUT


def test_nonfinite_calculation_output_is_rejected() -> None:
    with pytest.raises(CalculationError) as error:
        percentage_return(1e-300, 1e300)
    assert error.value.code is CalculationErrorCode.INVALID_RESULT


def test_moving_average_requires_a_complete_positive_window() -> None:
    assert moving_average([10, 20, 30, 40], 2) == pytest.approx([15, 25, 35])
    with pytest.raises(CalculationError) as error:
        moving_average([10, 20], 3)
    assert error.value.code is CalculationErrorCode.INSUFFICIENT_DATA
    with pytest.raises(CalculationError):
        moving_average([10, 20], 0)


def test_historical_volatility_uses_sample_standard_deviation() -> None:
    # Adjacent simple returns are +10% and -10%.
    prices = [100.0, 110.0, 99.0]
    daily = historical_volatility(prices, annualize=False)
    annualized = historical_volatility(prices, trading_periods_per_year=252)

    assert daily == pytest.approx(math.sqrt(0.02) * 100)
    assert annualized == pytest.approx(daily * math.sqrt(252))
    with pytest.raises(CalculationError) as error:
        historical_volatility([100, 101])
    assert error.value.code is CalculationErrorCode.INSUFFICIENT_DATA


def test_volume_change_uses_explicit_mean_baseline() -> None:
    assert volume_change(150, [100, 200]) == pytest.approx(0)
    assert volume_change(300, [100, 200]) == pytest.approx(100)
    with pytest.raises(CalculationError):
        volume_change(100, [0, 0])
    with pytest.raises(CalculationError):
        volume_change(100, [])


def test_correlation_aligns_common_dates_before_computing_returns() -> None:
    d1, d2, d3, d4, d5 = [date(2025, 1, day) for day in (2, 3, 6, 7, 8)]
    stock = {d1: 100, d2: 900, d3: 110, d4: 99}
    benchmark = {d1: 200, d3: 220, d4: 198, d5: 1000}

    result = return_correlation(stock, benchmark)

    assert result.correlation == pytest.approx(1)
    assert result.effective_start_date == d1
    assert result.effective_end_date == d4
    assert result.common_session_count == 3
    assert result.paired_return_count == 2


def test_correlation_rejects_short_or_constant_series() -> None:
    d1, d2, d3 = date(2025, 1, 2), date(2025, 1, 3), date(2025, 1, 6)
    with pytest.raises(CalculationError) as error:
        return_correlation({d1: 100, d2: 110}, {d1: 200, d2: 220})
    assert error.value.code is CalculationErrorCode.INSUFFICIENT_DATA
    with pytest.raises(CalculationError) as error:
        return_correlation(
            {d1: 100, d2: 100, d3: 100}, {d1: 200, d2: 210, d3: 220}
        )
    assert error.value.code is CalculationErrorCode.CONSTANT_SERIES


def test_calculator_tool_returns_m1_tool_result() -> None:
    action = ResearchAction(
        action_id="A1",
        tool_name="calculator",
        purpose="Measure the decline",
        arguments={
            "operation": "percentage_return",
            "starting_price": 100,
            "ending_price": 90,
            "price_basis": "adjusted_close",
            "effective_start_date": "2025-01-02",
            "effective_end_date": "2025-01-06",
            "source_result_ids": ["R1"],
        },
    )
    result = CalculatorTool().execute(action)

    assert result.success
    assert result.data["value"] == pytest.approx(-10)
    assert result.data["unit"] == "percent"
    assert result.data["inputs"]["starting_price"] == 100
    assert result.data["inputs"]["price_basis"] == "adjusted_close"
    assert result.data["inputs"]["effective_start_date"] == "2025-01-02"
    assert result.data["inputs"]["source_result_ids"] == ["R1"]
    assert result.metadata.api_call_count == 0
    assert result.model_validate_json(result.model_dump_json()) == result


def test_calculator_tool_rejects_unknown_operation_and_zero_denominator() -> None:
    unknown = ResearchAction(
        action_id="A1",
        tool_name="calculator",
        purpose="Unknown",
        arguments={"operation": "run_python", "code": "print(1)"},
    )
    result = CalculatorTool().execute(unknown)
    assert result.error.code == "INVALID_ARGUMENTS"

    invalid = ResearchAction(
        action_id="A2",
        tool_name="calculator",
        purpose="Invalid denominator",
        arguments={"operation": "percentage_return", "starting_price": 0, "ending_price": 90},
    )
    result = CalculatorTool().execute(invalid)
    assert result.error.code == "INVALID_INPUT"


def test_calculator_tool_correlation_accepts_json_date_keys() -> None:
    action = ResearchAction(
        action_id="A1",
        tool_name="calculator",
        purpose="Measure correlation",
        arguments={
            "operation": "return_correlation",
            "stock_prices": {"2025-01-02": 100, "2025-01-03": 110, "2025-01-06": 99},
            "benchmark_prices": {"2025-01-02": 200, "2025-01-03": 220, "2025-01-06": 198},
        },
    )
    result = CalculatorTool().execute(action)
    assert result.success
    assert result.data["correlation"] == pytest.approx(1)
    assert result.data["effective_end_date"] == "2025-01-06"


@pytest.mark.parametrize(
    "arguments,key,expected",
    [
        (
            {"operation": "relative_performance", "stock_return_pct": 5, "benchmark_return_pct": 2},
            "value",
            3,
        ),
        ({"operation": "moving_average", "prices": [10, 20, 30], "window": 2}, "values", [15, 25]),
        (
            {"operation": "historical_volatility", "prices": [100, 110, 99], "annualize": False},
            "value",
            math.sqrt(0.02) * 100,
        ),
        (
            {"operation": "volume_change", "current_volume": 300, "baseline_volumes": [100, 200]},
            "value",
            100,
        ),
    ],
)
def test_calculator_tool_dispatches_supported_operations(
    arguments: dict, key: str, expected: float | list[float]
) -> None:
    action = ResearchAction(
        action_id="A1", tool_name="calculator", purpose="Calculate", arguments=arguments
    )
    result = CalculatorTool().execute(action)
    assert result.success
    assert result.data[key] == pytest.approx(expected)

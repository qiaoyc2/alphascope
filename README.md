# AlphaScope

AlphaScope is a Python project for an autonomous financial research agent that will investigate stock questions using traceable evidence. The current milestone provides validated research state, a free historical market-data adapter, and deterministic calculations; agent planning and causal research will follow later.

Requires Python 3.11 or newer.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest
```

Milestone 2A has no required environment variables. To try the live market-data tools:

```bash
.venv/bin/python examples/market_data_demo.py --ticker GOOGL --benchmark QQQ
```

Dates passed with `--start` and `--end` are inclusive. Returns use adjusted close by default; raw close remains separately available. Yahoo data accessed through yfinance is subject to Yahoo's usage terms and may be unavailable or incomplete.

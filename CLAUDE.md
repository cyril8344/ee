# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

XAU/USD (Gold) scalping bot trading London (8–12h CET) and NY (14–18h CET) sessions only. Multi-timeframe strategy: H1 bias → M15 confirmation → M5 entry. Deployed on Railway; frontend served via nginx inside the same Docker build.

## Commands

```bash
# Install everything
make install          # pip install -r requirements.txt + npm install in frontend/

# Run locally (two terminals)
make backend          # uvicorn backend/main:app --reload --port 8000
make frontend         # vite dev server on :5173, proxies /api and /ws to :8000

# Tests (always use synthetic data, no network needed)
make test             # pytest backend/tests -q
XAU_DATA_PROVIDER=synthetic python -m pytest backend/tests/test_strategy.py -q  # single file
XAU_DATA_PROVIDER=synthetic python -m pytest backend/tests -k "test_risk" -q     # single test

# Docker (full stack)
make docker-up        # builds and starts backend + frontend via docker-compose
make docker-down

# Utilities
make clean            # removes __pycache__, .pyc, SQLite DB
python backend/news_filter.py   # print today's economic calendar
python backend/backtest.py      # quick 45-day backtest with summary
```

## Architecture

### Signal Pipeline (`strategy.py → evaluate()`)

The 10-stage filter runs in strict order — a rejection at any stage short-circuits the rest:

1. Bad timing (Mon < 10h, Fri > 16h CET)
2. Session gate (London 8–12h, NY 14–18h CET)
3. H1 EMA200 bias — NEUTRAL if EMA50 and EMA200 disagree
4. M15 EMA9/21 trend + RSI 35–65
5. M5 ATR ≥ 3.0 (volatility gate)
6. H1 ADX ≥ 25 (trend strength)
7. M5 EMA9 alignment (adaptive tolerance)
8. M5 RSI momentum (LONG > 45, SHORT < 55)
9. Candle patterns (≥ 2 patterns, each weight ≥ 0.60, sum ≥ 1.0)
10. ML Gate — logistic regression, activates after 15 trades

### ML Gate (`ml_gate.py`)

Online logistic regression with 6 features:

| Feature | Encoding |
|---------|---------|
| `atr_norm` | ATR / price |
| `rsi_norm` | (RSI−50) / 50 → [−1, +1] |
| `ema200_bias` | +1 LONG / −1 SHORT |
| `pattern_w_norm` | avg pattern weight / 2 |
| `adx_norm` | ADX H1 / 50 |
| `session_enc` | London=1.0, NY=0.5 |

Entry threshold: 0.55 (boosted when on a losing streak). **Reset ML weights (pass `reset=True`) whenever filters or features change.**

### Trade Management

- TP1 = 0.7R → exits 50% of position (pas de déplacement SL après TP1)
- TP2 = 1.4R → exits remaining 50%
- SL = 1.4 × ATR; pas de déplacement après TP1 — SL reste au niveau initial; timeout at 45 minutes
- Risk: 1% capital per trade (configurable), max 4 trades/day, daily stop at −2%

### Data Flow

```
data_provider.py  →  broker.py (M5 OHLCV, yfinance GC=F)
                  →  strategy.py (multi-TF indicators)
                  →  ml_gate.py (online learning)
                  →  risk_manager.py (position size)
                  →  broker.py (PaperBroker or MT5Broker)
                  →  database.py (SQLite)
                  →  main.py (WebSocket broadcast)
```

`data_provider.py` tries providers in order: Twelve Data → Polygon → Alpha Vantage → yfinance → synthetic fallback. Tests always force `XAU_DATA_PROVIDER=synthetic` (set in `conftest.py`).

### Pre-training (`pretrain.py`)

Bar-by-bar historical replay that trains the ML gate offline before live trading. Runs **without** the ML gate (so win-rate reported is the raw signal quality). Uses realistic lot sizing matching the live formula: `volume = capital × risk_pct% / (SL_dist × contract_size)`. Supports `strategy_mode="A"` (EMA/pattern) and `strategy_mode="B"` (ICT) — the dashboard automatically passes the current active strategy. Always re-run with `reset=True` after any strategy or feature change.

### Backend Entry Point (`main.py`)

FastAPI app with a background asyncio trading loop. Key endpoints:

| Endpoint | Purpose |
|----------|---------|
| `GET /api/state` | current bot state snapshot |
| `GET /api/chart?tf=M5\|M15\|H1` | OHLCV + indicator data |
| `GET /api/trades?scope=today\|all` | trade history + equity curve |
| `GET/POST /api/settings` | read / update bot config (stored in SQLite key-value) |
| `POST /api/bot/toggle` | pause / resume |
| `POST /api/mode` | switch paper ↔ live (requires double confirmation) |
| `POST /api/backtest` | trigger backtest run |
| `WebSocket /ws` | real-time state stream to dashboard |

### Frontend (`frontend/src/`)

- **Dashboard.jsx** — the only active page: live bot status, EMA chart (lightweight-charts), RSI/ATR gauges, news countdown, active trade, trade history, equity curve, settings panel, pretrain panel
- **BacktestPanel.jsx** — kept in code but **removed from navigation**
- **LoginPage.jsx** — JWT auth (token stored in localStorage)
- Vite dev server proxies `/api` and `/ws` to `:8000`; production nginx does the same

### Deployment

Railway auto-deploys from `main` via nixpacks. Build: Python 3.12 venv + `npm run build` in `frontend/`. Start: `uvicorn --app-dir backend main:app`. Frontend static files are served by nginx inside the frontend container; nginx also reverse-proxies `/api/` and `/ws` to the backend container.

After merging to `main`:
1. Wait ~2–3 min for Railway deploy
2. Re-run pretrain (with `reset=True`) if filters or ML features changed
3. Monitor WR/PF in the "Statut bot" panel

## Key Architecture Decisions

- **BacktestPanel removed from nav** — only the pretrain panel is exposed in the dashboard
- **Synthetic data** uses `vol=0.0004` (realistic for XAU/USD) — avoid drawing conclusions from synthetic backtest results
- **Volume filter removed** — unreliable across data sources
- **RSI M15 history**: 33/67 (original) → 40/60 (broke everything) → 35/65 → **40/60 (current)** — retesté avec RSI M5 45/55, résultats à valider
- **Pattern floor 0.65** blocks patterns that lose 65%+ of the time (was 0.60)
- **TREND_BIAS_DISTANCE = 0.5 ATR H1** blocks SHORT when price > EMA200 + 0.5×ATR and LONG when price < EMA200 − 0.5×ATR
- **EMA200_MIN_DIST = 0.3 ATR H1** requires price to be ≥ 0.3×ATR on the correct side of EMA200 (filters ambiguous zone — SL direct avg dist was 0.03 vs TP2 avg −0.5)
- **BAD_HOURS_CET = {10}** blocks 10h00-10h59 CET (London) — WR 38% over 37 trades in 6M backtest
- **ADX SHORT minimum = 30** (ADX_MIN + 5 = 25+5) vs 25 for LONG — stricter filter against shorting in uptrend (was mistakenly set to ADX_MIN+13=38, fixed)
- **MAX_TRADE_MINUTES = 45** (was 30) — more time for TP targets to be reached
- **TP1 = 0.7R**, **TP2 = 1.4R** — gap TP1→TP2 = 0.7R; TP1=1.0R testé mais a empiré PF (1.07→0.95) car trop difficile à atteindre
- **Pas de déplacement SL après TP1** — l'or pullback régulièrement sous l'entrée après TP1. Le SL reste au niveau initial (−1.4R). Pire cas après TP1=0.7R : +0.7×50% − 1.4×50% = −0.35R net.
- **ML Gate: 3 → 6 features** (June 2026) — ML weights must be reset after any feature count change
- **Strategy B (ICT)** selectable via `strategy` setting ("A" or "B"); "A" is default EMA/pattern strategy, "B" is SMC/ICT via `strategy_ict.py`
- `MT5Broker` in `broker.py` requires MetaTrader5 (Windows only, manual install); `PaperBroker` is the default everywhere else

## Secrets — Never Commit

Store only in Railway Variables (not in `.env` files committed to git):

- `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID`
- `FRED_API_KEY`
- `ADMIN_USERNAME` / `ADMIN_PASSWORD`
- `JWT_SECRET`

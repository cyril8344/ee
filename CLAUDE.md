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
5. M5 ATR ≥ 2.5 (volatility gate)
6. H1 ADX ≥ 20 (trend strength — LONG et SHORT identique)
7. M5 EMA9 alignment (adaptive tolerance)
8. M5 RSI momentum (LONG > 45, SHORT < 55)
9. VWAP alignment (close ≥ VWAP for LONG, ≤ VWAP for SHORT)
10. Candle patterns — soit 1 pattern fort (ancre, weight ≥ 0.85) soit 2+ patterns (sum ≥ 1.0 LONG / 1.5 SHORT) — ancre (ema9_pullback ou micro_breakout) toujours requise
11. ML Gate — logistic regression, activates after 15 trades

### ML Gate (`ml_gate.py`)

Online logistic regression with 8 features:

| Feature | Encoding |
|---------|---------|
| `atr_norm` | ATR / price |
| `rsi_norm` | (RSI−50) / 50 → [−1, +1] |
| `ema200_bias` | +1 LONG / −1 SHORT |
| `pattern_w_norm` | avg pattern weight / 2 |
| `adx_norm` | ADX H1 / 50 |
| `session_enc` | London=1.0, NY=0.5 |
| `h1_rsi_norm` | (RSI H1 − 50) / 50 → [−1, +1] |
| `hour_in_session` | 0.0 (début session) → 1.0 (fin session) |

Entry threshold: 0.50 (boosted when on a losing streak). **Reset ML weights (pass `reset=True`) whenever filters or features change.**

### Trade Management

- TP1 = 0.7R → exits 50% of position (pas de déplacement SL après TP1)
- TP2 = 1.8R → exits remaining 50%
- SL = dernier swing low/high M5 (lookback=10), plafonné à 1.4×ATR; pas de déplacement après TP1; timeout at 45 minutes. **SL sous mèche pattern testé → rejeté** (PF 1.36→1.26, SL direct 34.5→36.2% — l'or chasse les mèches avant de partir)
- **Early exit à 15 min** : si MFE < 0.2R après 3 bougies M5, sortie au prix actuel. Convertit les −1.4R (trades sans conviction) en petites pertes ~−0.3R.
- Risk: 5% capital per trade (configurable), max 4 trades/day, daily stop at −2%

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
- **RSI M15** : symétrique 40/60 (directional 40/60 LONG>40,SHORT<60 testé → WR 49%→40.4%, PF 1.20→0.96, rejeté)
- **RSI M5** : 45/55 (momentum minimal requis)
- **Pattern floor 0.67** blocks patterns that lose 67%+ of the time (was 0.65 → 0.67)
- **TREND_BIAS_DISTANCE = 0.5 ATR H1** blocks SHORT when price > EMA200 + 0.5×ATR and LONG when price < EMA200 − 0.5×ATR
- **EMA200_MIN_DIST supprimé** : entrée AT EMA200 valide en scalp M5 avec pattern + VWAP
- **BAD_HOURS_CET = {8, 10}** : 8h London open (manipulation pre-session) + 10h CET (WR 38% / 37 trades)
- **ADX_MIN = 20** LONG et SHORT identique (était 25 LONG / 35 SHORT — trop restrictif, ne discrimine pas SL vs TP2)
- **Mode momentum fort supprimé** : ADX H1 > 35/40 → 1 pattern testé → PF 1.34 vs 1.42, rejeté. Toujours 2 patterns requis.
- **MAX_TRADE_MINUTES = 45** (was 30) — more time for TP targets to be reached
- **TP1 = 0.7R**, **TP2 = 1.8R** — gap TP1→TP2 = 1.1R; TP2=1.4R testé mais moins bon, 1.8R optimal confirmé
- **SL → entrée (BE 0R) après TP1** — déplacé à l'entrée sur les bougies suivantes (pas de vérification intrabar). Pire cas : +0.7R×50% + 0×50% = +0.35R net. À comparer via pretrain avec "pas de déplacement" (−0.35R pire cas mais plus de trades TP2).
- **ML Gate: 8 features** (h1_rsi_norm + hour_in_session ajoutés en June 2026) — ML weights must be reset after any feature count change
- **Strategy B (EUR/USD) Order Block only** (June 2026) : biais H1 (EMA50 vs EMA200) + OB M5 non mitiguée + retest → TP1=0.7R, TP2=1.8R. Supprimé : AMD, FVG, Asian range, sweep, accumulation.
- **Strategy A (XAU/USD)** : EMA/patterns, toujours actif sur XAUUSD, non modifiable depuis le dashboard
- **Strategy B (EUR/USD)** : Order Block M5 via `strategy_ict.py`, toujours actif sur EURUSD, non modifiable depuis le dashboard
- `MT5Broker` in `broker.py` requires MetaTrader5 (Windows only, manual install); `PaperBroker` is the default everywhere else

## Règles anti-overfitting (OBLIGATOIRES)

Toute modification de paramètre stratégie (RSI, ATR, ADX, TREND_BIAS_DISTANCE, patterns, TP/SL…) **doit être validée en walk-forward avant merge**. Le pretrain in-sample seul ne prouve rien.

### Workflow obligatoire pour chaque changement

1. **Proposer** le changement avec une hypothèse claire ("RSI 48 → filtre les LONG à momentum faible")
2. **Tester en pretrain 6M** → noter PF et WR in-sample
3. **Lancer le walk-forward** (4 fenêtres × 1.5M) depuis le dashboard
4. **Critère de robustesse** : PF > 1.0 dans ≥ 75% des fenêtres ET `std_pf < 0.30`
5. **Merger uniquement si** OOS cohérent — un PF élevé in-sample avec variance inter-fenêtres élevée = curve-fitting, rejeter

### Règles absolues

- **Jamais de merge sur un résultat in-sample seul**, même si le PF est très bon (ex. PF 1.5 sur 6M peut être PF 0.8 en OOS)
- **Max 3 paramètres optimisés à la fois** — optimiser plus simultanément garantit le surapprentissage
- **Win rate in-sample > 58% sur 6M** = signal fort d'overfitting (la stratégie scalpe un régime particulier)
- **Optuna bayésien** (POST /api/optimize/bayesian) : utilise le walk-forward comme objectif → les paramètres trouvés sont validés OOS par construction
- **Après chaque changement de filtres ou features ML** : reset les poids ML (`reset=True`) et relancer pretrain

### Métriques cibles validées

| Métrique | Seuil acceptable | Seuil optimal |
|---------|-----------------|---------------|
| PF walk-forward (moy) | > 1.0 | > 1.15 |
| std_pf inter-fenêtres | < 0.30 | < 0.20 |
| % fenêtres rentables | ≥ 75% | 100% |
| SL direct | < 38% | < 32% |
| WR | 48–56% | 52–55% |

## Secrets — Never Commit

Store only in Railway Variables (not in `.env` files committed to git):

- `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID`
- `FRED_API_KEY`
- `ADMIN_USERNAME` / `ADMIN_PASSWORD`
- `JWT_SECRET`

# XAU/USD Scalping Bot — Guide Claude

## Architecture

```
backend/
  strategy.py      — Signal generation (evaluate, patterns, bias, filters)
  ml_gate.py       — Online logistic regression (6 features) + adaptive thresholds
  pretrain.py      — Historical replay engine (bar-by-bar, feeds ML systems)
  backtest.py      — Backtesting engine (BTTrade, _try_exit, _build_report)
  broker.py        — PaperBroker / live execution
  risk_manager.py  — Position sizing (1% risk), daily stop, max trades/day
  database.py      — SQLite persistence (trades, ML weights, pattern stats, equity)
  data_provider.py — Data sources (yfinance GC=F, synthetic fallback)
  main.py          — FastAPI app, WebSocket, trading loop, pretrain API
  news_filter.py   — Economic calendar filter (±30min around events)

frontend/
  src/Dashboard.jsx — React dashboard (statut bot + pré-entraînement uniquement)
  src/BacktestPanel.jsx — Removed from nav (kept in code but unused)
```

## Paramètres stratégie actuels (strategy.py)

| Paramètre | Valeur | Notes |
|-----------|--------|-------|
| ATR_MIN | 3.0 | Volatilité min M5 (XAU/USD ~$4150, ATR réel ~3.5$) |
| ADX_MIN | 25.0 | Force tendance H1 minimale |
| RSI_LOW | 35.0 | Borne basse M15 RSI (confirmation) |
| RSI_HIGH | 65.0 | Borne haute M15 RSI |
| RSI M5 LONG | > 45 | Momentum M5 (bloque si RSI < 45 pour LONG) |
| RSI M5 SHORT | < 55 | Momentum M5 (bloque si RSI > 55 pour SHORT) |
| SL_ATR_MULT | 1.2 | Stop loss = 1.2 × ATR |
| MAX_TRADE_MINUTES | 30 | Timeout trade |
| PATTERN_FLOOR | 0.60 | Poids pattern min (en dessous = ignoré) |
| Patterns requis | ≥ 2 | + sum(weights) ≥ 1.0 |

## ML Gate (ml_gate.py)

6 features :
1. `atr_norm` — ATR/prix (volatilité normalisée)
2. `rsi_norm` — (RSI-50)/50 centré [-1, +1]
3. `ema200_bias` — +1 LONG, -1 SHORT
4. `pattern_w_norm` — poids moyen patterns / 2 (qualité signal)
5. `adx_norm` — ADX H1 / 50 (force tendance)
6. `session_enc` — London=1.0, NY=0.5

Seuil entrée : 0.55 (+ streak boost si séries noires)
Activation : après 15 trades minimum

## Pipeline evaluate() (ordre des filtres)

1. Bad timing (lundi < 10h, vendredi > 16h CET)
2. Session gate (London 8-12h, NY 14-18h CET)
3. H1 EMA200 bias (NEUTRE si EMA50 et EMA200 en désaccord)
4. M15 EMA9/21 + RSI 35-65
5. M5 ATR ≥ 3.0
6. H1 ADX ≥ 25
7. M5 EMA9 alignement (tolérance adaptative)
8. M5 RSI momentum (LONG: >45, SHORT: <55)
9. Patterns chandelier (≥ 2, PATTERN_FLOOR ≥ 0.60, sum ≥ 1.0)
10. ML Gate (après 15 trades)

## Gestion des trades

- TP1 = 1R (sortie 60%), SL → breakeven
- TP2 = 2.5R (sortie 40% restant)
- Risque : 1% du capital par trade (configurable dashboard)
- Max 4 trades/jour, stop journalier -2%

## Performance cible (données réelles Railway)

| Métrique | Cible | Actuel (déc25-juin26) |
|----------|-------|----------------------|
| Trades/6 mois | 100-150 | ~127 |
| Win Rate | ≥ 28% | 26% |
| Profit Factor | ≥ 1.10 | 0.97 |
| Net PnL (1% risque) | +400$+ | ~-$210 |

## Pré-entraînement

- Toujours lancer avec `reset=True` après changement de filtres ou de features ML
- Le pré-entraînement tourne SANS ML gate → WR brut (live sera meilleur)
- Sizing fixe 0.01 lot dans pretrain → multiplier par ~24 pour estimer 1% risque réel

## Règles de sécurité ABSOLUES

**Ne jamais committer ces variables — uniquement dans Railway Variables :**
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`
- `FRED_API_KEY`
- `ADMIN_USERNAME` / `ADMIN_PASSWORD`
- `JWT_SECRET`

## Décisions architecturales importantes

- **Backtest tab supprimé** du dashboard — user utilise uniquement le panneau pré-entraînement
- **Données synthétiques** vol=0.0004 (réaliste) — éviter les conclusions du backtest synthétique
- **ML Gate 3→6 features** (juin 2026) — reset obligatoire après changement
- **RSI M15 : 33/67 original → 40/60 cassait tout → 35/65 actuel**
- **Volume filter supprimé** (peu fiable selon source données)
- **Pattern floor 0.60** = bloque les patterns perdant 70%+ du temps

## Déploiement

Railway auto-déploie depuis `main`. Après merge :
1. Attendre ~2-3 min le déploiement
2. Relancer le pré-entraînement si changement de filtres ou ML
3. Observer WR/PF dans panneau "Statut bot"

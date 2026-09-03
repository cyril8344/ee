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
6. H1 ADX ≥ 28 (trend strength — LONG et SHORT identique)
7. M5 EMA9 alignment (adaptive tolerance)
8. M5 RSI momentum (LONG > 49, SHORT < 57)
9. VWAP alignment (close ≥ VWAP for LONG, ≤ VWAP for SHORT)
10. Candle patterns — soit 1 pattern fort (ancre, weight ≥ 0.85) soit 2+ patterns (sum ≥ 1.0 LONG / 1.5 SHORT) — ancre (ema9_pullback ou micro_breakout) toujours requise

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
| `GET /api/trades/report?symbol=` | rapport historique filtré par symbole (XAUUSD/EURUSD) |
| `GET/POST /api/settings` | read / update bot config (stored in SQLite key-value) |
| `POST /api/bot/toggle` | pause / resume |
| `POST /api/mode` | switch paper ↔ live (requires double confirmation) |
| `POST /api/backtest` | trigger backtest run |
| `GET /api/report/weekly?week=0&symbol=` | rapport hebdomadaire (week=0 = cette sem., -1 = préc.) |
| `GET /api/report/monthly?month=0&symbol=` | rapport mensuel (month=0 = ce mois) |
| `GET /api/live-agent` | statut agent adaptatif live (RSI, ADX courants) |
| `POST /api/live-agent/params` | forcer des paramètres (annuler ajustement automatique) |
| `WebSocket /ws` | real-time state stream to dashboard |

### Frontend (`frontend/src/`)

- **Dashboard.jsx** — the only active page. Sections principales :
  - Graphique (candlesticks lightweight-charts + EMA9/21/200 + OB pour ICT)
  - Statut bot : RSI/ATR gauges, conditions ICT, agent adaptatif live (↩ Revenir en arrière)
  - Trade actif + historique du jour (tabs Tous/XAU/EUR)
  - Rapport historique (filtrable Tous/XAU/EUR)
  - Rapport hebdomadaire (navigation ← Préc. / Cette sem., filtre XAU/EUR, export PDF)
  - Rapport mensuel (navigation ← Préc. / Ce mois, filtre XAU/EUR, export PDF, breakdown par semaine)
  - Pretrain panel
  - Settings panel
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

- **LLM Gate supprimée** (juillet 2026) — ajoutait une latence API Anthropic externe dans la boucle de trading, pouvait bloquer silencieusement des signaux. Fichier `llm_gate.py` conservé mais non importé.
- **Agent adaptatif live — ajustement automatique DÉSACTIVÉ (août 2026)** (`live_agent.AUTO_ADJUST_ENABLED = False`). Il ajustait 4 paramètres tous les 10 trades sur un WR glissant de 20 trades — erreur-type ±11 points, donc il réagissait au bruit environ une évaluation sur cinq, et dépassait le maximum de 3 paramètres simultanés fixé par les règles anti-overfitting ci-dessous, sans aucune validation walk-forward. Surtout, son `_load()` retenait la valeur **la plus permissive** entre la valeur sauvegardée et le défaut module (`min()` pour les seuils plancher) : un desserrage était donc **irréversible**, aucun redémarrage ne le rattrapait. Effet constaté — `strategy.ADX_MIN` bloqué à **20** pendant des semaines alors que le code déclarait 28, sans alerte (`_near_bound` ne se déclenche qu'à 18). Bilan : un seul ajustement en plus d'un mois, pour cette divergence. L'agent ne fait plus que compter les trades et gérer la sortie de `BOOTSTRAP_MODE` ; il n'écrit plus jamais dans `strategy` de lui-même. Le forçage manuel (`force_params`, dashboard) reste actif — action humaine explicite et tracée — mais **ne survit pas à un redéploiement** : pour rendre une valeur permanente, l'écrire dans `strategy.py`.
- **Agents autonomes désactivés** (juillet 2026) — `researcher_agent.py` (grid-search RSI/ADX sur 1 mois de backtest **in-sample seul**, appliqué au live toutes les 4h sans walk-forward — violait directement les règles anti-overfitting ci-dessous) et `adaptive_agent.py` (LLM Claude Haiku en boucle toutes les 6h, faisait doublon avec `live_agent.py` sur les mêmes paramètres, avec des bornes différentes — même risque que le LLM Gate déjà retiré). Les trois agents modifiaient `strategy.RSI_M5_LONG_MIN`/`RSI_M5_SHORT_MAX`/`ADX_MIN` sans coordination ni verrou entre eux, ce qui rendait toute dérive impossible à expliquer. Fichiers conservés mais non importés dans `main.py`. Seul `live_agent.py` reste actif.
- **BacktestPanel removed from nav** — only the pretrain panel is exposed in the dashboard
- **Fenêtre du contexte live = `CONTEXT_BARS_M5` (5000 bougies M5)** — les frames M15/H1/H4 sont resamplées depuis cette série, donc c'est elle qui fixe leur longueur. À 500 bougies (ancienne valeur) le live n'avait que ~43 bougies H1 : `ema()` étant un `ewm(span=200)` sans `min_periods`, ~65% de l'« EMA200 H1 » n'était que sa valeur d'amorce, alors que le biais H1 (EMA50 vs EMA200) et `TREND_BIAS_DISTANCE` en dépendent. Le pretrain/walk-forward resample toute la période et obtient des milliers de bougies H1 — **live et backtest ne calculaient donc pas le même biais**, piste sérieuse pour l'écart PF live vs walk-forward. Aucun appel API supplémentaire (`MarketData._fetch()` récupérait déjà une série longue), +4.6 ms/tick. H4 reste à ~109 bougies (33% d'amorce) mais `H4_TREND_FILTER_ENABLED=False` le tient hors du chemin de décision. Le nombre de bougies H1 est exposé dans `snapshot["context_bars"]` et affiché dans « Conditions d'entrée » — en dessous de `H1_BARS_MIN_RELIABLE` (250) une alerte est levée, **sans jamais bloquer le trading** (une coupure fournisseur ne doit pas arrêter le bot en silence).
- **Jeux de timeframes (`pretrain.TIMEFRAME_SETS`)** — `"M5"` (défaut, comportement inchangé) et `"H1"` (sonde : entrée H1, confirmation H4, biais Daily). Sélectionnable dans le panel walk-forward. Le `base` est demandé **nativement** au fournisseur via `load_m5_data(interval=...)` — l'historique M5 de Twelve Data s'arrête vers 2022, donc un backtest H1 sur une décennie est impossible en resamplant. Les constantes en minutes (`MAX_TRADE_MINUTES`, `EARLY_EXIT_MINUTES`) sont multipliées par `scale`, celles en prix (`ATR_MIN`) par `√scale` (l'amplitude croît en √T), puis **restaurées** dans le `finally` commun. `_try_exit` porte `bar_minutes` sur le trade au lieu des 300 s codées en dur. Motivation : la friction passe de ~4 % du R en M5 à ~1 % en H1, mais il y a 12× moins de trades — le passage n'est gagnant que si le signal est intrinsèquement meilleur, ce que seul ce test tranche.
- **Couverture d'une plage : tolérance de bord** (`data_provider._coverage_ok`) — le marché étant fermé le week-end, une plage lundi→dimanche plafonne à 179j/181j (98.9 %), sous le seuil de 99 %, ce qui faisait rejeter un fetch complet et basculer en silence sur le repli synthétique. Tolérance **doublement bornée** : au plus un week-end manquant (2 jours) ET couverture ≥ 98 %, pour qu'un déficit de 2 jours sur 51 (96 %) reste rejeté.
- **Synthetic data** uses `vol=0.0004` (realistic for XAU/USD) — avoid drawing conclusions from synthetic backtest results
- **Volume filter removed** — unreliable across data sources
- **RSI M15** : zone 30/70 (était 35/65 — assoupli pour éviter blocage en oversold/overbought modéré)
- **RSI M5 LONG** : 49 (was 46 → agent adaptatif a validé 49 en live)
- **RSI M5 SHORT** : 57
- **Pattern floor 0.67** blocks patterns that lose 67%+ of the time (was 0.65 → 0.67)
- **TREND_BIAS_DISTANCE = 0.3 ATR H1** blocks SHORT when price > EMA200 + 0.3×ATR and LONG when price < EMA200 − 0.3×ATR
- **EMA200_MIN_DIST supprimé** : entrée AT EMA200 valide en scalp M5 avec pattern + VWAP
- **BAD_HOURS_CET = {8, 10}** : 8h London open (manipulation pre-session) + 10h CET (WR 38% / 37 trades)
- **ADX_MIN = 20** — valeur que le live appliquait **réellement** (imposée par l'agent adaptatif, cf. ci-dessus), désormais déclarée explicitement dans `strategy.py`. Le code affichait 28 sans l'appliquer. C'est aussi la valeur de `_PRETRAIN_OVERRIDES`, donc celle sous laquelle la baseline walk-forward (PF 0.91) a été mesurée. **20 vs 28 est une question ouverte à trancher en walk-forward** — pas un effet de bord d'un correctif.
- **Mode momentum fort supprimé** : ADX H1 > 35/40 → 1 pattern testé → PF 1.34 vs 1.42, rejeté. Toujours 2 patterns requis.
- **MAX_TRADE_MINUTES = 75** (30 → 45 → 75) — more time for TP targets to be reached. Attention : la boucle de trading **suspend toute la gestion de position** tant que le fournisseur de données est en repli synthétique (`main.py`, `continue` avant « Manage open position », pour ne jamais fermer sur un prix simulé). Le timeout n'est pas raté mais différé — la durée réelle d'un trade peut donc dépasser 75 min de la durée de la coupure. Mesurable via `GET /api/diagnostics/real-trades-duration`.
- **TP1 = 0.7R** (`TP1_R`), **TP2 = 1.8R** (`TP2_R`), fraction soldée à TP1 = 0.5 (`TP1_CLOSE_RATIO`), BE après TP1 (`BE_AFTER_TP1`) — c'étaient des littéraux codés en dur, donc **la seule partie de la stratégie qu'aucun walk-forward ne pouvait tester**. Désormais constantes, exposées en overrides walk-forward, valeurs par défaut inchangées. Motivation : sur 2 567 trades sans filtres de seuil, WR > 50 % mais PF < 1 dans les 4 fenêtres — les entrées sélectionnent, la géométrie des sorties perd (un gagnant type rapporte `TP1_R × ratio`, un perdant coûte 1R).
- **Niveau de TP2 : lisible dans la distribution du MFE, sans relancer de run** (`pretrain._diag_mfe_distribution()`). `mfe_r` étant le maximum atteint en faveur avant sortie, `P(MFE ≥ x)` **est** le taux de touche d'un TP placé à x, à SL identique. Espérance de la seconde moitié pour un TP2 candidat : `E(x) = p(x)·x − (1−p(x))`. **Censure** : un trade clôturé à TP2 n'enregistre jamais de MFE au-delà, donc les niveaux > `TP2_R` du run sont des planchers (marqués `fiable: False`, grisés au dashboard). Pour explorer plus loin, relancer avec `TP2_R` élevé (ex. 10) et `BE_AFTER_TP1=false`.
- **Retirer le BE après TP1 n'est gagnant que si `p(TP2 | TP1) > 1/(TP2_R+1)`** ≈ 35.7 % par défaut. Le seuil ne dépend ni de `TP1_R` ni de la fraction soldée (le gain encaissé à TP1 est acquis dans les deux scénarios). Mesuré par `pretrain._diag_tp1_to_tp2()` et affiché sur chaque fenêtre walk-forward — à consulter **avant** de lancer un test de cette famille.
- **SL → entrée (BE 0R) après TP1** — déplacé à l'entrée sur les bougies suivantes (pas de vérification intrabar). Pire cas : +0.7R×50% + 0×50% = +0.35R net. À comparer via pretrain avec "pas de déplacement" (−0.35R pire cas mais plus de trades TP2).
- **Strategy B (EUR/USD) Order Block only** (June 2026) : biais H1 (EMA50 vs EMA200) + ADX H1 ≥ 20 + OB M5 non mitiguée + retest → TP1=0.7R, **TP2=1.0R** (était 1.8R, quasi jamais atteint sur les rebonds OB EUR/USD — abaissé le 13/07, PR #243). Supprimé : AMD, FVG, Asian range, sweep, accumulation.
- **Strategy A (XAU/USD)** : EMA/patterns, toujours actif sur XAUUSD, non modifiable depuis le dashboard
- **Strategy B (EUR/USD)** : Order Block M5 via `strategy_ict.py`, toujours actif sur EURUSD, non modifiable depuis le dashboard
- **ATR M5 pour EUR/USD** : `snapshot()` utilise 5 décimales pour `atr_m5` et `atr_avg` (ATR EUR/USD ≈ 0.0003 — 3 décimales donnait 0.0). `AtrGauge` adapte automatiquement l'affichage (< 1 → 5 décimales).
- **Rapports hebdo/mensuel** : `entry_time` est comparé via `substr(entry_time, 1, 10)` en SQLite (comparaison lexicographique sur YYYY-MM-DD uniquement — robuste aux formats avec/sans microsecondes ou timezone). Les effects React ont un `setInterval(30s)` pour retry automatique.
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

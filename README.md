# 🟡 XAU/USD Scalping Bot

Bot de **scalping / day trading court terme sur l'Or (XAU/USD)** avec dashboard
temps réel (dark theme) et moteur de **backtest** complet.

> ⚠️ **Avertissement** : ce logiciel est fourni à des fins éducatives et de
> recherche. Le trading de l'or avec effet de levier comporte un risque élevé
> de perte. Le mode **paper** est activé par défaut ; le passage en **live**
> requiert une double confirmation. Utilisez-le à vos propres risques.

---

## ✨ Fonctionnalités

- **Actif unique** : XAU/USD, multi-timeframe **M5 / M15 / H1**.
- **Sessions filtrées** : trading uniquement London (08h–12h CET) et
  New York (14h–18h CET) ; veille en dehors.
- **Stratégie multi-timeframe** :
  - Biais H1 via EMA200 (zone de confusion EMA50–EMA200 ignorée).
  - Confirmation M15 : croisement EMA9/EMA21, RSI(14) 45–55, volume > moyenne 20.
  - Entrée M5 : bougie englobante / rebond EMA9 / cassure de micro-consolidation,
    filtre ATR(14) > 0.8.
- **Gestion de trade** : SL au dernier swing (max 1.2×ATR), TP1 = 1R (clôture 60 %),
  TP2 = 2R (40 %), sortie forcée à 45 min.
- **Risque strict** : 1 % par trade (fixe, changement confirmé), max 4 trades/jour,
  stop journalier −2 % → bot bloqué jusqu'au lendemain.
- **Filtre news** : blocage ±30 min autour des événements majeurs USD
  (NFP, CPI, FOMC…) via calendrier économique, avec repli hors-ligne.
- **Dashboard** : biais, session, P&L du jour, graphique bougies + EMA + S/R,
  RSI M5/M15, jauge ATR, countdown news, trade actif avec timer et P&L live,
  historique + équité intraday, alertes **sonores + visuelles**.
- **Backtest** : courbe d'équité, winrate global et par session, profit factor,
  max drawdown ($/%), heatmap horaire, durée moyenne gagnants vs perdants.
- **Broker** : PaperBroker simulé (par défaut) ou MetaTrader 5 (live/paper).

---

## 🗂️ Structure

```
.
├── requirements.txt
├── Makefile                # raccourcis : make install / backend / frontend / test / docker-up
├── docker-compose.yml      # stack complète (backend + frontend nginx)
├── .env.example            # configuration (clés data providers)
├── pytest.ini
├── backend/
│   ├── Dockerfile
│   ├── database.py         # SQLite (trades, équité, settings, stats jour)
│   ├── risk_manager.py     # 1%/trade, 4 trades/j, stop −2%, sizing lots
│   ├── news_filter.py      # calendrier économique + repli hors-ligne
│   ├── strategy.py         # indicateurs + logique multi-timeframe
│   ├── data_provider.py    # data unifiée (Twelve Data / Polygon / AlphaVantage / yfinance / synthétique)
│   ├── backtest.py         # moteur de backtest M5 + statistiques
│   ├── broker.py           # PaperBroker + MT5Broker + flux de données
│   ├── main.py             # API FastAPI + WebSocket + boucle de trading
│   └── tests/              # suite pytest (stratégie, risque, backtest, news, data)
└── frontend/
    ├── Dockerfile
    ├── nginx.conf
    ├── package.json
    ├── vite.config.js
    ├── index.html
    └── src/
        ├── main.jsx
        ├── Dashboard.jsx       # dashboard live
        └── BacktestPanel.jsx   # panneau de backtest
```

## ⚡ Démarrage rapide (Makefile / Docker)

```bash
cp .env.example .env       # (optionnel) ajouter vos clés API data
make install               # deps backend + frontend
make backend               # terminal 1 : API sur :8000
make frontend              # terminal 2 : dashboard sur :5173
make test                  # lance la suite pytest

# — ou tout en conteneurs —
make docker-up             # http://localhost:5173  (API: :8000/docs)
make docker-down
```

---

## 🚀 Installation & lancement

### 1. Backend (Python 3.10+)

```bash
# depuis la racine du projet
python -m venv venv
source venv/bin/activate          # Windows : venv\Scripts\activate
pip install -r requirements.txt

cd backend
python database.py                # initialise la base SQLite (optionnel)
uvicorn main:app --reload --port 8000
```

L'API tourne sur **http://localhost:8000** (docs interactives : `/docs`).
Le bot démarre en mode **paper** et boucle toutes les 5 secondes.

> **MetaTrader 5 (optionnel, live/paper réel — Windows uniquement)** :
> installez le terminal MT5 et `pip install MetaTrader5`, ouvrez une session,
> puis basculez en mode *live* depuis le dashboard. Sans MT5, le bot reste en
> simulation (PaperBroker) automatiquement.

### 2. Frontend (Node 18+)

```bash
cd frontend
npm install
npm run dev
```

Ouvrez **http://localhost:5173**. Le serveur Vite proxifie `/api` et `/ws`
vers le backend (port 8000). Pour pointer vers un backend distant :

```bash
VITE_API_URL=http://mon-serveur:8000 npm run dev
```

---

## 🔬 Utiliser le backtest

1. Onglet **Backtest** du dashboard.
2. Choisir la période (recommandé : 6–12 derniers mois), le capital, le risque
   (défaut 1 %), le spread (0.3 pip) et le slippage (0.1 pip).
3. **Lancer le backtest**.

> **Sources de données** : le bot utilise un connecteur unifié
> (`backend/data_provider.py`). Renseignez une clé dans `.env` pour des données
> réelles longue durée :
>
> | Provider | Clé `.env` | Note |
> |---|---|---|
> | Twelve Data | `TWELVEDATA_API_KEY` | symbole `XAU/USD` natif |
> | Polygon.io | `POLYGON_API_KEY` | ticker `X:XAUUSD` |
> | Alpha Vantage | `ALPHAVANTAGE_API_KEY` | FX intraday |
> | yfinance | — | sans clé, M5 limité à ~60 j (`GC=F`) |
> | synthétique | — | repli hors-ligne déterministe |
>
> `XAU_DATA_PROVIDER=auto` (défaut) essaie les providers dont la clé est
> présente, puis yfinance, puis le générateur **synthétique** afin que le
> backtest et le flux live ne plantent jamais. L'endpoint `GET /api/data-provider`
> indique quels providers sont actifs.

---

## ⚙️ Configuration (REST)

| Endpoint | Méthode | Rôle |
|---|---|---|
| `/api/state` | GET | État live complet (biais, session, risque, position…) |
| `/api/chart?tf=M5\|M15\|H1` | GET | Bougies + EMA + S/R + marqueurs |
| `/api/trades?scope=today\|all` | GET | Trades + courbe d'équité |
| `/api/settings` | GET/POST | Lecture/écriture des réglages |
| `/api/mode` | POST | Bascule paper/live (live ⇒ double confirmation) |
| `/api/close` | POST | Ferme la position ouverte |
| `/api/bot/toggle` | POST | Pause / reprise du bot |
| `/api/backtest` | POST | Lance un backtest |
| `/api/news` | GET | Statut du filtre news |
| `/api/data-provider` | GET | Provider de données configuré / disponibles |
| `/ws` | WS | Flux d'état temps réel |

Changer le **risque par trade** nécessite `confirm_risk_change=true` ;
passer en **live** nécessite `confirm=true` **et** `confirm_again=true`.

---

## 🧠 Détails de la stratégie

| Étape | Timeframe | Condition |
|---|---|---|
| Biais | H1 | Prix > EMA200 → LONG · < EMA200 → SHORT · entre EMA50/200 → NEUTRE |
| Confirmation | M15 | EMA9/EMA21 alignées/croisées · RSI 45–55 · volume > moy. 20 |
| Entrée | M5 | Englobante OU rebond EMA9 OU cassure micro-range · ATR > 0.8 |
| Stop | M5 | Dernier swing, plafonné à 1.2×ATR |
| Sorties | — | TP1 1R (60 %), TP2 2R (40 %), timeout 45 min |

**Spécifications contrat or** : 1 lot standard = 100 oz ; 1 pip = 0.1 ;
P&L = (sortie − entrée) × 100 × lots. Le sizing ajuste le volume pour risquer
exactement 1 % du capital sur la distance au stop.

---

## 🛡️ Garde-fous risque

- Risque/trade fixe à **1 %** (modification confirmée explicitement).
- **Max 4 trades/jour**.
- **Stop journalier −2 %** → bot bloqué jusqu'au lendemain.
- **Aucun trade** pendant la fenêtre news (±30 min).
- **Paper par défaut**, switch live à double confirmation.

---

## 🧪 Tests

Suite automatisée (pytest, données synthétiques hors-ligne — aucun réseau requis) :

```bash
make test
# ou
XAU_DATA_PROVIDER=synthetic python -m pytest backend/tests -q
```

Couverture : indicateurs & sessions, logique de stratégie, gestion du risque
(sizing 1 %, max trades/jour, stop journalier), backtest de bout en bout,
filtre news et connecteur de données.

Vérifications manuelles rapides :

```bash
cd backend
python database.py        # init DB + dump settings
python news_filter.py     # statut calendrier économique
python data_provider.py   # provider utilisé + dernières bougies
python backtest.py        # backtest 45 jours + résumé
```

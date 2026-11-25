# Bot XAU/USD — contexte projet

> Document de transfert pour un Projet Claude. Il complète `CLAUDE.md` (qui
> décrit le code) : ici on trouve **ce qui a été mesuré, ce qui a été rejeté, et
> les pièges qui ont déjà coûté cher**. Lire les deux.
>
> Dernière mise à jour : 25 novembre 2026.

---

## 1. Le projet en une page

Bot de scalping sur l'or (XAU/USD), déployé sur Railway, dépôt `cyril8344/ee`.
Backend FastAPI + boucle de trading asyncio, frontend React, SQLite.
Mode `paper` par défaut. Le propriétaire trade aussi **manuellement** en démo.

Quatre stratégies coexistent :

| Mode | Instrument | Méthode | État |
|---|---|---|---|
| **A** | — | EMA/patterns, pipeline 10 filtres | **Mesurée, perdante.** Débranchée de l'or le 25/11/2026 |
| **B** | EUR/USD | Order Block ICT (`strategy_ict.py`) | Active |
| **ES** | ES (proxy SPY×10) | `strategy_es.py` | Active |
| **C** | **XAU/USD** | Balayage/récupération D1→H1→M15 (`strategy_sweep.py`) | **Active en live, mesurée par rien** |

---

## 2. ⚠️ L'avertissement le plus important

**La stratégie C tourne en live sur l'or sans aucune validation.**

Pas de walk-forward. Pas même un pretrain in-sample. C'est une décision humaine
explicite, prise en connaissance de cause, documentée dans `MARKET_CONFIG`.

Le seul chiffre disponible — R:R médian de **0.59** au TP1 sur séries simulées —
penche du mauvais côté. **Première action utile pour quiconque reprend ce
projet : lancer le pretrain en mode C, puis le walk-forward 4 fenêtres.**

Revenir à la stratégie A : remettre `"default_strategy": "A"` dans
`MARKET_CONFIG["XAUUSD"]`, une seule ligne.

---

## 3. Ce qui a été mesuré

Tout ce tableau vient de mesures réelles faites dans ce projet. Rien n'y est
estimé ou supposé.

### Stratégie A (l'ancienne, sur l'or)

| Mesure | Valeur |
|---|---|
| PF walk-forward, 24 mois | **0.89 – 1.03** |
| Baseline de référence | PF 0.91 |
| Apport des 5 seuils d'entrée | **+0.05 PF** (0.84 sans filtres → 0.89) |
| Écart restant jusqu'à la rentabilité | 0.11 |
| Win rate | > 50 % |

**Lecture** : les entrées sélectionnent correctement, mais le PF reste < 1 dans
les 4 fenêtres. **C'est la géométrie des sorties qui perd**, pas le choix des
entrées. Un gagnant type rapporte `TP1_R × ratio`, un perdant coûte 1R.

Conséquence directe : **la famille « mieux choisir ses entrées » est
épuisée**. Elle vaut 1/20e de point de PF. Ne pas y retourner sans raison neuve.

### Balayage/récupération (la règle de la stratégie C)

Mesuré sur marches aléatoires — des séries **sans aucune structure** :

| Timeframe | Condition haussière remplie |
|---|---|
| D1 | **24.9 %** des bougies |
| H1 | 19.9 % |
| M15 | 13.3 % |

Le hasard pur produit ce motif en permanence. **Un exemple bien choisi ne prouve
rien.** La question utile n'est pas « est-ce que ça arrive ? » mais « est-ce que
le taux réel sur l'or s'écarte de ces 24.9 % ? ».

Si les trois étages étaient indépendants, l'alignement vaudrait **0.66 %**. Il
sera mesuré bien plus haut, et cet écart **est** la redondance entre étages.

### Comparaison des deux biais (sur séries calibrées or)

| | Donne un sens | Bloque |
|---|---|---|
| Biais actuel (EMA50) | **100 %** | 0 % |
| Biais balayage D1 | 44.4 % | **55.6 %** |

Et quand les deux donnent un sens, **ils sont d'accord 30 % du temps** — donc
ils se contredisent 7 fois sur 10. Logique : l'EMA est suiviste, le balayage est
un retournement. Ce ne sont pas deux réglages du même bot, ce sont deux bots.

### Conflit entre la stratégie C et les filtres du mode A

Sur les bougies où la règle dit LONG : clôture au-dessus du VWAP **49 %**,
au-dessus de l'EMA9 **49 %**.

Donc le filtre VWAP **seul** écarterait 51 % des signaux. Et les survivants
seraient les balayages qui ressemblent à de la continuation de tendance —
exactement le sous-ensemble qui n'est pas la méthode. **C'est pourquoi la C est
un mode séparé et pas un override du pipeline A.**

### Définitions du scanner SMC

| | Écart mesuré |
|---|---|
| Order Block : « dernière bougie contraire » vs « extrême de la jambe » | **bougie différente dans 67 %** des cassures (195 OB) |
| Hauteur de zone (bougie entière vs open→extrême) | **+74 %** en médiane |
| Pivots : fractale symétrique vs mode « jambe » | 0 à 19 % de pivots en moins, 7 séries sur 7 |

### Fumée de la stratégie C, 208 jours simulés

93 signaux · R:R médian au TP1 **0.59** · 54 long / 39 short.
Premier motif de rejet : `etape1_pas_de_balayage_d1` (2 674 fois).

---

## 4. Le défaut récurrent de ce dépôt : les divergences silencieuses

**C'est la section la plus importante du document.** Le même bug revient sous
dix formes : *le code fait autre chose que ce qu'il déclare, et rien ne le
signale.* Pas de crash, pas de log, juste des chiffres faux.

Catalogue de ce qui a été trouvé :

| Divergence | Effet |
|---|---|
| `ADX_MIN` déclaré 28, appliqué **20** (imposé par l'agent adaptatif) | des semaines de live sur un seuil non documenté |
| `compute_bias()` : la doc dit « NEUTRE si EMA50 et EMA200 divergent », le code n'utilise **que** l'EMA50 et ne renvoie **jamais** NEUTRE | l'étage 3 du pipeline ne bloque rien du tout |
| `CLAUDE.md` affirme qu'Optuna valide OOS « par construction » — **faux** : `optimizer.py:183` maximise ce score sur fenêtres fixes, ce qui brûle son statut OOS | fausse confiance dans les paramètres trouvés |
| `CONTEXT_BARS_M5` à 500 → ~43 bougies H1 → **65 % de l'« EMA200 H1 » n'était que sa valeur d'amorce** | live et backtest ne calculaient pas le même biais |
| `build_context()` agrège en `label="right"`, pretrain en `label="left"` | bornes décalées d'une bougie entre live et backtest |
| Durée max : le live lit `pos.meta` (défaut 75 min), pretrain lit `sig.max_duration_min` (240) | trades **trois fois plus courts** en live qu'en backtest |
| `market_symbol()` repli muet sur XAUUSD | un walk-forward « ES » chargeait des données or |
| `_coverage_ok` : seuil 99 % inatteignable (week-ends) | bascule silencieuse sur données synthétiques |
| Pillow déclaré dans le `requirements.txt` de la racine, que Railway ne lit pas | `No module named 'PIL'` en production |
| `_diag_mfe_distribution` : espérance fausse (ignorait les sorties intermédiaires) | **a inversé la conclusion** — annonçait TP2=1.0R 35× meilleur que 1.8R ; le walk-forward a donné 0.93 contre 1.03 |

### Et le même piège côté tests

Une **fixture dégénérée inverse la conclusion**, deux fois constaté :

- série où `open == close` → les deux modes de pivot paraissent équivalents (faux dès que les bougies ont un corps) ;
- bougies calmes **strictement identiques** → aucun plus-bas n'est strictement inférieur au suivant, aucune jambe ne bascule, et la fixture « prouve » une absence de signal qui ne vient que de sa platitude.

D'où `_respire()` à côté de `_calm()` dans les tests SMC.

### Règle de travail qui en découle

**Mesurer avant d'affirmer, et écrire le chiffre dans le code.** Plusieurs
commentaires du dépôt portent désormais leurs mesures, pour qu'on sache quand
elles cessent d'être vraies. Des tests verrouillent les chiffres annoncés.

---

## 5. Ce qui a été testé et REJETÉ — ne pas refaire

| Piste | Résultat | Statut |
|---|---|---|
| Timeframe H1 (entrée H1, confirmation H4, biais Daily) | PF 0.93, **0 fenêtre sur 4** | fermé |
| Timeframe M1 | ≈ **−0.10 PF** par la friction seule (6.2 % du R) | fermé, calculé |
| Seuils d'entrée (RSI, ADX, ATR, VWAP, patterns) | +0.05 PF cumulés | épuisé |
| SL sous la mèche du pattern | PF 1.36 → **1.26**, SL direct 34.5 → 36.2 % | rejeté |
| Mode momentum fort (ADX > 35 → 1 pattern) | PF 1.34 vs **1.42** | rejeté |
| TP2 = 1.0R (suggéré par le diagnostic MFE) | walk-forward **0.93 vs 1.03** | rejeté, et le diagnostic corrigé |
| Agents autonomes (`researcher_agent`, `adaptive_agent`) | grid-search in-sample appliqué au live sans walk-forward | désactivés |
| Agent adaptatif live (ajustement auto) | réagissait au bruit, desserrage **irréversible** | désactivé |
| LLM Gate | latence API dans la boucle, blocages silencieux | supprimée |
| Heures bloquées `[8, 10, 15]` | **jamais lancé** — sélection in-sample, le propriétaire a décliné à raison | ne pas lancer |

---

## 6. Les questions encore ouvertes

Par ordre d'intérêt mesuré :

1. **La stratégie C vaut-elle quelque chose ?** Rien ne le dit. Elle est en
   live. Priorité absolue : pretrain mode C, puis walk-forward.
2. **La géométrie des sorties.** C'est là que la stratégie A perdait, et c'est
   la seule famille vraiment non testée. `TP1_R`, `TP2_R`, `TP1_CLOSE_RATIO`,
   `BE_AFTER_TP1` sont désormais des constantes exposées en override.
   Repère utile : retirer le BE après TP1 n'est gagnant que si
   `p(TP2 | TP1) > 1/(TP2_R+1)` ≈ **35.7 %**. Mesuré par `_diag_tp1_to_tp2()`
   et affiché sur chaque fenêtre — **à consulter avant** de lancer un test de
   cette famille.
3. **Dimensionnement du SL** (`SL_ATR_MULT`, `SL_ATR_MULT_HIGH`,
   `SL_MIN_ATR_MULT`), exposé en override.
4. **ADX 20 vs 28** — question ouverte, pas un effet de bord.
5. **Aligner le scanner SMC sur l'écran** : les définitions ont été changées,
   mais **rien n'a été vérifié contre une capture réelle**. Test d'acceptation
   prévu : une capture MT5 avec les zones de l'indicateur dessus, le scanner
   recalcule les siennes depuis les pixels, elles doivent tomber au même
   endroit.

---

## 7. Détails de la stratégie C

Source : trois captures vidéo d'un créateur (Hyvirtus France), lues image par
image. Une seule règle, appliquée trois fois :

```
balayage      : low(bougie en cours) < low(bougie précédente)
récupération  : close(bougie en cours) > low(bougie précédente)
```

| Étage | TF | Rôle |
|---|---|---|
| 1 | Daily (PDH/PDL) | direction |
| 2 | H1 / M30 | setup |
| 3 | M15 → M1 | cassure du niveau, retour sur l'Order Block → entrée |

SL sous l'OB · TP1 un peu sous le plus haut H1 · TP2 au PDH.

### Trois constats structurels, trouvés en écrivant les tests

1. **La cassure M15 cherchée *à l'intérieur* de la bougie H1 du setup serait une
   condition vide** : la clôture H1 au-dessus du niveau *est* la clôture de sa
   dernière M15. L'étape 3 n'ajouterait rien à l'étape 2. Elle est donc cherchée
   **après** la clôture H1 — ce que l'anti look-ahead imposait de toute façon.
2. **L'Order Block ne filtre quasiment rien** : à 12 bougies de recul, une
   bougie de couleur contraire existe presque toujours. Ce qui lie réellement,
   c'est le **retour du prix** dans la zone.
3. **L'étape 2 n'est pas une porte binaire** : la recherche remonte 12 bougies,
   donc elle élargit la fenêtre autant qu'elle filtre.

Des trois étapes annoncées, **une seule sélectionne vraiment** (le balayage
journalier, ~25 % des jours), plus le retour dans la zone.

---

## 8. Le trading manuel du propriétaire

Compte **fictif** (démo MetaTrader 5, portefeuille de 100 000 €). Pas de compte
prop firm. Environ **+8 000 €** revendiqués, ~30 trades sur 3 jours.

Méthode : Order Block, alignement M5 / H1 / H4, pas de contre-tendance
journalière, R:R souvent 1:3 avec stop serré.

**Lecture statistique à garder en tête** : à 1:3, le seuil de rentabilité est
`(1+f)/4` ≈ **26 %** de réussite. Sur ~30 trades, un résultat positif n'est pas
distinguable du hasard. Ce n'est pas un jugement sur la méthode — c'est la
taille de l'échantillon qui ne permet encore rien de conclure.

À noter : **sa méthode manuelle et la stratégie C ont la même forme** (multi-TF,
Order Block, alignement). La C est, pour la première fois, une spécification
assez précise de ce qu'il fait à l'œil pour être testée.

---

## 9. Règles anti-overfitting (obligatoires)

Toute modification de paramètre stratégie **doit être validée en walk-forward
avant merge**. Le pretrain in-sample seul ne prouve rien.

1. Proposer le changement avec une hypothèse claire
2. Pretrain 6M → noter PF et WR in-sample
3. Walk-forward 4 fenêtres × 1.5M depuis le dashboard
4. Robustesse : **PF > 1 dans ≥ 75 % des fenêtres ET `std_pf` < 0.30**
5. Merger uniquement si l'OOS est cohérent

**Absolus** : jamais de merge sur in-sample seul · **max 3 paramètres à la
fois** · WR in-sample > 58 % sur 6M = signal fort d'overfitting · reset des
poids ML après tout changement de filtres ou de features.

⚠️ Et le rappel qui vaut pour tout ce document : **P(≥3 fenêtres sur 4
rentables) = 5/16 = 31 % pour une stratégie à edge nul.** Trois fenêtres vertes
ne sont pas une preuve.

---

## 10. Conventions de travail

Ce qui a bien marché dans ce projet, et qui mérite d'être reconduit :

- **Aucun verdict.** Les modules d'aide (`smc/plan.py`, panneau Niveaux,
  `candle_sweep`) renvoient des critères avec leur mesure et leur seuil, jamais
  un « VALIDE » ni une probabilité. Rien dans ce dépôt ne mesure si un trade va
  marcher. Un avis tranché se recopie dans la plateforme ; un décompte de
  critères, non.
- **Chaque refus porte son motif.** « Aucune zone » et « trois zones toutes
  mitigées » sont deux situations opposées ; un blanc se lit comme une panne.
- **Les commentaires portent leurs chiffres**, pour qu'on sache quand ils
  cessent d'être vrais.
- **Mesurer avant d'affirmer** — y compris pour se contredire soi-même. Deux
  justifications écrites dans ce projet ont dû être retirées après mesure.
- **Tester une méthode isolée avant de la greffer.** Un résultat obtenu à
  travers neuf filtres existants ne dit rien de la méthode testée.

## 11. Secrets — ne jamais commiter

Uniquement dans les variables Railway : `TELEGRAM_BOT_TOKEN`,
`TELEGRAM_CHAT_ID`, `FRED_API_KEY`, `ADMIN_USERNAME`, `ADMIN_PASSWORD`,
`JWT_SECRET`.

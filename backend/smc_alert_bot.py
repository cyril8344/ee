"""
smc_alert_bot.py — Robot d'ASSISTANCE (alerte seule, aucun ordre envoyé)
=========================================================================
Détecte des Order Blocks SMC/ICT en temps réel et envoie une alerte Telegram
détaillée quand toutes les conditions demandées sont réunies. Ne passe JAMAIS
d'ordre — c'est un script séparé, non branché sur la boucle de trading de
main.py/BotState, à lancer indépendamment.

⚠️ Lire avant d'activer les filtres FVG/sweep en mode strict :
Ce dépôt a DÉJÀ testé cette combinaison de filtres (FVG obligatoire + sweep de
liquidité + AMD + zone premium/discount) sur EUR/USD en conditions réelles de
walk-forward (voir strategy_ict.py / CLAUDE.md, "Strategy B Order Block only",
juin 2026) et les a retirés du pipeline live — trop de confluence exigée =
trop peu de signaux, sans gain de robustesse démontré en out-of-sample. Les
fonctions `find_fvgs()` / `liquidity_swept()` dans strategy.py existent déjà
mais ne sont appelées nulle part dans la logique live (`evaluate()` /
`evaluate_ict()`) — vérifié par grep, pas par supposition.

Ici c'est différent : ce bot n'ouvre AUCUNE position, toi seul valides —
donc le coût d'un filtre trop strict est juste "moins d'alertes", pas du
capital mal engagé. Les seuils ci-dessous (FVG_REQUIRED, SWEEP_REQUIRED,
ZONE_FILTER_REQUIRED) sont donc laissés activables/désactivables par CLI,
mais gardés à False par défaut — pour repartir de la même base validée que
le reste du repo (OB + retest + biais HTF) et n'ajouter FVG/sweep qu'en test
volontaire, pas comme prérequis caché.

Usage :
    python backend/smc_alert_bot.py --symbol EURUSD --entry-tf M15 --htf H4
    python backend/smc_alert_bot.py --symbol XAUUSD --require-fvg --require-sweep --require-zone
    python backend/smc_alert_bot.py --symbol EURUSD --dry-run   # affiche sans envoyer Telegram
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data_provider
import strategy
from strategy_ict import _find_order_blocks, _h1_bias  # réutilise le détecteur OB déjà calibré

STATE_FILE = Path(__file__).parent / "data" / "smc_alert_state.json"

TF_RESAMPLE = {
    "M5":  "5min",
    "M15": "15min",
    "H1":  "60min",
    "H4":  "240min",
}

# ──────────────────────────────────────────────────────────────────────────
# Config (surchargeable en CLI, voir main() en bas)
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    symbol: str = "EURUSD"
    entry_tf: str = "M15"          # timeframe où l'OB est cherché
    htf: str = "H4"                 # timeframe du biais directionnel
    poll_seconds: int = 300
    ob_lookback_bars: int = 60      # bougies entry_tf analysées pour les OB
    fvg_lookahead_bars: int = 3     # fenêtre après l'OB où chercher un FVG
    swing_lookback_bars: int = 100  # fenêtre pour le range Fibonacci discount/premium
    proximity_atr: float = 0.4      # tolérance "prix proche de l'OB" en ATR entry_tf
    sl_buffer_atr: float = 0.3      # buffer SL derrière l'extrême de l'OB
    require_fvg: bool = False       # FVG obligatoire juste après l'OB (désactivé par défaut, voir docstring)
    require_sweep: bool = False     # sweep de liquidité obligatoire avant l'impulsion (désactivé par défaut)
    require_zone: bool = False      # zone Fibonacci discount/premium obligatoire (désactivé par défaut)
    cooldown_minutes: int = 60      # ne pas ré-alerter le même OB avant ce délai
    dry_run: bool = False


# ──────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────
def fetch_frames(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Récupère le M5 brut (via le même fournisseur multi-provider que le bot
    principal — Twelve Data → Polygon → Alpha Vantage → yfinance → synthetic)
    puis resample en entry_tf et htf. Fenêtre de 45 jours : assez de bougies
    pour que l'EMA200 htf converge et pour un range Fibonacci significatif,
    même en H4 (45j ≈ 65-70 bougies H4)."""
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - pd.Timedelta(days=45)
    m5_raw, provider = data_provider.get_m5(
        start=start_dt.strftime("%Y-%m-%d"), end=end_dt.strftime("%Y-%m-%d"), symbol=cfg.symbol)
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}

    entry_rule = TF_RESAMPLE[cfg.entry_tf]
    htf_rule = TF_RESAMPLE[cfg.htf]

    entry_df = m5_raw if cfg.entry_tf == "M5" else (
        m5_raw.resample(entry_rule, label="right", closed="right").agg(agg).dropna()
    )
    htf_df = m5_raw.resample(htf_rule, label="right", closed="right").agg(agg).dropna()

    entry_df = strategy.add_indicators(entry_df)
    htf_df = strategy.add_indicators(htf_df)
    return entry_df, htf_df


# ──────────────────────────────────────────────────────────────────────────
# Filtres SMC additionnels (biais HTF réutilise strategy_ict._h1_bias)
# ──────────────────────────────────────────────────────────────────────────
def fvg_after_ob(entry_df: pd.DataFrame, ob: Dict[str, Any], direction: str,
                  lookahead: int) -> Optional[Dict[str, Any]]:
    """Cherche un FVG (imbalance 3 bougies) non comblé dans les `lookahead`
    bougies suivant la bougie d'OB. Réutilise strategy.find_fvgs() (déjà
    écrit et testé dans ce repo, jamais branché en live) plutôt que d'en
    réécrire une variante."""
    try:
        ob_pos = entry_df.index.get_loc(ob["ts"])
    except KeyError:
        return None
    window = entry_df.iloc[ob_pos: ob_pos + lookahead + 3]
    if len(window) < 3:
        return None
    fvg_type = "bullish" if direction == "LONG" else "bearish"
    for fvg in strategy.find_fvgs(window, lookback=len(window)):
        if fvg["type"] == fvg_type:
            return fvg
    return None


def sweep_before_ob(entry_df: pd.DataFrame, ob: Dict[str, Any], direction: str) -> bool:
    """True si un plus haut/bas antérieur a été balayé (mèche) juste avant la
    formation de l'OB. Réutilise strategy.liquidity_swept() sur la fenêtre
    qui précède l'OB."""
    try:
        ob_pos = entry_df.index.get_loc(ob["ts"])
    except KeyError:
        return False
    window = entry_df.iloc[max(0, ob_pos - 20): ob_pos + 1]
    if len(window) < 6:
        return False
    bias = "LONG" if direction == "LONG" else "SHORT"
    return strategy.liquidity_swept(window, bias, lookback=len(window))


def zone_filter(entry_df: pd.DataFrame, ob: Dict[str, Any], direction: str,
                 lookback: int) -> tuple[str, float]:
    """Position de l'OB dans le range Fibonacci [swing_low, swing_high] des
    `lookback` dernières bougies entry_tf. Retourne (zone, pct) où pct=0 =
    plus bas du range, pct=100 = plus haut. Discount = pct<50, Premium = pct>50."""
    window = entry_df.tail(lookback)
    range_low = float(window["low"].min())
    range_high = float(window["high"].max())
    if range_high <= range_low:
        return "equilibrium", 50.0
    ob_mid = (ob["low"] + ob["high"]) / 2
    pct = (ob_mid - range_low) / (range_high - range_low) * 100.0
    if pct < 50:
        zone = "discount"
    elif pct > 50:
        zone = "premium"
    else:
        zone = "equilibrium"
    return zone, pct


def price_near_ob(entry_df: pd.DataFrame, ob: Dict[str, Any], proximity_atr: float) -> bool:
    """True si le dernier close est dans la zone OB ou à proximity_atr×ATR de sa bordure."""
    last = entry_df.iloc[-1]
    atr_val = float(last.get("atr", 0) or 0)
    tol = proximity_atr * atr_val
    price = float(last["close"])
    return (ob["low"] - tol) <= price <= (ob["high"] + tol)


# ──────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ──────────────────────────────────────────────────────────────────────────
def build_candidates(entry_df: pd.DataFrame, htf_df: pd.DataFrame, cfg: Config) -> List[Dict[str, Any]]:
    """Retourne la liste des OBs valides (tous filtres actifs appliqués),
    les plus récents en premier, avec toutes les infos nécessaires à l'alerte."""
    direction = _h1_bias(htf_df)  # 'LONG' / 'SHORT' / None — EMA50 vs EMA200 sur htf
    if direction is None:
        return []

    atr_val = float(entry_df["atr"].iloc[-1] or 0)
    if atr_val <= 0:
        return []

    obs = _find_order_blocks(entry_df.tail(cfg.ob_lookback_bars + 5), direction, atr_val)
    if not obs:
        return []

    candidates: List[Dict[str, Any]] = []
    for ob in reversed(obs):  # plus récent d'abord
        fvg = fvg_after_ob(entry_df, ob, direction, cfg.fvg_lookahead_bars)
        if cfg.require_fvg and fvg is None:
            continue

        swept = sweep_before_ob(entry_df, ob, direction)
        if cfg.require_sweep and not swept:
            continue

        zone, zone_pct = zone_filter(entry_df, ob, direction, cfg.swing_lookback_bars)
        zone_ok = (direction == "LONG" and zone == "discount") or \
                  (direction == "SHORT" and zone == "premium")
        if cfg.require_zone and not zone_ok:
            continue

        if not price_near_ob(entry_df, ob, cfg.proximity_atr):
            continue  # OB valide mais prix pas encore revenu dessus — pas d'alerte pour l'instant

        candidates.append({
            "ob": ob,
            "direction": direction,
            "fvg": fvg,
            "swept": swept,
            "zone": zone,
            "zone_pct": zone_pct,
            "zone_ok": zone_ok,
            "atr": atr_val,
        })
    return candidates


def build_levels(entry_df: pd.DataFrame, cand: Dict[str, Any], cfg: Config) -> Dict[str, float]:
    ob = cand["ob"]
    direction = cand["direction"]
    atr_val = cand["atr"]
    price = float(entry_df["close"].iloc[-1])

    sr = strategy.swing_levels(entry_df, lookback=cfg.swing_lookback_bars)

    if direction == "LONG":
        sl = ob["low"] - cfg.sl_buffer_atr * atr_val
        tp = strategy.nearest_resistance_above(price, sr, min_gap=atr_val * 0.5)
    else:
        sl = ob["high"] + cfg.sl_buffer_atr * atr_val
        tp = strategy.nearest_support_below(price, sr, min_gap=atr_val * 0.5)

    if tp is None:
        # pas de structure claire trouvée → fallback 2R
        risk = abs(price - sl)
        tp = price + 2 * risk if direction == "LONG" else price - 2 * risk

    return {"entry_price": price, "sl": float(sl), "tp": float(tp)}


# ──────────────────────────────────────────────────────────────────────────
# Alerte
# ──────────────────────────────────────────────────────────────────────────
def format_alert(cfg: Config, cand: Dict[str, Any], levels: Dict[str, float]) -> str:
    ob = cand["ob"]
    direction = cand["direction"]
    arrow = "🟢 Bullish Order Block" if direction == "LONG" else "🔴 Bearish Order Block"
    risk = abs(levels["entry_price"] - levels["sl"])
    reward = abs(levels["tp"] - levels["entry_price"])
    rr = round(reward / risk, 2) if risk > 0 else 0.0

    fvg_txt = "OUI" if cand["fvg"] else "NON"
    sweep_txt = "OUI" if cand["swept"] else "NON"
    zone_txt = f'{cand["zone"]} ({cand["zone_pct"]:.0f}%)'

    lines = [
        f"<b>{cfg.symbol} — {cfg.entry_tf}</b>",
        f"{arrow}",
        "",
        f"Zone OB : <b>{ob['low']:.5f} — {ob['high']:.5f}</b> (50% = {((ob['low']+ob['high'])/2):.5f})",
        f"Prix actuel : {levels['entry_price']:.5f}",
        "",
        f"SL suggéré : {levels['sl']:.5f}",
        f"TP suggéré : {levels['tp']:.5f}  (R:R ≈ {rr})",
        "",
        f"FVG présent : {fvg_txt} | Sweep : {sweep_txt} | Zone : {zone_txt}",
        f"Biais HTF ({cfg.htf}) : {direction}",
        "",
        "⚠️ Alerte informative — aucun ordre n'a été envoyé, à valider manuellement.",
    ]
    return "\n".join(lines)


def send_telegram(text: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        print("[smc_alert_bot] TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID absents — alerte affichée seulement.")
        return
    payload = json.dumps({"chat_id": chat_id, "text": text, "parse_mode": "HTML"})
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=payload.encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(req, timeout=5)
    except Exception as exc:
        print(f"[smc_alert_bot] Échec envoi Telegram : {exc}")


# ──────────────────────────────────────────────────────────────────────────
# Dédoublonnage (ne pas ré-alerter le même OB à chaque cycle)
# ──────────────────────────────────────────────────────────────────────────
def _load_state() -> Dict[str, str]:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_state(state: Dict[str, str]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state))


def already_alerted_recently(state: Dict[str, str], key: str, cooldown_minutes: int) -> bool:
    ts = state.get(key)
    if ts is None:
        return False
    last = datetime.fromisoformat(ts)
    return (datetime.now(timezone.utc) - last).total_seconds() < cooldown_minutes * 60


# ──────────────────────────────────────────────────────────────────────────
# Boucle
# ──────────────────────────────────────────────────────────────────────────
def run_once(cfg: Config) -> int:
    """Un cycle complet : fetch → détection → alertes. Retourne le nombre d'alertes envoyées."""
    entry_df, htf_df = fetch_frames(cfg)
    if len(entry_df) < 50 or len(htf_df) < 50:
        print("[smc_alert_bot] Pas assez de données, cycle ignoré.")
        return 0

    candidates = build_candidates(entry_df, htf_df, cfg)
    state = _load_state()
    sent = 0
    for cand in candidates:
        ob = cand["ob"]
        key = f"{cfg.symbol}:{cfg.entry_tf}:{cand['direction']}:{ob['ts']}"
        if already_alerted_recently(state, key, cfg.cooldown_minutes):
            continue
        levels = build_levels(entry_df, cand, cfg)
        text = format_alert(cfg, cand, levels)
        print(text)
        print("-" * 40)
        if not cfg.dry_run:
            send_telegram(text)
        state[key] = datetime.now(timezone.utc).isoformat()
        sent += 1
    _save_state(state)
    return sent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--entry-tf", default="M15", choices=list(TF_RESAMPLE))
    parser.add_argument("--htf", default="H4", choices=list(TF_RESAMPLE))
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--require-fvg", action="store_true")
    parser.add_argument("--require-sweep", action="store_true")
    parser.add_argument("--require-zone", action="store_true")
    parser.add_argument("--cooldown-minutes", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true", help="n'envoie pas sur Telegram, affiche seulement")
    parser.add_argument("--once", action="store_true", help="un seul cycle puis quitte (utile pour tester)")
    args = parser.parse_args()

    cfg = Config(
        symbol=args.symbol, entry_tf=args.entry_tf, htf=args.htf,
        poll_seconds=args.poll_seconds,
        require_fvg=args.require_fvg, require_sweep=args.require_sweep,
        require_zone=args.require_zone, cooldown_minutes=args.cooldown_minutes,
        dry_run=args.dry_run,
    )

    print(f"[smc_alert_bot] {cfg.symbol} {cfg.entry_tf}/{cfg.htf} — "
          f"FVG={cfg.require_fvg} sweep={cfg.require_sweep} zone={cfg.require_zone} "
          f"— alerte seule, aucun ordre envoyé.")

    while True:
        try:
            n = run_once(cfg)
            if n:
                print(f"[smc_alert_bot] {n} alerte(s) envoyée(s).")
        except Exception as exc:
            print(f"[smc_alert_bot] Erreur cycle : {exc}")
        if args.once:
            break
        time.sleep(cfg.poll_seconds)


if __name__ == "__main__":
    main()

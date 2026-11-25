"""Stratégie C — la méthode du reel, seule, sans aucun filtre du mode A.

Ces tests ne disent pas qu'elle gagne de l'argent : rien ici ne le mesure, et
la condition de base tombe sur ~25 % des bougies journalières d'une marche
aléatoire. Ils vérifient qu'elle est implémentée **telle qu'énoncée**, que
chaque étape peut être mise en défaut séparément, et qu'elle ne lit pas le
futur.

Le test qui compte le plus est `test_a_forming_daily_candle_is_never_used` :
juger le balayage d'hier sur une bougie journalière encore ouverte produit un
backtest flatteur qui ne se reproduit jamais en live.
"""
import os
from datetime import timedelta

import pandas as pd
import pytest

os.environ.setdefault("XAU_DATA_PROVIDER", "synthetic")

import strategy_sweep as ss


def _m5(rows, start="2026-01-05 00:00"):
    idx = pd.date_range(start, periods=len(rows), freq="5min", tz="UTC")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 1.0
    return df


def _plat(p, n, amp=0.6):
    """Bougies de respiration, jamais strictement identiques : des lows égaux
    empêchent tout balayage de se former et la fixture « prouverait » une
    absence de signal qui ne vient que de sa platitude."""
    out = []
    for i in range(n):
        d = ((i * 37) % 11 - 5) * 0.02
        out.append((p + d, p + amp + d, p - amp + d, p + d))
    return out


# --------------------------------------------------------------------------- #
# La fixture : le cas exact de la vidéo, construit en M5
# --------------------------------------------------------------------------- #
def _m15_bar(o, h, l, c):
    """Trois bougies M5 dont l'agrégat vaut exactement (o, h, l, c).

    Construire la fixture en M5 et laisser `resample` fabriquer les M15/H1/D1
    est volontaire : c'est le même chemin que pretrain et que le live, donc la
    fixture ne peut pas valider une agrégation que la production ne ferait pas.
    """
    return [(o, h, o, o), (o, o, l, l), (l, max(l, c), min(l, c), c)]


def _heure(o, h, l, c):
    """Une bougie H1 = quatre M15 = douze M5."""
    mid = (o + c) / 2
    return (_m15_bar(o, h, o, mid) + _m15_bar(mid, mid, l, l)
            + _m15_bar(l, max(l, mid), l, mid) + _m15_bar(mid, max(mid, c), min(mid, c), c))


def _scenario(jour_balaye=True, setup_h1=True, cassure=True, retour=True,
              order_block=True):
    """Le cas exact de la vidéo, construit en M5, heure par heure.

    J1 pose la référence journalière (plus bas 97). J2 balaie 97 et clôture
    au-dessus → étape 1. J3 : une heure de référence (plus bas 103, plus haut
    108), l'heure suivante balaie 103 et récupère → étape 2 ; l'heure d'après
    casse 103 en M15 après un Order Block, et le prix y revient → étape 3.

    Chaque étape est débrayable, pour que les tests négatifs retirent **une
    seule** condition à la fois : c'est ce qui attribue une absence de signal à
    la bonne cause.
    """
    rows = []
    # --- J1 : 288 M5, un plus bas net à 97 ---------------------------------
    rows += _plat(100, 100)
    rows += [(100, 100.4, 97.0, 99.0)]
    rows += _plat(100, 187)

    # --- J2 : balaie 97 puis récupère (ou jamais n'y descend) --------------
    if jour_balaye:
        rows += _plat(99, 60)
        rows += [(99, 99.2, 95.0, 98.0)]
        rows += _plat(101, 227)
    else:
        rows += _plat(99, 288)

    # --- J3 ----------------------------------------------------------------
    rows += _heure(105, 105.6, 104.4, 105)              # H+0 : calme
    rows += _heure(105, 108.0, 103.0, 106.0)            # H+1 : RÉFÉRENCE
    if setup_h1:
        rows += _heure(106, 106.4, 101.0, 105.5)        # H+2 : balaie 103
    else:
        rows += _heure(106, 106.4, 104.0, 105.5)        # ne descend pas à 103

    # H+3 : repli sous 103, Order Block, puis cassure (le tout en M15)
    rows += _m15_bar(104, 104.2, 102.0, 102.5)
    if order_block:
        rows += _m15_bar(102.5, 102.8, 101.5, 102.0)    # M15 ROUGE = l'OB
    else:
        rows += _m15_bar(102.0, 102.8, 101.5, 102.5)    # verte : aucun OB
    rows += _m15_bar(102.0, 105.2, 102.0,
                     105.0 if cassure else 102.8)       # impulsion / rien
    if retour:
        rows += _m15_bar(105.0, 105.2, 102.2, 102.4)    # le prix revient
    else:
        rows += _m15_bar(105.0, 108.0, 104.8, 107.0)    # il ne revient pas
    return _m5(rows)


def _frames(m5):
    return (m5, ss.resample(m5, "15min"), ss.resample(m5, "1h"),
            ss.resample(m5, "1D"))


def _eval(m5, **kw):
    f = _frames(m5)
    return ss.evaluate_sweep(f[0], f[1], f[2], f[3],
                             now=m5.index[-1].to_pydatetime(), **kw)


# --------------------------------------------------------------------------- #
# Le chemin nominal
# --------------------------------------------------------------------------- #
def test_the_three_steps_together_produce_a_signal():
    """Sans cette preuve, « 0 trade » en backtest serait indistinguable d'un
    pipeline cassé — et on corrigerait les mauvais seuils. Exactement ce qui
    est arrivé au replay §11.3 du scanner."""
    motifs = {}
    sig = _eval(_scenario(), _reject_log=motifs)
    assert sig is not None, f"aucun signal, motifs = {motifs}"
    assert sig.direction == "long"
    assert sig.meta["niveau_h1"] == pytest.approx(103.0)


def test_the_levels_follow_the_documented_conventions():
    """Entrée au bord de l'Order Block, stop derrière lui moins la marge, TP1
    un peu sous le plus haut H1. Tout doit se retrouver à la main."""
    sig = _eval(_scenario())
    assert sig is not None
    m = sig.meta
    assert sig.entry == pytest.approx(m["ob_haut"])
    assert sig.stop_loss == pytest.approx(
        m["ob_bas"] - ss.SL_MARGIN_ATR * sig.atr)
    assert sig.take_profit1 == pytest.approx(
        m["extreme_h1"] - ss.TP1_MARGIN_ATR * sig.atr)
    assert sig.stop_loss < sig.entry < sig.take_profit1


def test_the_second_target_is_the_opposing_daily_liquidity():
    sig = _eval(_scenario())
    assert sig is not None
    assert sig.take_profit2 >= sig.take_profit1
    if sig.meta["tp2_sur_liquidite"]:
        assert sig.take_profit2 == pytest.approx(sig.meta["pdh"])


def test_the_risk_reward_is_reported_and_recomputable():
    sig = _eval(_scenario())
    assert sig is not None
    attendu = abs(sig.take_profit1 - sig.entry) / abs(sig.entry - sig.stop_loss)
    assert sig.meta["rr_tp1"] == pytest.approx(attendu)
    assert sig.risk_distance == pytest.approx(abs(sig.entry - sig.stop_loss))


# --------------------------------------------------------------------------- #
# Chaque étape, retirée seule
# --------------------------------------------------------------------------- #
def test_without_the_daily_sweep_there_is_no_direction():
    motifs = {}
    assert _eval(_scenario(jour_balaye=False), _reject_log=motifs) is None
    assert "etape1_pas_de_balayage_d1" in motifs


def test_without_the_h1_setup_nothing_fires():
    """Sans le balayage H1 au niveau de référence, le signal de la vidéo ne se
    produit pas.

    Nuance que ce test a révélée et qu'il faut garder sous les yeux : la
    recherche du setup remonte `H1_SETUP_MAX_BARS` bougies, donc elle peut
    tomber sur un balayage H1 **plus ancien** et poursuivre avec lui. C'est le
    comportement voulu — un setup vieux de quelques heures reste un setup —
    mais ça veut dire que l'étape 2 n'est pas une porte binaire : elle élargit
    la fenêtre autant qu'elle filtre. Le test vérifie donc l'absence de signal,
    pas un motif de rejet précis qui dépendrait du reste de la série.
    """
    motifs = {}
    assert _eval(_scenario(setup_h1=False), _reject_log=motifs) is None
    assert sum(motifs.values()) > 0
    assert "etape1_pas_de_balayage_d1" not in motifs   # la direction, elle, est là


def test_without_the_m15_break_nothing_fires():
    motifs = {}
    assert _eval(_scenario(cassure=False), _reject_log=motifs) is None
    assert motifs, "le rejet doit être tracé, pas silencieux"


def test_without_an_order_block_there_is_nothing_to_enter_on():
    """L'entrée est une limite posée sur une zone. Sans bougie contraire avant
    l'impulsion, il n'y a pas de zone — donc pas de prix d'entrée, pas de stop,
    pas de R:R. Mieux vaut aucun trade qu'une entrée au marché inventée."""
    motifs = {}
    assert _eval(_scenario(order_block=False), ob_lookback_bars=1,
                 _reject_log=motifs) is None
    assert "etape3_pas_d_order_block" in motifs

    # Et le constat inverse, qui doit rester visible : au recul par défaut
    # (12 bougies) une bougie contraire existe presque toujours, donc cette
    # condition ne filtre quasiment rien. Ce n'est pas elle qui sélectionne.
    assert _eval(_scenario(order_block=False)) is not None


def test_the_price_must_come_back_into_the_order_block():
    """« Le prix revient, certains entrent. » Tant qu'il n'est pas revenu, il
    n'y a pas d'entrée — c'est une limite, pas un marché."""
    motifs = {}
    assert _eval(_scenario(retour=False), _reject_log=motifs) is None
    assert "etape3_prix_hors_order_block" in motifs


def test_every_rejection_is_named():
    """Un blanc se lirait comme une panne. Chaque refus porte son motif, pour
    qu'un backtest à 0 trade soit diagnosticable."""
    for kw in ({"jour_balaye": False}, {"setup_h1": False},
               {"cassure": False}, {"retour": False}):
        motifs = {}
        _eval(_scenario(**kw), _reject_log=motifs)
        assert sum(motifs.values()) > 0


# --------------------------------------------------------------------------- #
# Anti look-ahead — le test le plus important du fichier
# --------------------------------------------------------------------------- #
def test_a_forming_daily_candle_is_never_used():
    """Le plus bas de la journée en cours n'est connu qu'à minuit. L'utiliser
    revient à lire le futur et produit un backtest qui ne se reproduit jamais.
    """
    m5 = _scenario()
    f = _frames(m5)
    # `now` au milieu du dernier jour : la bougie D1 courante n'est pas close.
    milieu = m5.index[-1].to_pydatetime()
    d1_vus = ss._closed(f[3], milieu, 24 * 60)
    assert len(d1_vus) < len(f[3])
    assert d1_vus.index[-1].date() < pd.Timestamp(milieu).date()


def test_evaluating_on_a_prefix_never_uses_later_bars():
    """La preuve générale : couper la série plus tôt ne peut pas faire
    apparaître d'information venue d'après."""
    m5 = _scenario()
    coupe = m5.iloc[: len(m5) - 3]          # on retire le retour sur l'OB
    f = _frames(coupe)
    sig = ss.evaluate_sweep(f[0], f[1], f[2], f[3],
                            now=coupe.index[-1].to_pydatetime())
    assert sig is None      # le prix n'était pas encore revenu dans la zone


def test_an_empty_series_is_refused_rather_than_guessed():
    vide = _m5([])
    assert ss.evaluate_sweep(vide, vide, vide, vide,
                             now=pd.Timestamp("2026-01-05", tz="UTC")
                             .to_pydatetime()) is None


# --------------------------------------------------------------------------- #
# Ce que la stratégie n'emprunte PAS au mode A
# --------------------------------------------------------------------------- #
def test_no_filter_of_the_existing_pipeline_is_applied():
    """Le cœur du choix de conception. Greffer la règle sur les neuf filtres du
    mode A répondrait à « survit-elle ? » et non à « vaut-elle quelque
    chose ? » — et les survivants seraient les balayages qui ressemblent à de
    la continuation, c'est-à-dire le sous-ensemble qui n'est pas la méthode."""
    source = open(ss.__file__, encoding="utf-8").read()
    for emprunt in ("active_session", "is_bad_timing", "compute_bias",
                    "ADX_MIN", "RSI_M5", "vwap", "find_patterns",
                    "TP1_R", "TP2_R"):
        assert emprunt not in source, f"« {emprunt} » ne doit pas être utilisé"


def test_the_signal_carries_the_numbers_that_produced_it():
    sig = _eval(_scenario())
    assert sig is not None
    for clef in ("niveau_h1", "extreme_h1", "pdh", "pdl", "ob_haut", "ob_bas",
                 "rr_tp1", "depassement_d1", "recuperation_d1"):
        assert clef in sig.meta, clef


def test_the_minimum_rr_is_off_by_default_and_can_be_switched_on():
    """La vidéo n'impose aucun R:R — elle dit seulement « SL proche, TP loin,
    moins de chances ». Le défaut reste donc fidèle, et le seuil est testable.
    """
    assert ss.MIN_RR == 0.0
    sig = _eval(_scenario())
    assert sig is not None
    assert _eval(_scenario(), min_rr=sig.meta["rr_tp1"] + 1.0) is None


def test_the_max_duration_travels_in_the_meta_too():
    """Le live lit la durée maximale dans `pos.meta` (défaut 75 min), le
    pretrain sur `sig.max_duration_min` (240). Sans la clé dans le meta, la
    même stratégie tiendrait ses trades trois fois moins longtemps en live
    qu'en backtest, sans la moindre erreur visible. C'est la famille de bug qui
    a déjà coûté le biais H1 à ce dépôt."""
    sig = _eval(_scenario())
    assert sig is not None
    assert sig.meta["max_duration_min"] == sig.max_duration_min == ss.MAX_DURATION_MIN

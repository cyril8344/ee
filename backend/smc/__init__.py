"""Scanner SMC multi-timeframes — analyse seule, aucune exécution.

Ce paquet ne passe AUCUN ordre et n'a aucun accès en écriture au compte. Il
détecte de la structure et des zones, puis alerte ; l'utilisateur passe ses
ordres à la main. Il est donc volontairement séparé de la boucle de trading de
`main.py` : rien ici ne doit pouvoir influencer une décision du bot.

Ordre d'implémentation (§11 de la spec) :
  1. data_feed + structure   ← livré
  2. zones
  3. confluence + sweep, validés en replay (2–6 signaux/semaine attendus)
  4. charting + alerting + logger
  5. main, puis 2 semaines d'observation avant de trader le moindre signal
"""

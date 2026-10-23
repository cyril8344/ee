"""Les dépendances du backend doivent être déclarées LÀ OÙ ELLES SONT INSTALLÉES.

Ce dépôt a **deux** fichiers de dépendances, et un seul compte en production :

- `requirements.txt` (racine) — utilisé par le `Makefile`, donc dev local seul ;
- `backend/requirements.txt` — celui qu'installent **`nixpacks.toml` (Railway)**
  et `backend/Dockerfile`.

Ils ont des épinglages différents (fastapi 0.111 contre 0.110, requests 2.32.3
contre 2.31.0), donc les fusionner changerait les versions déployées : ce n'est
pas une décision à prendre en passant. Mais leur ressemblance est un piège.

Cas réel : Pillow ajouté à la racine seulement. Tous les tests passaient en
local, le build Railway réussissait, et la fonctionnalité échouait chez
l'utilisateur avec « No module named 'PIL' ». Rien dans la chaîne ne l'a signalé
— exactement la famille de pannes silencieuses que ce dépôt traque ailleurs.
"""
import re
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[2]
DEPLOYE = RACINE / "backend" / "requirements.txt"
LOCAL = RACINE / "requirements.txt"

# Module importé par le backend → nom de distribution attendu dans le fichier.
# Uniquement des dépendances dont l'absence casse une fonctionnalité en
# production ; les outils de test n'ont pas à être déployés.
DEPENDANCES_RUNTIME = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "pydantic": "pydantic",
    "pandas": "pandas",
    "numpy": "numpy",
    "requests": "requests",
    "yfinance": "yfinance",
    "websockets": "websockets",
    "multipart": "python-multipart",   # uploads de fichiers (capture d'écran)
    "PIL": "pillow",                   # lecture des captures (smc/chart_image.py)
    "optuna": "optuna",
    "gymnasium": "gymnasium",
}


def _distributions(path: Path) -> set:
    noms = set()
    for ligne in path.read_text().splitlines():
        ligne = ligne.split("#")[0].strip()
        if not ligne:
            continue
        nom = re.split(r"[<>=!\[]", ligne)[0].strip()
        if nom:
            noms.add(nom.lower())
    return noms


@pytest.mark.parametrize("module,distribution", sorted(DEPENDANCES_RUNTIME.items()))
def test_every_runtime_dependency_is_in_the_deployed_file(module, distribution):
    """C'est `backend/requirements.txt` que Railway installe, pas celui de la
    racine. Une dépendance déclarée uniquement à la racine n'existe pas en
    production, et la panne n'apparaît qu'à l'usage."""
    assert distribution in _distributions(DEPLOYE), (
        f"{distribution} (importé comme `{module}`) manque dans "
        f"backend/requirements.txt — le fichier qu'installent nixpacks.toml et "
        f"backend/Dockerfile. L'ajouter à la racine ne suffit pas."
    )


@pytest.mark.parametrize("module,distribution", sorted(DEPENDANCES_RUNTIME.items()))
def test_the_two_files_agree_on_runtime_dependencies(module, distribution):
    """Les versions peuvent diverger — les paquets, non. Sans ça, un test qui
    passe en local ne dit plus rien sur ce qui tourne en production."""
    assert distribution in _distributions(LOCAL), (
        f"{distribution} manque dans requirements.txt (racine) : les tests "
        f"locaux ne couvriraient pas ce qui est déployé."
    )


def test_the_deployed_file_is_the_one_the_build_installs():
    """Garde-fou sur le lien lui-même : si le build changeait de fichier, les
    tests ci-dessus vérifieraient le mauvais."""
    nixpacks = (RACINE / "nixpacks.toml").read_text()
    assert "backend/requirements.txt" in nixpacks, (
        "nixpacks.toml n'installe plus backend/requirements.txt — mettre à jour "
        "DEPLOYE dans ce fichier de tests."
    )

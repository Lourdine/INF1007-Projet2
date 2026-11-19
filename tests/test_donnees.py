"""
test_donnees.py — Tests unitaires du chargement, nettoyage et validation des données
pour le projet INF1007 Projet 2.
"""

import pandas as pd
import pytest
from donnees import (
    COLONNES_REQUISES,
    VariableNumerique,
    VariableCategorielle,
    charger_donnees,
    nettoyer_donnees,
    obtenir_features_cible,
    JeuDeDonnees,
)
from exceptions import (
    FichierIntrouvableError,
    ColonneManquanteError,
    ValeurHorsIntervalleError,
    TypeInvalideError,
    DonneeInvalideError,
)


def _ligne_valide(**overrides) -> dict:
    """Construit une ligne conforme aux 13 colonnes attendues par donnees.py."""
    base = {
        "AGE": "26-39",
        "GENDER": "female",
        "DRIVING_EXPERIENCE": "10-19y",
        "CREDIT_SCORE": 0.72,
        "VEHICLE_YEAR": "after 2015",
        "MARRIED": 1,
        "VEHICLE_OWNERSHIP": 1,
        "ANNUAL_MILEAGE": 12000.0,
        "SPEEDING_VIOLATIONS": 1,
        "PAST_ACCIDENTS": 0,
        "DUIS": 0,
        "VEHICLE_TYPE": "sedan",
        "OUTCOME": 0,
    }
    base.update(overrides)
    return base


def test_charger_donnees_valide_et_erreurs(tmp_path):
    """Teste le chargement d'un CSV valide, d'un fichier absent et d'une colonne manquante."""
    csv_valide = tmp_path / "valide.csv"
    pd.DataFrame([_ligne_valide(), _ligne_valide(OUTCOME=1)]).to_csv(csv_valide, index=False)

    df = charger_donnees(str(csv_valide))
    assert list(df.columns) == COLONNES_REQUISES
    assert len(df) == 2

    with pytest.raises(FichierIntrouvableError):
        charger_donnees(str(tmp_path / "introuvable.csv"))

    csv_incomplet = tmp_path / "incomplet.csv"
    df_sans_cible = pd.DataFrame([_ligne_valide()]).drop(columns=["OUTCOME"])
    df_sans_cible.to_csv(csv_incomplet, index=False)

    with pytest.raises(ColonneManquanteError):
        charger_donnees(str(csv_incomplet))


def test_nettoyer_donnees_et_obtenir_features_cible():
    """Teste la suppression des NaN, la réindexation, les bornes limites et la séparation X, y."""
    df_brut = pd.DataFrame([
        _ligne_valide(CREDIT_SCORE=0.0, ANNUAL_MILEAGE=0.0, OUTCOME=0),
        _ligne_valide(CREDIT_SCORE=1.0, ANNUAL_MILEAGE=25000.0, OUTCOME=1),
        _ligne_valide(CREDIT_SCORE=None, OUTCOME=0),
    ])

    df_propre = nettoyer_donnees(df_brut)
    assert len(df_propre) == 2
    assert list(df_propre.index) == [0, 1]

    X, y = obtenir_features_cible(df_propre)
    assert "OUTCOME" not in X.columns
    assert X.shape[1] == len(COLONNES_REQUISES) - 1
    assert list(y) == [0, 1]


def test_nettoyer_donnees_valeurs_aberrantes():
    """Vérifie que CREDIT_SCORE hors [0, 1] ou ANNUAL_MILEAGE négatif lève ValeurHorsIntervalleError."""
    df_credit_invalide = pd.DataFrame([_ligne_valide(CREDIT_SCORE=1.2)])
    with pytest.raises(ValeurHorsIntervalleError):
        nettoyer_donnees(df_credit_invalide)

    df_kilometrage_negatif = pd.DataFrame([_ligne_valide(ANNUAL_MILEAGE=-10.0)])
    with pytest.raises(ValeurHorsIntervalleError):
        nettoyer_donnees(df_kilometrage_negatif)


def test_polymorphisme_variables_et_jeu_de_donnees(tmp_path):
    """Teste VariableNumerique, VariableCategorielle et le cycle complet de JeuDeDonnees."""
    var_num = VariableNumerique("CREDIT_SCORE", "Score de crédit", 0.0, 1.0)
    var_cat = VariableCategorielle("GENDER", "Genre", ["male", "female"])

    assert var_num.valider(0.0) is True
    assert var_num.valider(1.0) is True
    assert var_cat.valider("female") is True
    assert "CREDIT_SCORE" in str(var_num)
    assert "VariableNumerique" in repr(var_num)

    with pytest.raises(TypeInvalideError):
        var_num.valider("0.5")

    with pytest.raises(ValeurHorsIntervalleError):
        var_num.valider(2.5)

    with pytest.raises(DonneeInvalideError):
        var_cat.valider("inconnu")

    chemin_csv = tmp_path / "jeu.csv"
    pd.DataFrame([_ligne_valide(), _ligne_valide(OUTCOME=1)]).to_csv(chemin_csv, index=False)

    jeu = JeuDeDonnees(str(chemin_csv))
    with pytest.raises(ValueError):
        jeu.nettoyer()

    jeu.charger()
    jeu.nettoyer()
    X, y = jeu.obtenir_features_cible()
    assert len(X) == 2
    assert len(y) == 2
    assert "lignes=2" in str(jeu)
    
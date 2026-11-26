"""
tests/test_donnees.py — Tests unitaires pour donnees.py

On vérifie le chargement, le nettoyage, la séparation des données,
et les classes Variable, VariableNumerique, VariableCategorielle, JeuDeDonnees.
pytest découvre automatiquement les fonctions qui commencent par test_.
"""

import pytest
import pandas as pd
from exceptions import (
    FichierIntrouvableError,
    ValeurHorsIntervalleError,
    TypeInvalideError,
    DonneeInvalideError,
)
from donnees import (
    charger_donnees,
    nettoyer_donnees,
    obtenir_features_cible,
    VariableNumerique,
    VariableCategorielle,
    JeuDeDonnees,
)


def creer_df_valide():
    """DataFrame minimal valide pour les tests — 3 conducteurs, 13 colonnes."""
    return pd.DataFrame({
        "AGE":                ["16-25", "26-39", "40-64"],
        "GENDER":             ["male", "female", "male"],
        "DRIVING_EXPERIENCE": ["0-9y", "10-19y", "20-29y"],
        "CREDIT_SCORE":       [0.5, 0.7, 0.3],
        "VEHICLE_YEAR":       ["before 2015", "after 2015", "before 2015"],
        "MARRIED":            ["0", "1", "0"],
        "VEHICLE_OWNERSHIP":  [1, 0, 1],
        "ANNUAL_MILEAGE":     [12000.0, 15000.0, 10000.0],
        "SPEEDING_VIOLATIONS":[0, 1, 2],
        "PAST_ACCIDENTS":     [0, 0, 1],
        "DUIS":               [0, 0, 0],
        "VEHICLE_TYPE":       ["sedan", "sedan", "sports car"],
        "OUTCOME":            [0, 1, 0],
    })


# ── charger_donnees ────────────────────────────────────────────────────────────

def test_charger_donnees_fichier_introuvable():
    """
    charger_donnees() doit lever FichierIntrouvableError si le chemin n'existe pas.
    pytest.raises vérifie que l'exception est bien levée.
    """
    with pytest.raises(FichierIntrouvableError):
        charger_donnees("chemin/inexistant.csv")


# ── nettoyer_donnees ───────────────────────────────────────────────────────────

def test_nettoyer_donnees_supprime_lignes_nulles():
    """
    nettoyer_donnees() doit supprimer les lignes avec des valeurs manquantes (NaN).
    dropna() retire ces lignes — ici on ajoute une ligne vide et on vérifie qu'elle disparaît.
    """
    df = creer_df_valide()
    ligne_nulle = pd.DataFrame([{col: None for col in df.columns}])
    df_avec_null = pd.concat([df, ligne_nulle], ignore_index=True)

    df_nettoye = nettoyer_donnees(df_avec_null)
    assert df_nettoye.isnull().sum().sum() == 0


def test_nettoyer_donnees_leve_erreur_credit_score_hors_intervalle():
    """
    nettoyer_donnees() doit lever ValeurHorsIntervalleError si CREDIT_SCORE > 1.
    Cas d'erreur avec pytest.raises.
    """
    df = creer_df_valide()
    df.loc[0, "CREDIT_SCORE"] = 1.5  # valeur invalide
    with pytest.raises(ValeurHorsIntervalleError):
        nettoyer_donnees(df)


# ── obtenir_features_cible ─────────────────────────────────────────────────────

def test_obtenir_features_cible_separe_correctement():
    """
    obtenir_features_cible() doit retourner X sans OUTCOME et y = OUTCOME.
    """
    df = creer_df_valide()
    X, y = obtenir_features_cible(df)
    assert "OUTCOME" not in X.columns
    assert list(y) == [0, 1, 0]


def test_obtenir_features_cible_x_a_12_colonnes():
    """
    X doit contenir exactement 12 colonnes (toutes sauf OUTCOME).
    """
    df = creer_df_valide()
    X, _ = obtenir_features_cible(df)
    assert X.shape[1] == 12


# ── VariableNumerique ──────────────────────────────────────────────────────────

def test_variable_numerique_valide_valeur_correcte():
    """
    VariableNumerique.valider() doit retourner True pour une valeur dans l'intervalle.
    """
    v = VariableNumerique("CREDIT_SCORE", "Score de crédit", 0.0, 1.0)
    assert v.valider(0.5) is True


def test_variable_numerique_valide_borne_exacte():
    # 1.0 est pile sur la borne — doit être accepté (cas limite)
    v = VariableNumerique("CREDIT_SCORE", "Score de crédit", 0.0, 1.0)
    assert v.valider(1.0) is True


def test_variable_numerique_leve_erreur_hors_intervalle():
    """
    VariableNumerique.valider() doit lever ValeurHorsIntervalleError si hors intervalle.
    """
    v = VariableNumerique("CREDIT_SCORE", "Score de crédit", 0.0, 1.0)
    with pytest.raises(ValeurHorsIntervalleError):
        v.valider(1.5)


def test_variable_numerique_leve_erreur_type_invalide():
    """
    VariableNumerique.valider() doit lever TypeInvalideError si ce n'est pas un nombre.
    """
    v = VariableNumerique("CREDIT_SCORE", "Score de crédit", 0.0, 1.0)
    with pytest.raises(TypeInvalideError):
        v.valider("abc")


# ── VariableCategorielle ───────────────────────────────────────────────────────

def test_variable_categorielle_valide_valeur_correcte():
    """
    VariableCategorielle.valider() doit retourner True pour une valeur dans les catégories.
    """
    v = VariableCategorielle("GENDER", "Genre", ["male", "female"])
    assert v.valider("male") is True


def test_variable_categorielle_leve_erreur_valeur_inconnue():
    """
    VariableCategorielle.valider() doit lever DonneeInvalideError pour une valeur inconnue.
    """
    v = VariableCategorielle("GENDER", "Genre", ["male", "female"])
    with pytest.raises(DonneeInvalideError):
        v.valider("unknown")


# ── __str__ et __repr__ ────────────────────────────────────────────────────────

def test_variable_str_contient_nom():
    """
    __str__ de VariableNumerique doit retourner une chaîne contenant le nom de la variable.
    """
    v = VariableNumerique("CREDIT_SCORE", "Score de crédit", 0.0, 1.0)
    assert "CREDIT_SCORE" in str(v)


# ── JeuDeDonnees ───────────────────────────────────────────────────────────────

def test_jeu_de_donnees_str_contient_chemin():
    """
    __str__ de JeuDeDonnees doit mentionner le chemin passé au constructeur.
    """
    jdd = JeuDeDonnees("donnees/dataset.csv")
    assert "donnees/dataset.csv" in str(jdd)


def test_jeu_de_donnees_nettoyer_sans_charger_leve_erreur():
    """
    Appeler nettoyer() sans avoir appelé charger() doit lever ValueError.
    """
    jdd = JeuDeDonnees("donnees/dataset.csv")
    with pytest.raises(ValueError):
        jdd.nettoyer()
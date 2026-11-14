"""
donnees.py — Chargement et nettoyage du dataset pour INF1007 Projet 2.
Sujet : Prédiction de réclamation d'assurance auto (classification binaire).
"""

import pandas as pd
from exceptions import (
    FichierIntrouvableError,    # Levée si le fichier CSV est introuvable
    ColonneManquanteError,      # Levée si une colonne attendue est absente du CSV
    ValeurHorsIntervalleError,  # Levée si une valeur est hors de la plage autorisée
)

# Colonnes retenues pour le projet : 12 variables d'entrée + OUTCOME (cible).
# Exclusions volontaires : ID (inutile), RACE (sensible), INCOME, CHILDREN, EDUCATION, POSTAL_CODE.
# POSTAL_CODE exclu : variable géographique arbitraire, non pertinente pour prédire une réclamation.
# OUTCOME = 0 (pas de réclamation) ou 1 (réclamation) — c'est ce qu'on veut prédire.
COLONNES_REQUISES = [
    "AGE", "GENDER", "DRIVING_EXPERIENCE", "CREDIT_SCORE",
    "VEHICLE_YEAR", "MARRIED", "VEHICLE_OWNERSHIP", "ANNUAL_MILEAGE",
    "SPEEDING_VIOLATIONS", "PAST_ACCIDENTS", "DUIS",
    "VEHICLE_TYPE", "OUTCOME"
]


def charger_donnees(chemin: str) -> pd.DataFrame:
    """
    Lit le fichier CSV depuis le chemin donné et retourne un DataFrame Pandas.
    Utilise nos exceptions personnalisées pour afficher des messages clairs à l'utilisateur.
    """
    try:
        df = pd.read_csv(chemin)  # Pandas lit le CSV et le convertit en tableau (DataFrame)
    except FileNotFoundError:
        # On remplace l'erreur générique de Python par notre message personnalisé
        raise FichierIntrouvableError(chemin)

    # On vérifie que toutes les colonnes nécessaires sont présentes.
    # Un mauvais fichier CSV pourrait en manquer, ce qui planterait le modèle plus loin.
    for col in COLONNES_REQUISES:
        if col not in df.columns:
            raise ColonneManquanteError(col)

    return df[COLONNES_REQUISES]  # On garde seulement les 13 colonnes choisies


def nettoyer_donnees(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie le DataFrame avant l'entraînement du modèle.
    Un modèle ML ne peut pas traiter des valeurs manquantes ou invalides.
    """
    # Supprime les lignes incomplètes — une case vide rendrait les calculs du modèle incorrects
    df = df.dropna()

    # Valide les valeurs critiques ligne par ligne :
    # CREDIT_SCORE doit être entre 0 et 1 (score normalisé dans ce dataset)
    # ANNUAL_MILEAGE ne peut pas être négatif (impossible physiquement)
    for _, row in df.iterrows():  # _ = numéro de ligne ignoré, row = données de la ligne
        if not (0 <= row["CREDIT_SCORE"] <= 1):
            raise ValeurHorsIntervalleError("CREDIT_SCORE", row["CREDIT_SCORE"], 0, 1)
        if row["ANNUAL_MILEAGE"] < 0:
            raise ValeurHorsIntervalleError("ANNUAL_MILEAGE", row["ANNUAL_MILEAGE"], 0, float("inf"))

    # Après suppression de lignes, les index ont des trous (0,1,3,5...).
    # reset_index remet une numérotation propre 0,1,2,3... pour éviter des bugs plus loin.
    return df.reset_index(drop=True)


def obtenir_features_cible(df: pd.DataFrame):
    """
    Sépare le DataFrame en X (variables d'entrée) et y (cible) — séparation standard en ML.
    X = ce qu'on donne au modèle pour apprendre (12 variables).
    y = ce qu'on veut qu'il prédise (OUTCOME : 0 ou 1).
    """
    X = df.drop(columns=["OUTCOME"])  # Toutes les colonnes sauf OUTCOME
    y = df["OUTCOME"]                 # Seulement OUTCOME
    return X, y
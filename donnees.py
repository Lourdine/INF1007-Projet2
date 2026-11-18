"""
donnees.py — Chargement, nettoyage et classes de données pour INF1007 Projet 2.
Sujet : Prédiction de réclamation d'assurance auto (classification binaire).
"""

import pandas as pd                  # Bibliothèque pour manipuler les données sous forme de tableau
from abc import ABC, abstractmethod  # ABC = outil Python pour créer des classes abstraites (non instanciables)
from exceptions import (
    FichierIntrouvableError,         # Levée si le fichier CSV est introuvable
    ColonneManquanteError,           # Levée si une colonne attendue est absente du CSV
    ValeurHorsIntervalleError,       # Levée si une valeur est hors de la plage autorisée
    TypeInvalideError,               # Levée si une valeur n'est pas du bon type
    DonneeInvalideError,             # Levée si une valeur catégorielle est inconnue
)

COLONNES_REQUISES = [
    "AGE", "GENDER", "DRIVING_EXPERIENCE", "CREDIT_SCORE",
    "VEHICLE_YEAR", "MARRIED", "VEHICLE_OWNERSHIP", "ANNUAL_MILEAGE",
    "SPEEDING_VIOLATIONS", "PAST_ACCIDENTS", "DUIS",
    "VEHICLE_TYPE", "OUTCOME"
]


class Variable(ABC):                 # ABC = classe abstraite — ne peut pas être instanciée directement
    """Classe abstraite représentant une variable d'entrée du modèle."""

    def __init__(self, nom: str, description: str):
        self._nom = nom              # Nom de la variable ex: "CREDIT_SCORE"
        self._description = description  # Description lisible ex: "Score de crédit entre 0 et 1"

    @property
    def nom(self) -> str:            # @property = on accède à _nom comme un attribut, pas une méthode
        """Retourne le nom de la variable (lecture seule)."""
        return self._nom

    @abstractmethod                  # Force chaque sous-classe à redéfinir cette méthode
    def valider(self, valeur) -> bool:
        """Valide la valeur — chaque sous-classe implémente sa propre logique (polymorphisme)."""
        pass

    def __str__(self) -> str:        # Méthode spéciale appelée par print()
        return f"{self._nom} : {self._description}"

    def __repr__(self) -> str:       # Représentation technique pour le débogage
        return f"{self.__class__.__name__}(nom={self._nom!r})"


class VariableNumerique(Variable):   # Hérite de Variable — redéfinit valider() pour les nombres
    """Variable dont la valeur doit être un nombre dans [minimum, maximum]."""

    def __init__(self, nom: str, description: str, minimum: float, maximum: float):
        super().__init__(nom, description)  # Appelle le constructeur de Variable
        self._minimum = minimum      # Borne inférieure autorisée
        self._maximum = maximum      # Borne supérieure autorisée

    def valider(self, valeur) -> bool:  # Redéfinit valider() de Variable (polymorphisme)
        """Vérifie que la valeur est un nombre dans [minimum, maximum]."""
        if not isinstance(valeur, (int, float)):       # Vérifie que c'est un nombre
            raise TypeInvalideError(self._nom, "float", valeur)
        if not (self._minimum <= valeur <= self._maximum):  # Vérifie l'intervalle
            raise ValeurHorsIntervalleError(self._nom, valeur, self._minimum, self._maximum)
        return True

    def __str__(self) -> str:
        return f"{self._nom} : {self._description} [{self._minimum}, {self._maximum}]"


class VariableCategorielle(Variable):  # Hérite de Variable — redéfinit valider() pour les catégories
    """Variable dont la valeur doit appartenir à un ensemble fixe de catégories."""

    def __init__(self, nom: str, description: str, categories: list):
        super().__init__(nom, description)  # Appelle le constructeur de Variable
        self._categories = categories       # Liste des valeurs autorisées ex: ["male", "female"]

    def valider(self, valeur) -> bool:  # Redéfinit valider() de Variable (polymorphisme)
        """Vérifie que la valeur fait partie des catégories autorisées."""
        if valeur not in self._categories:  # Si la valeur n'est pas dans la liste
            raise DonneeInvalideError(
                f"'{self._nom}' doit être l'une des valeurs : {self._categories}, "
                f"mais a reçu : {repr(valeur)}."
            )
        return True

    def __str__(self) -> str:
        return f"{self._nom} : {self._description} {self._categories}"


def charger_donnees(chemin: str) -> pd.DataFrame:
    """Lit le fichier CSV et retourne un DataFrame Pandas."""
    try:
        df = pd.read_csv(chemin)     # Pandas lit le CSV et le convertit en tableau
    except FileNotFoundError:
        raise FichierIntrouvableError(chemin)  # Remplace l'erreur générique par notre message
    for col in COLONNES_REQUISES:    # Vérifie que toutes les colonnes nécessaires sont présentes
        if col not in df.columns:
            raise ColonneManquanteError(col)
    return df[COLONNES_REQUISES]     # Garde seulement les 13 colonnes choisies


def nettoyer_donnees(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie le DataFrame avant l'entraînement du modèle."""
    df = df.dropna()                 # Supprime les lignes avec des valeurs manquantes
    for _, row in df.iterrows():     # _ = numéro de ligne ignoré, row = données de la ligne
        if not (0 <= row["CREDIT_SCORE"] <= 1):         # CREDIT_SCORE doit être entre 0 et 1
            raise ValeurHorsIntervalleError("CREDIT_SCORE", row["CREDIT_SCORE"], 0, 1)
        if row["ANNUAL_MILEAGE"] < 0:                   # ANNUAL_MILEAGE ne peut pas être négatif
            raise ValeurHorsIntervalleError("ANNUAL_MILEAGE", row["ANNUAL_MILEAGE"], 0, float("inf"))
    return df.reset_index(drop=True)  # Remet une numérotation propre 0,1,2,3... après dropna()


def obtenir_features_cible(df: pd.DataFrame) -> tuple:
    """Sépare le DataFrame en X (variables d'entrée) et y (cible OUTCOME)."""
    X = df.drop(columns=["OUTCOME"])  # Toutes les colonnes sauf OUTCOME
    y = df["OUTCOME"]                 # Seulement OUTCOME (0 ou 1)
    return X, y                       # Retourne un tuple (X, y)


class JeuDeDonnees:
    """Encapsule le chargement, le nettoyage et la préparation des données."""

    def __init__(self, chemin: str):
        self._chemin = chemin        # Chemin vers le fichier CSV
        self._df = None              # DataFrame brut — rempli par charger()
        self._df_propre = None       # DataFrame nettoyé — rempli par nettoyer()

    def charger(self) -> None:
        """Charge le fichier CSV dans _df."""
        self._df = charger_donnees(self._chemin)  # Réutilise la fonction charger_donnees()

    def nettoyer(self) -> None:
        """Nettoie _df et stocke le résultat dans _df_propre."""
        if self._df is None:         # Vérifie qu'on a chargé les données avant
            raise ValueError("Les données ne sont pas chargées. Appelez charger() d'abord.")
        self._df_propre = nettoyer_donnees(self._df)  # Réutilise nettoyer_donnees()

    def obtenir_features_cible(self) -> tuple:
        """Retourne (X, y) prêts pour l'entraînement."""
        if self._df_propre is None:  # Vérifie qu'on a nettoyé avant
            raise ValueError("Les données ne sont pas nettoyées. Appelez nettoyer() d'abord.")
        return obtenir_features_cible(self._df_propre)  # Réutilise obtenir_features_cible()

    def __str__(self) -> str:        # Méthode spéciale appelée par print()
        nb_lignes = len(self._df_propre) if self._df_propre is not None else 0
        return f"JeuDeDonnees(chemin='{self._chemin}', lignes={nb_lignes})"

    def __repr__(self) -> str:       # Représentation technique pour le débogage
        return f"JeuDeDonnees(chemin={self._chemin!r})"
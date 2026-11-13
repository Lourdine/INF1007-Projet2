"""
modele.py — Entraînement et prédiction du modèle ML pour INF1007 Projet 2.
Sujet : Prédiction de réclamation d'assurance auto (classification binaire).
"""

import joblib  # Sauvegarde et chargement du modèle sur disque
import pandas as pd
from sklearn.ensemble import RandomForestClassifier  # 100 arbres de décision qui votent ensemble
from sklearn.model_selection import train_test_split  # Sépare les données 80% train / 20% test
from sklearn.metrics import accuracy_score  # Calcule le % de bonnes prédictions
from sklearn.preprocessing import LabelEncoder  # Convertit le texte en chiffres (male→0, female→1)
from exceptions import ModeleNonEntraineError


class Modele:
    """Encapsule le modèle ML : entraînement, prédiction, sauvegarde."""

    def __init__(self):
        self._modele = None  # Pas encore entraîné — sera créé dans entrainer()
        self._encodeurs = {}  # Clé = nom de colonne, valeur = son LabelEncoder

    def _encoder_donnees(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convertit les colonnes texte en chiffres pour que le modèle puisse les lire."""
        X = X.copy()  # On copie pour ne pas modifier le DataFrame original
        for col in X.select_dtypes(include="object").columns:  # Pour chaque colonne texte
            if col not in self._encodeurs:
                self._encodeurs[col] = LabelEncoder()  # Crée un encodeur pour cette colonne
                X[col] = self._encodeurs[col].fit_transform(X[col])  # Apprend ET convertit
            else:
                X[col] = self._encodeurs[col].transform(X[col])  # Réutilise le même encodeur
        return X

    def entrainer(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Entraîne le modèle et retourne la précision obtenue (entre 0 et 1)."""
        X_encode = self._encoder_donnees(X)  # Étape 1 : convertir le texte en chiffres
        X_train, X_test, y_train, y_test = train_test_split(
            X_encode, y, test_size=0.2, random_state=42  # 80% entraînement, 20% test
        )
        self._modele = RandomForestClassifier(n_estimators=100, random_state=42)
        self._modele.fit(X_train, y_train)  # Le modèle apprend le lien entre X et y
        predictions = self._modele.predict(X_test)  # Teste sur les 20% mis de côté
        return accuracy_score(y_test, predictions)  # % de bonnes prédictions

    def predire(self, X: pd.DataFrame) -> list:
        """Prédit 0 (pas de réclamation) ou 1 (réclamation) pour chaque ligne de X."""
        if self._modele is None:
            raise ModeleNonEntraineError()  # On ne peut pas prédire sans modèle entraîné
        X_encode = self._encoder_donnees(X)  # Même encodage qu'à l'entraînement
        return self._modele.predict(X_encode).tolist()  # Retourne une liste Python de 0 et 1

    def sauvegarder(self, chemin: str) -> None:
        """Sauvegarde le modèle dans un fichier pour ne pas réentraîner à chaque fois."""
        joblib.dump(self._modele, chemin)  # Écrit le modèle sur disque

    def charger(self, chemin: str) -> None:
        """Recharge un modèle sauvegardé depuis un fichier."""
        self._modele = joblib.load(chemin)  # Remet le modèle en mémoire

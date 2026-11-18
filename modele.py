
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
import json  # Pour sauvegarder les métadonnées du modèle en JSON
import numpy as np 
import os

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
        accuracy = accuracy_score(y_test, predictions)  # % de bonnes prédictions

        # Modèle trivial — prédit toujours la classe majoritaire (point de comparaison)
        
        classe_majoritaire = np.bincount(y_test).argmax()
        predictions_triviales = [classe_majoritaire] * len(y_test)
        accuracy_trivial = accuracy_score(y_test, predictions_triviales)
        print(f"Modèle trivial     — précision : {accuracy_trivial:.2%}")
        print(f"Modèle RandomForest — précision : {accuracy:.2%}")
        print(f"Gain par rapport au trivial : +{(accuracy - accuracy_trivial):.2%}")

        return accuracy

    def predire(self, X: pd.DataFrame) -> list:
        """Prédit 0 (pas de réclamation) ou 1 (réclamation) pour chaque ligne de X."""
        if self._modele is None:
            raise ModeleNonEntraineError()  # On ne peut pas prédire sans modèle entraîné
        X_encode = self._encoder_donnees(X)  # Même encodage qu'à l'entraînement
        return self._modele.predict(X_encode).tolist()  # Retourne une liste Python de 0 et 1
    def predire_probabilite(self, X: pd.DataFrame) -> float:
        """Retourne la probabilité de réclamation en pourcentage (0 à 100)."""
        if self._modele is None:
            raise ModeleNonEntraineError()  # On ne peut pas prédire sans modèle entraîné
        X_encode = self._encoder_donnees(X)              # Même encodage qu'à l'entraînement
        proba = self._modele.predict_proba(X_encode)[0]  # [proba_classe_0, proba_classe_1]
        return round(proba[1] * 100, 1)                  # % de probabilité de réclamation
    
    def sauvegarder(self, chemin: str) -> None:
        """Sauvegarde le modèle dans un fichier pour ne pas réentraîner à chaque fois."""
        joblib.dump(self._modele, chemin)  # Écrit le modèle sur disque
    
    def sauvegarder_metadonnees(self, X: pd.DataFrame, precision: float, chemin: str) -> None:
        """Sauvegarde les métadonnées du modèle dans un fichier JSON.
        Contient : noms des colonnes, valeurs catégorielles, intervalles numériques, précision."""
    
        colonnes_categories = X.select_dtypes(include="object").columns.tolist()
        colonnes_numeriques = X.select_dtypes(exclude="object").columns.tolist()

        metadonnees = {
            "colonnes": X.columns.tolist(),           # Ordre exact des colonnes
            "metrique": "accuracy",
            "precision": round(precision, 4),
            "categories": {
                col: sorted(X[col].unique().tolist()) # Valeurs possibles pour chaque catégorie
                for col in colonnes_categories
            },
            "numeriques": {
                col: {
                    "min": round(float(X[col].min()), 4),
                    "max": round(float(X[col].max()), 4)
                }
                for col in colonnes_numeriques
            }
        }

        os.makedirs(os.path.dirname(chemin), exist_ok=True)  # Crée le dossier modeles/ si absent
        with open(chemin, "w", encoding="utf-8") as f:
            json.dump(metadonnees, f, indent=4, ensure_ascii=False)
    def charger(self, chemin: str) -> None:
        """Recharge un modèle sauvegardé depuis un fichier."""
        self._modele = joblib.load(chemin)  # Remet le modèle en mémoire




    def __str__(self) -> str:
        """Représentation lisible du modèle — affiche son état et le nombre d'arbres."""
        if self._modele is None:
            return "Modele(non entraîné)"
        n_arbres = self._modele.n_estimators
        return f"Modele(RandomForest, {n_arbres} arbres, entraîné)"

    def __repr__(self) -> str:
        """Représentation technique pour le débogage."""
        entraine = self._modele is not None
        return f"Modele(entraine={entraine})"
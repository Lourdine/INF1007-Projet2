"""
test_modele.py — Tests unitaires de la classe Modele (entraînement, évaluation,
prédiction, sauvegarde/chargement et métadonnées) pour INF1007 Projet 2.
"""

import json
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score
from modele import Modele
from exceptions import ModeleNonEntraineError


def _creer_dataset_synthetique(n: int = 100) -> tuple[pd.DataFrame, pd.Series]:
    """Génère un jeu de données synthétique reproductible pour tester Modele."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame({
        "AGE": rng.choice(["16-25", "26-39", "40-64", "65+"], size=n),
        "GENDER": rng.choice(["male", "female"], size=n),
        "DRIVING_EXPERIENCE": rng.choice(["0-9y", "10-19y", "20-29y", "30y+"], size=n),
        "CREDIT_SCORE": np.round(rng.uniform(0.2, 0.95, size=n), 3),
        "VEHICLE_YEAR": rng.choice(["before 2015", "after 2015"], size=n),
        "MARRIED": rng.integers(0, 2, size=n),
        "VEHICLE_OWNERSHIP": rng.integers(0, 2, size=n),
        "ANNUAL_MILEAGE": rng.integers(3000, 25000, size=n).astype(float),
        "SPEEDING_VIOLATIONS": rng.integers(0, 5, size=n),
        "PAST_ACCIDENTS": rng.integers(0, 4, size=n),
        "DUIS": rng.integers(0, 2, size=n),
        "VEHICLE_TYPE": rng.choice(["sedan", "sports car"], size=n),
    })
    # Règle déterministe simple pour que le modèle apprenne un signal clair
    score_risque = (X["PAST_ACCIDENTS"] >= 2) | (X["DUIS"] == 1) | (X["CREDIT_SCORE"] < 0.35)
    y = pd.Series(score_risque.astype(int), name="OUTCOME")
    return X, y


def test_modele_non_entraine_leve_erreur():
    """Vérifie que prédire ou calculer une probabilité sans entraînement lève ModeleNonEntraineError."""
    modele = Modele()
    X, _ = _creer_dataset_synthetique(5)

    assert "non entraîné" in str(modele)
    assert "entraine=False" in repr(modele)

    with pytest.raises(ModeleNonEntraineError):
        modele.predire(X)

    with pytest.raises(ModeleNonEntraineError):
        modele.predire_probabilite(X)


def test_entrainement_prediction_et_coherence_numpy():
    """Teste l'entraînement, la précision avec pytest.approx, la vectorisation NumPy et les prédictions."""
    X, y = _creer_dataset_synthetique(120)
    modele = Modele()

    precision = modele.entrainer(X, y)
    assert 0.0 <= precision <= 1.0
    assert precision == pytest.approx(accuracy_score(y, modele.predire(X)), abs=0.25)

    # Vérification vectorisée NumPy sur un exemple connu
    y_reel = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    acc_np = float(np.mean(y_pred == y_reel))
    assert acc_np == pytest.approx(accuracy_score(y_reel, y_pred), rel=1e-9)

    preds = modele.predire(X.iloc[:3])
    assert len(preds) == 3
    assert set(preds).issubset({0, 1})

    proba = modele.predire_probabilite(X.iloc[[0]])
    assert 0.0 <= proba <= 100.0
    assert "entraîné" in str(modele)
    assert "entraine=True" in repr(modele)


def test_sauvegarde_chargement_et_metadonnees(tmp_path):
    """Vérifie la sauvegarde/recharge du modèle (.pkl) et la génération du JSON de métadonnées."""
    X, y = _creer_dataset_synthetique(60)
    modele = Modele()
    precision = modele.entrainer(X, y)

    chemin_pkl = tmp_path / "modele_test.pkl"
    chemin_json = tmp_path / "sous_dossier" / "metadonnees.json"

    modele.sauvegarder(str(chemin_pkl))
    modele.sauvegarder_metadonnees(X, precision, str(chemin_json))

    assert chemin_pkl.exists()
    assert chemin_json.exists()

    with open(chemin_json, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["colonnes"] == list(X.columns)
    assert meta["metrique"] == "accuracy"
    assert meta["precision"] == pytest.approx(round(precision, 4), rel=1e-6)
    assert "GENDER" in meta["categories"]
    assert "CREDIT_SCORE" in meta["numeriques"]

    modele_recharge = Modele()
    modele_recharge.charger(str(chemin_pkl))
    assert modele.predire(X.iloc[:5]) == modele_recharge.predire(X.iloc[:5])

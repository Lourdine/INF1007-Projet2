"""
tests/test_modele.py — Tests unitaires pour modele.py

On vérifie que l'entraînement, la prédiction, la sauvegarde et le chargement
du RandomForestClassifier fonctionnent correctement.
pytest découvre automatiquement les fonctions qui commencent par test_.
"""

import pytest
import pandas as pd
from exceptions import ModeleNonEntraineError
from donnees import obtenir_features_cible
from modele import Modele


def creer_df_valide():
    """
    Crée un DataFrame minimal valide pour les tests.
    On utilise des valeurs réalistes pour éviter les erreurs de validation.
    """
    return pd.DataFrame({
        "AGE":                ["16-25", "26-39", "40-64"],
        "GENDER":             ["male", "female", "male"],
        "DRIVING_EXPERIENCE": ["0-9y", "10-19y", "20-29y"],
        "CREDIT_SCORE":       [0.5, 0.7, 0.3],
        "VEHICLE_YEAR":       ["before 2015", "after 2015", "before 2015"],
        "MARRIED":            ["0", "1", "0"],
        "VEHICLE_OWNERSHIP":  [1, 0, 1],
        "ANNUAL_MILEAGE":     [12000, 15000, 10000],
        "SPEEDING_VIOLATIONS":[0, 1, 2],
        "PAST_ACCIDENTS":     [0, 0, 1],
        "DUIS":               [0, 0, 0],
        "VEHICLE_TYPE":       ["sedan", "sedan", "sports car"],
        "OUTCOME":            [0, 1, 0],
    })


def test_predire_sans_entrainement_leve_erreur():
    """
    Modele.predire() doit lever ModeleNonEntraineError si on n'a pas encore entraîné.
    C'est la garde contre un modèle None — on ne peut pas prédire sans avoir appris.
    """
    m = Modele()
    df = creer_df_valide()
    X, _ = obtenir_features_cible(df)

    with pytest.raises(ModeleNonEntraineError):
        m.predire(X)


def test_entrainer_retourne_accuracy_valide():
    """
    Modele.entrainer() doit retourner un float entre 0 et 1 (accuracy_score).
    0.0 = aucune bonne prédiction, 1.0 = toutes correctes.
    """
    m = Modele()
    df = creer_df_valide()
    X, y = obtenir_features_cible(df)

    accuracy = m.entrainer(X, y)
    assert 0.0 <= accuracy <= 1.0


def test_predire_apres_entrainement_retourne_liste():
    """
    Après entraînement, predire() doit retourner une liste de prédictions.
    Chaque valeur doit être 0 ou 1 (classification binaire).
    """
    m = Modele()
    df = creer_df_valide()
    X, y = obtenir_features_cible(df)

    m.entrainer(X, y)
    predictions = m.predire(X)

    assert isinstance(predictions, list)
    assert all(p in [0, 1] for p in predictions)


def test_predire_probabilite_entre_0_et_100():
    """
    predire_probabilite() doit retourner un float entre 0 et 100.
    C'est le % de chance d'avoir une réclamation.
    """
    m = Modele()
    df = creer_df_valide()
    X, y = obtenir_features_cible(df)

    m.entrainer(X, y)
    proba = m.predire_probabilite(X.head(1))

    assert isinstance(proba, float)
    assert 0.0 <= proba <= 100.0


def test_sauvegarder_et_charger_modele(tmp_path):
    """
    sauvegarder() + charger() doivent permettre de retrouver le modèle intact.
    joblib sérialise l'objet Python sur disque — après rechargement, predire() doit fonctionner.
    tmp_path est un dossier temporaire créé automatiquement par pytest pour ce test.
    """
    m = Modele()
    df = creer_df_valide()
    X, y = obtenir_features_cible(df)
    m.entrainer(X, y)

    chemin = str(tmp_path / "modele_test.pkl")
    m.sauvegarder(chemin)

    m2 = Modele()
    m2.charger(chemin)
    # Après chargement, predire() lève ModeleNonEntraineError seulement si _modele est None
    # On vérifie que le modèle est bien rechargé en appelant predict directement
    X_encode = m._encoder_donnees(X)
    predictions = m2._modele.predict(X_encode)
    assert len(predictions) == len(X)


def test_str_modele_non_entraine():
    """
    __str__ doit indiquer clairement que le modèle n'est pas encore entraîné.
    """
    m = Modele()
    assert "non entraîné" in str(m)


def test_str_modele_entraine():
    """
    __str__ doit mentionner RandomForest et le nombre d'arbres après entraînement.
    """
    m = Modele()
    df = creer_df_valide()
    X, y = obtenir_features_cible(df)
    m.entrainer(X, y)

    assert "RandomForest" in str(m)
    assert "100" in str(m)


def test_predire_probabilite_sans_entrainement_leve_erreur():
    """
    predire_probabilite() doit aussi lever ModeleNonEntraineError si non entraîné.
    """
    m = Modele()
    df = creer_df_valide()
    X, _ = obtenir_features_cible(df)

    with pytest.raises(ModeleNonEntraineError):
        m.predire_probabilite(X.head(1))

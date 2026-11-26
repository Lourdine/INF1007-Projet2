
"""
tests/test_exceptions.py — Tests unitaires pour exceptions.py

On vérifie que chaque exception personnalisée :
- hérite de la bonne classe parente (polymorphisme)
- produit un message d'erreur clair contenant les informations utiles
pytest découvre automatiquement les fonctions qui commencent par test_.
"""

import pytest
from exceptions import (
    DonneeInvalideError,
    ValeurNegativeError,
    TypeInvalideError,
    ValeurHorsIntervalleError,
    FichierIntrouvableError,
    ColonneManquanteError,
    ModeleNonEntraineError,
)


def test_valeur_negative_error_herite_de_donnee_invalide():
    """
    ValeurNegativeError doit hériter de DonneeInvalideError.
    Polymorphisme : on peut attraper la sous-classe avec except DonneeInvalideError.
    """
    err = ValeurNegativeError("ANNUAL_MILEAGE", -500)
    assert isinstance(err, DonneeInvalideError)


def test_valeur_negative_error_message_contient_champ_et_valeur():
    """
    Le message doit contenir le nom du champ et la valeur négative reçue.
    """
    err = ValeurNegativeError("ANNUAL_MILEAGE", -500)
    assert "ANNUAL_MILEAGE" in str(err)
    assert "-500" in str(err)


def test_valeur_hors_intervalle_herite_de_donnee_invalide():
    """
    ValeurHorsIntervalleError doit hériter de DonneeInvalideError.
    """
    err = ValeurHorsIntervalleError("CREDIT_SCORE", 1.5, 0, 1)
    assert isinstance(err, DonneeInvalideError)


def test_valeur_hors_intervalle_message_contient_champ_et_valeur():
    """
    Le message doit mentionner le champ et la valeur hors intervalle.
    """
    err = ValeurHorsIntervalleError("CREDIT_SCORE", 1.5, 0, 1)
    assert "CREDIT_SCORE" in str(err)
    assert "1.5" in str(err)


def test_type_invalide_error_herite_de_donnee_invalide():
    """
    TypeInvalideError doit hériter de DonneeInvalideError.
    """
    err = TypeInvalideError("CREDIT_SCORE", "float", "abc")
    assert isinstance(err, DonneeInvalideError)


def test_fichier_introuvable_herite_de_exception():
    """
    FichierIntrouvableError doit hériter de Exception et mentionner le chemin.
    """
    err = FichierIntrouvableError("chemin/inexistant.csv")
    assert isinstance(err, Exception)
    assert "chemin/inexistant.csv" in str(err)


def test_modele_non_entraine_message_non_vide():
    """
    ModeleNonEntraineError doit avoir un message clair — pas un message vide.
    Levée quand on appelle predire() sans avoir entraîné le modèle.
    """
    err = ModeleNonEntraineError()
    assert isinstance(err, Exception)
    assert len(str(err)) > 0


def test_colonne_manquante_message_contient_colonne():
    """
    ColonneManquanteError doit mentionner le nom de la colonne absente.
    """
    err = ColonneManquanteError("CREDIT_SCORE")
    assert "CREDIT_SCORE" in str(err)
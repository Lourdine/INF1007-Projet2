"""
test_exceptions.py — Tests unitaires des exceptions personnalisées (INF1007 Projet 2).
Vérifie l'héritage, les messages d'erreur et la levée avec pytest.raises.
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


def test_hierarchie_exceptions():
    """Vérifie que les exceptions d'entrée héritent de DonneeInvalideError et Exception."""
    assert issubclass(DonneeInvalideError, Exception)
    assert issubclass(ValeurNegativeError, DonneeInvalideError)
    assert issubclass(TypeInvalideError, DonneeInvalideError)
    assert issubclass(ValeurHorsIntervalleError, DonneeInvalideError)
    assert issubclass(FichierIntrouvableError, Exception)
    assert issubclass(ColonneManquanteError, Exception)
    assert issubclass(ModeleNonEntraineError, Exception)


def test_messages_exceptions_usuels_et_limites():
    """Vérifie le formatage des messages pour des cas usuels et limites."""
    err_neg = ValeurNegativeError("ANNUAL_MILEAGE", -500.0)
    assert "ANNUAL_MILEAGE" in str(err_neg)
    assert "-500.0" in str(err_neg)

    err_type = TypeInvalideError("CREDIT_SCORE", "float", "abc")
    assert "CREDIT_SCORE" in str(err_type)
    assert "float" in str(err_type)
    assert "'abc'" in str(err_type)

    err_borne = ValeurHorsIntervalleError("CREDIT_SCORE", 1.5, 0, 1)
    assert "0" in str(err_borne) and "1" in str(err_borne) and "1.5" in str(err_borne)

    err_fichier = FichierIntrouvableError("donnees/absent.csv")
    assert "donnees/absent.csv" in str(err_fichier)

    err_col = ColonneManquanteError("OUTCOME")
    assert "OUTCOME" in str(err_col)

    err_modele = ModeleNonEntraineError()
    assert "Aucun modèle" in str(err_modele)


def test_levee_exceptions_avec_pytest_raises():
    """Vérifie que chaque exception personnalisée peut être levée et interceptée."""
    with pytest.raises(DonneeInvalideError):
        raise ValeurNegativeError("PAST_ACCIDENTS", -1)

    with pytest.raises(TypeInvalideError):
        raise TypeInvalideError("DUIS", "int", None)

    with pytest.raises(ValeurHorsIntervalleError):
        raise ValeurHorsIntervalleError("CREDIT_SCORE", -0.1, 0.0, 1.0)

    with pytest.raises(FichierIntrouvableError):
        raise FichierIntrouvableError("inexistant.csv")

    with pytest.raises(ColonneManquanteError):
        raise ColonneManquanteError("AGE")

    with pytest.raises(ModeleNonEntraineError):
        raise ModeleNonEntraineError()
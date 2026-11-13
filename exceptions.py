"""
exceptions.py — Exceptions personnalisées pour le projet INF1007 Projet 2.
Toutes les exceptions héritent de DonneeInvalideError ou directement de Exception.
"""


class DonneeInvalideError(Exception):
    """Erreur de base pour toute donnée invalide dans le projet."""

    def __init__(self, message: str = "Donnée invalide."):
        super().__init__(message)


class ValeurNegativeError(DonneeInvalideError):
    """Levée quand une valeur numérique est négative alors qu'elle doit être positive."""

    def __init__(self, champ: str, valeur: float):
        super().__init__(
            f"La valeur de '{champ}' ne peut pas être négative : {valeur}."
        )


class TypeInvalideError(DonneeInvalideError):
    """Levée quand une valeur n'est pas du bon type attendu."""

    def __init__(self, champ: str, type_attendu: str, valeur_recue):
        super().__init__(
            f"'{champ}' doit être de type {type_attendu}, "
            f"mais a reçu : {repr(valeur_recue)}."
        )


class ValeurHorsIntervalleError(DonneeInvalideError):
    """Levée quand une valeur numérique est hors de l'intervalle autorisé."""

    def __init__(self, champ: str, valeur: float, minimum: float, maximum: float):
        super().__init__(
            f"'{champ}' doit être entre {minimum} et {maximum}, "
            f"mais a reçu : {valeur}."
        )


class FichierIntrouvableError(Exception):
    """Levée quand un fichier requis (CSV ou modèle) est introuvable."""

    def __init__(self, chemin: str):
        super().__init__(
            f"Le fichier est introuvable : '{chemin}'. "
            f"Vérifiez que le fichier existe à cet emplacement."
        )


class ColonneManquanteError(Exception):
    """Levée quand une colonne attendue est absente du fichier CSV chargé."""

    def __init__(self, colonne: str):
        super().__init__(
            f"La colonne '{colonne}' est absente du fichier de données. "
            f"Vérifiez que le bon fichier CSV a été chargé."
        )


class ModeleNonEntraineError(Exception):
    """Levée quand on tente une prédiction sans modèle chargé ou entraîné."""

    def __init__(self):
        super().__init__(
            "Aucun modèle n'est chargé. "
            "Veuillez d'abord entraîner ou charger un modèle avant de prédire."
        )
      

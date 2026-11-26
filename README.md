# INF1007 Projet 2 — Prédiction de réclamation d'assurance auto

## Description

Application de classification binaire qui prédit si un conducteur est susceptible
de faire une réclamation d'assurance auto. L'utilisateur saisit ses informations
(âge, genre, expérience de conduite, score de crédit, kilométrage annuel, etc.)
via une interface graphique Tkinter. Le modèle retourne une prédiction (0 ou 1)
accompagnée d'un pourcentage de probabilité.

## Source et licence des données

- **Jeu de données** : Car Insurance Claim Prediction
- **Source** : [Kaggle](https://www.kaggle.com/datasets/sagnik1511/car-insurance-data)
- **Fichier** : `donnees/Car_Insurance_Claim.csv`
- **Licence** : CC0 — domaine public

## Installation

```bash
pip install -r requirements.txt
```

## Lancement

```bash
python main.py
```

## Structure du code

```
INF1007-Projet2/
├── README.md
├── DECISIONS.md
├── JOURNAL_IA.md
├── requirements.txt
├── main.py                        # Point d'entrée — lance l'application
├── donnees.py                     # Variable (ABC), VariableNumerique, VariableCategorielle, JeuDeDonnees
├── modele.py                      # RandomForestClassifier — entraînement, prédiction, sauvegarde
├── exceptions.py                  # 7 exceptions personnalisées
├── interface.py                   # Interface graphique Tkinter (thème navy professionnel)
├── donnees/
│   └── Car_Insurance_Claim.csv    # Jeu de données d'entraînement (10 302 lignes)
├── modeles/
│   └── metadonnees.json           # Colonnes, intervalles, valeurs catégorielles, précision
└── tests/
    ├── test_exceptions.py         # 8 tests — exceptions personnalisées
    ├── test_donnees.py            # 14 tests — chargement, nettoyage, classes Variable
    └── test_modele.py             # 8 tests — entraînement, prédiction, sauvegarde
```

## Résultats du modèle

| Modèle | Précision |
|---|---|
| Modèle trivial (classe majoritaire) | 68,53 % |
| RandomForestClassifier (100 arbres) | 81,41 % |
| **Gain** | **+12,88 %** |

Entraîné sur 80 % des données, évalué sur les 20 % restants (`random_state=42`).

## Tests et couverture

Commande : `python -m pytest --cov=. tests/`

| Fichier | Couverture |
|---|---|
| `exceptions.py` | 100 % |
| `modele.py` | 89 % |
| `donnees.py` | 80 % |
| `interface.py` | 0 % (interface graphique — non testable avec pytest) |
| **Total** | **54 %** |

30 tests au total — tous réussis.

## Concepts du cours utilisés

| Chapitre | Concept | Où |
|---|---|---|
| Ch. 7 | Fonction lambda comme clé de tri | `interface.py` — `key=lambda i: importances[i]` |
| Ch. 10 | Classe abstraite et polymorphisme | `donnees.py` — `Variable` → `VariableNumerique`, `VariableCategorielle` |
| Ch. 11 | Exceptions personnalisées | `exceptions.py` — 7 classes héritant de `Exception` |
| Ch. 12 | Calcul vectorisé NumPy | `modele.py` ligne 51 — `np.mean(np.array(predictions) == np.array(y_test))` |
| Ch. 13 | Graphiques Matplotlib | Graphique d'exploration + graphique d'importance dans l'interface |

## Références

- [Documentation pandas](https://pandas.pydata.org/docs/)
- [Documentation scikit-learn](https://scikit-learn.org/stable/)
- [Documentation Tkinter](https://docs.python.org/3/library/tkinter.html)
- [RandomForestClassifier — scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Intégrer Matplotlib dans Tkinter](https://matplotlib.org/stable/gallery/user_interfaces/embedding_in_tk_sgskip.html)
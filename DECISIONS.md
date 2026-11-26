# DECISIONS.md — Fiche de décisions

## INF1007 Projet 2 — Prédiction de réclamation d'assurance auto

| Choix | Options envisagées | Choix retenu et justification | Validé par l'auxiliaire |
|---|---|---|---|
| **Domaine** | Dentaire (prédiction du coût d'un traitement — dataset introuvable), hypothèque, assurance auto | Assurance auto : domaine des besoins essentiels, connaissance du domaine, dataset facilement trouvable avec licence CC0 | Oui (validation A) |
| **Jeu de données** | Plusieurs jeux Kaggle sur l'assurance ; certains avec licence non publique ou colonnes insuffisantes | Car Insurance Claim Prediction (Kaggle, CC0) : 10 302 lignes, 19 colonnes originales dont 12 retenues après nettoyage, variables numériques et catégorielles, cible binaire OUTCOME | Oui (validation B) |
| **Variable cible et métrique** | OUTCOME (0 = pas de réclamation, 1 = réclamation) ; métriques envisagées : F1, précision, exactitude | Exactitude (accuracy) retenue : classification binaire avec classes relativement équilibrées ; un modèle trivial (classe majoritaire 68,53 %) sert de point de comparaison | Oui (validation C) |
| **Modèle** | Arbre de décision, forêt aléatoire | RandomForestClassifier (100 arbres, random_state=42) : robuste aux valeurs aberrantes, gère les variables mixtes encodées, interprétable via l'importance des variables | Oui (validation C) |
| **Technologie d'interface** | Tkinter, Streamlit, Gradio | Tkinter : intégré à Python (aucune installation supplémentaire), supporte FigureCanvasTkAgg pour intégrer Matplotlib directement dans la fenêtre, fonctionne en local sans serveur | Oui (validation D) |
| **Outil du chapitre 7 (lambda)** | Décorateur, générateur, lambda | Lambda dans `interface.py` pour trier les indices par importance des variables : `key=lambda i: importances[i]` — concis et adapté à une clé de tri ponctuelle | — |
| **Gestion des exceptions** | `except Exception` générique, exceptions précises par module | 7 exceptions personnalisées héritant de `Exception` dans `exceptions.py` : `DonneeInvalideError`, `ValeurNegativeError`, `TypeInvalideError`, `ValeurHorsIntervalleError`, `FichierIntrouvableError`, `ColonneManquanteError`, `ModeleNonEntraineError` | — |
| **Sauvegarde du modèle** | Pickle, joblib | joblib : recommandé par scikit-learn pour sérialiser les modèles NumPy/sklearn efficacement ; métadonnées séparées en JSON | — |
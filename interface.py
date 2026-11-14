"""
interface.py — Interface graphique Tkinter pour INF1007 Projet 2.
Sujet : Prédiction de réclamation d'assurance auto (classification binaire).
"""

import tkinter as tk                                    # Bibliothèque interface graphique incluse dans Python
from tkinter import ttk, messagebox                    # ttk = widgets modernes, messagebox = fenêtres d'erreur
import pandas as pd                                    # Pour créer le DataFrame d'une ligne à prédire
import matplotlib.pyplot as plt                        # Pour tracer le graphique d'importance des variables
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # Intègre matplotlib dans Tkinter
import csv                                             # Pour écrire/lire le fichier historique
import os                                              # Pour vérifier si un fichier existe sur le disque
from datetime import datetime                          # Pour horodater chaque prédiction dans l'historique

from donnees import charger_donnees, nettoyer_donnees, obtenir_features_cible  # Nos fonctions de données
from modele import Modele                              # Notre classe modèle ML
from exceptions import DonneeInvalideError, ModeleNonEntraineError  # Nos exceptions personnalisées

CHEMIN_CSV = "donnees/Car_Insurance_Claim.csv"         # Chemin vers le dataset
CHEMIN_MODELE = "modele_sauvegarde.pkl"                # Fichier où on sauvegarde le modèle entraîné
CHEMIN_HISTORIQUE = "historique.csv"                   # Fichier où on sauvegarde chaque prédiction

# Valeurs possibles pour chaque menu déroulant
OPTIONS_GENDER = ["male", "female"]
OPTIONS_DRIVING = ["0-9y", "10-19y", "20-29y", "30y+"]
OPTIONS_VEHICLE_YEAR = ["before 2015", "after 2015"]
OPTIONS_MARRIED = ["0", "1"]                           # 0 = non marié, 1 = marié
OPTIONS_OWNERSHIP = ["0", "1"]                         # 0 = pas propriétaire, 1 = propriétaire
OPTIONS_VEHICLE_TYPE = ["sedan", "sports car"]


class Interface:
    """Classe principale de l'interface graphique."""

    def __init__(self, root: tk.Tk):
        self.root = root                               # root = la fenêtre principale Tkinter
        self.root.title("Prédiction de réclamation d'assurance auto")  # Titre de la fenêtre
        self.root.geometry("900x700")                  # Taille de la fenêtre : 900px large, 700px haut
        self._modele = Modele()                        # Crée un objet Modele (pas encore entraîné)
        self._entrainer_modele()                       # Charge les données et entraîne le modèle
        self._construire_interface()                   # Construit tous les éléments visuels

    def _entrainer_modele(self):
        """Orchestre le chargement, nettoyage et entraînement — utilise donnees.py et modele.py."""
        df = charger_donnees(CHEMIN_CSV)               # Charge le CSV → DataFrame
        df = nettoyer_donnees(df)                      # Supprime les lignes vides, valide les valeurs
        X, y = obtenir_features_cible(df)              # Sépare les 13 variables (X) et OUTCOME (y)
        precision = self._modele.entrainer(X, y)       # RandomForest apprend le lien entre X et y (80/20)
        self._modele.sauvegarder(CHEMIN_MODELE)        # Sauvegarde le modèle pour ne pas réentraîner
        print(f"Modèle entraîné — précision : {precision:.2%}")  # Affiche ex: "précision : 85.00%"

    def _construire_interface(self):
        """Construit tous les éléments visuels de la fenêtre."""
        titre = tk.Label(self.root,
                         text="Prédiction de réclamation d'assurance auto",
                         font=("Helvetica", 14, "bold"))
        titre.pack(pady=8)                             # pack() place l'élément dans la fenêtre (sans pack = invisible)

        avertissement = tk.Label(self.root,
                                 text="⚠ Outil pédagogique — pas un outil de décision réelle",
                                 fg="red", font=("Helvetica", 10))
        avertissement.pack()                           # Placé sous le titre

        cadre = tk.Frame(self.root)                    # Conteneur invisible qui divise la fenêtre en 2 colonnes
        cadre.pack(pady=10, padx=20, fill="both", expand=True)

        colonne_gauche = tk.Frame(cadre)               # Colonne gauche : champs de saisie
        colonne_gauche.pack(side="left", fill="y", padx=10)

        self._colonne_droite = tk.Frame(cadre)         # Colonne droite : résultat + graphique
        self._colonne_droite.pack(side="right", fill="both", expand=True)

        self._champs = {}                              # Dictionnaire : clé = nom variable, valeur = champ/menu

        # Création des champs — champ texte pour les numériques, menu déroulant pour les catégorielles
        self._creer_champ_texte(colonne_gauche, "AGE")
        self._creer_menu(colonne_gauche, "GENDER", OPTIONS_GENDER)
        self._creer_menu(colonne_gauche, "DRIVING_EXPERIENCE", OPTIONS_DRIVING)
        self._creer_champ_texte(colonne_gauche, "CREDIT_SCORE")
        self._creer_champ_texte(colonne_gauche, "ANNUAL_MILEAGE")
        self._creer_champ_texte(colonne_gauche, "SPEEDING_VIOLATIONS")
        self._creer_champ_texte(colonne_gauche, "PAST_ACCIDENTS")
        self._creer_champ_texte(colonne_gauche, "DUIS")
        self._creer_menu(colonne_gauche, "VEHICLE_YEAR", OPTIONS_VEHICLE_YEAR)
        self._creer_champ_texte(colonne_gauche, "POSTAL_CODE")
        self._creer_menu(colonne_gauche, "MARRIED", OPTIONS_MARRIED)
        self._creer_menu(colonne_gauche, "VEHICLE_OWNERSHIP", OPTIONS_OWNERSHIP)
        self._creer_menu(colonne_gauche, "VEHICLE_TYPE", OPTIONS_VEHICLE_TYPE)

        btn_predire = tk.Button(colonne_gauche, text="Prédire",
                                command=self._predire,  # Appelle _predire() quand on clique
                                bg="#1a3a5c", fg="white", font=("Helvetica", 11, "bold"))
        btn_predire.pack(pady=10)

        btn_historique = tk.Button(colonne_gauche, text="Voir historique",
                                   command=self._afficher_historique)  # Appelle _afficher_historique()
        btn_historique.pack()

        self._label_resultat = tk.Label(self._colonne_droite, text="",
                                        font=("Helvetica", 12), wraplength=300)
        self._label_resultat.pack(pady=10)             # Zone où le résultat de la prédiction s'affiche

    def _creer_champ_texte(self, parent, nom: str):
        """Crée un label + champ texte pour une variable numérique."""
        tk.Label(parent, text=nom).pack(anchor="w")    # Label avec le nom de la variable
        champ = tk.Entry(parent)                       # Champ texte où l'utilisateur tape
        champ.pack(fill="x", pady=2)
        self._champs[nom] = champ                      # Stocke dans le dictionnaire pour lecture plus tard

    def _creer_menu(self, parent, nom: str, options: list):
        """Crée un label + menu déroulant pour une variable catégorielle."""
        tk.Label(parent, text=nom).pack(anchor="w")    # Label avec le nom de la variable
        var = tk.StringVar(value=options[0])           # Variable qui stocke la valeur sélectionnée
        menu = ttk.Combobox(parent, textvariable=var, values=options, state="readonly")
        menu.pack(fill="x", pady=2)
        self._champs[nom] = var                        # Stocke la StringVar (pas le menu) pour lire la valeur

    def _lire_saisies(self) -> dict:
        """Lit et valide toutes les saisies. Lève DonneeInvalideError si invalide."""
        valeurs = {}
        champs_numeriques = [                          # Ces champs doivent contenir des chiffres
            "AGE", "CREDIT_SCORE", "ANNUAL_MILEAGE",
            "SPEEDING_VIOLATIONS", "PAST_ACCIDENTS", "DUIS", "POSTAL_CODE"
        ]
        for nom, champ in self._champs.items():        # Parcourt tous les champs du dictionnaire
            valeur = champ.get()                       # Lit la valeur tapée ou sélectionnée
            if valeur == "":
                raise DonneeInvalideError(f"Le champ '{nom}' est vide.")
            if nom in champs_numeriques:
                try:
                    valeurs[nom] = float(valeur)       # Convertit en nombre — plante si texte
                except ValueError:
                    raise DonneeInvalideError(f"'{nom}' doit être un nombre, pas '{valeur}'.")
            else:
                valeurs[nom] = valeur                  # Valeur texte pour les menus déroulants
        return valeurs

    def _predire(self):
        """Lit les saisies, appelle le modèle et affiche le résultat."""
        try:
            valeurs = self._lire_saisies()             # Lit et valide tous les champs
            X = pd.DataFrame([valeurs])                # Crée un DataFrame d'une seule ligne
            prediction = self._modele.predire(X)[0]   # Le modèle retourne 0 ou 1

            if prediction == 1:                        # 1 = réclamation probable
                msg = "⚠ Ce client va probablement faire une réclamation."
                couleur = "red"
            else:                                      # 0 = pas de réclamation probable
                msg = "✅ Ce client ne va probablement pas faire une réclamation."
                couleur = "green"

            self._label_resultat.config(text=msg, fg=couleur)  # Affiche le résultat
            self._sauvegarder_historique(valeurs, prediction)   # Sauvegarde dans historique.csv
            self._afficher_graphique()                          # Affiche le graphique

        except DonneeInvalideError as e:
            messagebox.showerror("Erreur de saisie", str(e))   # Message d'erreur clair, pas de crash
        except ModeleNonEntraineError as e:
            messagebox.showerror("Erreur modèle", str(e))

    def _afficher_graphique(self):
        """Affiche un graphique matplotlib de l'importance des variables."""
        df = charger_donnees(CHEMIN_CSV)
        colonnes = [c for c in df.columns if c != "OUTCOME"]   # Noms des 13 variables
        importances = self._modele._modele.feature_importances_ # Importance calculée par RandomForest

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.barh(colonnes, importances, color="#1a3a5c")         # Graphique horizontal
        ax.set_title("Importance des variables")
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self._colonne_droite)  # Intègre dans Tkinter
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)                                          # Libère la mémoire

    def _sauvegarder_historique(self, valeurs: dict, prediction: int):
        """Sauvegarde chaque prédiction dans historique.csv."""
        fichier_existe = os.path.exists(CHEMIN_HISTORIQUE)     # Vérifie si le fichier existe déjà
        with open(CHEMIN_HISTORIQUE, "a", newline="") as f:    # Ouvre en mode ajout
            writer = csv.DictWriter(f, fieldnames=["date"] + list(valeurs.keys()) + ["prediction"])
            if not fichier_existe:
                writer.writeheader()                           # Écrit les entêtes seulement la 1re fois
            ligne = {"date": datetime.now().strftime("%Y-%m-%d %H:%M")}
            ligne.update(valeurs)
            ligne["prediction"] = prediction
            writer.writerow(ligne)                             # Écrit la ligne dans le fichier

    def _afficher_historique(self):
        """Ouvre une nouvelle fenêtre avec le tableau de l'historique."""
        if not os.path.exists(CHEMIN_HISTORIQUE):
            messagebox.showinfo("Historique", "Aucune prédiction enregistrée.")
            return

        fenetre = tk.Toplevel(self.root)                       # Nouvelle fenêtre par-dessus la principale
        fenetre.title("Historique des prédictions")
        fenetre.geometry("800x400")

        frame = tk.Frame(fenetre)
        frame.pack(fill="both", expand=True)

        df = pd.read_csv(CHEMIN_HISTORIQUE)                    # Charge l'historique
        tree = ttk.Treeview(frame, columns=list(df.columns), show="headings")  # Tableau avec colonnes
        for col in df.columns:
            tree.heading(col, text=col)                        # Entête de chaque colonne
            tree.column(col, width=80)
        for _, row in df.iterrows():
            tree.insert("", "end", values=list(row))           # Ajoute chaque ligne au tableau

        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)           # Barre de défilement verticale
        tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

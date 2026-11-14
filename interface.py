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

CHEMIN_CSV        = "donnees/Car_Insurance_Claim.csv"  # Chemin vers le dataset
CHEMIN_MODELE     = "modele_sauvegarde.pkl"            # Fichier où on sauvegarde le modèle entraîné
CHEMIN_HISTORIQUE = "historique.csv"                   # Fichier où on sauvegarde chaque prédiction

# ── Thème Navy professionnel ───────────────────────────────────────────────────
# Chaque constante est une couleur hexadécimale utilisée dans toute l'interface
NAVY        = "#0f172a"                                # Fond principal très foncé (arrière-plan fenêtre)
NAVY_PANEL  = "#1e293b"                                # Fond des panneaux (légèrement plus clair)
NAVY_BORDER = "#334155"                                # Couleur des bordures et séparateurs
BLEU        = "#3b82f6"                                # Bleu vif — bouton Prédire et barres du graphique
BLEU_HOVER  = "#2563eb"                                # Bleu plus foncé quand on survole le bouton
BLEU_2      = "#1e3a5f"                                # Bleu sombre — bouton Voir historique
TEXTE       = "#e2e8f0"                                # Texte principal (blanc cassé, lisible sur fond sombre)
TEXTE_MUT   = "#94a3b8"                                # Texte secondaire (gris, pour labels et placeholders)
VERT        = "#10b981"                                # Couleur résultat : pas de réclamation
ROUGE       = "#ef4444"                                # Couleur résultat : réclamation probable
JAUNE       = "#f59e0b"                                # Couleur avertissement dans le header

# ── Options des menus déroulants ──────────────────────────────────────────────
# Ces listes définissent les valeurs possibles pour les variables catégorielles
# Elles correspondent aux valeurs réelles du dataset Car_Insurance_Claim.csv
OPTIONS_AGE          = ["16-25", "26-39", "40-64", "65+"]
OPTIONS_GENDER       = ["male", "female"]
OPTIONS_DRIVING      = ["0-9y", "10-19y", "20-29y", "30y+"]
OPTIONS_VEHICLE_YEAR = ["before 2015", "after 2015"]
OPTIONS_MARRIED      = ["0", "1"]                      # 0 = non marié, 1 = marié
OPTIONS_VEHICLE_TYPE = ["sedan", "sports car"]
OPTIONS_OWNERSHIP    = ["0", "1"]   


class Interface:
    """
    Classe principale de l'interface graphique — thème Navy professionnel.

    Responsabilités :
    - Entraîner le modèle au démarrage (via donnees.py + modele.py)
    - Afficher les 12 champs de saisie en 2 colonnes
    - Lire et valider les saisies, appeler le modèle, afficher le résultat
    - Afficher le graphique d'importance des variables après chaque prédiction
    - Sauvegarder chaque prédiction dans historique.csv
    """

    def __init__(self, root: tk.Tk):
        self.root = root                               # root = la fenêtre principale Tkinter
        self.root.title("Prédiction de réclamation d'assurance auto")
        self.root.state("zoomed")                      # Fenêtre maximisée au lancement (Windows)
        self.root.configure(bg=NAVY)                   # Fond noir marine pour toute la fenêtre
        self._modele = Modele()                        # Crée un objet Modele (pas encore entraîné)
        self._entrainer_modele()                       # Charge les données et entraîne le modèle
        self._appliquer_style_ttk()                    # Configure le style sombre des menus déroulants
        self._construire_interface()                   # Construit tous les éléments visuels

    # ── Entraînement ──────────────────────────────────────────────────────────

    def _entrainer_modele(self):
        """
        Orchestre le chargement, nettoyage et entraînement.
        Appelée une seule fois au démarrage — utilise donnees.py et modele.py.
        """
        df = charger_donnees(CHEMIN_CSV)               # Charge le CSV → DataFrame pandas
        df = nettoyer_donnees(df)                      # Supprime les lignes vides, valide CREDIT_SCORE
        X, y = obtenir_features_cible(df)              # X = 12 variables d'entrée, y = OUTCOME (0 ou 1)
        precision = self._modele.entrainer(X, y)       # RandomForest apprend le lien entre X et y (split 80/20)
        self._modele.sauvegarder(CHEMIN_MODELE)        # Sauvegarde le modèle sur disque (.pkl) pour ne pas réentraîner
        print(f"Modèle entraîné — précision : {precision:.2%}")  # Affiche ex: "précision : 85.00%"

    # ── Style ttk ─────────────────────────────────────────────────────────────

    def _appliquer_style_ttk(self):
        """
        Applique un style sombre aux widgets ttk (Combobox = menu déroulant).
        Par défaut, les Combobox sont blancs — on les force à suivre notre thème navy.
        """
        style = ttk.Style()
        style.theme_use("default")                     # Thème de base sur lequel on applique nos couleurs
        style.configure("TCombobox",
                        fieldbackground=NAVY_PANEL,    # Fond du champ texte du menu
                        background=NAVY_PANEL,         # Fond du bouton déroulant
                        foreground=TEXTE,              # Couleur du texte sélectionné
                        selectbackground=BLEU,         # Fond de la valeur sélectionnée dans la liste
                        selectforeground=TEXTE,
                        bordercolor=NAVY_BORDER,
                        lightcolor=NAVY_BORDER,
                        darkcolor=NAVY_BORDER)
        style.map("TCombobox",
                  fieldbackground=[("readonly", NAVY_PANEL)],   # État lecture seule (notre cas)
                  foreground=[("readonly", TEXTE)])

    # ── Construction de l'interface ───────────────────────────────────────────

    def _construire_interface(self):
        """
        Construit tous les éléments visuels de la fenêtre.
        Structure : header en haut, puis zone principale = panneau gauche + panneau droit.
        """

        # ── Bandeau header ────────────────────────────────────────────────────
        # Barre horizontale en haut : logo bleu + titre + avertissement
        header = tk.Frame(self.root, bg=NAVY_PANEL)
        header.pack(fill="x")                          # fill="x" = étire sur toute la largeur

        inner_header = tk.Frame(header, bg=NAVY_PANEL)
        inner_header.pack(fill="x", padx=24, pady=14)

        # Carré bleu avec l'emoji bouclier (simule un logo)
        logo_frame = tk.Frame(inner_header, bg=BLEU, width=38, height=38)
        logo_frame.pack(side="left")
        logo_frame.pack_propagate(False)               # Empêche le Frame de rétrécir autour du Label
        tk.Label(logo_frame, text="🛡", font=("Segoe UI Emoji", 18),
                 bg=BLEU, fg=TEXTE).place(relx=0.5, rely=0.5, anchor="center")  # Centré dans le carré

        # Titre + sous-titre à droite du logo
        texte_header = tk.Frame(inner_header, bg=NAVY_PANEL)
        texte_header.pack(side="left", padx=(12, 0))
        tk.Label(texte_header,
                 text="Prédiction de réclamation d'assurance auto",
                 font=("Segoe UI", 14, "bold"),
                 fg=TEXTE, bg=NAVY_PANEL).pack(anchor="w")
        tk.Label(texte_header,
                 text="Outil pédagogique — pas un outil de décision réelle",
                 font=("Segoe UI", 9),
                 fg=JAUNE, bg=NAVY_PANEL).pack(anchor="w")  # Jaune = avertissement visible

        # Ligne séparatrice sous le header
        tk.Frame(self.root, bg=NAVY_BORDER, height=1).pack(fill="x")

        # ── Zone principale ───────────────────────────────────────────────────
        # Conteneur horizontal : panneau gauche (variables) + panneau droit (résultat)
        zone = tk.Frame(self.root, bg=NAVY)
        zone.pack(fill="both", expand=True, padx=24, pady=20)

        # ── Panneau gauche : variables ────────────────────────────────────────
        panneau_vars = tk.Frame(zone, bg=NAVY_PANEL,
                                highlightbackground=NAVY_BORDER,
                                highlightthickness=1)  # Bordure fine autour du panneau
        panneau_vars.pack(side="left", fill="y", padx=(0, 16))  # fill="y" = hauteur du contenu

        self._titre_panneau(panneau_vars, "PROFIL CONDUCTEUR")

        # Deux sous-colonnes de 6 champs chacune (total = 12 variables)
        sous_cadre = tk.Frame(panneau_vars, bg=NAVY_PANEL)
        sous_cadre.pack(padx=18, pady=(0, 10))

        col_g = tk.Frame(sous_cadre, bg=NAVY_PANEL)   # Colonne gauche : 6 champs
        col_g.pack(side="left", padx=(0, 18))

        col_d = tk.Frame(sous_cadre, bg=NAVY_PANEL)   # Colonne droite : 6 champs
        col_d.pack(side="left")

        self._champs = {}                              # Dictionnaire : clé = nom variable, valeur = champ/StringVar

        # ── 6 champs colonne gauche ───────────────────────────────────────────
        self._creer_menu(col_g, "AGE", OPTIONS_AGE)
        self._creer_menu(col_g, "GENDER", OPTIONS_GENDER)
        self._creer_menu(col_g, "DRIVING_EXPERIENCE", OPTIONS_DRIVING)
        self._creer_champ_texte(col_g, "CREDIT_SCORE", "ex: 0.72")      # Entre 0 et 1
        self._creer_champ_texte(col_g, "ANNUAL_MILEAGE", "ex: 12000")   # Kilomètres par an
        self._creer_champ_texte(col_g, "SPEEDING_VIOLATIONS", "ex: 1")  # Nombre d'infractions

        # ── 6 champs colonne droite ───────────────────────────────────────────
        self._creer_champ_texte(col_d, "PAST_ACCIDENTS", "ex: 0")       # Nombre d'accidents passés
        self._creer_champ_texte(col_d, "DUIS", "ex: 0")                 # Conduite en état d'ivresse
        self._creer_menu(col_d, "VEHICLE_YEAR", OPTIONS_VEHICLE_YEAR)
        self._creer_menu(col_d, "VEHICLE_OWNERSHIP", OPTIONS_OWNERSHIP)  # 0 = loué, 1 = propriétaire
        self._creer_menu(col_d, "MARRIED", OPTIONS_MARRIED)
        self._creer_menu(col_d, "VEHICLE_TYPE", OPTIONS_VEHICLE_TYPE)

        # ── Boutons sous les deux colonnes ────────────────────────────────────
        zone_boutons = tk.Frame(panneau_vars, bg=NAVY_PANEL)
        zone_boutons.pack(pady=14)

        tk.Button(zone_boutons,
                  text="  Prédire  →  ",
                  command=self._predire,               # Appelle _predire() au clic
                  bg=BLEU, fg=TEXTE,
                  font=("Segoe UI", 11, "bold"),
                  relief="flat", padx=18, pady=8,
                  cursor="hand2", bd=0,
                  activebackground=BLEU_HOVER,
                  activeforeground=TEXTE).pack(side="left", padx=(0, 10))

        tk.Button(zone_boutons,
                  text="Voir historique",
                  command=self._afficher_historique,   # Appelle _afficher_historique() au clic
                  bg=BLEU_2, fg=TEXTE,
                  font=("Segoe UI", 10),
                  relief="flat", padx=12, pady=8,
                  cursor="hand2", bd=0,
                  activebackground=NAVY_BORDER,
                  activeforeground=TEXTE).pack(side="left")

        # ── Panneau droit : résultat + graphique ──────────────────────────────
        # expand=True → prend tout l'espace restant à droite
        self._colonne_droite = tk.Frame(zone, bg=NAVY_PANEL,
                                        highlightbackground=NAVY_BORDER,
                                        highlightthickness=1)
        self._colonne_droite.pack(side="right", fill="both", expand=True)

        self._titre_panneau(self._colonne_droite, "RÉSULTAT & ANALYSE")

        # Label qui affiche le résultat de la prédiction (mis à jour après chaque clic Prédire)
        self._label_resultat = tk.Label(self._colonne_droite,
                                        text="Remplissez les champs et cliquez sur Prédire.",
                                        font=("Segoe UI", 11),
                                        wraplength=400,            # Retour à la ligne automatique
                                        fg=TEXTE_MUT, bg=NAVY_PANEL)
        self._label_resultat.pack(pady=(6, 10), padx=16)

    # ── Helpers visuels ───────────────────────────────────────────────────────

    def _titre_panneau(self, parent, texte: str):
        """Affiche un titre de section en lettres grises + une ligne séparatrice."""
        tk.Label(parent, text=texte,
                 font=("Segoe UI", 9, "bold"),
                 fg=TEXTE_MUT, bg=NAVY_PANEL).pack(pady=(14, 4), padx=18, anchor="w")
        tk.Frame(parent, bg=NAVY_BORDER, height=1).pack(fill="x", padx=18, pady=(0, 10))

    def _creer_champ_texte(self, parent, nom: str, placeholder: str = ""):
        """
        Crée un label + champ texte pour une variable numérique.
        Le placeholder (texte fantôme) disparaît quand l'utilisateur clique dans le champ.
        """
        tk.Label(parent, text=nom,
                 font=("Segoe UI", 8, "bold"),
                 fg=TEXTE_MUT, bg=NAVY_PANEL).pack(anchor="w", pady=(6, 2))

        champ = tk.Entry(parent, width=22,
                         bg=NAVY_PANEL, fg=TEXTE,       # Fond sombre, texte clair
                         insertbackground=TEXTE,         # Curseur blanc (sinon invisible sur fond sombre)
                         relief="flat",
                         highlightbackground=NAVY_BORDER,
                         highlightcolor=BLEU,            # Bordure bleue quand le champ est actif
                         highlightthickness=1,
                         font=("Segoe UI", 10))
        champ.pack(fill="x", ipady=5)                   # ipady = padding interne vertical

        if placeholder:
            champ.insert(0, placeholder)               # Affiche le placeholder au départ
            champ.config(fg=TEXTE_MUT)                 # En gris pour indiquer que c'est un exemple
            # FocusIn = l'utilisateur clique dans le champ → on efface le placeholder
            champ.bind("<FocusIn>",
                       lambda e, c=champ, p=placeholder: self._effacer_placeholder(c, p))
            # FocusOut = l'utilisateur quitte le champ → on remet le placeholder si vide
            champ.bind("<FocusOut>",
                       lambda e, c=champ, p=placeholder: self._restaurer_placeholder(c, p))

        self._champs[nom] = champ                       # Stocke dans le dictionnaire pour lecture plus tard

    def _effacer_placeholder(self, champ, placeholder):
        """Efface le placeholder quand l'utilisateur entre dans le champ."""
        if champ.get() == placeholder:                 # Seulement si c'est encore le placeholder
            champ.delete(0, tk.END)
            champ.config(fg=TEXTE)                     # Passe en blanc = texte réel

    def _restaurer_placeholder(self, champ, placeholder):
        """Remet le placeholder si l'utilisateur quitte le champ sans rien saisir."""
        if champ.get() == "":
            champ.insert(0, placeholder)
            champ.config(fg=TEXTE_MUT)                 # Remet en gris

    def _creer_menu(self, parent, nom: str, options: list):
        """
        Crée un label + menu déroulant (Combobox) pour une variable catégorielle.
        state="readonly" = l'utilisateur ne peut que choisir, pas taper du texte libre.
        """
        tk.Label(parent, text=nom,
                 font=("Segoe UI", 8, "bold"),
                 fg=TEXTE_MUT, bg=NAVY_PANEL).pack(anchor="w", pady=(6, 2))

        var = tk.StringVar(value=options[0])           # Variable Tkinter qui stocke la valeur choisie
        menu = ttk.Combobox(parent, textvariable=var,
                            values=options,
                            state="readonly",          # Lecture seule : on ne peut que sélectionner
                            width=20, font=("Segoe UI", 10))
        menu.pack(fill="x", ipady=3)

        self._champs[nom] = var                        # On stocke la StringVar (pas le widget) pour lire la valeur

    # ── Lecture et validation des saisies ─────────────────────────────────────

    def _lire_saisies(self) -> dict:
        """
        Lit toutes les valeurs saisies dans les champs et les valide.
        Retourne un dictionnaire {nom_variable: valeur}.
        Lève DonneeInvalideError si un champ est vide ou contient un non-nombre.
        """
        # Placeholders à ignorer : si l'utilisateur n'a pas touché au champ, ce texte est encore là
        PLACEHOLDERS = {
            "CREDIT_SCORE":        "ex: 0.72",
            "ANNUAL_MILEAGE":      "ex: 12000",
            "SPEEDING_VIOLATIONS": "ex: 1",
            "PAST_ACCIDENTS":      "ex: 0",
            "DUIS":                "ex: 0",
          
        }
        valeurs = {}
        champs_numeriques = [                          # Ces variables doivent être des nombres
            "CREDIT_SCORE", "ANNUAL_MILEAGE",
            "SPEEDING_VIOLATIONS", "PAST_ACCIDENTS", "DUIS"
            # POSTAL_CODE retiré : accepte lettres et chiffres (ex: H3S1W7)
        ]

        for nom, champ in self._champs.items():        # Parcourt les 12 variables
            valeur = champ.get()                       # Lit la valeur tapée ou sélectionnée

            # Vide ou placeholder non effacé = champ non rempli
            if valeur == "" or valeur == PLACEHOLDERS.get(nom, ""):
                raise DonneeInvalideError(f"Le champ '{nom}' est vide.")

            if nom in champs_numeriques:
                try:
                    valeurs[nom] = float(valeur)       # Convertit en nombre décimal
                except ValueError:
                    raise DonneeInvalideError(f"'{nom}' doit être un nombre, pas '{valeur}'.")
            else:
                valeurs[nom] = valeur                  # Valeur texte pour les menus et POSTAL_CODE

        return valeurs

    # ── Prédiction ────────────────────────────────────────────────────────────

    def _predire(self):
        """
        Appelée au clic sur Prédire.
        Lit les saisies → crée un DataFrame → appelle le modèle → affiche le résultat.
        """
        try:
            valeurs = self._lire_saisies()             # Lit et valide tous les champs
            # On force l'ordre des colonnes = même ordre qu'à l'entraînement dans donnees.py
            ORDRE_COLONNES = [
                "AGE", "GENDER", "DRIVING_EXPERIENCE", "CREDIT_SCORE",
                "VEHICLE_YEAR", "MARRIED", "VEHICLE_OWNERSHIP", "ANNUAL_MILEAGE",
                "SPEEDING_VIOLATIONS", "PAST_ACCIDENTS", "DUIS", "VEHICLE_TYPE"
            ]
            X = pd.DataFrame([valeurs])[ORDRE_COLONNES]  # Réordonne les colonnes avant la prédiction               # Crée un DataFrame d'une seule ligne (1 client)
            prediction = self._modele.predire(X)[0]   # Le modèle retourne [0] ou [1] → on prend le premier

            if prediction == 1:                        # 1 = réclamation probable
                msg      = "⚠  Réclamation probable"
                sous_msg = "Ce profil présente un risque élevé de réclamation."
                couleur  = ROUGE
            else:                                      # 0 = pas de réclamation probable
                msg      = "✓  Pas de réclamation"
                sous_msg = "Ce profil présente un faible risque de réclamation."
                couleur  = VERT

            self._label_resultat.config(text=f"{msg}\n{sous_msg}", fg=couleur)
            self._sauvegarder_historique(valeurs, prediction)  # Enregistre dans historique.csv
            self._afficher_graphique()                         # Redessine le graphique

        except DonneeInvalideError as e:
            messagebox.showerror("Erreur de saisie", str(e))  # Fenêtre d'erreur, pas de crash
        except ModeleNonEntraineError as e:
            messagebox.showerror("Erreur modèle", str(e))

    # ── Graphique ─────────────────────────────────────────────────────────────

    def _afficher_graphique(self):
        """
        Affiche un graphique matplotlib de l'importance des variables.
        Appelée après chaque prédiction — vide d'abord l'ancien graphique.
        """
        # Vide tous les widgets de la colonne droite sauf le label résultat
        for widget in self._colonne_droite.winfo_children():
            if widget is self._label_resultat:
                continue                               # On garde le label résultat
            widget.destroy()                           # On détruit tout le reste

        df = charger_donnees(CHEMIN_CSV)
        colonnes = [c for c in df.columns if c != "OUTCOME"]   # Noms des 12 variables d'entrée
        importances = self._modele._modele.feature_importances_ # Tableau calculé par RandomForest

        # Création du graphique avec le thème navy
        fig, ax = plt.subplots(figsize=(5, 4.5))
        fig.patch.set_facecolor(NAVY_PANEL)            # Fond de la figure
        ax.set_facecolor(NAVY_PANEL)                   # Fond de la zone de dessin

        ax.barh(colonnes, importances,                 # barh = barres horizontales
                color=BLEU, alpha=0.85, height=0.6)
        ax.set_title("Importance des variables",
                     fontsize=10, color=TEXTE, pad=10, fontweight="bold")
        ax.tick_params(labelsize=8, colors=TEXTE_MUT)  # Labels des axes en gris
        ax.spines[:].set_color(NAVY_BORDER)            # Bordures du graphique en gris foncé
        ax.set_xlabel("Importance (RandomForest)", fontsize=8, color=TEXTE_MUT)
        fig.tight_layout(pad=1.5)

        # Intègre le graphique matplotlib dans Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self._colonne_droite)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=12, pady=(0, 12))
        plt.close(fig)                                 # Libère la mémoire

    # ── Historique ────────────────────────────────────────────────────────────

    def _sauvegarder_historique(self, valeurs: dict, prediction: int):
        """
        Sauvegarde chaque prédiction dans historique.csv.
        Si le fichier n'existe pas encore, on écrit d'abord les entêtes.
        """
        fichier_existe = os.path.exists(CHEMIN_HISTORIQUE)
        with open(CHEMIN_HISTORIQUE, "a", newline="") as f:    # "a" = append = ajoute sans écraser
            writer = csv.DictWriter(f, fieldnames=["date"] + list(valeurs.keys()) + ["prediction"])
            if not fichier_existe:
                writer.writeheader()                   # Écrit "date,AGE,GENDER,..." seulement la 1re fois
            ligne = {"date": datetime.now().strftime("%Y-%m-%d %H:%M")}
            ligne.update(valeurs)                      # Ajoute toutes les valeurs du client
            ligne["prediction"] = prediction           # Ajoute le résultat (0 ou 1)
            writer.writerow(ligne)

    def _afficher_historique(self):
        """
        Ouvre une nouvelle fenêtre (Toplevel) avec un tableau de toutes les prédictions passées.
        Utilise ttk.Treeview pour afficher les lignes de historique.csv sous forme de tableau.
        """
        if not os.path.exists(CHEMIN_HISTORIQUE):
            messagebox.showinfo("Historique", "Aucune prédiction enregistrée.")
            return

        fenetre = tk.Toplevel(self.root)               # Toplevel = nouvelle fenêtre par-dessus la principale
        fenetre.title("Historique des prédictions")
        fenetre.geometry("900x400")
        fenetre.configure(bg=NAVY_PANEL)

        frame = tk.Frame(fenetre, bg=NAVY_PANEL)
        frame.pack(fill="both", expand=True, padx=16, pady=16)

        df = pd.read_csv(CHEMIN_HISTORIQUE)            # Charge toutes les prédictions passées

        # Treeview = tableau avec colonnes et lignes
        tree = ttk.Treeview(frame, columns=list(df.columns), show="headings")
        for col in df.columns:
            tree.heading(col, text=col)                # Entête de chaque colonne
            tree.column(col, width=90)
        for _, row in df.iterrows():
            tree.insert("", "end", values=list(row))  # Ajoute chaque ligne au tableau

        # Barre de défilement verticale (si beaucoup de prédictions)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
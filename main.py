"""
main.py — Point d'entrée de l'application INF1007 Projet 2.
C'est ce fichier qu'on lance pour démarrer l'application.
"""

import tkinter as tk        # Importe Tkinter pour créer la fenêtre principale
from interface import Interface  # Importe notre classe Interface qui gère tout l'affichage

if __name__ == "__main__":  # Exécute ce bloc seulement si on lance main.py directement
    root = tk.Tk()          # Crée la fenêtre principale (convention : on l'appelle root)
    app = Interface(root)   # Crée l'interface et la connecte à la fenêtre (app = convention)
    root.mainloop()         # Garde la fenêtre ouverte — attend que l'utilisateur la ferme

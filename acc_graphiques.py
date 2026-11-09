"""
acc_graphiques.py — Graphiques ACC (version 3.2)
================================================

Chaque fonction renvoie une figure matplotlib, sans l'afficher :
    - dans le notebook :     fig = fig_xxx(...); plt.show()   (ou display(fig))
    - dans l'application :   st.pyplot(fig_xxx(...))

Trois familles :
    1. Données des bâtiments    fig_bilan_batiments, fig_productible,
                                fig_profil_mensuel, fig_profil_journalier,
                                fig_carte_chaleur, fig_carte
    2. Optimisation             fig_nuage_objectifs, fig_meilleur_par_taille
    3. Boucle retenue           fig_bilan_mensuel_boucle, fig_profil_journalier_boucle,
                                fig_par_periode, fig_gains_participants
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from acc_moteur import INDICATEURS, DonneesACC

# Code couleur unique dans tout le projet
COULEURS = {
    "consommation": "#3478D4",
    "production": "#E59A32",
    "autoconso": "#8E6CC4",
    "partage": "#23A36D",
    "non_valorise": "#F2C14E",
    "complement": "#9AA5B1",
    "accise": "#23A36D",
    "energie": "#3478D4",
    "vente": "#E59A32",
    "frais": "#D3524B",
}
FUSEAU = "Europe/Paris"


def _libelle(colonne: str) -> str:
    libelle, unite, _ = INDICATEURS.get(colonne, (colonne, "", ""))
    return f"{libelle} ({unite})" if unite else libelle


def _finaliser(ax, titre, xlabel=None, ylabel=None):
    ax.set_title(titre, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)


def _local(index, fuseau=FUSEAU):
    return pd.DatetimeIndex(index).tz_convert(fuseau)


# ============================================================================
# 1. DONNÉES DES BÂTIMENTS
# ============================================================================
def fig_bilan_batiments(bilan: pd.DataFrame):
    """Consommation, production et autoconsommation individuelle annuelles."""
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(bilan))
    largeur = 0.27
    for k, (col, nom, cle) in enumerate([
        ("Consommation_MWh", "Consommation", "consommation"),
        ("Production_MWh", "Production PV", "production"),
        ("Autoconso_ind_MWh", "Autoconsommation individuelle", "autoconso"),
    ]):
        ax.bar(x + (k - 1) * largeur, bilan[col], largeur, label=nom, color=COULEURS[cle])
    ax.set_xticks(x, bilan["Batiment"])
    ax.legend()
    _finaliser(ax, "Bilan énergétique annuel par bâtiment", ylabel="Énergie (MWh/an)")
    fig.tight_layout()
    return fig


def fig_productible(bilan: pd.DataFrame, plage=(1000, 1400)):
    """Productible des producteurs : repère les puissances crêtes incohérentes."""
    prod = bilan[bilan["Puissance_PV_kWc"] > 0]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if prod.empty:
        ax.text(0.5, 0.5, "Aucun producteur PV", ha="center", va="center")
        ax.axis("off")
        return fig
    ax.axhspan(*plage, color="#DDDDDD", alpha=0.6,
               label=f"Plage indicative France ({plage[0]}–{plage[1]} kWh/kWc)")
    hors = (prod["Productible_kWh_kWc"] < plage[0]) | (prod["Productible_kWh_kWc"] > plage[1])
    ax.bar(prod["Batiment"], prod["Productible_kWh_kWc"],
           color=np.where(hors, COULEURS["frais"], COULEURS["production"]))
    for xb, v in zip(prod["Batiment"], prod["Productible_kWh_kWc"]):
        ax.annotate(f"{v:,.0f}".replace(",", " "), (xb, v), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=9)
    ax.legend(loc="upper left", fontsize=9)
    _finaliser(ax, "Productible annuel (rouge : hors plage, vérifier la puissance crête)",
               ylabel="kWh produits par kWc installé")
    fig.tight_layout()
    return fig


def fig_profil_mensuel(donnees: DonneesACC, fuseau=FUSEAU):
    """Consommation et production mensuelles, tous bâtiments confondus."""
    mois = _local(donnees.index, fuseau).month
    conso = pd.Series((donnees.A + donnees.D).sum(axis=1)).groupby(mois).sum() / 1000
    prod = pd.Series((donnees.A + donnees.S).sum(axis=1)).groupby(mois).sum() / 1000
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(1, 13)
    ax.bar(x - 0.2, conso.reindex(x, fill_value=0), 0.4, label="Consommation",
           color=COULEURS["consommation"])
    ax.bar(x + 0.2, prod.reindex(x, fill_value=0), 0.4, label="Production PV",
           color=COULEURS["production"])
    ax.set_xticks(x, ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août",
                      "Sep", "Oct", "Nov", "Déc"])
    ax.legend()
    _finaliser(ax, "Consommation et production mensuelles (tous les bâtiments)",
               ylabel="Énergie (MWh/mois)")
    fig.tight_layout()
    return fig


def fig_profil_journalier(donnees: DonneesACC, numeros=None, fuseau=FUSEAU):
    """Journée moyenne (heure locale) : consommation et production."""
    pos = list(range(donnees.n)) if numeros is None else donnees.positions(numeros)
    local = _local(donnees.index, fuseau)
    heure = local.hour + local.minute / 60
    conso = pd.Series((donnees.A + donnees.D)[:, pos].sum(axis=1)).groupby(heure).mean()
    prod = pd.Series((donnees.A + donnees.S)[:, pos].sum(axis=1)).groupby(heure).mean()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(conso.index, conso, color=COULEURS["consommation"], lw=2, label="Consommation")
    ax.plot(prod.index, prod, color=COULEURS["production"], lw=2, label="Production PV")
    ax.fill_between(prod.index, np.minimum(conso, prod), color=COULEURS["partage"],
                    alpha=0.25, label="Recouvrement production / consommation")
    ax.set_xticks(range(0, 25, 2))
    ax.set_xlim(0, 24)
    ax.legend()
    qui = "tous les bâtiments" if numeros is None else ", ".join(f"B{b}" for b in numeros)
    _finaliser(ax, f"Journée moyenne sur l'année ({qui})", xlabel="Heure locale",
               ylabel="Énergie moyenne par pas (kWh)")
    fig.tight_layout()
    return fig


def fig_carte_chaleur(donnees: DonneesACC, numero: int, variable="Consommation",
                      fuseau=FUSEAU):
    """Jour de l'année × heure : où et quand l'énergie est consommée ou produite."""
    p = donnees.positions([numero])[0]
    valeurs = (donnees.A + donnees.D)[:, p] if variable == "Consommation" \
        else (donnees.A + donnees.S)[:, p]
    local = _local(donnees.index, fuseau)
    table = pd.DataFrame({"jour": local.normalize().tz_localize(None),
                          "heure": local.hour + local.minute / 60, "v": valeurs})
    grille = table.pivot_table(index="heure", columns="jour", values="v", aggfunc="mean")
    # Jours incomplets (premier/dernier jour après conversion UTC -> heure locale)
    grille = grille.loc[:, grille.notna().sum() >= 0.9 * grille.notna().sum().max()]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    image = ax.imshow(grille.to_numpy(), aspect="auto", origin="lower",
                      cmap="viridis" if variable == "Consommation" else "inferno",
                      extent=[0, grille.shape[1], grille.index.min(),
                              grille.index.max() + (grille.index[1] - grille.index[0]
                                                    if len(grille.index) > 1 else 1)])
    fig.colorbar(image, ax=ax, label="kWh par pas")
    debuts_mois = [i for i, j in enumerate(grille.columns) if j.day == 1]
    ax.set_xticks(debuts_mois, [grille.columns[i].strftime("%b") for i in debuts_mois])
    ax.set_title(f"B{numero} — {variable} par jour et par heure", fontweight="bold")
    ax.set_ylabel("Heure locale")
    fig.tight_layout()
    return fig


def fig_carte(donnees: DonneesACC, d_max_km: float, selection=None, titre=None):
    """Positions, liaisons admissibles (≤ d_max_km) et boucle sélectionnée."""
    choisis = set(donnees.positions(selection)) if selection else set()
    fig, ax = plt.subplots(figsize=(9, 7))
    for i in range(donnees.n):
        for j in range(i + 1, donnees.n):
            if donnees.distances[i, j] <= d_max_km:
                interne = i in choisis and j in choisis
                ax.plot([donnees.lon[i], donnees.lon[j]], [donnees.lat[i], donnees.lat[j]],
                        color="darkorange" if interne else "#CCCCCC",
                        lw=2 if interne else 0.8, zorder=1)
    for i in range(donnees.n):
        prod = donnees.pv[i] > 0
        actif = not choisis or i in choisis
        ax.scatter(donnees.lon[i], donnees.lat[i], s=200 if i in choisis else 110,
                   marker="^" if prod else "o",
                   color=(COULEURS["production"] if prod else COULEURS["consommation"])
                   if actif else "#DDDDDD", edgecolors="black", zorder=3)
        ax.annotate(f"B{donnees.numeros[i]}", (donnees.lon[i], donnees.lat[i]),
                    xytext=(6, 6), textcoords="offset points",
                    fontweight="bold" if i in choisis else "normal")
    ax.set_aspect(1 / np.cos(np.radians(donnees.lat.mean())), adjustable="datalim")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.legend(handles=[
        Line2D([0], [0], marker="^", ls="None", mfc=COULEURS["production"], mec="black",
               ms=10, label="Producteur"),
        Line2D([0], [0], marker="o", ls="None", mfc=COULEURS["consommation"], mec="black",
               ms=10, label="Consommateur"),
        Line2D([0], [0], color="#CCCCCC", label=f"Liaison possible (≤ {d_max_km:g} km)"),
    ] + ([Line2D([0], [0], color="darkorange", lw=2, label="Boucle sélectionnée")]
         if choisis else []), loc="best", fontsize=9)
    ax.set_title(titre or "Localisation des bâtiments et liaisons admissibles",
                 fontweight="bold")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


# ============================================================================
# 2. OPTIMISATION
# ============================================================================
def fig_nuage_objectifs(candidats: pd.DataFrame, x: str, y: str, solution=None,
                        couleur: str = "Nb_batiments", tableau_complet=None):
    """Deux indicateurs face à face : front de Pareto et boucle retenue."""
    fig, ax = plt.subplots(figsize=(11, 7))
    if tableau_complet is not None:
        exclues = tableau_complet[~tableau_complet["Solution"].isin(candidats["Solution"])]
        if not exclues.empty:
            ax.scatter(exclues[x], exclues[y], marker="x", color="#C0392B", alpha=0.4, s=30,
                       label="Écartée par un seuil")
    dominees = candidats[~candidats["Pareto"]]
    front = candidats[candidats["Pareto"]]
    ax.scatter(dominees[x], dominees[y], color="#BBBBBB", alpha=0.6, s=35,
               label="Dominée")
    nuage = ax.scatter(front[x], front[y], c=front[couleur], cmap="viridis", s=110,
                       edgecolors="black", linewidths=0.6, zorder=3,
                       label="Front de Pareto (objectifs choisis)")
    fig.colorbar(nuage, ax=ax, label=_libelle(couleur))
    if solution is not None:
        ax.scatter(solution[x], solution[y], marker="*", s=480, color="#D62728",
                   edgecolors="black", zorder=5, label=f"Boucle retenue : {solution['Solution']}")
        x_min, x_max = candidats[x].min(), candidats[x].max()
        a_droite = x_max > x_min and (solution[x] - x_min) > 0.7 * (x_max - x_min)
        ax.annotate(solution["Solution"], (solution[x], solution[y]),
                    xytext=(-12 if a_droite else 12, 10), textcoords="offset points",
                    ha="right" if a_droite else "left", fontsize=10, fontweight="bold")
    ax.set_xlabel(_libelle(x))
    ax.set_ylabel(_libelle(y))
    ax.set_title("Boucles admissibles : compromis entre deux indicateurs\n"
                 "(le front est calculé sur tous les objectifs choisis, pas seulement ces deux)",
                 fontweight="bold")
    ax.grid(alpha=0.25)
    ax.margins(0.06)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    return fig


def fig_meilleur_par_taille(tableau: pd.DataFrame, colonnes: list):
    """Meilleure valeur de chaque objectif selon le nombre de participants."""
    colonnes = list(colonnes)[:4]
    fig, axes = plt.subplots(1, len(colonnes), figsize=(4.6 * len(colonnes), 4), squeeze=False)
    for ax, col in zip(axes[0], colonnes):
        sens = INDICATEURS.get(col, ("", "", "max"))[2]
        groupe = tableau.groupby("Nb_batiments")[col]
        meilleur = groupe.max() if sens == "max" else groupe.min()
        ax.bar(meilleur.index, meilleur, color=COULEURS["consommation"], alpha=0.85)
        ax.set_xticks(meilleur.index)
        _finaliser(ax, f"{'Max' if sens == 'max' else 'Min'} : {INDICATEURS.get(col, (col,))[0]}",
                   xlabel="Nombre de participants", ylabel=_libelle(col))
        ax.title.set_fontsize(10)
    fig.suptitle("Effet de la taille de la boucle sur chaque objectif", fontweight="bold")
    fig.tight_layout()
    return fig


# ============================================================================
# 3. BOUCLE RETENUE
# ============================================================================
def fig_bilan_mensuel_boucle(series: pd.DataFrame, fuseau=FUSEAU):
    """Chaque mois : surplus partagé ou perdu, besoin couvert ou acheté."""
    mois = _local(series.index, fuseau).month
    m = series.groupby(mois)[["E_ACC_kWh", "Surplus_non_valorise_kWh",
                              "Achat_complementaire_kWh"]].sum().reindex(range(1, 13),
                                                                       fill_value=0) / 1000
    x = np.arange(1, 13)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - 0.2, m["E_ACC_kWh"], 0.4, color=COULEURS["partage"], label="Partagé dans la boucle")
    ax.bar(x - 0.2, m["Surplus_non_valorise_kWh"], 0.4, bottom=m["E_ACC_kWh"],
           color=COULEURS["non_valorise"], label="Surplus non valorisé (barre gauche)")
    ax.bar(x + 0.2, m["E_ACC_kWh"], 0.4, color=COULEURS["partage"])
    ax.bar(x + 0.2, m["Achat_complementaire_kWh"], 0.4, bottom=m["E_ACC_kWh"],
           color=COULEURS["complement"], label="Acheté au fournisseur (barre droite)")
    ax.set_xticks(x, ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août",
                      "Sep", "Oct", "Nov", "Déc"])
    ax.legend(fontsize=9)
    _finaliser(ax, "Bilan mensuel de la boucle : à gauche le surplus PV, à droite le besoin",
               ylabel="Énergie (MWh/mois)")
    fig.tight_layout()
    return fig


def fig_profil_journalier_boucle(series: pd.DataFrame, fuseau=FUSEAU):
    """Journée moyenne de la boucle : surplus, besoin et énergie partagée."""
    local = _local(series.index, fuseau)
    heure = local.hour + local.minute / 60
    moy = series.groupby(heure)[["Surplus_kWh", "Demande_kWh", "E_ACC_kWh"]].mean()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(moy.index, moy["Demande_kWh"], color=COULEURS["consommation"], lw=2,
            label="Besoin résiduel")
    ax.plot(moy.index, moy["Surplus_kWh"], color=COULEURS["production"], lw=2,
            label="Surplus PV disponible")
    ax.fill_between(moy.index, moy["E_ACC_kWh"], color=COULEURS["partage"], alpha=0.35,
                    label="Énergie partagée")
    ax.set_xticks(range(0, 25, 2))
    ax.set_xlim(0, 24)
    ax.legend()
    _finaliser(ax, "Journée moyenne de la boucle", xlabel="Heure locale",
               ylabel="Énergie moyenne par pas (kWh)")
    fig.tight_layout()
    return fig


def fig_par_periode(par_periode: pd.DataFrame):
    """Surplus, besoin et énergie partagée par plage tarifaire."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(par_periode))
    for k, (col, nom, cle) in enumerate([("Surplus_MWh", "Surplus PV", "production"),
                                         ("Demande_MWh", "Besoin résiduel", "consommation"),
                                         ("E_ACC_MWh", "Énergie partagée", "partage")]):
        ax.bar(x + (k - 1) * 0.27, par_periode[col], 0.27, label=nom, color=COULEURS[cle])
    ax.set_xticks(x, par_periode.index)
    ax.legend()
    _finaliser(ax, "Bilan par plage tarifaire", ylabel="Énergie (MWh/an)")
    fig.tight_layout()
    return fig


def fig_gains_participants(participants: pd.DataFrame, titre="Gain annuel par participant"):
    """Gains décomposés : énergie, accise, vente ACC, frais."""
    fig, ax = plt.subplots(figsize=(max(9, 1.3 * len(participants)), 5.5))
    x = np.arange(len(participants))
    haut, bas = np.zeros(len(x)), np.zeros(len(x))
    for col, nom, cle in [("Gain_energie_EUR", "Énergie (consommateur)", "energie"),
                          ("Accise_evitee_EUR", "Accise évitée (consommateur)", "accise"),
                          ("Gain_producteur_EUR", "Vente ACC (producteur)", "vente")]:
        v = participants[col].to_numpy(dtype=float)
        ax.bar(x, v, bottom=np.where(v >= 0, haut, bas), color=COULEURS[cle], label=nom)
        haut, bas = haut + np.maximum(v, 0), bas + np.minimum(v, 0)
    frais = participants["Frais_EUR"].to_numpy(dtype=float)
    if (frais > 0).any():
        ax.bar(x, -frais, bottom=bas, color=COULEURS["frais"], label="Frais de gestion")
        bas = bas - frais
    for k, total in enumerate(participants["Gain_total_EUR"].to_numpy(dtype=float)):
        ax.annotate(f"{total:,.0f} €".replace(",", " "), (k, haut[k] if total >= 0 else bas[k]),
                    xytext=(0, 4 if total >= 0 else -12), textcoords="offset points",
                    ha="center", fontsize=9, fontweight="bold")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, participants["Batiment"])
    ax.set_ylim(min(0.0, bas.min() * 1.2), max(1.0, haut.max() * 1.15))
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4)
    _finaliser(ax, titre, ylabel="Gain annuel (€/an)")
    fig.tight_layout()
    return fig


def fig_comparaison_strategies(tableau: pd.DataFrame, choix: pd.DataFrame,
                               x: str = "E_ACC_MWh", y: str = "Valeur_collective_EUR"):
    """Toutes les boucles en gris, et la boucle choisie par chaque stratégie."""
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(tableau[x], tableau[y], color="#CCCCCC", s=30, label="Boucles admissibles")
    marqueurs = ["*", "D", "s", "^", "P", "X", "o", "v"]
    for k, (strategie, ligne) in enumerate(choix.iterrows()):
        ax.scatter(ligne[x], ligne[y], s=260, marker=marqueurs[k % len(marqueurs)],
                   edgecolors="black", zorder=3 + k,
                   label=f"{strategie} : {ligne['Solution']} ({int(ligne['Nb_batiments'])} bât.)")
    ax.set_xlabel(_libelle(x))
    ax.set_ylabel(_libelle(y))
    ax.set_title("Boucle choisie par chaque stratégie d'optimisation", fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    return fig

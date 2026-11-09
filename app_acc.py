"""
app_acc.py — Prototype d'application ACC (version 3.2)
=====================================================

Lancement, depuis le dossier qui contient ce fichier :

    streamlit run app_acc.py

Fichiers nécessaires dans le même dossier :
    acc_donnees.py     (votre module de chargement, inchangé)
    acc_moteur.py      (calculs)
    acc_graphiques.py  (graphiques)

Prototype volontairement simple : l'objectif est de valider les écrans,
les tableaux et les graphiques avant de construire une vraie application.
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import acc_moteur as m
import acc_graphiques as g

st.set_page_config(page_title="ACC — prototype d'optimisation", layout="wide")
st.title("Autoconsommation collective — prototype d'optimisation")


def afficher(fig):
    st.pyplot(fig)
    plt.close(fig)


def libelle(colonne):
    nom, unite, _ = m.INDICATEURS.get(colonne, (colonne, "", ""))
    return f"{nom} ({unite})" if unite else nom


def csv(df):
    return df.drop(columns=["Indices"], errors="ignore").to_csv(
        sep=";", index=False).encode("utf-8-sig")


# ============================================================================
# BARRE LATÉRALE : DONNÉES, CONTRAINTES, PRIX
# ============================================================================
st.sidebar.header("Données")
dossier = st.sidebar.text_input("Dossier des fichiers Excel", "donnees/batiments")
pas = st.sidebar.selectbox("Pas de mesure (minutes)", [60, 30, 15], index=0)
fuseau_naif = st.sidebar.selectbox("Fuseau des dates sans fuseau", ["UTC", "Europe/Paris"])
controle_pv = st.sidebar.selectbox("Puissance PV incohérente", ["avertir", "erreur"])
if st.sidebar.button("Charger les données", type="primary"):
    st.session_state["parametres_chargement"] = (dossier, pas, fuseau_naif, controle_pv)
    st.session_state.pop("resultat", None)


@st.cache_resource(show_spinner="Lecture des fichiers Excel…")
def charger(dossier, pas, fuseau_naif, controle_pv):
    from acc_donnees import ConfigurationACC, charger_donnees_acc
    with warnings.catch_warnings(record=True) as avertissements:
        warnings.simplefilter("always", UserWarning)
        jeu = charger_donnees_acc(ConfigurationACC(
            dossier=Path(dossier), pas_minutes=pas, fuseau_horaires_naifs=fuseau_naif,
            strict=True, controle_pv=controle_pv))
        jeu.exiger_complet()
    messages = [str(a.message) for a in avertissements if issubclass(a.category, UserWarning)]
    return jeu, m.depuis_jeu_acc(jeu), messages


if "parametres_chargement" not in st.session_state:
    st.info("Indiquez le dossier des fichiers Excel dans la barre latérale, "
            "puis cliquez sur « Charger les données ».")
    st.stop()

try:
    jeu, donnees, avertissements = charger(*st.session_state["parametres_chargement"])
except ModuleNotFoundError:
    st.error("Module acc_donnees.py introuvable : placez-le dans le même dossier que app_acc.py.")
    st.stop()
except Exception as erreur:
    st.error(f"Chargement impossible : {erreur}")
    st.stop()

st.sidebar.header("Contraintes de la boucle")
contraintes = dict(
    d_max_km=st.sidebar.number_input("Distance max entre participants (km)", 0.1, 50.0, 10.0, 0.5),
    p_pv_max_kwc=st.sidebar.number_input("Puissance PV totale strictement inférieure à (kWc)",
                                         1.0, 100000.0, 500.0, 50.0),
    nb_min_batiments=2,
    nb_max_batiments=st.sidebar.number_input("Nombre max de participants (0 = sans limite)",
                                             0, donnees.n, 0, 1),
    exiger_consommateur_sans_pv=st.sidebar.checkbox("Au moins un consommateur sans PV", True),
)

st.sidebar.header("Prix (€/kWh HT)")
st.sidebar.caption("Valeurs par défaut fictives, sauf l'accise.")
with st.sidebar.expander("Prix de l'énergie", expanded=False):
    prix_fournisseur = {p: st.number_input(f"Fournisseur, part énergie — {p}", 0.0, 1.0,
                                           m.Tarifs().prix_fournisseur[p], 0.005, format="%.4f")
                        for p in m.PERIODES}
    prix_acc = {p: st.number_input(f"Prix ACC — {p}", 0.0, 1.0, m.Tarifs().prix_acc[p], 0.005,
                                   format="%.4f") for p in m.PERIODES}
    valeur_surplus = st.number_input("Valeur du surplus sans ACC", 0.0, 1.0, 0.0, 0.005,
                                     format="%.4f")
with st.sidebar.expander("Taxes et frais", expanded=False):
    taux_accise = st.number_input("Accise (€/kWh)", 0.0, 0.1, 0.02635, 0.001, format="%.5f")
    tva = st.number_input("TVA non récupérable (0 ou 0.20)", 0.0, 0.3, 0.0, 0.05)
    frais = st.number_input("Frais de gestion (€/an par participant)", 0.0, 100000.0, 0.0, 10.0)
with st.sidebar.expander("Plages horaires", expanded=False):
    plages = dict(
        creuses_debut=st.number_input("Heures creuses : début", 0.0, 23.5, 22.0, 0.5),
        creuses_fin=st.number_input("Heures creuses : fin", 0.0, 23.5, 6.0, 0.5),
        pointe_debut=st.number_input("Pointe : début", 0.0, 23.5, 18.0, 0.5),
        pointe_fin=st.number_input("Pointe : fin", 0.0, 23.5, 20.0, 0.5),
        pointe_jours_ouvres=st.checkbox("Pointe en jours ouvrés seulement", True),
    )
tarifs = dict(prix_fournisseur=prix_fournisseur, prix_acc=prix_acc,
              valeur_surplus=valeur_surplus, taux_accise=taux_accise,
              tva_non_recuperable=tva, frais_par_participant=frais)


@st.cache_data(show_spinner="Calcul de toutes les boucles admissibles…")
def calculer_tableau(_donnees, cle_donnees, contraintes, tarifs, plages):
    periodes = m.classer_periodes(_donnees.index, m.PlagesHoraires(**plages))
    boucles = m.enumerer_boucles(_donnees, m.Contraintes(**contraintes))
    return m.evaluer_boucles(_donnees, boucles, m.Tarifs(**tarifs), periodes), periodes


try:
    tableau, periodes = calculer_tableau(donnees, st.session_state["parametres_chargement"],
                                         contraintes, tarifs, plages)
except Exception as erreur:
    st.error(f"Calcul impossible : {erreur}")
    st.stop()

# Un résultat d'optimisation n'est valable que pour les paramètres qui l'ont produit.
cle_parametres = repr((st.session_state["parametres_chargement"], contraintes, tarifs, plages))
if st.session_state.get("cle_resultat") != cle_parametres:
    st.session_state.pop("resultat", None)

onglet_donnees, onglet_boucles, onglet_optim, onglet_boucle = st.tabs(
    ["Données", "Boucles admissibles", "Optimisation", "Boucle retenue"])

# ============================================================================
# ONGLET 1 : DONNÉES
# ============================================================================
with onglet_donnees:
    bilan = m.bilan_batiments(donnees)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bâtiments", donnees.n)
    c2.metric("Producteurs PV", int(donnees.est_producteur.sum()))
    c3.metric("Pas de mesure", f"{jeu.pas_minutes} min")
    c4.metric("Période", f"{donnees.index[0]:%d/%m/%Y} → {donnees.index[-1]:%d/%m/%Y}")
    for message in avertissements:
        st.warning(message)

    st.subheader("Bilan annuel par bâtiment")
    st.dataframe(bilan.round(2), hide_index=True)
    afficher(g.fig_bilan_batiments(bilan))
    afficher(g.fig_productible(bilan))
    afficher(g.fig_profil_mensuel(donnees))
    afficher(g.fig_profil_journalier(donnees))

    st.subheader("Profil détaillé d'un bâtiment")
    col_a, col_b = st.columns(2)
    numero = col_a.selectbox("Bâtiment", donnees.numeros, format_func=lambda b: f"B{b}")
    variable = col_b.radio("Grandeur", ["Consommation", "Production"], horizontal=True)
    afficher(g.fig_carte_chaleur(donnees, int(numero), variable))

    st.subheader("Localisation")
    afficher(g.fig_carte(donnees, contraintes["d_max_km"]))

# ============================================================================
# ONGLET 2 : BOUCLES ADMISSIBLES
# ============================================================================
with onglet_boucles:
    st.metric("Boucles admissibles", len(tableau))
    colonnes_courtes = ["Solution", "Participants", "Nb_batiments", "Distance_max_km",
                        "Puissance_PV_kWc", "E_ACC_MWh", "TAC_pct", "TAP_pct",
                        "Valeur_collective_EUR", "Gain_moyen_EUR", "Gain_max_EUR",
                        "Gain_min_EUR", "Participant_gain_min"]
    tout = st.checkbox("Afficher toutes les colonnes", False)
    vue = tableau.drop(columns=["Indices"]) if tout else tableau[colonnes_courtes]
    st.dataframe(vue.round(2), hide_index=True, height=450)
    st.download_button("Télécharger le tableau (CSV)", csv(tableau),
                       "boucles_admissibles.csv", "text/csv")
    st.caption("Gains en €/an. Valeur_collective = somme des gains de tous les participants ; "
               "le prix ACC la répartit sans la modifier.")
    afficher(g.fig_meilleur_par_taille(
        tableau, ["E_ACC_MWh", "Valeur_collective_EUR", "Gain_moyen_EUR", "Gain_min_EUR"]))

# ============================================================================
# ONGLET 3 : OPTIMISATION MULTI-OBJECTIF
# ============================================================================
with onglet_optim:
    st.write("Choisissez les objectifs **dans l'ordre de priorité**, puis les seuils à respecter. "
             "Le calcul garde les boucles non dominées, puis les départage par priorité : "
             "aucune pondération n'est utilisée.")

    colonnes_indic = list(m.INDICATEURS)
    choisis = st.multiselect("Objectifs (le premier choisi est prioritaire)", colonnes_indic,
                             default=["Valeur_collective_EUR", "Gain_min_EUR"],
                             format_func=libelle)
    objectifs = []
    if choisis:
        colonnes_sens = st.columns(len(choisis))
        for k, (col, colonne) in enumerate(zip(colonnes_sens, choisis)):
            defaut = m.INDICATEURS[colonne][2]
            sens = col.radio(f"{k + 1}. {m.INDICATEURS[colonne][0]}", ["max", "min"],
                             index=0 if defaut == "max" else 1, key=f"sens_{colonne}",
                             format_func=lambda s: "Maximiser" if s == "max" else "Minimiser")
            objectifs.append(m.Objectif(colonne, sens))

    st.write("Seuils (ajoutez ou supprimez des lignes) :")
    libelles = {libelle(c): c for c in colonnes_indic}
    seuils_saisis = st.data_editor(
        pd.DataFrame({"Indicateur": [libelle("Gain_min_EUR"), libelle("TAC_pct")],
                      "Opérateur": [">=", ">="], "Valeur": [0.0, 0.0]}),
        num_rows="dynamic", hide_index=True, key="seuils",
        column_config={
            "Indicateur": st.column_config.SelectboxColumn(options=list(libelles), required=True),
            "Opérateur": st.column_config.SelectboxColumn(options=[">=", "<="], required=True),
            "Valeur": st.column_config.NumberColumn(required=True),
        })
    tolerance = st.slider("Tolérance sur les objectifs prioritaires (%)", 0.0, 20.0, 5.0, 0.5,
                          help="0 % : ordre strict. 5 % : on accepte de perdre jusqu'à 5 % sur "
                               "l'objectif 1 pour gagner sur l'objectif 2, etc.")

    if st.button("Lancer l'optimisation", type="primary"):
        seuils = [m.Seuil(libelles[r["Indicateur"]], r["Opérateur"], float(r["Valeur"]))
                  for _, r in seuils_saisis.dropna().iterrows()]
        try:
            st.session_state["resultat"] = m.optimiser(tableau, objectifs, seuils, tolerance)
            st.session_state["cle_resultat"] = cle_parametres
        except ValueError as erreur:
            st.session_state.pop("resultat", None)
            st.error(str(erreur))

    resultat = st.session_state.get("resultat")
    if resultat is None:
        st.info("Choisissez les objectifs et les seuils, puis cliquez sur « Lancer l'optimisation ». "
                "Un changement de contraintes ou de prix efface le résultat précédent.")
    else:
        sol = resultat.solution
        st.subheader(f"Boucle retenue : {sol['Solution']} — {sol['Participants']}")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Énergie partagée", f"{sol['E_ACC_MWh']:.1f} MWh")
        c2.metric("TAC / TAP", f"{sol['TAC_pct']:.1f} % / {sol['TAP_pct']:.1f} %")
        c3.metric("Gain total", f"{sol['Valeur_collective_EUR']:,.0f} €/an".replace(",", " "))
        c4.metric("Gain moyen", f"{sol['Gain_moyen_EUR']:,.0f} €/an".replace(",", " "))
        c5.metric("Gain minimal", f"{sol['Gain_min_EUR']:,.0f} €/an".replace(",", " "))

        st.write(f"{len(resultat.candidats)} boucles respectent les seuils, dont "
                 f"{int(resultat.candidats['Pareto'].sum())} non dominées. "
                 "Étapes de la sélection par priorité :")
        st.dataframe(resultat.etapes.round(2), hide_index=True)

        front = resultat.candidats[resultat.candidats["Pareto"]].sort_values(
            objectifs[0].colonne, ascending=objectifs[0].sens == "min")
        st.write("Front de Pareto :")
        colonnes_front = list(dict.fromkeys(["Solution", "Participants", "Nb_batiments"]
                                            + [o.colonne for o in resultat.objectifs]))
        st.dataframe(front[colonnes_front].round(2), hide_index=True)
        st.download_button("Télécharger le front (CSV)", csv(front), "front_pareto.csv",
                           "text/csv")

        axes = [o.colonne for o in resultat.objectifs]
        col_x, col_y = st.columns(2)
        axe_x = col_x.selectbox("Axe horizontal", colonnes_indic, index=colonnes_indic.index(axes[0]),
                                format_func=libelle)
        axe_y = col_y.selectbox("Axe vertical", colonnes_indic,
                                index=colonnes_indic.index(axes[1] if len(axes) > 1 else "TAC_pct"),
                                format_func=libelle)
        afficher(g.fig_nuage_objectifs(resultat.candidats, axe_x, axe_y, sol,
                                       tableau_complet=tableau))
        afficher(g.fig_meilleur_par_taille(resultat.candidats, axes))

# ============================================================================
# ONGLET 4 : BOUCLE RETENUE (OU TOUTE AUTRE BOUCLE)
# ============================================================================
with onglet_boucle:
    resultat = st.session_state.get("resultat")
    options = tableau["Solution"].tolist()
    defaut = options.index(resultat.solution["Solution"]) \
        if resultat is not None and resultat.solution["Solution"] in options else 0
    participants_par_solution = dict(zip(tableau["Solution"], tableau["Participants"]))
    choix = st.selectbox("Boucle à analyser", options, index=defaut,
                         format_func=lambda s: f"{s} — {participants_par_solution[s]}")
    ligne = tableau.set_index("Solution").loc[choix]
    numeros = [int(b.strip()[1:]) for b in ligne["Participants"].split(",")]
    analyse = m.analyser_boucle(donnees, numeros, m.Tarifs(**tarifs), periodes)
    r = analyse["resume"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Énergie partagée", f"{r['E_ACC_MWh']:.1f} MWh/an")
    c2.metric("TAC", f"{r['TAC_pct']:.1f} %")
    c3.metric("TAP", f"{r['TAP_pct']:.1f} %")
    c4.metric("Gain total", f"{r['Valeur_collective_EUR']:,.0f} €/an".replace(",", " "))

    st.subheader("Gains par participant")
    st.dataframe(analyse["participants"].round(1), hide_index=True)
    st.download_button("Télécharger les gains (CSV)", csv(analyse["participants"]),
                       f"gains_{choix}.csv", "text/csv")
    afficher(g.fig_gains_participants(analyse["participants"],
                                      f"{choix} — gain annuel par participant"))
    afficher(g.fig_bilan_mensuel_boucle(analyse["series"]))
    afficher(g.fig_profil_journalier_boucle(analyse["series"]))
    st.dataframe(analyse["par_periode"].round(2))
    afficher(g.fig_par_periode(analyse["par_periode"]))
    afficher(g.fig_carte(donnees, contraintes["d_max_km"], numeros, f"Localisation de {choix}"))

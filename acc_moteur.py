"""
acc_moteur.py — Moteur de calcul de l'autoconsommation collective (version 3.2)
==============================================================================

Un seul endroit pour tous les calculs, utilisé à la fois par le notebook
ACC_MODEL_3_2.ipynb et par l'application app_acc.py :

    1. Préparation des données      preparer_donnees(), depuis_jeu_acc()
    2. Plages tarifaires            classer_periodes()
    3. Boucles admissibles          enumerer_boucles()
    4. Énergie et gains             evaluer_boucles(), analyser_boucle()
    5. Optimisation multi-objectif  optimiser()

Le chargement des fichiers Excel reste confié à acc_donnees.py (inchangé).
Aucun graphique ici : voir acc_graphiques.py.

Conventions (identiques aux versions 3.x) :
    - S : surplus PV injecté après autoconsommation individuelle (kWh/pas)
    - D : soutirage résiduel après autoconsommation individuelle (kWh/pas)
    - A : autoconsommation individuelle (kWh/pas)
    - Énergie partagée à chaque pas : E_t = min(somme S, somme D)
    - Répartition : prorata de la consommation résiduelle (consommateurs)
      et prorata du surplus injecté (producteurs).

LES PRIX PAR DÉFAUT SONT FICTIFS, sauf le taux d'accise (catégorie > 36 kVA,
en vigueur depuis le 1er août 2026) qui doit être mis à jour à chaque révision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

PERIODES = ("Creuses", "Pleines", "Pointe")


# ============================================================================
# 1. PARAMÈTRES
# ============================================================================
@dataclass
class Contraintes:
    """Contraintes de constitution d'une boucle (hypothèses du modèle)."""
    d_max_km: float = 10.0                  # distance max entre deux participants
    p_pv_max_kwc: float = 500.0             # puissance PV totale STRICTEMENT inférieure
    nb_min_batiments: int = 2
    nb_max_batiments: int = 0               # 0 = pas de limite
    exiger_consommateur_sans_pv: bool = True

    def verifier(self) -> None:
        if self.d_max_km <= 0 or self.p_pv_max_kwc <= 0:
            raise ValueError("Distance maximale et puissance maximale doivent être positives.")
        if self.nb_min_batiments < 2:
            raise ValueError("Une boucle compte au moins 2 bâtiments.")
        if self.nb_max_batiments and self.nb_max_batiments < self.nb_min_batiments:
            raise ValueError("Le nombre maximal de bâtiments est inférieur au minimal.")


@dataclass
class PlagesHoraires:
    """Plages tarifaires en heure locale (mêmes valeurs que le Bloc 11 v3)."""
    fuseau: str = "Europe/Paris"
    creuses_debut: float = 22.0             # inclus
    creuses_fin: float = 6.0                # exclu
    pointe_debut: float = 18.0              # inclus
    pointe_fin: float = 20.0                # exclu
    pointe_jours_ouvres: bool = True


@dataclass
class Tarifs:
    """Prix en €/kWh hors TVA. Prix de l'énergie FICTIFS par défaut."""
    # Part ÉNERGIE seule du fournisseur : sans TURPE, sans accise, sans TVA.
    prix_fournisseur: dict = field(default_factory=lambda: {
        "Creuses": 0.10, "Pleines": 0.13, "Pointe": 0.13})
    # Prix de l'énergie vendue au sein de l'ACC.
    prix_acc: dict = field(default_factory=lambda: {
        "Creuses": 0.10, "Pleines": 0.10, "Pointe": 0.10})
    # Valeur du surplus pour le producteur SANS ACC (0 = non rémunéré).
    valeur_surplus: float = 0.0
    # Accise : 0,02635 €/kWh (> 36 kVA depuis le 1er août 2026) ; 0,03062 si ≤ 36 kVA.
    taux_accise: float = 0.02635
    # Accise nulle sur les kWh partagés issus d'une installation < 1 MWc.
    seuil_accise_kwc: float = 1000.0
    # TVA non récupérable par les consommateurs (0.20 pour des particuliers).
    tva_non_recuperable: float = 0.0
    # Frais annuels de gestion par participant (€/an).
    frais_par_participant: float = 0.0

    def verifier(self) -> None:
        for nom, dico in [("prix_fournisseur", self.prix_fournisseur),
                          ("prix_acc", self.prix_acc)]:
            if set(dico) != set(PERIODES):
                raise ValueError(f"{nom} doit contenir exactement {PERIODES}.")
            if not all(np.isfinite(v) and v >= 0 for v in dico.values()):
                raise ValueError(f"{nom} : prix négatifs ou invalides.")
        for nom in ["valeur_surplus", "taux_accise", "seuil_accise_kwc",
                    "tva_non_recuperable", "frais_par_participant"]:
            v = getattr(self, nom)
            if not (np.isfinite(v) and v >= 0):
                raise ValueError(f"{nom} doit être un nombre positif ou nul.")
        if self.taux_accise > 0.1:
            raise ValueError("Taux d'accise attendu en €/kWh (ex. 0.02635), pas en €/MWh.")


# ============================================================================
# 2. DONNÉES
# ============================================================================
@dataclass
class DonneesACC:
    """Données prêtes au calcul : bâtiments triés et matrices T × n alignées."""
    batiments: pd.DataFrame
    index: pd.DatetimeIndex
    numeros: np.ndarray
    S: np.ndarray
    D: np.ndarray
    A: np.ndarray
    pv: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    distances: np.ndarray

    @property
    def n(self) -> int:
        return len(self.numeros)

    @property
    def est_producteur(self) -> np.ndarray:
        return self.pv > 0

    def positions(self, numeros) -> list:
        """Numéros de bâtiments -> positions dans les matrices."""
        correspondance = {int(b): i for i, b in enumerate(self.numeros)}
        try:
            return [correspondance[int(b)] for b in numeros]
        except KeyError as erreur:
            raise ValueError(f"Bâtiment inconnu : B{erreur.args[0]}") from None


def distance_haversine(lat1, lon1, lat2, lon2):
    """Distance à vol d'oiseau en km (même formule que le Bloc 8 v3)."""
    rayon = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * rayon * np.arcsin(np.minimum(1, np.sqrt(a)))


def preparer_donnees(batiments: pd.DataFrame, mat_S: pd.DataFrame,
                     mat_D: pd.DataFrame, mat_A: pd.DataFrame) -> DonneesACC:
    """Contrôle et aligne les données issues de acc_donnees."""
    colonnes = ["Batiment", "Latitude", "Longitude", "Puissance_PV_kWc"]
    manquantes = [c for c in colonnes if c not in batiments.columns]
    if manquantes:
        raise ValueError(f"Colonnes absentes du tableau des bâtiments : {manquantes}")

    bat = batiments.copy()
    bat["Batiment"] = pd.to_numeric(bat["Batiment"], errors="raise").astype(int)
    for c in colonnes[1:]:
        bat[c] = pd.to_numeric(bat[c], errors="raise").astype(float)
    if bat[colonnes].isna().any().any():
        raise ValueError("Coordonnées GPS ou puissance PV manquantes.")
    if bat["Batiment"].duplicated().any():
        raise ValueError("Numéros de bâtiments dupliqués.")
    if (bat["Puissance_PV_kWc"] < 0).any():
        raise ValueError("Puissance PV négative.")
    bat = bat.sort_values("Batiment").reset_index(drop=True)
    numeros = bat["Batiment"].to_numpy()

    for nom, mat in [("S", mat_S), ("D", mat_D), ("A", mat_A)]:
        if not mat.index.equals(mat_S.index):
            raise ValueError(f"Horodatages de la matrice {nom} désynchronisés.")
        absents = set(numeros) - set(pd.to_numeric(mat.columns).astype(int))
        if absents:
            raise ValueError(f"Matrice {nom} : bâtiments absents {sorted(absents)}.")

    index = pd.DatetimeIndex(mat_S.index)
    if index.tz is None:
        raise ValueError("Horodatages sans fuseau : vérifier le chargement (UTC attendu).")

    def aligner(mat):
        m = mat.copy()
        m.columns = pd.to_numeric(m.columns).astype(int)
        arr = m.loc[:, numeros].to_numpy(dtype=float)
        if np.isnan(arr).any():
            raise ValueError("Valeurs manquantes dans les matrices énergétiques.")
        if (arr < -1e-8).any():
            raise ValueError("Énergies nettes négatives incohérentes.")
        return np.clip(arr, 0.0, None)

    lat = bat["Latitude"].to_numpy()
    lon = bat["Longitude"].to_numpy()
    n = len(numeros)
    distances = np.zeros((n, n))
    for i in range(n):
        distances[i] = distance_haversine(lat[i], lon[i], lat, lon)
    np.fill_diagonal(distances, 0.0)

    return DonneesACC(batiments=bat, index=index, numeros=numeros,
                      S=aligner(mat_S), D=aligner(mat_D), A=aligner(mat_A),
                      pv=bat["Puissance_PV_kWc"].to_numpy(), lat=lat, lon=lon,
                      distances=distances)


def depuis_jeu_acc(jeu_acc) -> DonneesACC:
    """Raccourci : jeu chargé par acc_donnees.charger_donnees_acc()."""
    mat_S, mat_D, mat_A = jeu_acc.matrices()
    return preparer_donnees(jeu_acc.batiments, mat_S, mat_D, mat_A)


def bilan_batiments(donnees: DonneesACC) -> pd.DataFrame:
    """Bilan annuel par bâtiment (MWh) et productible (kWh/kWc)."""
    production = (donnees.A + donnees.S).sum(axis=0)
    consommation = (donnees.A + donnees.D).sum(axis=0)
    productible = np.divide(production, donnees.pv, out=np.full(donnees.n, np.nan),
                            where=donnees.pv > 0)
    return pd.DataFrame({
        "Batiment": [f"B{b}" for b in donnees.numeros],
        "Type": np.where(donnees.est_producteur, "Producteur", "Consommateur"),
        "Puissance_PV_kWc": donnees.pv,
        "Consommation_MWh": consommation / 1000,
        "Production_MWh": production / 1000,
        "Autoconso_ind_MWh": donnees.A.sum(axis=0) / 1000,
        "Surplus_MWh": donnees.S.sum(axis=0) / 1000,
        "Soutirage_residuel_MWh": donnees.D.sum(axis=0) / 1000,
        "Productible_kWh_kWc": productible,
    })


# ============================================================================
# 3. PLAGES TARIFAIRES
# ============================================================================
def _masque_plage(h, debut, fin):
    if debut == fin:
        raise ValueError("Une plage horaire doit avoir une durée non nulle.")
    return (h >= debut) & (h < fin) if debut < fin else (h >= debut) | (h < fin)


def classer_periodes(index: pd.DatetimeIndex,
                     plages: Optional[PlagesHoraires] = None) -> np.ndarray:
    """Étiquette chaque pas : 'Creuses', 'Pleines' ou 'Pointe'."""
    plages = plages or PlagesHoraires()
    dates = pd.DatetimeIndex(index)
    if dates.tz is None:
        raise ValueError("Horodatages sans fuseau : UTC attendu.")
    local = dates.tz_convert(plages.fuseau)
    heures = local.hour + local.minute / 60 + local.second / 3600
    creuses = _masque_plage(heures, plages.creuses_debut, plages.creuses_fin)
    pointe = _masque_plage(heures, plages.pointe_debut, plages.pointe_fin)
    if plages.pointe_jours_ouvres:
        pointe = pointe & (local.day_of_week < 5)
    if np.any(creuses & pointe):
        raise ValueError("Les plages creuses et pointe se chevauchent.")
    return np.select([creuses, pointe], ["Creuses", "Pointe"], default="Pleines")


# ============================================================================
# 4. BOUCLES ADMISSIBLES
# ============================================================================
def enumerer_boucles(donnees: DonneesACC, contraintes: Contraintes) -> list:
    """
    Toutes les boucles admissibles, sous forme de tuples de positions.

    Une boucle respectant la distance est une « clique » du graphe où deux
    bâtiments sont reliés s'ils sont à moins de d_max_km : on ne construit que
    ces ensembles, au lieu de tester les 2^n combinaisons. Même résultat que
    l'exploration exhaustive du Bloc 8 v3, dans le même ordre.
    """
    contraintes.verifier()
    n = donnees.n
    compatible = donnees.distances <= contraintes.d_max_km
    pv, prod = donnees.pv, donnees.est_producteur
    nb_max = contraintes.nb_max_batiments or n
    resultats = []

    def composition_valide(boucle):
        nb_prod = int(prod[list(boucle)].sum())
        nb_cons = len(boucle) - nb_prod
        return nb_prod >= 1 and (nb_cons >= 1 or not contraintes.exiger_consommateur_sans_pv)

    def etendre(boucle, candidats, puissance):
        if len(boucle) >= contraintes.nb_min_batiments and composition_valide(boucle):
            resultats.append(tuple(boucle))
        if len(boucle) >= nb_max:
            return
        for j in candidats:
            p = puissance + pv[j]
            if p >= contraintes.p_pv_max_kwc:          # strictement inférieure
                continue
            suivants = [k for k in candidats if k > j and compatible[j, k]]
            etendre(boucle + [j], suivants, p)

    for i in range(n):
        if pv[i] < contraintes.p_pv_max_kwc:
            etendre([i], [k for k in range(i + 1, n) if compatible[i, k]], pv[i])

    return sorted(resultats, key=lambda b: (len(b), b))


# ============================================================================
# 5. ÉNERGIE ET GAINS D'UNE BOUCLE
# ============================================================================
def _prix_par_pas(periodes: np.ndarray, tarifs: Tarifs):
    serie = pd.Series(periodes)
    pf = serie.map(tarifs.prix_fournisseur).to_numpy(dtype=float)
    pacc = serie.map(tarifs.prix_acc).to_numpy(dtype=float)
    if np.isnan(pf).any() or np.isnan(pacc).any():
        raise ValueError("Périodes tarifaires non reconnues.")
    return pf, pacc


def _calculer(donnees, positions, tarifs, pf, pacc, avec_details=False):
    """Cœur du calcul pour une boucle (positions dans les matrices)."""
    pos = list(positions)
    S, D = donnees.S[:, pos], donnees.D[:, pos]
    s_t, d_t = S.sum(axis=1), D.sum(axis=1)
    e_t = np.minimum(s_t, d_t)                               # énergie partagée
    s_tot, d_tot, e_tot = s_t.sum(), d_t.sum(), e_t.sum()
    if s_tot <= 0 or d_tot <= 0:
        return None                                          # TAC ou TAP indéfini

    r_d = np.divide(e_t, d_t, out=np.zeros_like(e_t), where=d_t > 0)
    r_s = np.divide(e_t, s_t, out=np.zeros_like(e_t), where=s_t > 0)

    pv = donnees.pv[pos]
    eligibles = (pv > 0) & (pv < tarifs.seuil_accise_kwc)
    part_eligible = np.divide(S[:, eligibles].sum(axis=1), s_t,
                              out=np.zeros_like(s_t), where=s_t > 0)

    tva = 1.0 + tarifs.tva_non_recuperable
    gain_energie = (D.T @ (r_d * (pf - pacc))) * tva
    accise = (D.T @ (r_d * part_eligible)) * tarifs.taux_accise * tva
    gain_producteur = S.T @ (r_s * (pacc - tarifs.valeur_surplus))
    gain_total = gain_energie + accise + gain_producteur - tarifs.frais_par_participant

    numeros = donnees.numeros[pos]
    noms = [f"B{b}" for b in numeros]
    i_max, i_min = int(np.argmax(gain_total)), int(np.argmin(gain_total))
    dist = donnees.distances[np.ix_(pos, pos)].max()
    kappa = e_tot / min(s_tot, d_tot)

    resume = {
        "Indices": tuple(pos),
        "Participants": ", ".join(noms),
        "Producteurs": ", ".join(n for n, p in zip(noms, pv) if p > 0),
        "Consommateurs": ", ".join(n for n, p in zip(noms, pv) if p <= 0),
        "Nb_batiments": len(pos),
        "Nb_producteurs": int((pv > 0).sum()),
        "Nb_consommateurs": int((pv <= 0).sum()),
        "Distance_max_km": float(dist),
        "Puissance_PV_kWc": float(pv.sum()),
        "Surplus_MWh": s_tot / 1000,
        "Demande_MWh": d_tot / 1000,
        "Autoconso_ind_MWh": donnees.A[:, pos].sum() / 1000,
        "E_ACC_MWh": e_tot / 1000,
        "TAC_pct": 100 * e_tot / s_tot,
        "TAP_pct": 100 * e_tot / d_tot,
        "Adequation_pct": 100 * kappa,
        "Surplus_non_valorise_MWh": (s_tot - e_tot) / 1000,
        "Achat_complementaire_MWh": (d_tot - e_tot) / 1000,
        "Valeur_collective_EUR": float(gain_total.sum()),
        "Gain_moyen_EUR": float(gain_total.mean()),
        "Gain_max_EUR": float(gain_total[i_max]),
        "Gain_min_EUR": float(gain_total[i_min]),
        "Participant_gain_max": noms[i_max],
        "Participant_gain_min": noms[i_min],
        "Gains_consommateurs_EUR": float((gain_energie + accise).sum()),
        "Gains_producteurs_EUR": float(gain_producteur.sum()),
        "Accise_evitee_EUR": float(accise.sum()),
        "Valeur_par_kWh_EUR": float(gain_total.sum() / e_tot) if e_tot > 0 else np.nan,
    }
    if not avec_details:
        return resume

    participants = pd.DataFrame({
        "Batiment": noms,
        "Role": np.where(pv > 0, "Producteur-consommateur", "Consommateur"),
        "Conso_nette_kWh": D.sum(axis=0),
        "E_ACC_recue_kWh": D.T @ r_d,
        "Gain_energie_EUR": gain_energie,
        "Accise_evitee_EUR": accise,
        "E_ACC_vendue_kWh": S.T @ r_s,
        "Gain_producteur_EUR": gain_producteur,
        "Frais_EUR": np.full(len(pos), tarifs.frais_par_participant),
        "Gain_total_EUR": gain_total,
    })
    series = pd.DataFrame({
        "Surplus_kWh": s_t,
        "Demande_kWh": d_t,
        "E_ACC_kWh": e_t,
        "Surplus_non_valorise_kWh": s_t - e_t,
        "Achat_complementaire_kWh": d_t - e_t,
        "Autoconso_ind_kWh": donnees.A[:, pos].sum(axis=1),
    }, index=donnees.index)
    return resume, participants, series


def evaluer_boucles(donnees: DonneesACC, boucles: list, tarifs: Tarifs,
                    periodes: np.ndarray) -> pd.DataFrame:
    """
    Tableau de TOUTES les boucles admissibles : énergie et gains.

    Les boucles sans surplus ou sans demande sont écartées (TAC/TAP indéfinis).
    Tri et noms ACC1, ACC2… identiques au Bloc 8 v3 (taille décroissante,
    puis distance croissante).
    """
    tarifs.verifier()
    pf, pacc = _prix_par_pas(periodes, tarifs)
    lignes = [r for b in boucles
              if (r := _calculer(donnees, b, tarifs, pf, pacc)) is not None]
    if not lignes:
        raise ValueError("Aucune boucle admissible avec ces contraintes.")
    tableau = (pd.DataFrame(lignes)
               .sort_values(["Nb_batiments", "Distance_max_km"], ascending=[False, True])
               .reset_index(drop=True))
    tableau.insert(0, "Solution", [f"ACC{i + 1}" for i in range(len(tableau))])
    return tableau


def analyser_boucle(donnees: DonneesACC, numeros, tarifs: Tarifs,
                    periodes: np.ndarray) -> dict:
    """Analyse détaillée d'une boucle donnée par ses numéros de bâtiments."""
    tarifs.verifier()
    pf, pacc = _prix_par_pas(periodes, tarifs)
    positions = donnees.positions(numeros)
    if len(set(positions)) != len(positions):
        raise ValueError("Bâtiment présent deux fois dans la boucle.")
    sortie = _calculer(donnees, positions, tarifs, pf, pacc, avec_details=True)
    if sortie is None:
        raise ValueError("Boucle sans surplus ou sans demande : rien à partager.")
    resume, participants, series = sortie
    series["Periode"] = periodes

    # Contrôles : conservation et absence de double comptage du prix ACC.
    e = series["E_ACC_kWh"].sum()
    if not (np.isclose(participants["E_ACC_recue_kWh"].sum(), e)
            and np.isclose(participants["E_ACC_vendue_kWh"].sum(), e)):
        raise AssertionError("Énergie reçue ou vendue différente de l'énergie partagée.")
    if tarifs.tva_non_recuperable == 0:
        valeur = (np.sum(series["E_ACC_kWh"] * (pf - tarifs.valeur_surplus))
                  + resume["Accise_evitee_EUR"]
                  - tarifs.frais_par_participant * len(positions))
        if not np.isclose(resume["Valeur_collective_EUR"], valeur, rtol=1e-9, atol=1e-6):
            raise AssertionError("Double comptage détecté dans les gains.")

    par_periode = (series.groupby("Periode")[["Surplus_kWh", "Demande_kWh", "E_ACC_kWh"]]
                   .sum().reindex(list(PERIODES), fill_value=0.0) / 1000)
    par_periode.columns = ["Surplus_MWh", "Demande_MWh", "E_ACC_MWh"]
    return {"resume": resume, "participants": participants,
            "series": series, "par_periode": par_periode}


# ============================================================================
# 6. OPTIMISATION MULTI-OBJECTIF
# ============================================================================
# Indicateurs utilisables comme objectifs ou comme seuils :
# colonne -> (libellé, unité, sens par défaut)
INDICATEURS = {
    "E_ACC_MWh": ("Énergie autoconsommée collectivement", "MWh", "max"),
    "TAC_pct": ("Taux d'autoconsommation collective (TAC)", "%", "max"),
    "TAP_pct": ("Taux d'autoproduction collective (TAP)", "%", "max"),
    "Adequation_pct": ("Adéquation temporelle production/consommation", "%", "max"),
    "Surplus_non_valorise_MWh": ("Surplus PV non valorisé localement", "MWh", "min"),
    "Valeur_collective_EUR": ("Gain financier total annuel", "€/an", "max"),
    "Gain_moyen_EUR": ("Gain moyen par participant", "€/an", "max"),
    "Gain_min_EUR": ("Gain du participant le moins favorisé", "€/an", "max"),
    "Gain_max_EUR": ("Gain du participant le plus favorisé", "€/an", "max"),
    "Nb_batiments": ("Nombre de participants", "", "min"),
    "Distance_max_km": ("Distance maximale entre participants", "km", "min"),
    "Puissance_PV_kWc": ("Puissance PV totale", "kWc", "max"),
}


@dataclass
class Objectif:
    colonne: str
    sens: str = "max"                       # "max" ou "min"


@dataclass
class Seuil:
    colonne: str
    operateur: str                          # ">=" ou "<="
    valeur: float


@dataclass
class ResultatOptimisation:
    candidats: pd.DataFrame                 # boucles respectant les seuils, avec colonne Pareto
    solution: pd.Series                     # boucle retenue
    etapes: pd.DataFrame                    # trace de la sélection lexicographique
    objectifs: list
    seuils: list


def _verifier_objectifs(tableau, objectifs, seuils):
    if not objectifs:
        raise ValueError("Choisir au moins un objectif.")
    colonnes = [o.colonne for o in objectifs]
    if len(set(colonnes)) != len(colonnes):
        raise ValueError("Un même indicateur est choisi deux fois comme objectif.")
    for o in objectifs:
        if o.colonne not in tableau.columns:
            raise ValueError(f"Indicateur inconnu : {o.colonne}")
        if o.sens not in ("max", "min"):
            raise ValueError(f"Sens invalide pour {o.colonne} : 'max' ou 'min'.")
    for s in seuils:
        if s.colonne not in tableau.columns:
            raise ValueError(f"Indicateur inconnu dans les seuils : {s.colonne}")
        if s.operateur not in (">=", "<="):
            raise ValueError(f"Opérateur invalide : {s.operateur} ('>=' ou '<=').")


def appliquer_seuils(tableau: pd.DataFrame, seuils: list) -> pd.Series:
    masque = pd.Series(True, index=tableau.index)
    for s in seuils:
        valeurs = tableau[s.colonne].astype(float)
        masque &= valeurs >= s.valeur if s.operateur == ">=" else valeurs <= s.valeur
    return masque


def front_pareto(tableau: pd.DataFrame, objectifs: list) -> pd.Series:
    """
    Boucles non dominées pour les objectifs choisis.

    Tri lexicographique décroissant : une boucle ne peut être dominée que par
    une boucle placée avant elle, donc on la compare au seul front déjà
    construit (rapide même avec des dizaines de milliers de boucles).
    """
    if tableau.empty:
        return pd.Series(dtype=bool)
    signes = np.array([1.0 if o.sens == "max" else -1.0 for o in objectifs])
    F = tableau[[o.colonne for o in objectifs]].to_numpy(dtype=float) * signes
    # Arrondi : deux valeurs égales à 1e-9 près sont traitées comme égales
    # (sinon un TAC de 100 % peut valoir 99,999999999999 selon l'ordre de calcul).
    F = np.round(F, 9)
    ordre = np.lexsort(F.T[::-1])[::-1]
    front = []
    masque = np.zeros(len(F), dtype=bool)
    for i in ordre:
        if front:
            P = F[front]
            if np.any(np.all(P >= F[i], axis=1) & np.any(P > F[i], axis=1)):
                continue
        front.append(i)
        masque[i] = True
    return pd.Series(masque, index=tableau.index)


def selection_lexicographique(candidats: pd.DataFrame, objectifs: list,
                              tolerance_pct: float = 0.0):
    """
    Sélection par priorités : on garde les boucles proches du meilleur sur
    l'objectif 1 (à tolerance_pct près), puis parmi elles sur l'objectif 2, etc.
    Le dernier objectif est optimisé exactement. Aucune pondération.
    """
    restant = candidats
    etapes = []
    for k, o in enumerate(objectifs):
        valeurs = restant[o.colonne].astype(float)
        meilleur = valeurs.max() if o.sens == "max" else valeurs.min()
        tol = 0.0 if k == len(objectifs) - 1 else abs(meilleur) * tolerance_pct / 100
        garde = valeurs >= meilleur - tol if o.sens == "max" else valeurs <= meilleur + tol
        restant = restant[garde]
        etapes.append({"Priorite": k + 1, "Objectif": o.colonne, "Sens": o.sens,
                       "Meilleure_valeur": meilleur, "Tolerance": tol,
                       "Boucles_restantes": len(restant)})
    cles = [o.colonne for o in objectifs]
    ordre = [o.sens == "min" for o in objectifs]
    solution = restant.sort_values(cles, ascending=ordre, kind="mergesort").iloc[0]
    return solution, pd.DataFrame(etapes)


def optimiser(tableau: pd.DataFrame, objectifs: list, seuils: Optional[list] = None,
              tolerance_pct: float = 0.0) -> ResultatOptimisation:
    """
    Optimisation multi-objectif sans somme pondérée :
      1. seuils (contraintes sur les indicateurs) ;
      2. front de Pareto des objectifs choisis ;
      3. sélection lexicographique sur le front, dans l'ordre des priorités.
    """
    seuils = seuils or []
    _verifier_objectifs(tableau, objectifs, seuils)
    if not 0 <= tolerance_pct <= 100:
        raise ValueError("La tolérance doit être comprise entre 0 et 100 %.")
    candidats = tableau[appliquer_seuils(tableau, seuils)].copy()
    if candidats.empty:
        raise ValueError("Aucune boucle ne respecte les seuils : les assouplir.")
    candidats["Pareto"] = front_pareto(candidats, objectifs)
    solution, etapes = selection_lexicographique(
        candidats[candidats["Pareto"]], objectifs, tolerance_pct)
    return ResultatOptimisation(candidats=candidats, solution=solution, etapes=etapes,
                                objectifs=list(objectifs), seuils=list(seuils))


def numeros_de(solution: pd.Series) -> list:
    """Numéros de bâtiments d'une ligne du tableau (colonne Participants)."""
    return [int(x.strip().lstrip("B")) for x in str(solution["Participants"]).split(",")]

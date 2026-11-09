"""Chargement, validation, synchronisation et prétraitement des données ACC.

Un fichier .xlsx par bâtiment, avec au choix :
- une feuille Mesures contenant des colonnes Horodatage, Consommation (kWh),
  [Production (kWh)], Latitude, Longitude, Puissance_crete_kWc, Id_batiment ;
- une feuille Mesures contenant uniquement les séries, plus une feuille Metadonnees
  au format deux colonnes (champ, valeur) pour Id_batiment, Latitude, Longitude,
  Puissance_crete_kWc.

Horodatages ISO/Excel avec fuseau, ou heures locales naïves (Europe/Paris,
configurable) ; toutes les données de sortie sont en UTC. Zéro synthétique
UNIQUEMENT si le bâtiment est véritablement non producteur.

Les deux vecteurs nets et l'autoconsommation individuelle sont calculés UNE fois
au chargement. Les sommes n'ignorent jamais silencieusement les trous de données.
"""
from __future__ import annotations

import re
import unicodedata
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class ErreurDonneesACC(ValueError):
    """Erreur explicite, accompagnée du fichier ou bâtiment concerné."""


@dataclass(frozen=True)
class ConfigurationACC:
    dossier: Path | str = Path("donnees/batiments")
    pas_minutes: int = 15
    fuseau_horaires_naifs: str = "Europe/Paris"
    strict: bool = True
    # La comparaison porte sur le chevauchement temporel commun.
    # Les mesures extérieures sont conservées dans donnees_brutes.
    refuser_periodes_non_identiques: bool = False
    # Contrôle du maximum théorique d'énergie produite (25 % de tolérance
    # pour les écarts de mesure/transitoires). "erreur" est recommandé.
    controle_pv: str = "erreur"  # "erreur" | "avertir"

    def __post_init__(self):
        if self.controle_pv not in {"erreur", "avertir"}:
            raise ValueError("controle_pv doit être 'erreur' ou 'avertir'.")
        if self.pas_minutes <= 0 or 1440 % self.pas_minutes != 0:
            raise ValueError("Le pas_minutes doit diviser exactement une journée (ex. 15 ou 60).")


@dataclass
class BatimentCharge:
    identifiant: str
    latitude: float
    longitude: float
    puissance_crete_kwc: float
    type_batiment: str
    donnees: pd.DataFrame
    nom_fichier: str
    avec_production: bool
    remarques: list[str] = field(default_factory=list)


@dataclass
class DonneesACC:
    """Sorties : batiments, profils (synchronisés), donnees_brutes, rapport."""
    batiments: pd.DataFrame
    profils: pd.DataFrame
    donnees_brutes: dict[str, pd.DataFrame]
    rapport: pd.DataFrame
    pas_minutes: int
    debut: pd.Timestamp
    fin: pd.Timestamp
    pret_optimisation: bool

    def exiger_complet(self) -> None:
        """Refuse les simulations ACC en présence d'une seule mesure manquante."""
        if not self.pret_optimisation:
            anormaux = self.rapport.loc[self.rapport["valeurs_manquantes"] > 0,
                                         ["identifiant_batiment", "valeurs_manquantes"]]
            raise ErreurDonneesACC(
                "Impossible d'optimiser avec des données incomplètes. "
                "Corriger les mesures manquantes (aucune interpolation implicite).\n"
                + anormaux.to_string(index=False)
            )

    def matrices(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """(surplus PV, demande résiduelle, autoconsommation individuelle) en kWh."""
        self.exiger_complet()
        nums = self.batiments["Batiment"].tolist()
        def pivoter(nom: str) -> pd.DataFrame:
            return (self.profils.pivot(index="Horodatage", columns="Batiment", values=nom)
                    .reindex(columns=nums).sort_index())
        return (pivoter("production_nette"), pivoter("consommation_nette"),
                pivoter("autoconsommation_individuelle"))


def _normaliser(texte: Any) -> str:
    texte = unicodedata.normalize("NFKD", str(texte))
    texte = "".join(c for c in texte if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", texte.casefold())


_ALIAS = {
    "identifiant": ["Id_batiment", "Identifiant_batiment", "Identifiant du bâtiment",
                    "Identifiant", "ID", "Batiment", "Bâtiment", "Building_ID"],
    "horodatage": ["Horodatage", "Date_heure", "Date et heure", "Datetime",
                    "Timestamp", "Date", "Date_heure_mesure"],
    "consommation": ["Consommation", "Consommation_kWh", "Consommation (kWh)",
                     "Consommation [kWh]", "Energie_consommée", "Energie_consommée_kWh",
                     "Conso", "Conso_kWh", "Consumption", "Consumption_kWh",
                     "Consommation_MWh", "Consommation_Wh"],
    "production": ["Production", "Production_PV", "Production PV", "Production_kWh",
                   "Production photovoltaïque", "Production photovoltaïque kWh",
                   "Production solaire", "Production_MWh", "Production_Wh",
                   "Energie_produite", "Energie_produite_kWh"],
    "latitude": ["Latitude", "GPS_Lat", "GPS_Latitude", "Lat", "Latitude_GPS"],
    "longitude": ["Longitude", "GPS_Lon", "GPS_Longitude", "Lon", "Long", "Longitude_GPS"],
    "puissance_crete_kwc": ["Puissance_crete_kWc", "Puissance crête (kWc)",
                            "Puissance crête", "Puissance PV Cret Kwc", "Puissance PV kWc",
                            "Puissance_PV_kWc", "Puissance PV crête kWc", "Pmax_kWc",
                            "Pmax", "Puissance_installee_kWc", "Peak_power_kWp",
                            "Puissance crete kWp", "Puissance_crete_kWp"],
}
_ALIASES = {field: {_normaliser(name) for name in names} for field, names in _ALIAS.items()}


def _identifier_colonne(nom: Any) -> str | None:
    n = _normaliser(nom)
    for champ, aliases in _ALIASES.items():
        if n in aliases:
            return champ
    return None


def _trouver_colonnes(colonnes: Any, fichier: Path) -> dict[str, Any]:
    rep: dict[str, Any] = {}
    for colonne in colonnes:
        champ = _identifier_colonne(colonne)
        if champ:
            if champ in rep:
                raise ErreurDonneesACC(f"{fichier.name}: colonnes ambiguës pour '{champ}': "
                                      f"{rep[champ]!r} et {colonne!r}.")
            rep[champ] = colonne
    return rep


def _lire_metadonnees(classeur: pd.ExcelFile, fichier: Path,
                       feuille_mesures: str, ligne_entete: int) -> dict[str, Any]:
    """Cherche les paires clé-valeur dans les premières lignes d'une zone de métadonnées."""
    meta = {}
    for feuille in classeur.sheet_names:
        est_mesures = feuille == feuille_mesures
        est_meta = "meta" in _normaliser(feuille) or "info" in _normaliser(feuille)
        # Accepte une zone clairement séparée en tête de la feuille Mesures.
        # Ne cherche JAMAIS des valeurs dans les lignes de mesures elles-mêmes.
        if not est_meta and not (est_mesures and ligne_entete > 0):
            continue
        hauteur = min(25, ligne_entete) if est_mesures else 25
        if hauteur == 0:
            continue
        head = pd.read_excel(classeur, sheet_name=feuille, header=None,
                             nrows=hauteur, dtype=object)
        if head.empty:
            continue
        candidates = 0
        for r in range(min(25, len(head))):
            for c in range(min(head.shape[1], 12)):
                champ = _identifier_colonne(head.iat[r, c]) if pd.notna(head.iat[r, c]) else None
                if champ not in {"identifiant", "latitude", "longitude", "puissance_crete_kwc"}:
                    continue
                # Priorité au format métadonnées 'champ | valeur' dans la même ligne.
                v = head.iat[r, c + 1] if c + 1 < head.shape[1] else None
                # Sinon, 'en-tête / valeurs' : la cellule directement en dessous.
                if pd.isna(v) or _identifier_colonne(v) is not None:
                    v = head.iat[r + 1, c] if r + 1 < len(head) else None
                if pd.notna(v) and _identifier_colonne(v) is None:
                    candidates += 1
                    if champ in meta and str(meta[champ]).strip() != str(v).strip():
                        # Une feuille de mesures avec des métadonnées répétées peut
                        # apparaître; seules les feuilles metadata sont ici autorisées.
                        if "meta" in _normaliser(feuille) or "info" in _normaliser(feuille):
                            raise ErreurDonneesACC(f"{fichier.name}: métadonnée '{champ}' contradictoire.")
                    else:
                        meta.setdefault(champ, v)
        # Les feuilles Mesures sont traitées en tant que tableaux, pas méta.
    return meta


def _nombre(valeur: Any, champ: str, fichier: Path) -> float:
    if pd.isna(valeur) or str(valeur).strip() == "":
        raise ErreurDonneesACC(f"{fichier.name}: donnée obligatoire '{champ}' vide.")
    try:
        texte = str(valeur).strip().replace("\u202f", "").replace("\xa0", "").replace(" ", "")
        n = float(texte.replace(",", "."))
    except (TypeError, ValueError) as e:
        raise ErreurDonneesACC(f"{fichier.name}: valeur invalide de '{champ}': {valeur!r}") from e
    if not np.isfinite(n):
        raise ErreurDonneesACC(f"{fichier.name}: '{champ}' doit être fini: {valeur!r}.")
    return n


def _identifiant(valeur: Any, fichier: Path) -> str:
    if pd.isna(valeur) or str(valeur).strip() == "":
        raise ErreurDonneesACC(f"{fichier.name}: identifiant du bâtiment manquant.")
    s = str(valeur).strip()
    if re.fullmatch(r"\d+\.0+", s):
        s = s.split(".", 1)[0]
    return s


def _valeur_constante(df: pd.DataFrame, col: Any, champ: str, fichier: Path,
                      meta: Any | None = None) -> Any:
    vals = df[col].dropna().tolist() if col is not None else []
    if col is not None and len(vals) != len(df):
        raise ErreurDonneesACC(f"{fichier.name}: '{champ}' est partiellement absent (valeurs par ligne).")
    if champ == "identifiant":
        uniques = {_identifiant(x, fichier) for x in vals}
        if meta is not None:
            uniques.add(_identifiant(meta, fichier))
    else:
        uniques_num = [_nombre(x, champ, fichier) for x in vals]
        if meta is not None:
            uniques_num.append(_nombre(meta, champ, fichier))
        uniques = set(round(x, 9) for x in uniques_num)
    if not uniques:
        raise ErreurDonneesACC(f"{fichier.name}: donnée obligatoire manquante: '{champ}'. "
                              "L'ajouter en colonne répétée ou dans 'Metadonnees'.")
    if len(uniques) != 1:
        raise ErreurDonneesACC(f"{fichier.name}: '{champ}' doit être constant sur le bâtiment. "
                              f"Valeurs rencontrées: {list(uniques)[:5]}.")
    if champ == "identifiant":
        return next(iter(uniques))
    return float(next(iter(uniques)))


def _energie_kwh(serie: pd.Series, colonne: str, champ: str, fichier: Path) -> pd.Series:
    # Si aucune unité n'est inscrite, la convention du contrat est kWh/par pas.
    fragments = re.findall(r"(?<![a-z])(?:mwh|kwh|wh|kw|mw|w)(?![a-z])",
                           unicodedata.normalize('NFKD', str(colonne)).lower())
    units = set(fragments)
    if len(units) > 1 or (units and not units.issubset({"kwh", "mwh", "wh"})):
        raise ErreurDonneesACC(f"{fichier.name}: colonne '{colonne}' : unités non énergétiques "
                              "ou contradictoires. Fournir des kWh par intervalle (pas une puissance en kW).")
    facteur = {"kwh": 1., "mwh": 1000., "wh": 0.001}.get(next(iter(units), "kwh"), 1.)
    brut = (serie.astype("string").str.strip().str.replace("\u202f", "", regex=False)
            .str.replace("\xa0", "", regex=False).str.replace(" ", "", regex=False)
            .str.replace(",", ".", regex=False))
    vals = pd.to_numeric(brut, errors="coerce")
    faux = vals.isna() | ~np.isfinite(vals.to_numpy(dtype=float, na_value=np.nan)) | (vals < 0)
    if bool(faux.any()):
        exemples = (serie.loc[faux].head(3)).tolist()
        raise ErreurDonneesACC(f"{fichier.name}: {champ} contient {int(faux.sum())} "
                              f"valeur(s) manquante(s), négative(s) ou invalide(s): {exemples}.")
    return vals.astype(float) * facteur


def _dates_utc(serie: pd.Series, fichier: Path, fuseau: str) -> pd.Series:
    # ISO avec UTC/offset, dates Excel et timestamps locaux naïfs.
    # L'absence d'information de fuseau signifie explicitement Europe/Paris.
    def parse(v: Any):
        if pd.isna(v) or str(v).strip() == "":
            return pd.NaT
        try:
            # Excel peut stocker les dates comme nombres de jours depuis 1899-12-30.
            if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
                if not 1 <= float(v) <= 100000:
                    raise ValueError("Date numérique hors plage Excel plausible")
                t = pd.Timestamp("1899-12-30") + pd.to_timedelta(float(v), unit="D")
                # Erreurs de flottants binaires Excel limitées à quelques microsecondes.
                t = t.round("s")
            else:
                t = pd.Timestamp(v)
            if t.tzinfo is None:
                t = t.tz_localize(fuseau, ambiguous="raise", nonexistent="raise")
            return t.tz_convert("UTC")
        except Exception as exc:
            raise ErreurDonneesACC(f"{fichier.name}: horodatage invalide/ambigu en fuseau "
                                  f"{fuseau}: {v!r} ({exc}).") from exc
    vals = pd.to_datetime([parse(v) for v in serie], utc=True)
    if vals.isna().any():
        raise ErreurDonneesACC(f"{fichier.name}: {int(vals.isna().sum())} horodatage(s) manquant(s).")
    return pd.Series(vals, index=serie.index)


def decouvrir_fichiers(dossier: Path | str) -> list[Path]:
    rep = Path(dossier).expanduser().resolve()
    if not rep.is_dir():
        raise ErreurDonneesACC(f"Dossier de bâtiments introuvable : {rep}")
    fichiers = sorted((f for f in rep.iterdir() if f.is_file() and
                      f.suffix.lower() in {".xlsx", ".xlsm"} and
                      not f.name.startswith("~$")), key=lambda p: p.name.casefold())
    if not fichiers:
        raise ErreurDonneesACC(f"Aucun fichier .xlsx/.xlsm trouvé dans : {rep}")
    return fichiers


def charger_batiment(fichier: Path | str, config: ConfigurationACC) -> BatimentCharge:
    fichier = Path(fichier)
    try:
        classeur = pd.ExcelFile(fichier)
    except Exception as e:
        raise ErreurDonneesACC(f"{fichier.name}: fichier Excel illisible : {e}") from e
    choix: list[tuple[str, int, dict[str, Any]]] = []
    for feuille in classeur.sheet_names:
        premiere = pd.read_excel(classeur, sheet_name=feuille, header=None, nrows=18, dtype=object)
        for ligne in range(len(premiere)):
            cols = _trouver_colonnes(premiere.iloc[ligne].tolist(), fichier)
            if "horodatage" in cols and "consommation" in cols:
                choix.append((feuille, ligne, cols))
                break
    if len(choix) != 1:
        raise ErreurDonneesACC(
            f"{fichier.name}: {len(choix)} feuille(s) de mesures identifiée(s) (une seule attendue). "
            "Vérifier les en-têtes 'Horodatage' et 'Consommation'."
        )
    feuille, ligne_entete, _ = choix[0]
    df = pd.read_excel(classeur, sheet_name=feuille, header=ligne_entete)
    df = df.dropna(how="all").reset_index(drop=True)
    if df.empty:
        raise ErreurDonneesACC(f"{fichier.name}: aucune mesure dans '{feuille}'.")
    colonnes = _trouver_colonnes(df.columns, fichier)
    meta = _lire_metadonnees(classeur, fichier, feuille, ligne_entete)
    props = {}
    for champ in ("identifiant", "latitude", "longitude", "puissance_crete_kwc"):
        props[champ] = _valeur_constante(df, colonnes.get(champ), champ, fichier,
                                        meta.get(champ))
    lat, lon, p = (props["latitude"], props["longitude"], props["puissance_crete_kwc"])
    if not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise ErreurDonneesACC(f"{fichier.name}: coordonnées GPS invalides (lat={lat}, lon={lon}).")
    if lat == 0.0 and lon == 0.0:
        raise ErreurDonneesACC(f"{fichier.name}: coordonnées GPS (0, 0) suspectes/non renseignées.")
    if p < 0:
        raise ErreurDonneesACC(f"{fichier.name}: puissance crête négative ({p} kWc).")
    # Si l'en-tête explicite une puissance en kW, ne pas l'assimiler à des kWc.
    if "puissance_crete_kwc" in colonnes:
        nom_p = str(colonnes["puissance_crete_kwc"])
        if re.search(r"(?<![a-z])(?:kw|mw)(?![a-z])", nom_p.lower()) and not re.search(r"(?i)kwc|kwp", nom_p):
            raise ErreurDonneesACC(f"{fichier.name}: la puissance PV doit être une puissance CRÊTE (kWc), pas kW.")
    a_col_pv = "production" in colonnes
    remarques = []
    if len(df) < 2:
        raise ErreurDonneesACC(f"{fichier.name}: au moins deux horodatages sont nécessaires "
                              "pour vérifier le pas de mesure.")
    if p > 0 and not a_col_pv:
        raise ErreurDonneesACC(f"{fichier.name}: puissance crête {p} kWc, mais colonne Production absente.")
    consommation = _energie_kwh(df[colonnes["consommation"]], str(colonnes["consommation"]),
                                "Consommation", fichier)
    if a_col_pv:
        production = _energie_kwh(df[colonnes["production"]], str(colonnes["production"]),
                                 "Production", fichier)
        if p == 0 and bool((production > 0).any()):
            raise ErreurDonneesACC(f"{fichier.name}: production positive avec puissance crête 0 kWc.")
        if p == 0:
            remarques.append("Colonne Production tout à zéro, bâtiment non producteur.")
        if p > 0 and not bool((production > 0).any()):
            remarques.append("PV déclaré mais production entièrement nulle : arrêt ou panne à vérifier.")
        if p > 0:
            limite_kwh = p * config.pas_minutes / 60.0 * 1.25
            dep = production > limite_kwh
            if bool(dep.any()):
                erreur_pv = (f"{fichier.name}: {int(dep.sum())} mesure(s) de production "
                             f"excèdent 125 % de {p:g} kWc × {config.pas_minutes}/60 h "
                             f"(max. observé {float(production.max()):.2f} kWh/pas, "
                             f"borne {limite_kwh:.2f} kWh/pas). Vérifier unités, "
                             "puissance crête et capteurs.")
                if config.controle_pv == "erreur":
                    raise ErreurDonneesACC(erreur_pv)
                warnings.warn(erreur_pv, UserWarning, stacklevel=2)
                remarques.append(erreur_pv)
    else:
        # Le seul zéro autorisé est une production physiquement absente.
        production = pd.Series(np.zeros(len(df)), index=df.index)
    dates = _dates_utc(df[colonnes["horodatage"]], fichier, config.fuseau_horaires_naifs)
    if dates.duplicated().any():
        exemples = dates[dates.duplicated(keep=False)].head(3).astype(str).tolist()
        raise ErreurDonneesACC(f"{fichier.name}: horodatages dupliqués, ex. {exemples}.")
    pas = pd.Timedelta(minutes=config.pas_minutes)
    if ((dates - pd.Timestamp("1970-01-01", tz="UTC")) % pas != pd.Timedelta(0)).any():
        exemple = dates[((dates - pd.Timestamp("1970-01-01", tz="UTC")) % pas != pd.Timedelta(0))].iloc[0]
        raise ErreurDonneesACC(f"{fichier.name}: horodatage {exemple} non aligné sur le pas de "
                              f"{config.pas_minutes} min.")
    base = pd.DataFrame({"Horodatage": dates, "Consommation": consommation,
                         "Production": production}).sort_values("Horodatage").reset_index(drop=True)
    if p > 0:
        bat_type = "Producteur-consommateur"
    else:
        bat_type = "Consommateur"
    return BatimentCharge(props["identifiant"], lat, lon, p, bat_type, base,
                          fichier.name, a_col_pv, remarques)


def charger_donnees_acc(config: ConfigurationACC | None = None) -> DonneesACC:
    config = config or ConfigurationACC()
    fichiers = decouvrir_fichiers(config.dossier)
    objets = []
    for f in fichiers:
        try:
            objets.append(charger_batiment(f, config))
        except ErreurDonneesACC:
            raise
        except Exception as e:
            raise ErreurDonneesACC(f"{f.name}: impossible de traiter le classeur : {e}") from e
    ids = [o.identifiant for o in objets]
    dups = sorted({i for i in ids if ids.count(i) > 1})
    if dups:
        raise ErreurDonneesACC(f"Identifiants de bâtiment présents dans plusieurs fichiers : {dups}.")
    if len(objets) < 2:
        warnings.warn("Moins de deux bâtiments : aucune boucle ACC possible.")
    # Numérotation technique stable pour compatibilité des figures/optimiseurs.
    # ID d'origine inchangé dans identifiant_batiment.
    def cle(tri):
        s = tri.identifiant
        return (0, int(s)) if s.isdigit() else (1, s.casefold())
    objets.sort(key=cle)
    debut = max(o.donnees["Horodatage"].min() for o in objets)
    fin = min(o.donnees["Horodatage"].max() for o in objets)
    if debut > fin:
        raise ErreurDonneesACC("Aucun chevauchement temporel entre les fichiers des bâtiments.")
    grille = pd.date_range(debut, fin, freq=f"{config.pas_minutes}min", tz="UTC")
    if len(grille) == 0:
        raise ErreurDonneesACC("Période commune vide après synchronisation.")
    bruts: dict[str, pd.DataFrame] = {}
    profils_liste = []
    lignes_bat = []
    rapports = []
    for num, bat in enumerate(objets, 1):
        src = bat.donnees.copy()
        bruts[bat.identifiant] = src.copy()
        commun = src.set_index("Horodatage").reindex(grille)
        manque = commun["Consommation"].isna() | commun["Production"].isna()
        n_manque = int(manque.sum())
        n_hors = int(((src["Horodatage"] < debut) | (src["Horodatage"] > fin)).sum())
        if n_hors and not config.refuser_periodes_non_identiques:
            warnings.warn(f"{bat.nom_fichier}: {n_hors} mesures hors période commune "
                          "(exclues des comparaisons, conservées dans donnees_brutes).",
                          UserWarning, stacklevel=2)
        if config.refuser_periodes_non_identiques and n_hors:
            raise ErreurDonneesACC(f"{bat.nom_fichier}: {n_hors} mesure(s) hors de la "
                                  "période commune; périodes différentes.")
        if config.strict and n_manque:
            dates_abs = grille[manque.to_numpy()]
            ex = [str(x) for x in dates_abs[:3]]
            raise ErreurDonneesACC(
                f"{bat.nom_fichier} ({bat.identifiant}): {n_manque} pas de "
                f"{config.pas_minutes} min manquant(s) sur {len(grille)} attendus "
                f"entre {debut} et {fin}; exemples : {ex}. "
                "Pas de remplissage artificiel. Pour une archive horaire, "
                "choisir explicitement pas_minutes=60 sans la qualifier de données 15 min."
            )
        # Calculs uniques ; np.maximum laisse les NaN intacts.
        c = commun["Consommation"].astype(float)
        p = commun["Production"].astype(float)
        commun["autoconsommation_individuelle"] = np.minimum(c, p)
        commun["consommation_nette"] = np.maximum(0., c - p)
        commun["production_nette"] = np.maximum(0., p - c)
        commun["Horodatage"] = grille
        commun["Batiment"] = num
        commun["identifiant_batiment"] = bat.identifiant
        # Aliases legacy, sans recalcul physique en aval.
        commun["Surplus_kWh"] = commun["production_nette"]
        commun["Demande_kWh"] = commun["consommation_nette"]
        commun["Autoconso_ind_kWh"] = commun["autoconsommation_individuelle"]
        commun["Consommation_nette_kWh"] = commun["consommation_nette"]
        commun["Autoconsommation_locale_kWh"] = commun["autoconsommation_individuelle"]
        commun["Autoconso_kWh"] = commun["autoconsommation_individuelle"]
        # Import sans ACC (ne PAS confondre avec le soutirage final après partage).
        commun["Import_kWh"] = commun["consommation_nette"]
        profils_liste.append(commun.reset_index(drop=True))
        # Les sommes sont NaN si l'intervalle commun manque : ne jamais masquer le problème.
        somme = lambda v: float(v.sum()) if not n_manque else float("nan")
        lignes_bat.append({"Batiment": num, "identifiant_batiment": bat.identifiant,
                           "Nom_Batiment": bat.identifiant, "Latitude": bat.latitude,
                           "Longitude": bat.longitude, "Puissance_PV_kWc": bat.puissance_crete_kwc,
                           "puissance_crete_kwc": bat.puissance_crete_kwc,
                           "latitude": bat.latitude, "longitude": bat.longitude,
                           "Type": bat.type_batiment,
                           "Consommation_kWh": somme(commun["Consommation"]),
                           "Production_kWh": somme(commun["Production"]),
                           "Consommation_MWh": somme(commun["Consommation"]) / 1000.,
                           "Production_MWh": somme(commun["Production"]) / 1000.,
                           "Nombre_mesures": len(grille) - n_manque,
                           "Fichier": bat.nom_fichier})
        rapports.append({"fichier": bat.nom_fichier,
                         "identifiant_batiment": bat.identifiant,
                         "type_batiment": bat.type_batiment,
                         "mesures_originales": len(src),
                         "mesures_communes_attendues": len(grille),
                         "valeurs_manquantes": n_manque,
                         "hors_periode_commune": n_hors,
                         "remarques": "; ".join(bat.remarques)})
    profils = pd.concat(profils_liste, ignore_index=True).sort_values(
        ["Horodatage", "Batiment"]).reset_index(drop=True)
    batiments = pd.DataFrame(lignes_bat).sort_values("Batiment").reset_index(drop=True)
    rapport = pd.DataFrame(rapports)
    complet = bool((rapport["valeurs_manquantes"] == 0).all())
    return DonneesACC(batiments, profils, bruts, rapport, config.pas_minutes, debut, fin,
                      pret_optimisation=complet)

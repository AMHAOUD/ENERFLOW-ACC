# ACC — version 3.2

## Installation

- `acc_moteur.py` : tous les calculs (contraintes, énergie, gains, optimisation)
- `acc_graphiques.py` : tous les graphiques
- `ACC_MODEL_3_2.ipynb` : le notebook 
- `app_acc.py` : l'application

Bibliothèque  pour l'application uniquement :

```
pip install streamlit
```

## Lancer l'application

Dans un terminal ouvert dans ce dossier :

```
streamlit run app_acc.py
```

L'application s'ouvre dans le navigateur. Dans la barre latérale, indiquer le dossier des fichiers Excel (par défaut `donnees/batiments`), puis cliquer sur « Charger les données ».

## Les quatre onglets

1. **Données** : bilan par bâtiment, productible, profils mensuels et journaliers, carte de chaleur, carte.
2. **Boucles admissibles** : toutes les boucles avec énergie et gains (total, moyen, maximal, minimal), téléchargeables en CSV.
3. **Optimisation** : objectifs choisis par ordre de priorité, seuils, tolérance ; front de Pareto et boucle retenue.
4. **Boucle retenue** : gains par participant, bilans mensuels et journaliers, plages tarifaires, carte.

Les contraintes et les prix se règlent dans la barre latérale ; tout changement efface le résultat d'optimisation précédent.

## À savoir

- Les prix de l'énergie par défaut sont fictifs. Seul le taux d'accise est réel (catégorie > 36 kVA, depuis le 1er août 2026).

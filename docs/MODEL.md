# Documentation technique du modèle

## Vue d'ensemble

| | |
|---|---|
| **Algorithme** | `GradientBoostingClassifier` (scikit-learn) |
| **Tâche** | Classification binaire — prédiction du risque de départ d'un salarié |
| **Origine** | Réentraîné à partir du pipeline validé dans le projet Technova Partners |
| **Format de sérialisation** | `joblib` (`ml/artifacts/attrition_model.joblib`) |
| **Dataset d'entraînement** | `technova_hr_clean.csv` — 1470 salariés, 16% de taux d'attrition |

## Architecture du pipeline

Le modèle sérialisé est un pipeline scikit-learn **entièrement autonome**, en 3 étapes :

1. **Feature engineering** (`FunctionTransformer`) — calcule 6 variables dérivées à partir des données brutes :
   - `satisfaction_moyenne` (moyenne de 4 scores de satisfaction)
   - `experience_avant_entreprise` (expérience totale − ancienneté dans l'entreprise)
   - `nouvel_arrivant`, `nouveau_poste` (indicateurs binaires)
   - `anciennete_poste_relative`, `anciennete_manager_relative` (ratios d'ancienneté)
2. **Preprocessing** (`ColumnTransformer`) — imputation médiane + standardisation (numériques), imputation par mode + one-hot encoding (catégorielles)
3. **Classification** (`GradientBoostingClassifier`)

Cette encapsulation signifie que le modèle accepte directement des données **brutes** (mêmes colonnes que le dataset d'origine) en entrée — aucun prétraitement externe n'est nécessaire côté API.

### Hyperparamètres

Optimisés par `GridSearchCV` (validation croisée stratifiée, 5 folds, optimisation du recall) :

```python
{
    "learning_rate": 0.05,
    "max_depth": 2,
    "n_estimators": 200,
    "subsample": 0.8,
}
```

### Gestion du déséquilibre de classes

Le dataset est déséquilibré (16% de départs). Le modèle utilise une pondération des classes (`sample_weight`, stratégie `"balanced"`) plutôt qu'un sur/sous-échantillonnage, pour ne pas altérer la distribution réelle des données.

## Performances

Évaluées sur un split stratifié 80/20 (`random_state=42`, reproductible) :

| Métrique | Valeur | Interprétation |
|---|---|---|
| **Recall** | 0,681 | 68% des départs réels sont détectés |
| **Precision** | 0,427 | 43% des alertes du modèle sont de vrais départs |
| **F1-score** | 0,525 | Compromis recall/precision |
| **AUC ROC** | 0,800 | Bonne capacité de discrimination globale |
| **Accuracy** | 0,803 | À interpréter avec prudence (voir ci-dessous) |

### Pourquoi le recall est priorisé

Un modèle prédisant systématiquement "aucun départ" atteindrait 84% d'accuracy tout en étant inutile (0% de recall). Le coût métier d'un départ non anticipé (faux négatif) est jugé plus élevé que celui d'une fausse alerte RH (faux positif) — le seuil de décision (0,5) et l'optimisation ont été choisis en conséquence.

### Facteurs les plus influents (interprétabilité SHAP)

D'après l'analyse du notebook d'origine :

- **Augmentent le risque** : heures supplémentaires, déplacements fréquents, expériences professionnelles antérieures nombreuses
- **Réduisent le risque** : participation à l'épargne salariale (PEE), satisfaction élevée, rémunération élevée

## Limites connues

- **`augmentation_salaire_precedente` traitée comme catégorielle** (valeurs `"11 %"` à `"25 %"`) plutôt que numérique — héritage du pipeline d'origine. Une valeur hors de cette plage est rejetée par l'API (erreur 422) plutôt que silencieusement ignorée par le modèle. Amélioration future possible : conversion en variable numérique continue.
- **Causalité non établie** — le modèle identifie des facteurs associés statistiquement aux départs, pas des causes prouvées. Les résultats doivent être interprétés comme des signaux, pas des certitudes (cf. contexte métier du projet Technova Partners).
- **Faux positifs/négatifs** — sur le jeu de test (294 salariés), le modèle produit 43 fausses alertes et manque 15 départs réels. Le seuil de décision (0,5) peut être ajusté selon la capacité des RH à traiter les alertes.
- **Dataset relativement restreint** (1470 lignes) — les performances sur une population significativement différente (autre secteur, autre pays) ne sont pas garanties.

## Reproductibilité

`RANDOM_STATE = 42` est fixé à toutes les étapes générant de l'aléatoire (split train/test, entraînement du modèle). Un test automatisé (`tests/test_model_performance.py::test_training_is_reproducible`) vérifie que deux entraînements successifs produisent des prédictions identiques.

## Maintenance et protocole de mise à jour

### Quand réentraîner le modèle

- **Dérive des données** : si la distribution des nouvelles données RH s'écarte significativement du dataset d'entraînement (ex. nouveaux départements, changement de politique salariale)
- **Dégradation observée** : si le taux de fausses alertes remonté par les RH augmente notablement
- **Nouvelles données disponibles** : périodiquement (ex. annuellement), en intégrant les données RH de l'année écoulée via `prediction_logs` et les retours terrain

### Comment réentraîner

```bash
python train.py
```

**Important** : toujours utiliser ce point d'entrée (jamais `python ml/train_model.py` ni `python -m ml.train_model` directement) — voir README pour l'explication technique (piège de sérialisation `FunctionTransformer`/pickling).

Avant de déployer un nouveau modèle réentraîné :
1. Vérifier que `tests/test_model_performance.py` passe (seuils minimaux de recall/precision/F1/AUC)
2. Comparer les nouvelles métriques à celles documentées ici
3. Mettre à jour cette documentation si les métriques ou les facteurs influents changent significativement

### Suivi en production

Chaque prédiction est journalisée dans la table `prediction_logs` (voir README, section Base de données) avec horodatage — cette table constitue la base d'un futur monitoring de dérive (comparaison de la distribution des `probabilite_depart` dans le temps).
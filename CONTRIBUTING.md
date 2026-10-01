# Standards de code et d'expérimentation ML

Ce document définit les conventions suivies dans ce projet pour garder un code cohérent et un historique clair, à mesure que le projet évolue.

## Conventions de commits

On suit le format [Conventional Commits](https://www.conventionalcommits.org/) :

- `feat:` → nouvelle fonctionnalité
- `fix:` → correction de bug
- `test:` → ajout ou modification de tests
- `ci:` → changement lié au pipeline CI/CD
- `docs:` → changement de documentation
- `chore:` → tâche technique sans impact fonctionnel (config, dépendances...)
- `refactor:` → restructuration de code sans changement de comportement fonctionnel

Exemple : `feat: ajout endpoint /predict`

## Convention de branches

- `main` → code stable, déployé automatiquement sur Render
- `feature/nom-fonctionnalite` → une branche par fonctionnalité, fusionnée via Pull Request

## Règles de fusion

- Toute modification passe par une Pull Request vers `main`
- Les tests automatiques (GitHub Actions) doivent être au vert avant de pouvoir fusionner (règle de protection de branche activée)

## Style de code Python

- Suivre les conventions [PEP 8](https://peps.python.org/pep-0008/)
- Noms de fonctions et variables en `snake_case`
- Docstrings pour les fonctions non triviales

## Standards d'expérimentation ML

- **Modèle** : `GradientBoostingClassifier` (scikit-learn), choisi après comparaison avec LogisticRegression, RandomForest et DecisionTree (voir `docs/MODEL.md` pour le détail)
- **Métriques suivies** : recall, precision, F1-score, AUC ROC, le recall est priorisé (seuil de décision et optimisation orientés vers la détection maximale des départs, le coût d'un faux négatif étant jugé supérieur à celui d'un faux positif)
- **Gestion du déséquilibre de classes** (16% de taux d'attrition) : pondération des classes (`sample_weight`, stratégie `"balanced"`), pas de sur/sous-échantillonnage
- **Seuil de décision** : 0,5, ajustable selon la capacité des RH à traiter les alertes
- **Hyperparamètres** : optimisés par `GridSearchCV` (validation croisée stratifiée, 5 folds)
- **Reproductibilité** : `RANDOM_STATE = 42` fixé à toutes les étapes aléatoires (split, entraînement) — vérifiée automatiquement par `tests/test_model_performance.py::test_training_is_reproducible`
- **Limite connue** : `augmentation_salaire_precedente` traitée comme catégorielle plutôt que numérique — détail complet dans [`docs/MODEL.md`](./docs/MODEL.md)

## Tests

- Chaque nouvel endpoint de l'API doit avoir au moins un test associé dans `tests/`
- Lancer les tests en local avant de pousser : `pytest --cov=app --cov=ml --cov=db`
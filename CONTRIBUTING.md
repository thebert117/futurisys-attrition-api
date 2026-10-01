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

(à compléter à l'étape d'entraînement du modèle — décisions à documenter ici : choix du modèle, métriques suivies, gestion du déséquilibre de classes, seuil de décision retenu, etc.)

## Tests

- Chaque nouvel endpoint de l'API doit avoir au moins un test associé dans `tests/`
- Lancer les tests en local avant de pousser : `pytest --cov=app`
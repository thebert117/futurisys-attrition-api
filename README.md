# Futurisys - API de prédiction d'attrition

POC de mise en production d'un modèle de machine learning (prédiction du risque de départ des salariés) via une API FastAPI, pour le compte de Futurisys.

Le modèle est un `GradientBoostingClassifier` réentraîné à partir du projet Technova Partners, exposé via une API REST validée par Pydantic, avec traçabilité des prédictions dans une base PostgreSQL (en environnement local).

### Built With

- [Python 3.11](https://www.python.org/)
- [FastAPI](https://fastapi.tiangolo.com/) — framework API
- [Pydantic](https://docs.pydantic.dev/) — validation des données
- [scikit-learn](https://scikit-learn.org/) — modèle de machine learning
- [PostgreSQL](https://www.postgresql.org/) + [SQLAlchemy](https://www.sqlalchemy.org/) — base de données
- [Pytest](https://docs.pytest.org/) + [pytest-cov](https://coverage.readthedocs.io/) — tests
- [Docker](https://www.docker.com/) — conteneurisation
- [GitHub Actions](https://docs.github.com/actions) — CI/CD
- [Render](https://render.com/) — hébergement

## Sommaire

- [Built With](#built-with)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Authentification](#authentification)
- [CI/CD](#cicd)
- [Base de données](#base-de-données)
- [Déploiement](#déploiement)
- [Tests](#tests)
- [Standards de code](#standards-de-code)
- [Documentation du modèle](./docs/MODEL.md)
- [License](#license)
- [Contact](#contact)

## Installation

```bash
git clone https://github.com/thebert117/futurisys-attrition-api.git
cd futurisys-attrition-api
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate sous Windows
pip install -r requirements.txt
```

### Configuration de la base de données

Copie `.env.example` en `.env` et renseigne tes identifiants PostgreSQL locaux :

```
DB_USER=postgres
DB_PASSWORD=ton_mot_de_passe
DB_HOST=localhost
DB_PORT=5432
DB_NAME=futurisys_attrition
```

Le fichier `.env` n'est jamais versionné (voir `.gitignore`) - il contient des identifiants propres à chaque poste de travail.

### Entraînement du modèle

Le modèle entraîné (`ml/artifacts/attrition_model.joblib`) est inclus dans le dépôt — aucune action requise pour l'utiliser directement après un `git clone`.

Un réentraînement n'est nécessaire que dans ces cas précis :
- Mise à jour du dataset (nouvelles données RH disponibles)
- Modification du pipeline de feature engineering ou des hyperparamètres
- Mise à jour majeure de scikit-learn créant une incompatibilité de version avec le `.joblib` existant (`InconsistentVersionWarning` au chargement)

Pour réentraîner :

1. Placer le fichier `technova_hr_clean.csv` dans `data/`
2. Lancer : `python train.py`

**Important** : toujours utiliser `python train.py`, jamais `python ml/train_model.py` ni `python -m ml.train_model` directement. Le pipeline encapsule le feature engineering via un `FunctionTransformer` ; si `train_model.py` est exécuté comme point d'entrée direct, la fonction est sérialisée sous le module `__main__`, ce qui casse le rechargement du modèle ailleurs (API, tests, Render).

Cela génère `ml/artifacts/attrition_model.joblib`, utilisé par l'API.

## Utilisation

Lancer l'API en local :

```bash
uvicorn app.main:app --reload
```

Documentation interactive (Swagger) : http://127.0.0.1:8000/docs

### Endpoints

| Endpoint | Méthode | Description |
|---|---|---|
| `/health` | GET | Vérifie que l'API est active |
| `/predict` | POST | Retourne une prédiction de risque de départ pour un salarié |

Le endpoint `/predict` valide strictement les données d'entrée via Pydantic (types, bornes, valeurs catégorielles autorisées) et retourne une erreur `422` en cas de donnée invalide.

**Limite connue** : la variable `augmentation_salaire_precedente` est traitée comme catégorielle (valeurs `"11 %"` à `"25 %"`) plutôt que numérique, héritage du modèle d'origine. Une valeur hors de cette plage est rejetée par l'API plutôt que silencieusement ignorée par le modèle - un choix délibéré, détaillé dans `docs/MODEL.md`.

### Exemple d'appel à /predict

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: votre_cle_api" \
  -d '{
    "age": 41, "genre": "F", "revenu_mensuel": 5993,
    "statut_marital": "Célibataire", "departement": "Commercial",
    "poste": "Cadre Commercial", "nombre_experiences_precedentes": 8,
    "annee_experience_totale": 8, "annees_dans_l_entreprise": 6,
    "annees_dans_le_poste_actuel": 4, "satisfaction_employee_environnement": 2,
    "note_evaluation_precedente": 3, "niveau_hierarchique_poste": 2,
    "satisfaction_employee_nature_travail": 4, "satisfaction_employee_equipe": 1,
    "satisfaction_employee_equilibre_pro_perso": 1, "note_evaluation_actuelle": 3,
    "heure_supplementaires": "Oui", "augmentation_salaire_precedente": "11 %",
    "nombre_participation_pee": 0, "nb_formations_suivies": 0,
    "distance_domicile_travail": 1, "niveau_education": 2,
    "domaine_etude": "Infra & Cloud", "frequence_deplacement": "Occasionnel",
    "annees_depuis_la_derniere_promotion": 0, "annees_sous_responsable_actuel": 5
  }'
```

Réponse :
```json
{
  "risque_depart": true,
  "probabilite_depart": 0.737
}
```

Documentation technique complète du modèle (performances, limites, maintenance) : [`docs/MODEL.md`](./docs/MODEL.md)

## Authentification

L'endpoint `/predict` est protégé par une clé API, transmise via l'en-tête HTTP `X-API-Key`. L'endpoint `/health` reste public (convention standard pour un endpoint de monitoring).

Choix volontairement simple pour ce POC : une clé secrète unique et partagée, plutôt qu'un système de comptes utilisateurs — adapté à un usage où l'API est consommée par un service interne (SIRH, tableau de bord RH), sans besoin de droits différenciés entre utilisateurs.

### Configuration

La clé est définie dans `.env` (jamais versionnée) :

```
API_KEY=une_chaine_aleatoire_longue
```

Génération d'une clé sécurisée :

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### Exemple d'appel authentifié

Voir l'exemple complet dans la section [Utilisation](#utilisation) — l'en-tête `X-API-Key` y est déjà inclus.

Sans clé (401) ou avec une clé incorrecte (403), l'API refuse la requête.

### Gestion des accès en production

Sur Render, la clé est définie comme variable d'environnement (Dashboard → Settings → Environment), jamais en clair dans le code ou le Dockerfile. Une rotation de la clé (en cas de compromission) consiste à en générer une nouvelle et à mettre à jour la variable d'environnement sur Render, sans redéploiement de code nécessaire.

## CI/CD

Le pipeline GitHub Actions (`.github/workflows/ci.yml`) se déclenche à chaque push et Pull Request :

1. Installation des dépendances
2. Démarrage d'un service PostgreSQL éphémère (nécessaire pour tester `/predict`, qui écrit en base)
3. Création des tables via `db/create_db.py`
4. Exécution des tests Pytest avec rapport de couverture (`pytest-cov`)

La branche `main` est protégée : aucune fusion n'est possible tant que les tests ne sont pas au vert (règle de protection de branche GitHub, dépôt public requis pour que la règle soit appliquée sur le plan gratuit).

### Environnements

- **Développement** : poste local, base PostgreSQL locale, variables d'environnement via `.env`
- **Test / CI** : GitHub Actions, base PostgreSQL éphémère créée pour la durée du pipeline, variables d'environnement définies dans `ci.yml`
- **Production** : Render (voir [Déploiement](#déploiement))

## Base de données

Base PostgreSQL utilisée **uniquement en local**, conformément au cadrage du projet.

### Schéma

Schéma détaillé (diagramme UML) : [`docs/schema-bdd.md`](./docs/schema-bdd.md)

**`employees`** - dataset RH importé depuis le CSV d'origine (1470 salariés), sert de référence.

**`prediction_logs`** - un enregistrement par appel à `/predict` : les 27 champs d'entrée envoyés à l'API, la prédiction retournée (`risque_depart`, `probabilite_depart`), et un horodatage automatique (`created_at`).

Les deux tables ne sont **pas reliées par clé étrangère** : une prédiction peut concerner un profil hypothétique qui n'existe pas dans le dataset historique (simulation, nouveau candidat...). `employees` est une donnée de référence, `prediction_logs` est un journal d'utilisation de l'API.

### Choix de conception : pas de 3NF stricte

Le schéma est volontairement dénormalisé plutôt que d'extraire les variables catégorielles (département, poste...) en tables de dimension séparées. Justification :

- Volume très faible (1470 lignes, 3 départements, 9 postes) : le gain en normalisation serait négligeable
- Les valeurs catégorielles sont déjà validées en amont par Pydantic (`Literal`), rendant une contrainte de clé étrangère redondante
- `prediction_logs` est un journal d'événements (audit log), un pattern où la dénormalisation est une pratique standard, même dans des systèmes matures

### Besoins analytiques

La table `prediction_logs` (horodatage inclus) permet d'alimenter une analyse a posteriori de l'usage du modèle :

- Suivi du volume de prédictions dans le temps
- Distribution des scores de risque prédits (`probabilite_depart`), utile pour détecter une dérive du modèle
- Croisement avec `employees` pour comparer les prédictions aux départs réels observés (une fois ces données disponibles)

Un tableau de bord (ex. Metabase, Superset, ou simple notebook Python connecté à PostgreSQL) pourrait exploiter directement ces deux tables sans transformation supplémentaire, la structure actuelle étant déjà adaptée à une exploration analytique simple (SQL direct, pas de dénormalisation supplémentaire nécessaire pour ce volume).

### Mise en place

```bash
python -m db.create_db       # crée la base et les tables
python -m db.insert_dataset  # insère le dataset dans employees
```

**Important** : ces scripts doivent être lancés avec `python -m` (et non `python db/create_db.py`), pour que les imports entre modules fonctionnent correctement.

### Logging tolérant aux pannes

L'enregistrement en base dans `/predict` est entouré d'un `try/except` : si la base est injoignable, l'API continue de répondre à la prédiction plutôt que de planter, avec un simple avertissement dans les logs serveur. Ce comportement est nécessaire car l'API est déployée publiquement sur Render, qui n'a pas accès à la base PostgreSQL locale.

## Déploiement

L'API est déployée sur **Render** (plan gratuit), alternative à Hugging Face Spaces suite à la restriction de son SDK Docker gratuit courant 2026.

Le déploiement est automatique à chaque push sur `main` (Render détecte le changement et reconstruit l'image Docker).

URL de production : https://futurisys-attrition-api.onrender.com

**Limite connue** : le plan gratuit Render met le service en veille après 15 minutes d'inactivité (redémarrage à froid de 30-60 secondes au premier appel suivant).

## Tests

La suite de tests est organisée par nature :

- `tests/test_health.py`, `tests/test_predict_api.py` — tests fonctionnels (via l'API réelle, TestClient)
- `tests/test_schemas.py`, `tests/test_feature_engineering.py` — tests unitaires (composants isolés)
- `tests/test_database.py` — tests de la base de données (structure, intégrité, idempotence)
- `tests/test_model_performance.py` — tests de performance et de reproductibilité du modèle

Lancer la suite complète avec couverture :

```bash
pytest --cov=app --cov=ml --cov=db --cov-report=term-missing
```

Le rapport de couverture HTML est généré automatiquement dans `htmlcov/` et publié comme artefact téléchargeable à chaque exécution du pipeline CI (onglet Actions de GitHub, section "Artifacts" du run).

## Standards de code

Voir [`CONTRIBUTING.md`](./CONTRIBUTING.md) pour les conventions de commit, de nommage de branches, et les standards d'expérimentation ML.

## License

Distribué sous licence MIT. Voir [`LICENSE`](./LICENSE) pour plus de détails.

## Contact

Thomas Hébert — [github.com/thebert117](https://github.com/thebert117)

Lien du projet : [github.com/thebert117/futurisys-attrition-api](https://github.com/thebert117/futurisys-attrition-api)
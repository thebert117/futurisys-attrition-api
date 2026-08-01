# Futurisys - API de prédiction d'attrition

POC de mise en production d'un modèle de machine learning (prédiction du risque de départ des salariés) via une API FastAPI, pour le compte de Futurisys.

Le modèle est un `GradientBoostingClassifier` réentraîné à partir du projet Technova Partners, exposé via une API REST validée par Pydantic, avec traçabilité complète des prédictions dans une base PostgreSQL.

## Sommaire

- [Installation](#installation)
- [Utilisation](#utilisation)
- [CI/CD](#cicd)
- [Base de données](#base-de-données)
- [Déploiement](#déploiement)
- [Standards de code](#standards-de-code)

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

Le modèle n'est pas fourni tel quel avec le dépôt, il faut le réentraîner localement :

1. Placer le fichier `technova_hr_clean.csv` dans `data/`
2. Lancer : `python ml/train_model.py`

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

**Limite connue** : la variable `augmentation_salaire_precedente` est traitée comme catégorielle (valeurs `"11 %"` à `"25 %"`) plutôt que numérique, héritage du modèle d'origine. Une valeur hors de cette plage est rejetée par l'API plutôt que silencieusement ignorée par le modèle - un choix délibéré, détaillé dans `CONTRIBUTING.md`.

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

**`employees`** - dataset RH brut importé depuis le CSV d'origine (1470 salariés), sert de référence historique.

**`prediction_logs`** - un enregistrement par appel à `/predict` : les 27 champs d'entrée envoyés à l'API, la prédiction retournée (`risque_depart`, `probabilite_depart`), et un horodatage automatique (`created_at`).

Les deux tables ne sont **pas reliées par clé étrangère** : une prédiction peut concerner un profil hypothétique qui n'existe pas dans le dataset historique (simulation, nouveau candidat...). `employees` est une donnée de référence, `prediction_logs` est un journal d'utilisation de l'API.

### Choix de conception : pas de 3NF stricte

Le schéma est volontairement dénormalisé plutôt que d'extraire les variables catégorielles (département, poste...) en tables de dimension séparées. Justification :

- Volume très faible (1470 lignes, 3 départements, 9 postes) : le gain en normalisation serait négligeable
- Les valeurs catégorielles sont déjà validées en amont par Pydantic (`Literal`), rendant une contrainte de clé étrangère redondante
- `prediction_logs` est un journal d'événements (audit log), un pattern où la dénormalisation est une pratique standard, même dans des systèmes matures

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

## Standards de code

Voir [`CONTRIBUTING.md`](./CONTRIBUTING.md) pour les conventions de commit, de nommage de branches, et les standards d'expérimentation ML.
"""
Script d'entraînement du modèle de prédiction d'attrition Futurisys.

Reproduit fidèlement le pipeline validé dans le notebook Technova Partners :
- feature engineering (6 variables métier), encapsulé dans le pipeline via FunctionTransformer
- preprocessing (imputation + standardisation + one-hot encoding)
- GradientBoostingClassifier avec sample_weight (gestion du déséquilibre)
- hyperparamètres optimisés par GridSearchCV

Le modèle est entraîné directement sur 100% des données disponibles (meilleure
généralisation pour la production). La validation des performances (recall,
precision, F1, AUC) n'est plus refaite ici : elle est assurée de façon
automatisée et reproductible par tests/test_model_performance.py, qui
reconstruit le même split 80/20 pour vérifier que les seuils attendus sont
respectés. Ce script n'a donc plus besoin de dupliquer cette vérification.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

import joblib

RANDOM_STATE = 42

# Hyperparamètres retenus par GridSearchCV dans le notebook d'origine
BEST_PARAMS = {
    "learning_rate": 0.05,
    "max_depth": 2,
    "n_estimators": 200,
    "subsample": 0.8,
}

# Colonnes à exclure des features (identifiant, cible, doublon texte de la cible)
COLS_TO_EXCLUDE = ["id_employee", "a_quitte_l_entreprise", "target_attrition"]


def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les 6 features complémentaires définies dans le notebook d'origine."""
    data = data.copy()

    satisfaction_cols = [
        "satisfaction_employee_environnement",
        "satisfaction_employee_nature_travail",
        "satisfaction_employee_equipe",
        "satisfaction_employee_equilibre_pro_perso",
    ]
    data["satisfaction_moyenne"] = data[satisfaction_cols].mean(axis=1)

    data["experience_avant_entreprise"] = (
        data["annee_experience_totale"] - data["annees_dans_l_entreprise"]
    )

    data["nouvel_arrivant"] = (data["annees_dans_l_entreprise"] == 0).astype(int)
    data["nouveau_poste"] = (data["annees_dans_le_poste_actuel"] == 0).astype(int)

    data["anciennete_poste_relative"] = (
        data["annees_dans_le_poste_actuel"]
        / data["annees_dans_l_entreprise"].replace(0, np.nan)
    )
    data["anciennete_manager_relative"] = (
        data["annees_sous_responsable_actuel"]
        / data["annees_dans_l_entreprise"].replace(0, np.nan)
    )

    return data


def build_preprocessor(numeric_cols, categorical_cols) -> ColumnTransformer:
    """Reconstruit le pipeline de preprocessing exact du notebook."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )


def build_full_pipeline(numeric_cols, categorical_cols) -> Pipeline:
    """Construit le pipeline complet et autonome : feature engineering inclus.

    Le feature engineering est encapsulé dans le pipeline (FunctionTransformer)
    plutôt qu'appelé séparément : le .joblib résultant est autonome, il accepte
    des données brutes (mêmes colonnes que le CSV d'origine) et gère lui-même
    tout le traitement jusqu'à la prédiction finale.
    """
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    return Pipeline(steps=[
        ("feature_engineering", FunctionTransformer(add_engineered_features)),
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(random_state=RANDOM_STATE, **BEST_PARAMS)),
    ])


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "technova_hr_clean.csv"
    artifacts_dir = project_root / "ml" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print(f"Chargement du dataset depuis {data_path}...")
    df = pd.read_csv(data_path)

    # X reste en données BRUTES (mêmes colonnes que le CSV) : le feature
    # engineering est réalisé à l'intérieur du pipeline lui-même, pas ici.
    X = df.drop(columns=COLS_TO_EXCLUDE)
    y = df["target_attrition"]

    # On a besoin de connaître les colonnes numériques/catégorielles APRÈS
    # feature engineering pour construire le ColumnTransformer.
    X_preview = add_engineered_features(X)
    numeric_cols = X_preview.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X_preview.select_dtypes(include=["object", "str"]).columns.tolist()

    print(f"X (brut) : {X.shape[0]} lignes x {X.shape[1]} colonnes")
    print(f"Après feature engineering : {len(numeric_cols)} numériques, {len(categorical_cols)} catégorielles")

    print("\nEntraînement du modèle final sur 100% des données disponibles...")
    print("(Validation des performances assurée séparément par tests/test_model_performance.py)")

    pipe = build_full_pipeline(numeric_cols, categorical_cols)
    sample_weights = compute_sample_weight(class_weight="balanced", y=y)
    pipe.fit(X, y, classifier__sample_weight=sample_weights)

    model_path = artifacts_dir / "attrition_model.joblib"
    joblib.dump(pipe, model_path)
    print(f"\nModèle sauvegardé : {model_path}")

    raw_input_cols = [c for c in df.columns if c not in COLS_TO_EXCLUDE]
    columns_path = artifacts_dir / "input_columns.txt"
    columns_path.write_text("\n".join(raw_input_cols), encoding="utf-8")
    print(f"Liste des colonnes d'entrée sauvegardée : {columns_path}")


if __name__ == "__main__":
    main()
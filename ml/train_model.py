"""
Script d'entraînement du modèle de prédiction d'attrition Futurisys.

Reproduit fidèlement le pipeline validé dans le notebook Technova Partners :
- feature engineering (6 variables métier)
- preprocessing (imputation + standardisation + one-hot encoding)
- GradientBoostingClassifier avec sample_weight (gestion du déséquilibre)
- hyperparamètres optimisés par GridSearchCV

Étape 1 : réentraîne sur le split 80/20 pour revalider les métriques
Étape 2 : réentraîne le modèle final sur 100% des données pour la production
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

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


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "technova_hr_clean.csv"
    artifacts_dir = project_root / "ml" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print(f"Chargement du dataset depuis {data_path}...")
    df = pd.read_csv(data_path)
    df_model = add_engineered_features(df)

    X = df_model.drop(columns=COLS_TO_EXCLUDE)
    y = df_model["target_attrition"]

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "str"]).columns.tolist()

    print(f"X : {X.shape[0]} lignes x {X.shape[1]} colonnes")
    print(f"Colonnes numériques ({len(numeric_cols)}), catégorielles ({len(categorical_cols)})")

    # --- Étape 1 : revalidation sur le split 80/20, comme dans le notebook ---
    print("\n=== Étape 1 : revalidation sur split train/test (80/20) ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE,
    )

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(random_state=RANDOM_STATE, **BEST_PARAMS)),
    ])

    sample_weights_train = compute_sample_weight(class_weight="balanced", y=y_train)
    pipe.fit(X_train, y_train, classifier__sample_weight=sample_weights_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision : {precision_score(y_test, y_pred):.3f}")
    print(f"Recall    : {recall_score(y_test, y_pred):.3f}")
    print(f"F1-score  : {f1_score(y_test, y_pred):.3f}")
    print(f"ROC AUC   : {roc_auc_score(y_test, y_proba):.3f}")
    print("(Attendu d'après le notebook : Recall ~0.681, Precision ~0.427, F1 ~0.525, AUC ~0.800)")

    # --- Étape 2 : modèle final entraîné sur 100% des données ---
    print("\n=== Étape 2 : entraînement du modèle final sur 100% des données ===")
    preprocessor_final = build_preprocessor(numeric_cols, categorical_cols)
    final_pipe = Pipeline(steps=[
        ("preprocessor", preprocessor_final),
        ("classifier", GradientBoostingClassifier(random_state=RANDOM_STATE, **BEST_PARAMS)),
    ])

    sample_weights_full = compute_sample_weight(class_weight="balanced", y=y)
    final_pipe.fit(X, y, classifier__sample_weight=sample_weights_full)

    model_path = artifacts_dir / "attrition_model.joblib"
    joblib.dump(final_pipe, model_path)
    print(f"\nModèle final sauvegardé : {model_path}")

    # On sauvegarde aussi la liste exacte des colonnes attendues en entrée,
    # utile pour construire le schéma Pydantic de l'API sans risque d'erreur
    raw_input_cols = [c for c in df.columns if c not in COLS_TO_EXCLUDE]
    columns_path = artifacts_dir / "input_columns.txt"
    columns_path.write_text("\n".join(raw_input_cols), encoding="utf-8")
    print(f"Liste des colonnes d'entrée sauvegardée : {columns_path}")


if __name__ == "__main__":
    main()
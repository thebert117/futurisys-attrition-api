"""
Tests fonctionnels de performance et de reproductibilité du modèle.

On reconstruit le même split train/test que ml/train_model.py (même random_state)
pour vérifier que les métriques restent stables dans le temps et respectent des
seuils minimaux. Un écart important signalerait une régression (nouvelles données,
modification accidentelle du pipeline, etc.).

Ces tests sont plus lents que les autres (ils réentraînent un modèle) : c'est
attendu pour des tests de performance, qui n'ont pas vocation à tourner aussi
souvent que les tests unitaires.
"""

from pathlib import Path

import pandas as pd
import pytest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from ml.train_model import COLS_TO_EXCLUDE, RANDOM_STATE, add_engineered_features, build_full_pipeline

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "technova_hr_clean.csv"

MIN_RECALL = 0.60
MIN_PRECISION = 0.35
MIN_F1 = 0.45
MIN_AUC = 0.75


def _load_raw_X_y():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=COLS_TO_EXCLUDE)
    y = df["target_attrition"]
    return X, y


def _column_lists(X):
    preview = add_engineered_features(X)
    numeric_cols = preview.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = preview.select_dtypes(include=["object", "str"]).columns.tolist()
    return numeric_cols, categorical_cols


@pytest.fixture(scope="module")
def evaluation_metrics():
    """Réentraîne le pipeline complet (feature engineering inclus) sur le split 80/20."""
    if not DATA_PATH.exists():
        pytest.skip("technova_hr_clean.csv absent : impossible de valider les performances du modèle")

    X, y = _load_raw_X_y()
    numeric_cols, categorical_cols = _column_lists(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE,
    )

    pipe = build_full_pipeline(numeric_cols, categorical_cols)
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    pipe.fit(X_train, y_train, classifier__sample_weight=sample_weights)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    return {
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_proba),
    }


def test_recall_meets_minimum_threshold(evaluation_metrics):
    assert evaluation_metrics["recall"] >= MIN_RECALL


def test_precision_meets_minimum_threshold(evaluation_metrics):
    assert evaluation_metrics["precision"] >= MIN_PRECISION


def test_f1_meets_minimum_threshold(evaluation_metrics):
    assert evaluation_metrics["f1"] >= MIN_F1


def test_auc_meets_minimum_threshold(evaluation_metrics):
    assert evaluation_metrics["auc"] >= MIN_AUC


def test_training_is_reproducible():
    """Deux entraînements avec le même random_state doivent produire
    exactement les mêmes prédictions (reproductibilité)."""
    if not DATA_PATH.exists():
        pytest.skip("technova_hr_clean.csv absent")

    X, y = _load_raw_X_y()
    numeric_cols, categorical_cols = _column_lists(X)

    def train_and_predict():
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE,
        )
        pipe = build_full_pipeline(numeric_cols, categorical_cols)
        weights = compute_sample_weight(class_weight="balanced", y=y_train)
        pipe.fit(X_train, y_train, classifier__sample_weight=weights)
        return pipe.predict(X_test)

    predictions_run_1 = train_and_predict()
    predictions_run_2 = train_and_predict()

    assert (predictions_run_1 == predictions_run_2).all()


def test_pipeline_accepts_raw_input_without_manual_feature_engineering():
    """Vérifie que le pipeline encapsulé fonctionne directement sur des données
    brutes, sans appel externe à add_engineered_features (test de non-régression
    pour l'encapsulation du feature engineering dans le pipeline)."""
    if not DATA_PATH.exists():
        pytest.skip("technova_hr_clean.csv absent")

    X, y = _load_raw_X_y()
    numeric_cols, categorical_cols = _column_lists(X)

    pipe = build_full_pipeline(numeric_cols, categorical_cols)
    weights = compute_sample_weight(class_weight="balanced", y=y)
    pipe.fit(X, y, classifier__sample_weight=weights)

    assert "satisfaction_moyenne" not in X.columns
    prediction = pipe.predict(X.iloc[[0]])
    assert prediction[0] in (0, 1)
"""
Tests unitaires de add_engineered_features (ml/train_model.py).

On construit des DataFrames à la main pour vérifier que chaque variable
calculée est correcte, indépendamment du modèle ou de l'API.
"""

import pandas as pd

from ml.train_model import add_engineered_features


def make_base_row(**overrides):
    """Construit une ligne minimale avec toutes les colonnes nécessaires au feature engineering."""
    row = {
        "satisfaction_employee_environnement": 2,
        "satisfaction_employee_nature_travail": 4,
        "satisfaction_employee_equipe": 1,
        "satisfaction_employee_equilibre_pro_perso": 1,
        "annee_experience_totale": 10,
        "annees_dans_l_entreprise": 6,
        "annees_dans_le_poste_actuel": 3,
        "annees_sous_responsable_actuel": 4,
    }
    row.update(overrides)
    return row


def test_satisfaction_moyenne_is_correct_average():
    df = pd.DataFrame([make_base_row(
        satisfaction_employee_environnement=2,
        satisfaction_employee_nature_travail=4,
        satisfaction_employee_equipe=1,
        satisfaction_employee_equilibre_pro_perso=1,
    )])
    result = add_engineered_features(df)
    assert result["satisfaction_moyenne"].iloc[0] == (2 + 4 + 1 + 1) / 4


def test_experience_avant_entreprise_is_correct():
    df = pd.DataFrame([make_base_row(annee_experience_totale=10, annees_dans_l_entreprise=6)])
    result = add_engineered_features(df)
    assert result["experience_avant_entreprise"].iloc[0] == 4


def test_nouvel_arrivant_true_when_zero_years():
    df = pd.DataFrame([make_base_row(annees_dans_l_entreprise=0)])
    result = add_engineered_features(df)
    assert result["nouvel_arrivant"].iloc[0] == 1


def test_nouvel_arrivant_false_when_tenure_positive():
    df = pd.DataFrame([make_base_row(annees_dans_l_entreprise=6)])
    result = add_engineered_features(df)
    assert result["nouvel_arrivant"].iloc[0] == 0


def test_nouveau_poste_true_when_zero_years_in_role():
    df = pd.DataFrame([make_base_row(annees_dans_le_poste_actuel=0)])
    result = add_engineered_features(df)
    assert result["nouveau_poste"].iloc[0] == 1


def test_anciennete_poste_relative_is_ratio():
    df = pd.DataFrame([make_base_row(annees_dans_le_poste_actuel=3, annees_dans_l_entreprise=6)])
    result = add_engineered_features(df)
    assert result["anciennete_poste_relative"].iloc[0] == 0.5


def test_anciennete_relative_does_not_crash_on_zero_tenure():
    """Cas limite : un nouvel arrivant (0 an dans l'entreprise) ne doit pas provoquer
    d'erreur de division par zéro, mais produire une valeur manquante (NaN) gérée
    ensuite par l'imputation du preprocessing."""
    df = pd.DataFrame([make_base_row(annees_dans_l_entreprise=0, annees_dans_le_poste_actuel=0)])
    result = add_engineered_features(df)
    assert pd.isna(result["anciennete_poste_relative"].iloc[0])


def test_original_dataframe_is_not_mutated():
    """add_engineered_features ne doit pas modifier le DataFrame d'entrée (effet de bord)."""
    df = pd.DataFrame([make_base_row()])
    original_columns = list(df.columns)
    add_engineered_features(df)
    assert list(df.columns) == original_columns
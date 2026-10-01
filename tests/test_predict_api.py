"""
Tests fonctionnels de l'endpoint POST /predict.

Contrairement à test_schemas.py (unitaire), ces tests passent par la vraie API
(TestClient), donc valident l'ensemble de la chaîne : validation Pydantic,
feature engineering, appel au modèle, et réponse HTTP.
"""

from fastapi.testclient import TestClient

from app.main import app
from .conftest import VALID_EMPLOYEE_DATA

client = TestClient(app)

API_KEY_HEADERS = {"X-API-Key": "test-secret-key-12345"}


def test_predict_without_api_key_returns_401():
    response = client.post("/predict", json=VALID_EMPLOYEE_DATA)
    assert response.status_code == 401


def test_predict_with_wrong_api_key_returns_403():
    response = client.post("/predict", json=VALID_EMPLOYEE_DATA, headers={"X-API-Key": "mauvaise-cle"})
    assert response.status_code == 403


def test_health_does_not_require_api_key():
    response = client.get("/health")
    assert response.status_code == 200


def test_predict_returns_200_with_valid_data():
    response = client.post("/predict", json=VALID_EMPLOYEE_DATA, headers=API_KEY_HEADERS)
    assert response.status_code == 200


def test_predict_returns_expected_fields():
    response = client.post("/predict", json=VALID_EMPLOYEE_DATA, headers=API_KEY_HEADERS)
    body = response.json()
    assert "risque_depart" in body
    assert "probabilite_depart" in body
    assert isinstance(body["risque_depart"], bool)
    assert 0.0 <= body["probabilite_depart"] <= 1.0


def test_predict_rejects_invalid_genre():
    invalid_data = dict(VALID_EMPLOYEE_DATA)
    invalid_data["genre"] = "X"
    response = client.post("/predict", json=invalid_data, headers=API_KEY_HEADERS)
    assert response.status_code == 422


def test_predict_rejects_missing_field():
    incomplete_data = dict(VALID_EMPLOYEE_DATA)
    del incomplete_data["age"]
    response = client.post("/predict", json=incomplete_data, headers=API_KEY_HEADERS)
    assert response.status_code == 422


def test_predict_rejects_negative_age():
    invalid_data = dict(VALID_EMPLOYEE_DATA)
    invalid_data["age"] = -5
    response = client.post("/predict", json=invalid_data, headers=API_KEY_HEADERS)
    assert response.status_code == 422


def test_predict_accepts_boundary_age_18():
    data = dict(VALID_EMPLOYEE_DATA)
    data["age"] = 18
    response = client.post("/predict", json=data, headers=API_KEY_HEADERS)
    assert response.status_code == 200


def test_predict_accepts_boundary_age_70():
    data = dict(VALID_EMPLOYEE_DATA)
    data["age"] = 70
    response = client.post("/predict", json=data, headers=API_KEY_HEADERS)
    assert response.status_code == 200


def test_predict_rejects_age_just_above_boundary():
    data = dict(VALID_EMPLOYEE_DATA)
    data["age"] = 71
    response = client.post("/predict", json=data, headers=API_KEY_HEADERS)
    assert response.status_code == 422


def test_predict_is_deterministic():
    response_1 = client.post("/predict", json=VALID_EMPLOYEE_DATA, headers=API_KEY_HEADERS)
    response_2 = client.post("/predict", json=VALID_EMPLOYEE_DATA, headers=API_KEY_HEADERS)
    assert response_1.json() == response_2.json()


def test_predict_high_risk_profile_has_higher_probability_than_low_risk_profile():
    high_risk = dict(VALID_EMPLOYEE_DATA)
    high_risk.update({
        "heure_supplementaires": "Oui",
        "satisfaction_employee_environnement": 1,
        "satisfaction_employee_nature_travail": 1,
        "satisfaction_employee_equipe": 1,
        "satisfaction_employee_equilibre_pro_perso": 1,
        "nombre_participation_pee": 0,
        "revenu_mensuel": 2500,
    })

    low_risk = dict(VALID_EMPLOYEE_DATA)
    low_risk.update({
        "heure_supplementaires": "Non",
        "satisfaction_employee_environnement": 4,
        "satisfaction_employee_nature_travail": 4,
        "satisfaction_employee_equipe": 4,
        "satisfaction_employee_equilibre_pro_perso": 4,
        "nombre_participation_pee": 5,
        "revenu_mensuel": 12000,
    })

    proba_high = client.post("/predict", json=high_risk, headers=API_KEY_HEADERS).json()["probabilite_depart"]
    proba_low = client.post("/predict", json=low_risk, headers=API_KEY_HEADERS).json()["probabilite_depart"]

    assert proba_high > proba_low
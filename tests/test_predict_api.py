"""
Tests fonctionnels de l'endpoint POST /predict.

Contrairement à test_schemas.py (unitaire), ces tests passent par la vraie API
(TestClient), donc valident l'ensemble de la chaîne : validation Pydantic,
feature engineering, appel au modèle, et réponse HTTP.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

API_KEY_HEADERS = {"X-API-Key": "test-secret-key-12345"}

VALID_EMPLOYEE = {
    "age": 41,
    "genre": "F",
    "revenu_mensuel": 5993,
    "statut_marital": "Célibataire",
    "departement": "Commercial",
    "poste": "Cadre Commercial",
    "nombre_experiences_precedentes": 8,
    "annee_experience_totale": 8,
    "annees_dans_l_entreprise": 6,
    "annees_dans_le_poste_actuel": 4,
    "satisfaction_employee_environnement": 2,
    "note_evaluation_precedente": 3,
    "niveau_hierarchique_poste": 2,
    "satisfaction_employee_nature_travail": 4,
    "satisfaction_employee_equipe": 1,
    "satisfaction_employee_equilibre_pro_perso": 1,
    "note_evaluation_actuelle": 3,
    "heure_supplementaires": "Oui",
    "augmentation_salaire_precedente": "11 %",
    "nombre_participation_pee": 0,
    "nb_formations_suivies": 0,
    "distance_domicile_travail": 1,
    "niveau_education": 2,
    "domaine_etude": "Infra & Cloud",
    "frequence_deplacement": "Occasionnel",
    "annees_depuis_la_derniere_promotion": 0,
    "annees_sous_responsable_actuel": 5,
}


def test_predict_without_api_key_returns_401():
    """Aucune clé fournie : accès refusé (401)."""
    response = client.post("/predict", json=VALID_EMPLOYEE)
    assert response.status_code == 401


def test_predict_with_wrong_api_key_returns_403():
    """Clé fournie mais incorrecte : accès refusé (403)."""
    response = client.post("/predict", json=VALID_EMPLOYEE, headers={"X-API-Key": "mauvaise-cle"})
    assert response.status_code == 403


def test_health_does_not_require_api_key():
    """/health reste public, sans authentification (convention standard pour un endpoint de monitoring)."""
    response = client.get("/health")
    assert response.status_code == 200


def test_predict_returns_200_with_valid_data():
    response = client.post("/predict", json=VALID_EMPLOYEE, headers=API_KEY_HEADERS)
    assert response.status_code == 200


def test_predict_returns_expected_fields():
    response = client.post("/predict", json=VALID_EMPLOYEE, headers=API_KEY_HEADERS)
    body = response.json()
    assert "risque_depart" in body
    assert "probabilite_depart" in body
    assert isinstance(body["risque_depart"], bool)
    assert 0.0 <= body["probabilite_depart"] <= 1.0


def test_predict_rejects_invalid_genre():
    invalid_data = dict(VALID_EMPLOYEE)
    invalid_data["genre"] = "X"
    response = client.post("/predict", json=invalid_data, headers=API_KEY_HEADERS)
    assert response.status_code == 422


def test_predict_rejects_missing_field():
    incomplete_data = dict(VALID_EMPLOYEE)
    del incomplete_data["age"]
    response = client.post("/predict", json=incomplete_data, headers=API_KEY_HEADERS)
    assert response.status_code == 422


def test_predict_rejects_negative_age():
    invalid_data = dict(VALID_EMPLOYEE)
    invalid_data["age"] = -5
    response = client.post("/predict", json=invalid_data, headers=API_KEY_HEADERS)
    assert response.status_code == 422


def test_predict_accepts_boundary_age_18():
    """Cas limite : l'âge minimum autorisé (18) doit être accepté."""
    data = dict(VALID_EMPLOYEE)
    data["age"] = 18
    response = client.post("/predict", json=data, headers=API_KEY_HEADERS)
    assert response.status_code == 200


def test_predict_accepts_boundary_age_70():
    """Cas limite : l'âge maximum autorisé (70) doit être accepté."""
    data = dict(VALID_EMPLOYEE)
    data["age"] = 70
    response = client.post("/predict", json=data, headers=API_KEY_HEADERS)
    assert response.status_code == 200


def test_predict_rejects_age_just_above_boundary():
    """Cas limite : 71 ans (juste au-dessus de la borne) doit être rejeté."""
    data = dict(VALID_EMPLOYEE)
    data["age"] = 71
    response = client.post("/predict", json=data, headers=API_KEY_HEADERS)
    assert response.status_code == 422


def test_predict_is_deterministic():
    """La même requête envoyée deux fois doit produire exactement la même prédiction
    (le modèle ne doit pas avoir de composante aléatoire non maîtrisée)."""
    response_1 = client.post("/predict", json=VALID_EMPLOYEE, headers=API_KEY_HEADERS)
    response_2 = client.post("/predict", json=VALID_EMPLOYEE, headers=API_KEY_HEADERS)
    assert response_1.json() == response_2.json()


def test_predict_high_risk_profile_has_higher_probability_than_low_risk_profile():
    """Test de cohérence métier : un profil cumulant des facteurs de risque connus
    (heures sup, faible satisfaction, pas de PEE) doit recevoir une probabilité de
    départ plus élevée qu'un profil aux facteurs protecteurs (d'après l'analyse SHAP
    du notebook d'origine)."""
    high_risk = dict(VALID_EMPLOYEE)
    high_risk.update({
        "heure_supplementaires": "Oui",
        "satisfaction_employee_environnement": 1,
        "satisfaction_employee_nature_travail": 1,
        "satisfaction_employee_equipe": 1,
        "satisfaction_employee_equilibre_pro_perso": 1,
        "nombre_participation_pee": 0,
        "revenu_mensuel": 2500,
    })

    low_risk = dict(VALID_EMPLOYEE)
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
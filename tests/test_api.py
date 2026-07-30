from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200


def test_health_returns_ok_status():
    response = client.get("/health")
    assert response.json() == {"status": "ok"}


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


def test_predict_returns_200_with_valid_data():
    response = client.post("/predict", json=VALID_EMPLOYEE)
    assert response.status_code == 200


def test_predict_returns_expected_fields():
    response = client.post("/predict", json=VALID_EMPLOYEE)
    body = response.json()
    assert "risque_depart" in body
    assert "probabilite_depart" in body
    assert isinstance(body["risque_depart"], bool)
    assert 0.0 <= body["probabilite_depart"] <= 1.0


def test_predict_rejects_invalid_genre():
    invalid_data = dict(VALID_EMPLOYEE)
    invalid_data["genre"] = "X"  # valeur non autorisée par le schéma Pydantic
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422


def test_predict_rejects_missing_field():
    incomplete_data = dict(VALID_EMPLOYEE)
    del incomplete_data["age"]  # champ obligatoire manquant
    response = client.post("/predict", json=incomplete_data)
    assert response.status_code == 422


def test_predict_rejects_negative_age():
    invalid_data = dict(VALID_EMPLOYEE)
    invalid_data["age"] = -5  # viole la contrainte ge=18
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422
"""
Tests unitaires du schéma Pydantic EmployeeInput.

Contrairement aux tests fonctionnels de test_predict_api.py, on instancie
directement le modèle Pydantic ici, sans passer par l'API ni le modèle ML.
Cela isole la validation des données de tout le reste du système.
"""

import pytest
from pydantic import ValidationError

from app.schemas import EmployeeInput

VALID_DATA = {
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


def test_valid_data_is_accepted():
    """Un profil valide ne doit lever aucune erreur."""
    employee = EmployeeInput(**VALID_DATA)
    assert employee.age == 41


@pytest.mark.parametrize(
    "field, invalid_value",
    [
        ("age", 5),  # trop jeune (< 18)
        ("age", 150),  # trop âgé (> 70)
        ("genre", "X"),  # valeur hors Literal["F", "M"]
        ("departement", "Marketing"),  # département inexistant
        ("poste", "PDG"),  # poste inexistant
        ("heure_supplementaires", "yes"),  # doit être "Oui" ou "Non", pas "yes"
        ("augmentation_salaire_precedente", "9 %"),  # hors plage connue (11%-25%)
        ("frequence_deplacement", "Souvent"),  # valeur hors Literal
        ("revenu_mensuel", -100),  # revenu négatif
        ("satisfaction_employee_environnement", 10),  # hors échelle 1-4
        ("niveau_education", 0),  # hors échelle 1-5
    ],
)
def test_invalid_field_is_rejected(field, invalid_value):
    """Chaque contrainte du schéma doit rejeter une valeur qui la viole."""
    data = dict(VALID_DATA)
    data[field] = invalid_value
    with pytest.raises(ValidationError):
        EmployeeInput(**data)


def test_missing_required_field_is_rejected():
    """Un champ obligatoire manquant doit être rejeté."""
    data = dict(VALID_DATA)
    del data["age"]
    with pytest.raises(ValidationError):
        EmployeeInput(**data)


def test_wrong_type_is_rejected():
    """Un type incorrect (texte au lieu d'un entier) doit être rejeté."""
    data = dict(VALID_DATA)
    data["age"] = "quarante et un"
    with pytest.raises(ValidationError):
        EmployeeInput(**data)
"""
Tests unitaires du schéma Pydantic EmployeeInput.

Contrairement aux tests fonctionnels de test_predict_api.py, on instancie
directement le modèle Pydantic ici, sans passer par l'API ni le modèle ML.
Cela isole la validation des données de tout le reste du système.
"""

import pytest
from pydantic import ValidationError

from app.schemas import EmployeeInput

from .conftest import VALID_EMPLOYEE_DATA


def test_valid_data_is_accepted():
    """Un profil valide ne doit lever aucune erreur."""
    employee = EmployeeInput(**VALID_EMPLOYEE_DATA)
    assert employee.age == 41


@pytest.mark.parametrize(
    "field, invalid_value",
    [
        ("age", 5),  # trop jeune (< 18)
        ("age", 150),  # trop âgé (> 70)
        ("age", "quarante et un"),  # type incorrect (texte au lieu d'un entier)
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
    data = dict(VALID_EMPLOYEE_DATA)
    data[field] = invalid_value
    with pytest.raises(ValidationError):
        EmployeeInput(**data)


def test_missing_required_field_is_rejected():
    """Un champ obligatoire manquant doit être rejeté."""
    data = dict(VALID_EMPLOYEE_DATA)
    del data["age"]
    with pytest.raises(ValidationError):
        EmployeeInput(**data)
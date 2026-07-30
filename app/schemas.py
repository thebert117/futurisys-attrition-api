"""
Schémas Pydantic pour la validation des données de l'API.

Le schéma EmployeeInput reprend exactement les colonnes brutes attendues
par le pipeline de preprocessing du modèle (voir ml/train_model.py).
"""

from typing import Literal

from pydantic import BaseModel, Field


class EmployeeInput(BaseModel):
    """Données d'un salarié, telles qu'attendues en entrée du modèle."""

    age: int = Field(..., ge=18, le=70, description="Âge du salarié")
    genre: Literal["F", "M"]
    revenu_mensuel: int = Field(..., gt=0, description="Revenu mensuel en euros")
    statut_marital: Literal["Célibataire", "Divorcé(e)", "Marié(e)"]
    departement: Literal["Commercial", "Consulting", "Ressources Humaines"]
    poste: Literal[
        "Assistant de Direction",
        "Cadre Commercial",
        "Consultant",
        "Directeur Technique",
        "Manager",
        "Représentant Commercial",
        "Ressources Humaines",
        "Senior Manager",
        "Tech Lead",
    ]
    nombre_experiences_precedentes: int = Field(..., ge=0)
    annee_experience_totale: int = Field(..., ge=0)
    annees_dans_l_entreprise: int = Field(..., ge=0)
    annees_dans_le_poste_actuel: int = Field(..., ge=0)
    satisfaction_employee_environnement: int = Field(..., ge=1, le=4)
    note_evaluation_precedente: int = Field(..., ge=1, le=4)
    niveau_hierarchique_poste: int = Field(..., ge=1)
    satisfaction_employee_nature_travail: int = Field(..., ge=1, le=4)
    satisfaction_employee_equipe: int = Field(..., ge=1, le=4)
    satisfaction_employee_equilibre_pro_perso: int = Field(..., ge=1, le=4)
    note_evaluation_actuelle: int = Field(..., ge=1, le=4)
    heure_supplementaires: Literal["Oui", "Non"]
    augmentation_salaire_precedente: Literal[
        "11 %", "12 %", "13 %", "14 %", "15 %", "16 %", "17 %", "18 %",
        "19 %", "20 %", "21 %", "22 %", "23 %", "24 %", "25 %",
    ]
    nombre_participation_pee: int = Field(..., ge=0)
    nb_formations_suivies: int = Field(..., ge=0)
    distance_domicile_travail: int = Field(..., ge=0)
    niveau_education: int = Field(..., ge=1, le=5)
    domaine_etude: Literal[
        "Autre",
        "Entrepreunariat",
        "Infra & Cloud",
        "Marketing",
        "Ressources Humaines",
        "Transformation Digitale",
    ]
    frequence_deplacement: Literal["Aucun", "Frequent", "Occasionnel"]
    annees_depuis_la_derniere_promotion: int = Field(..., ge=0)
    annees_sous_responsable_actuel: int = Field(..., ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredictionOutput(BaseModel):
    """Réponse renvoyée par l'endpoint /predict."""

    risque_depart: bool = Field(..., description="True si le salarié est prédit à risque de départ")
    probabilite_depart: float = Field(..., description="Probabilité de départ estimée par le modèle (entre 0 et 1)")
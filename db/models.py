"""
Modèles SQLAlchemy décrivant les tables de la base de données Futurisys.

- Employee : le dataset RH
- PredictionLog : un enregistrement par appel à /predict (input + output)
"""

from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Employee(Base):
    """Un salarié du dataset RH d'origine (import du CSV)."""

    __tablename__ = "employees"

    id_employee = Column(Integer, primary_key=True)
    age = Column(Integer, nullable=False)
    genre = Column(String(1), nullable=False)
    revenu_mensuel = Column(Integer, nullable=False)
    statut_marital = Column(String(50), nullable=False)
    departement = Column(String(50), nullable=False)
    poste = Column(String(50), nullable=False)
    nombre_experiences_precedentes = Column(Integer, nullable=False)
    annee_experience_totale = Column(Integer, nullable=False)
    annees_dans_l_entreprise = Column(Integer, nullable=False)
    annees_dans_le_poste_actuel = Column(Integer, nullable=False)
    satisfaction_employee_environnement = Column(Integer, nullable=False)
    note_evaluation_precedente = Column(Integer, nullable=False)
    niveau_hierarchique_poste = Column(Integer, nullable=False)
    satisfaction_employee_nature_travail = Column(Integer, nullable=False)
    satisfaction_employee_equipe = Column(Integer, nullable=False)
    satisfaction_employee_equilibre_pro_perso = Column(Integer, nullable=False)
    note_evaluation_actuelle = Column(Integer, nullable=False)
    heure_supplementaires = Column(String(3), nullable=False)
    augmentation_salaire_precedente = Column(String(5), nullable=False)
    nombre_participation_pee = Column(Integer, nullable=False)
    nb_formations_suivies = Column(Integer, nullable=False)
    distance_domicile_travail = Column(Integer, nullable=False)
    niveau_education = Column(Integer, nullable=False)
    domaine_etude = Column(String(50), nullable=False)
    frequence_deplacement = Column(String(20), nullable=False)
    annees_depuis_la_derniere_promotion = Column(Integer, nullable=False)
    annees_sous_responsable_actuel = Column(Integer, nullable=False)
    a_quitte_l_entreprise = Column(String(3), nullable=True)
    target_attrition = Column(Integer, nullable=True)


class PredictionLog(Base):
    """Trace de chaque appel à l'endpoint /predict : input envoyé + output du modèle."""

    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    # Mêmes colonnes d'entrée que le schéma Pydantic EmployeeInput
    age = Column(Integer, nullable=False)
    genre = Column(String(1), nullable=False)
    revenu_mensuel = Column(Integer, nullable=False)
    statut_marital = Column(String(50), nullable=False)
    departement = Column(String(50), nullable=False)
    poste = Column(String(50), nullable=False)
    nombre_experiences_precedentes = Column(Integer, nullable=False)
    annee_experience_totale = Column(Integer, nullable=False)
    annees_dans_l_entreprise = Column(Integer, nullable=False)
    annees_dans_le_poste_actuel = Column(Integer, nullable=False)
    satisfaction_employee_environnement = Column(Integer, nullable=False)
    note_evaluation_precedente = Column(Integer, nullable=False)
    niveau_hierarchique_poste = Column(Integer, nullable=False)
    satisfaction_employee_nature_travail = Column(Integer, nullable=False)
    satisfaction_employee_equipe = Column(Integer, nullable=False)
    satisfaction_employee_equilibre_pro_perso = Column(Integer, nullable=False)
    note_evaluation_actuelle = Column(Integer, nullable=False)
    heure_supplementaires = Column(String(3), nullable=False)
    augmentation_salaire_precedente = Column(String(5), nullable=False)
    nombre_participation_pee = Column(Integer, nullable=False)
    nb_formations_suivies = Column(Integer, nullable=False)
    distance_domicile_travail = Column(Integer, nullable=False)
    niveau_education = Column(Integer, nullable=False)
    domaine_etude = Column(String(50), nullable=False)
    frequence_deplacement = Column(String(20), nullable=False)
    annees_depuis_la_derniere_promotion = Column(Integer, nullable=False)
    annees_sous_responsable_actuel = Column(Integer, nullable=False)

    # Output du modèle
    risque_depart = Column(Boolean, nullable=False)
    probabilite_depart = Column(Float, nullable=False)
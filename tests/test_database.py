"""
Tests de la base de données : structure des tables, intégrité des données,
et comportement des scripts d'initialisation/insertion.

Contrairement aux autres tests, ceux-ci vérifient directement la BDD
(via SQLAlchemy), indépendamment de l'API.
"""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError

from db import insert_dataset
from db.create_db import create_tables
from db.database import SessionLocal, engine
from db.models import Employee, PredictionLog


def test_create_tables_is_idempotent():
    """Relancer create_tables() ne doit pas lever d'erreur si les tables existent déjà
    (Base.metadata.create_all est conçu pour être idempotent)."""
    create_tables()  # ne doit pas planter, même appelé après la création initiale
    inspector = inspect(engine)
    assert "employees" in inspector.get_table_names()
    assert "prediction_logs" in inspector.get_table_names()


def test_employees_table_exists():
    inspector = inspect(engine)
    assert "employees" in inspector.get_table_names()


def test_prediction_logs_table_exists():
    inspector = inspect(engine)
    assert "prediction_logs" in inspector.get_table_names()


def test_employees_table_has_expected_columns():
    inspector = inspect(engine)
    columns = {col["name"] for col in inspector.get_columns("employees")}
    expected = {"id_employee", "age", "genre", "revenu_mensuel", "departement", "poste"}
    assert expected.issubset(columns)


def test_employees_id_employee_is_primary_key():
    inspector = inspect(engine)
    pk = inspector.get_pk_constraint("employees")
    assert pk["constrained_columns"] == ["id_employee"]


def test_insert_dataset_populates_employees_table():
    """Le script d'insertion doit remplir la table avec le dataset complet (1470 lignes)."""
    insert_dataset.main()

    session = SessionLocal()
    try:
        count = session.query(Employee).count()
    finally:
        session.close()

    assert count == 1470


def test_insert_dataset_is_idempotent():
    """Relancer le script une deuxième fois ne doit PAS dupliquer les données
    (protection contre les doublons déjà présente dans insert_dataset.py)."""
    insert_dataset.main()  # premier appel (peut être un no-op si déjà peuplé)
    insert_dataset.main()  # deuxième appel : ne doit rien ajouter de plus

    session = SessionLocal()
    try:
        count = session.query(Employee).count()
    finally:
        session.close()

    assert count == 1470


def test_employee_missing_required_field_raises_integrity_error():
    """La contrainte NOT NULL doit être appliquée par la base elle-même,
    indépendamment de toute validation applicative (Pydantic)."""
    session = SessionLocal()
    try:
        incomplete_employee = Employee(
            id_employee=999999,
            # 'age' volontairement omis : colonne nullable=False
            genre="F",
            revenu_mensuel=5000,
            statut_marital="Célibataire",
            departement="Commercial",
            poste="Consultant",
            nombre_experiences_precedentes=0,
            annee_experience_totale=0,
            annees_dans_l_entreprise=0,
            annees_dans_le_poste_actuel=0,
            satisfaction_employee_environnement=1,
            note_evaluation_precedente=1,
            niveau_hierarchique_poste=1,
            satisfaction_employee_nature_travail=1,
            satisfaction_employee_equipe=1,
            satisfaction_employee_equilibre_pro_perso=1,
            note_evaluation_actuelle=1,
            heure_supplementaires="Non",
            augmentation_salaire_precedente="11 %",
            nombre_participation_pee=0,
            nb_formations_suivies=0,
            distance_domicile_travail=0,
            niveau_education=1,
            domaine_etude="Autre",
            frequence_deplacement="Aucun",
            annees_depuis_la_derniere_promotion=0,
            annees_sous_responsable_actuel=0,
        )
        session.add(incomplete_employee)
        with pytest.raises(IntegrityError):
            session.commit()
    finally:
        session.rollback()
        session.close()


def test_prediction_log_created_at_is_set_automatically():
    """created_at doit être renseigné automatiquement, sans intervention explicite."""
    session = SessionLocal()
    try:
        log = PredictionLog(
            age=30, genre="M", revenu_mensuel=3000, statut_marital="Célibataire",
            departement="Commercial", poste="Consultant",
            nombre_experiences_precedentes=0, annee_experience_totale=5,
            annees_dans_l_entreprise=2, annees_dans_le_poste_actuel=1,
            satisfaction_employee_environnement=2, note_evaluation_precedente=2,
            niveau_hierarchique_poste=1, satisfaction_employee_nature_travail=2,
            satisfaction_employee_equipe=2, satisfaction_employee_equilibre_pro_perso=2,
            note_evaluation_actuelle=2, heure_supplementaires="Non",
            augmentation_salaire_precedente="11 %", nombre_participation_pee=0,
            nb_formations_suivies=0, distance_domicile_travail=5, niveau_education=2,
            domaine_etude="Autre", frequence_deplacement="Aucun",
            annees_depuis_la_derniere_promotion=0, annees_sous_responsable_actuel=1,
            risque_depart=False, probabilite_depart=0.1,
        )
        session.add(log)
        session.commit()
        session.refresh(log)

        assert log.created_at is not None
        # L'horodatage doit être récent (quelques secondes maximum), preuve
        # qu'il a bien été généré à l'insertion et non laissé à une valeur par défaut fixe.
        now = datetime.now(timezone.utc)
        created_at = log.created_at.replace(tzinfo=timezone.utc) if log.created_at.tzinfo is None else log.created_at
        assert now - created_at < timedelta(minutes=1)

        # Nettoyage : on supprime la ligne de test pour ne pas polluer la table
        session.delete(log)
        session.commit()
    finally:
        session.close()
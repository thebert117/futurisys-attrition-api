"""
Tests de la base de données : structure des tables, intégrité des données,
et comportement des scripts d'initialisation/insertion.

Contrairement aux autres tests, ceux-ci vérifient directement la BDD
(via SQLAlchemy), indépendamment de l'API.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError

from db import insert_dataset
from db.create_db import create_tables
from db.database import SessionLocal, engine
from db.models import Employee, PredictionLog

from .conftest import VALID_EMPLOYEE_DATA

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "technova_hr_clean.csv"


def test_create_tables_is_idempotent():
    create_tables()
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
    if not DATA_PATH.exists():
        pytest.skip("technova_hr_clean.csv absent (fichier non versionné, volontairement exclu de Git)")

    insert_dataset.main()

    session = SessionLocal()
    try:
        count = session.query(Employee).count()
    finally:
        session.close()

    assert count == 1470


def test_insert_dataset_is_idempotent():
    if not DATA_PATH.exists():
        pytest.skip("technova_hr_clean.csv absent (fichier non versionné, volontairement exclu de Git)")

    insert_dataset.main()
    insert_dataset.main()

    session = SessionLocal()
    try:
        count = session.query(Employee).count()
    finally:
        session.close()

    assert count == 1470


def test_employee_missing_required_field_raises_integrity_error():
    """La contrainte NOT NULL doit être appliquée par la base elle-même,
    indépendamment de toute validation applicative (Pydantic)."""
    incomplete_data = dict(VALID_EMPLOYEE_DATA)
    del incomplete_data["age"]  # colonne nullable=False volontairement omise

    session = SessionLocal()
    try:
        incomplete_employee = Employee(id_employee=999999, **incomplete_data)
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
            **VALID_EMPLOYEE_DATA,
            risque_depart=False,
            probabilite_depart=0.1,
        )
        session.add(log)
        session.commit()
        session.refresh(log)

        assert log.created_at is not None
        now = datetime.now(timezone.utc)
        created_at = log.created_at.replace(tzinfo=timezone.utc) if log.created_at.tzinfo is None else log.created_at
        assert now - created_at < timedelta(minutes=1)

        session.delete(log)
        session.commit()
    finally:
        session.close()
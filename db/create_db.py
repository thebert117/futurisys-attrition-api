"""
Script de création de la base de données PostgreSQL et de ses tables.

Usage : python db/create_db.py
"""

import os

import psycopg2
from dotenv import load_dotenv
from sqlalchemy import create_engine

from db.models import Base

load_dotenv()

DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "futurisys_attrition")


def create_database_if_not_exists():
    """Se connecte à la base 'postgres' par défaut pour créer notre base si besoin."""
    conn = psycopg2.connect(
        dbname="postgres", user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT,
    )
    conn.autocommit = True  # nécessaire pour pouvoir exécuter CREATE DATABASE
    cursor = conn.cursor()

    cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,))
    exists = cursor.fetchone()

    if not exists:
        cursor.execute(f'CREATE DATABASE "{DB_NAME}"')
        print(f"Base de données '{DB_NAME}' créée.")
    else:
        print(f"La base de données '{DB_NAME}' existe déjà, rien à faire.")

    cursor.close()
    conn.close()


def create_tables():
    """Crée les tables définies dans db/models.py, si elles n'existent pas déjà."""
    db_url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    print("Tables créées (ou déjà existantes) : employees, prediction_logs")


if __name__ == "__main__":
    create_database_if_not_exists()
    create_tables()
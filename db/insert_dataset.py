"""
Insère le dataset technova_hr_clean.csv dans la table employees.

Usage : python -m db.insert_dataset
"""

from pathlib import Path

import pandas as pd

from db.database import SessionLocal
from db.models import Employee


def main():
    csv_path = Path(__file__).resolve().parent.parent / "data" / "technova_hr_clean.csv"
    df = pd.read_csv(csv_path)

    session = SessionLocal()

    existing_count = session.query(Employee).count()
    if existing_count > 0:
        print(f"La table employees contient déjà {existing_count} lignes. Import annulé pour éviter les doublons.")
        print("Si tu veux réimporter, vide d'abord la table (voir README).")
        session.close()
        return

    employees = [Employee(**row.to_dict()) for _, row in df.iterrows()]
    session.bulk_save_objects(employees)
    session.commit()

    count = session.query(Employee).count()
    print(f"{count} salariés insérés dans la table employees.")

    session.close()


if __name__ == "__main__":
    main()
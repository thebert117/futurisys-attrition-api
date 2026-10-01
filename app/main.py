from pathlib import Path

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException

from app.schemas import EmployeeInput, PredictionOutput
from app.security import verify_api_key
from db.database import SessionLocal
from db.models import PredictionLog

app = FastAPI(title="Futurisys Attrition API")

MODEL_PATH = Path(__file__).resolve().parent.parent / "ml" / "artifacts" / "attrition_model.joblib"

# Le modèle est chargé une seule fois au démarrage de l'API, pas à chaque requête
# (le charger à chaque appel serait lent et inutile, il ne change pas entre deux requêtes)
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput, dependencies=[Depends(verify_api_key)])
def predict(employee: EmployeeInput):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non disponible. Avez-vous lancé train.py ?",
        )

    # Le modèle chargé encapsule tout : feature engineering, preprocessing et
    # classification. On lui passe directement les données brutes validées par Pydantic.
    input_df = pd.DataFrame([employee.model_dump()])

    probabilite_brute = model.predict_proba(input_df)[0, 1]

    # Calculées une seule fois, réutilisées pour la réponse ET l'enregistrement
    # en base, pour éviter toute divergence entre les deux.
    risque_depart = bool(probabilite_brute >= 0.5)
    probabilite_depart = round(float(probabilite_brute), 4)

    # Enregistrement systématique de l'input et de l'output en base de données.
    # Ce logging ne doit jamais empêcher l'API de répondre : si la base est
    # injoignable (ex. déploiement public alors que la BDD reste en local,
    # conformément à l'énoncé), on avertit dans les logs sans faire planter la requête.
    try:
        session = SessionLocal()
        try:
            log_entry = PredictionLog(
                **employee.model_dump(),
                risque_depart=risque_depart,
                probabilite_depart=probabilite_depart,
            )
            session.add(log_entry)
            session.commit()
        finally:
            session.close()
    except Exception as e:
        print(f"[avertissement] Échec de l'enregistrement en base de données : {e}")

    return PredictionOutput(
        risque_depart=risque_depart,
        probabilite_depart=probabilite_depart,
    )
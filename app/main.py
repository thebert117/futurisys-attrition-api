from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import EmployeeInput, PredictionOutput
from db.database import SessionLocal
from db.models import PredictionLog
from ml.train_model import add_engineered_features

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


@app.post("/predict", response_model=PredictionOutput)
def predict(employee: EmployeeInput):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non disponible. Avez-vous lancé ml/train_model.py ?",
        )

    # On convertit les données validées par Pydantic en DataFrame à une ligne,
    # exactement comme le format attendu par le pipeline scikit-learn
    input_df = pd.DataFrame([employee.model_dump()])

    # On applique le même feature engineering que lors de l'entraînement
    input_df = add_engineered_features(input_df)

    probabilite = model.predict_proba(input_df)[0, 1]
    risque = probabilite >= 0.5

    # Enregistrement systématique de l'input et de l'output en base de données.
    # Ce logging ne doit jamais empêcher l'API de répondre : si la base est
    # injoignable (ex. déploiement public alors que la BDD reste en local,
    # conformément à l'énoncé), on avertit dans les logs sans faire planter la requête.
    try:
        session = SessionLocal()
        try:
            log_entry = PredictionLog(
                **employee.model_dump(),
                risque_depart=bool(risque),
                probabilite_depart=round(float(probabilite), 4),
            )
            session.add(log_entry)
            session.commit()
        finally:
            session.close()
    except Exception as e:
        print(f"[avertissement] Échec de l'enregistrement en base de données : {e}")

    return PredictionOutput(
        risque_depart=bool(risque),
        probabilite_depart=round(float(probabilite), 4),
    )
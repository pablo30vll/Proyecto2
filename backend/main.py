from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SodAI Drinks API 🥤",
    description="API para predicciones de compras de productos SodAI",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "/app/models/model.pkl"
model = None

# ==== 1. SOLO LOS 10 FEATURES ====
class PredictionRequest(BaseModel):
    customer_type: str
    Y: float
    X: float
    num_deliver_per_week: int
    brand: str
    sub_category: str
    segment: str
    package: str
    size: float
    week_num: int

class BulkPredictionRequest(BaseModel):
    data: List[PredictionRequest]

@app.on_event("startup")
async def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        logger.info("✅ Modelo cargado correctamente.")
    else:
        logger.error("❌ No se encontró el modelo en /app/models/model.pkl")

@app.get("/")
def root():
    return {"message": "SodAI Drinks API 🥤", "status": "active", "model_loaded": model is not None}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado")
    try:
        features = {
            'Y': request.Y,
            'X': request.X,
            'num_deliver_per_week': request.num_deliver_per_week,
            'size': request.size,
            'customer_type': request.customer_type,
            'brand': request.brand,
            'sub_category': request.sub_category,
            'segment': request.segment,
            'package': request.package,
            'week_num': request.week_num,
        }
        df = pd.DataFrame([features])
        if 'week_num' in df.columns:
            df = df.rename(columns={'week_num': 'week_of_year'})
        probability = model.predict_proba(df)[0, 1]
        prediction = probability > 0.5
        return {
            "probability": float(probability),
            "prediction": bool(prediction),
            "week": request.week_num
        }
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

# ==== 2. BULK PREDICTION ====
@app.post("/predict/bulk")
def predict_bulk(request: BulkPredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado")
    try:
        df = pd.DataFrame([r.dict() for r in request.data])
        proba = model.predict_proba(df)[:, 1]
        preds = proba > 0.5
        results = []
        for i, row in enumerate(df.itertuples(index=False)):
            results.append({
                "probability": float(proba[i]),
                "prediction": bool(preds[i]),
                "week": getattr(row, "week_num", 0)
            })
        total = len(results)
        positive = sum(r["prediction"] for r in results)
        return {
            "predictions": results,
            "total_predictions": total,
            "positive_predictions": positive
        }
    except Exception as e:
        logger.error(f"Error en predicción masiva: {e}")
        raise HTTPException(status_code=500, detail=f"Error en predicción bulk: {str(e)}")

@app.get("/model/info")
def get_model_info():
    if model is None:
        return {"model_loaded": False, "message": "No hay modelo cargado"}
    return {
        "model_loaded": True,
        "model_type": str(type(model).__name__)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_pipeline.pkl")

pipeline = joblib.load(MODEL_PATH)

app = FastAPI()

@app.get("/")
def home():
    return {
        "message": "Customer Churn API"
    }

class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict")
def predict(customer: Customer):

    data = customer.model_dump()

    
    if data["tenure"] < 0:
        return {"error": "tenure cannot be negative"}

    if data["MonthlyCharges"] < 0:
        return {"error": "invalid monthly charges"}

    customer_df = pd.DataFrame([data])

    probability = pipeline.predict_proba(customer_df)[0][1]
    prediction = int(probability >= 0.5)

    logger.info(f"Received request: {data}")
    logger.info(f"Churn probability: {probability}")

    risk_level = (
    "high" if probability > 0.7 else
    "medium" if probability > 0.4 else
    "low"
)

    return {        
        "churn_probability": round(float(probability), 4),
        "risk_level": risk_level
    }
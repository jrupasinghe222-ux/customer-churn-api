# Customer Churn Prediction API

## Overview
This project predicts customer churn using machine learning and exposes predictions via a FastAPI service.

## Tech Stack
- Python
- Scikit-learn
- Pandas
- FastAPI
- Uvicorn

## ML Approach
- Logistic Regression with class balancing
- One-hot encoding for categorical variables
- Pipeline-based preprocessing
- Evaluation using precision, recall, F1-score

## API Endpoints

### POST /predict

Input:
```json
{
  "gender": "Male",
  "tenure": 12,
  ...
}

Output:

{
  "churn_probability": 0.78,
  "risk_level": "high"
}

Run Locally
uvicorn api.main:app --reload

Model Performance:
F1-score improved using class balancing
Focus on recall for churn detection
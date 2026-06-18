# Customer Churn Prediction System

## Overview

This project predicts whether a telecom customer will churn using machine learning and exposes predictions via an API and web interface.

## Live Demo
https://customer-churn-api.streamlit.app

## Tech Stack

- Python
- Scikit-learn
- Pandas
- FastAPI
- Streamlit
- Uvicorn

## Problem Statement

Telecom companies lose revenue when customers leave. The goal is to predict churn early so businesses can take preventive actions.

## Machine Learning Approach

- Logistic Regression (final model)
- Class imbalance handled using `class_weight="balanced"`
- One-hot encoding for categorical features
- Pipeline used for preprocessing + model bundling


## Evaluation Metrics

Focus was on churn class (minority class):

- Precision
- Recall
- F1-score

Final model prioritized recall to reduce missed churners.

## API Endpoints

### POST /predict

#### Input

```json
{
  "gender": "Male",
  "tenure": 12,
  "Contract": "Month-to-month"
}
```

<img width="500" height="372" alt="image" src="https://github.com/user-attachments/assets/59802f9e-bd40-4c78-8d9b-dcb366ec18d1" />

#### Output

```json
{
  "churn_probability": 0.78,
  "risk_level": "high"
}
```

<img width="526" height="266" alt="image" src="https://github.com/user-attachments/assets/34b2a823-2004-4a40-99c5-9795b46e2847" />


## Streamlit UI

Interactive dashboard for:

- Entering customer data
- Getting real-time churn prediction
- Viewing risk level

## How to Run Locally

### Run Streamlit App

```bash
pip install -r requirements.txt

streamlit run streamlit_app.py
```

### Run FastAPI API

```bash
pip install -r requirements.txt

uvicorn api.main:app --reload
```

## System Architecture
<img width="500" height="750" alt="architecture diagram customer churn" src="https://github.com/user-attachments/assets/d45b3d96-cd2b-42bf-af56-d56c9572bc42" />


## Key Learnings

- End-to-end ML pipeline design
- Handling imbalanced datasets
- Model evaluation beyond accuracy
- API development with FastAPI
- ML deployment concepts
- Docker containerization
- Building interactive ML UI

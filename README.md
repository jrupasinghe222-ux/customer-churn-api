# 📌 Customer Churn Prediction System

## 🔍 Overview

This project predicts whether a telecom customer will churn using machine learning and exposes predictions via an API and web interface.

---

## ⚙️ Tech Stack

- Python
- Scikit-learn
- Pandas
- FastAPI
- Streamlit
- Uvicorn

---

## 📊 Problem Statement

Telecom companies lose revenue when customers leave. The goal is to predict churn early so businesses can take preventive actions.

---

## 🧠 Machine Learning Approach

- Logistic Regression (final model)
- Class imbalance handled using `class_weight="balanced"`
- One-hot encoding for categorical features
- Pipeline used for preprocessing + model bundling

---

## 📈 Evaluation Metrics

Focus was on churn class (minority class):

- Precision
- Recall
- F1-score

Final model prioritized recall to reduce missed churners.

---

## 🚀 API Endpoints

### POST /predict

#### Input

```json
{
  "gender": "Male",
  "tenure": 12,
  "Contract": "Month-to-month"
}
```

#### Output

```json
{
  "churn_probability": 0.78,
  "risk_level": "high"
}
```

---

## 🖥️ Streamlit UI

Interactive dashboard for:

- Entering customer data
- Getting real-time churn prediction
- Viewing risk level

---

## ▶️ How to Run Locally

```bash
pip install -r requirements.txt

uvicorn api.main:app --reload

streamlit run streamlit_app.py
```

---

## 📌 Key Learnings

- End-to-end ML pipeline design
- Handling imbalanced datasets
- Model evaluation beyond accuracy
- API development with FastAPI
- ML deployment concepts
- Building interactive ML UI
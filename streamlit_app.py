import streamlit as st
import joblib
import pandas as pd

pipeline = joblib.load("models/churn_pipeline.pkl")

st.set_page_config(page_title="Customer Churn Predictor")

st.title("Customer Churn Prediction")

# Demographics
gender = st.selectbox("Gender", ["Male", "Female"])

SeniorCitizen = st.selectbox(
    "Senior Citizen",
    [0, 1]
)

Partner = st.selectbox(
    "Partner",
    ["Yes", "No"]
)

Dependents = st.selectbox(
    "Dependents",
    ["Yes", "No"]
)

# Account Information
tenure = st.number_input(
    "Tenure (Months)",
    min_value=0,
    max_value=100,
    value=12
)

# Phone Services
PhoneService = st.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

MultipleLines = st.selectbox(
    "Multiple Lines",
    ["Yes", "No", "No phone service"]
)

# Internet Services
InternetService = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

# Online Features
OnlineSecurity = st.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)

OnlineBackup = st.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"]
)

DeviceProtection = st.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"]
)

TechSupport = st.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)

StreamingTV = st.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
)

StreamingMovies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)

# Billing Information
Contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

PaperlessBilling = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

MonthlyCharges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

TotalCharges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=1000.0
)

if st.button("Predict Churn"):

    payload = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }

    df = pd.DataFrame([payload])

    probability = pipeline.predict_proba(df)[0][1]

    if probability < 0.4:
        risk_level = "low"
    elif probability < 0.7:
        risk_level = "medium"
    else:
        risk_level = "high"

    st.subheader("Prediction Result")

    st.metric(
        "Churn Probability",
        f"{probability:.2%}"
    )


    if risk_level == "low":
        st.success("🟢 Low Risk Customer")

    elif risk_level == "medium":
        st.warning("🟡 Medium Risk Customer")

    else:
        st.error("🔴 High Risk Customer")
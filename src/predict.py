import joblib
import pandas as pd

pipeline = joblib.load(
    "../models/churn_pipeline.pkl"
)

new_customer = pd.DataFrame([
    {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 100,
        "TotalCharges": 100
    }
])

prediction = pipeline.predict(
    new_customer
)

print(prediction)


probability = pipeline.predict_proba(
    new_customer
)

churn_probability = probability[0][1]

print(
    f"Churn Probability: {churn_probability:.2%}"
)
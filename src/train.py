import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# load dataset
df = pd.read_csv("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Data Preprocessing

# Converting data type to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Removing rows with null values
df = df.dropna()

# removing the customer ID column
df = df.drop("customerID", axis=1)

# Encoding target variable
df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

# Split into input and output variables
X = df.drop("Churn", axis=1)
y = df["Churn"]

# encoding categorical variables
X = pd.get_dummies(X, drop_first=True)

# Split training an dtesting data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model using LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions for test data
predictions = model.predict(X_test)

# Accuracy evaluation
accuracy = accuracy_score(
    y_test,
    predictions
)

print("Accuracy:", accuracy)

print(
    classification_report(
        y_test,
        predictions
    )
)

cm = confusion_matrix(y_test, predictions)
print(cm)

# Train model using Randomforest
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Evaluation
print(
    accuracy_score(
        y_test,
        rf_predictions
    )
)

print(
    classification_report(
        y_test,
        rf_predictions
    )
)

# Feature importance

importance = rf_model.feature_importances_

feature_importance = sorted(
    zip(X.columns, importance),
    key=lambda x: x[1],
    reverse=True
)

for feature, score in feature_importance[:10]:
    print(feature, score)

# Improving model performance
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Make predictions for test data
predictions = model.predict(X_test)

# Accuracy evaluation
accuracy = accuracy_score(
    y_test,
    predictions
)

print(
    classification_report(
        y_test,
        predictions
    )
)

# save model
joblib.dump(
    model,
    "../models/churn_model.pkl"
)


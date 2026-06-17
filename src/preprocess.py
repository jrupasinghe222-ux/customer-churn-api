import pandas as pd

df = pd.read_csv(
    "../data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

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

print(df.head())

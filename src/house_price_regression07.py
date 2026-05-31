import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load dataset
houses = pd.read_csv("../data/houses.csv")

# Split columns into numerical and categorical features
num_features = houses.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = houses.select_dtypes(include=['object', 'category']).columns.tolist()

# Data imputation
num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

# Transform missing values
houses[num_features] = num_imputer.fit_transform(houses[num_features])
houses[cat_features] = cat_imputer.fit_transform(houses[cat_features])

# OUTLIER REMOVAL
# Calculate quartiles and IQR for the price variable
Q1 = houses['Cena'].quantile(0.25)
Q3 = houses['Cena'].quantile(0.75)
IQR = Q3 - Q1

# Define outlier boundaries
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove observations outside the accepted range
houses = houses[(houses['Cena'] >= lower_bound) & (houses['Cena'] <= upper_bound)]

# Initialize OneHotEncoder
# sparse_output=False -> return dense array instead of sparse matrix
# handle_unknown="ignore" -> ignore unseen categories in future data
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

# Fit and transform categorical variables
encoded = pd.DataFrame(encoder.fit_transform(houses[cat_features]))

encoded.index = houses.index
encoded.columns = encoder.get_feature_names_out(cat_features)

# Remove original categorical columns
houses = houses.drop(columns=cat_features).join(encoded)

# Split data into features and target
X = houses.drop(columns=["Cena"])
y = houses["Cena"]

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(X_train, X_test, y_train, y_test)

# Standard Scaler
scaler = StandardScaler()
num_features.remove("Cena")
X_train[num_features] = scaler.fit_transform((X_train[num_features]))
X_test[num_features] = scaler.transform(X_test[num_features])

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=1.0, max_iter=10000),
    "XGBoost": xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8, random_state=42)
}

# Słowniki do przechowywania wyników
metrics = {"Model": [], "MAE": [], "RMSE": [], "R² Score": []}
predictions = {name: None for name in models}
coefficients = {name: None for name in models}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    predictions[name] = y_predict

    metrics["Model"].append(name)
    metrics["MAE"].append(mean_absolute_error(y_test, y_predict))
    metrics["RMSE"].append(
        mean_squared_error(y_test, y_predict, squared=False)
    )
    metrics["R² Score"].append(r2_score(y_test, y_predict))

    if hasattr(model, 'coef_'):
        coefficients[name] = model.coef_

results = pd.DataFrame(metrics)

print(results)

print("\nModel coefficients:")
print(coefficients)


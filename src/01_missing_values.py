import pandas as pd
from sklearn.impute import SimpleImputer

houses = pd.read_csv("../data/houses.csv")

num_inputer = SimpleImputer(strategy="median")
cat_inputer = SimpleImputer(strategy="most_frequent")

num_features = houses.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = houses.select_dtypes(include=['object']).columns.tolist()

houses[num_features] = num_inputer.fit_transform(houses[num_features])
houses[cat_features] = cat_inputer.fit_transform(houses[cat_features])

print(houses.isnull().sum())
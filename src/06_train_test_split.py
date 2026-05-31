import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

houses = pd.read_csv("../data/houses.csv")

num_inputer = SimpleImputer(strategy="median")
cat_inputer = SimpleImputer(strategy="most_frequent")

num_features = houses.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = houses.select_dtypes(include=['object']).columns.tolist()

houses[num_features] = num_inputer.fit_transform(houses[num_features])
houses[cat_features] = cat_inputer.fit_transform(houses[cat_features])

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded = pd.DataFrame(encoder.fit_transform(houses[cat_features]))
encoded.index = houses.index
encoded.columns = encoder.get_feature_names_out(cat_features)
houses = houses.drop(columns=cat_features).join(encoded)

X = houses.drop(columns=["Cena"])
y = houses["Cena"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print(X_train, X_test, y_train, y_test)


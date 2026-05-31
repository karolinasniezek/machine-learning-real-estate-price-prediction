import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from scipy.stats import alpha
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

houses = pd.read_csv("../data/houses.csv")

num_inputer = SimpleImputer(strategy="median")
cat_inputer = SimpleImputer(strategy="most_frequent")

num_features = houses.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = houses.select_dtypes(include=['object']).columns.tolist()

houses[num_features] = num_inputer.fit_transform(houses[num_features])
houses[cat_features] = cat_inputer.fit_transform(houses[cat_features])

print(houses.isnull().sum())

plt.figure(figsize=(12,8))
sns.heatmap(houses[num_features].corr(), annot=True, cmap="coolwarm", fmt='.2f', linewidths=0.5)
plt.title("House Price Correlations")
plt.savefig('../figures/heatmap-house-price-correlations.png')

plt.figure(figsize=(8,6))
sns.histplot(houses["Cena"], bins=30, kde=True, color="blue")
plt.title("Histogram features")
plt.xlabel("Cena")
plt.ylabel("Ilosc domow")
plt.savefig("../figures/histogram-features.png")

plt.figure(figsize=(10,6))
sns.boxplot(x=houses["Jakosc"], y=houses["Cena"], color="skyblue")
plt.title("Quality vs Price")
plt.xlabel("Jakosc")
plt.ylabel("Cena")
plt.savefig("../figures/boxplot-quality-vs-price.png")

plt.figure(figsize=(8,6))
sns.scatterplot(x=houses["Powierzchnia"], y=houses["Cena"], color="purple", alpha=0.6)
plt.title("Area vs Price")
plt.xlabel("Powierzchnia")
plt.ylabel("Cena")
plt.savefig("../figures/scatterplot-area-vs-price.png")

plt.figure(figsize=(10,6))
sns.boxplot(x=houses["Cena"], color="skyblue")
plt.title("Price")
plt.savefig("../figures/boxplot-price.png")

plt.figure(figsize=(12,6))
sns.boxplot(x=houses["Sasiedztwo"], y=houses["Cena"], color="skyblue")
plt.title("Neighborhood vs Price")
plt.xlabel("Sasiedztwo")
plt.xticks(rotation=90)
plt.ylabel("Cena")
plt.savefig("../figures/boxplot-neighborhood-vs-price.png")

Q1 = houses["Cena"].quantile((0.25))
Q3 = houses["Cena"].quantile((0.75))
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_count = houses[(houses["Cena"] <= lower_bound) | (houses["Cena"] >= upper_bound)].shape[0]
houses = houses[(houses["Cena"] >= lower_bound) & (houses["Cena"] <= upper_bound)]
print(houses)
print(outliers_count)

# plt.figure(figsize=(10,6))
# sns.boxplot(x=houses["Cena"], color="skyblue")
# plt.title("Price without outliers")
# plt.savefig('../figures/boxplot-price-without-outliers.png')

houses["LogCena"] = np.log(houses["Cena"])

plt.figure(figsize=(10,6))

sns.histplot(houses["LogCena"], bins=30, kde=True, color="skyblue")

plt.title("Log-transformed House Prices")
plt.xlabel("Log Price")
plt.ylabel("Count")

plt.savefig("../figures/log-price-histogram.png")

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded = pd.DataFrame(encoder.fit_transform(houses[cat_features]))
encoded.index = houses.index
encoded.columns = encoder.get_feature_names_out(cat_features)
houses = houses.drop(columns=cat_features).join(encoded)
print(houses)

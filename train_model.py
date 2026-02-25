import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
data = pd.read_csv("crime_merged.csv")

# Encode State column
le = LabelEncoder()
data["State"] = le.fit_transform(data["State"])

# Features
X = data[["State", "Year"]]

# Target
y = data["Total Crimes"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model + encoder together
pickle.dump((model, le), open("crime_model.pkl", "wb"))

print("Model trained and saved successfully!")
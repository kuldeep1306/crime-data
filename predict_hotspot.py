import pandas as pd
import joblib

model = joblib.load("crime_model.pkl")
df = pd.read_csv("crime_merged.csv")

state_info = df.groupby("State")[["Latitude", "Longitude"]].mean().reset_index()
state = input("Enter the state name: ").strip()
year = int(input("Enter Year to predict: "))

future_input = pd.DataFrame({'Year': [year]})
pred = model.predict(future_input)[0]
murder, rape, robbery, theft, riots, total = pred

print("\n--------PREDICTION RESULT---------")
print("State:", state)
print("Year:", year)
print("Predicted Murder:", int(murder))
print("Predicted Rape:", int(rape))
print("Predicted Robbery:", int(robbery))
print("Predicted Theft:", int(theft))
print("Predicted Riots:", int(riots))
print("Predicted Total Crimes:", int(total))

low = df["Total Crimes"].quantile(0.33)
high = df["Total Crimes"].quantile(0.66)

if total <= low:
    color = "green"
else:
    color = "red"

row = state_info[state_info["State"].str.lower() == state.lower()]
if row.empty:
    print("\nState not found in the dataset.")
else:
    lat = float(row["Latitude"].values[0])
    lon = float(row["Longitude"].values[0])

    print("\n------HOTSPOT DATA FOR MAP---------")
    print("Latitude:", lat)
    print("Longitude:", lon)
    print("Color:", color)

    hotspot_output = {
        "state": state,
        "year": year,
        "total_predicted": int(total),
        "color": color,
        "latitude": lat,
        "longitude": lon
    }

    print("\nHotspot Output Data:", hotspot_output)
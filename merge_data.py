import pandas as pd
crime = pd.read_csv('Crimecases2.csv')
coords = pd.read_csv("india_state_coordinates.csv")
merged = pd.merge(crime, coords, on='State', how='left')
merged.to_csv('crime_merged.csv', index=False)
print("Merging compelete! File saved as crime_merged1.csv")
print(merged.head())
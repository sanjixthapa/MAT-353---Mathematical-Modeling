import pandas as pd
import random
from collections import Counter

# Define horse odds
horse_odds = {
    "Euler's Folly": 7,
    "Leapin' Leibniz": 5,
    "Newton Lobell": 9,
    "Count Cauchy": 12,
    "Pumped up Poisson": 4,
    "Loping L’Hôpital": 35,
    "Steamin’ Stokes": 15,
    "Dancin’ Dantzig": 4
}

# Convert odds to probabilities
horse_weights = {name: 1 / (odds + 1) for name, odds in horse_odds.items()}
names = list(horse_weights.keys())
weights = list(horse_weights.values())

# Normalize weights so their sum equals 1
weights = [w / sum(weights) for w in weights]

results = []

winners = random.choices(names, weights=weights, k=1000)
result = dict(Counter(winners))
results.append(result)

# Create DataFrame of results
df_results = pd.DataFrame(results).T.fillna(0).astype(int)

# Print the result table
print(df_results)
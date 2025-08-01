import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Data
weights_female = np.array([48, 50.5, 60.5, 75, 80])
lifts_female = np.array([185, 225, 242.5, 245, 300])

weights_male = np.array([56, 62, 69, 77, 85, 94, 105, 110])
lifts_male = np.array([305, 325, 357.5, 376.5, 390, 405, 425, 472.5])

# Model Definition
def model(weight, k):
    return k * weight**(2/3)


# Fit the models
params_female, _ = curve_fit(model, weights_female, lifts_female)
k_female = params_female[0]

params_male, _ = curve_fit(model, weights_male, lifts_male)
k_male = params_male[0]

print(f"FEMALE model: Lift ≈ {k_female:.2f} × (Weight)^(2/3)")
print(f"MALE model: Lift ≈ {k_male:.2f} × (Weight)^(2/3)")

# Plot the fits
plt.figure(figsize=(10,6))
plt.scatter(weights_female, lifts_female, label='Female Actual Data', color='hotpink', s=80)
plt.plot(weights_female, model(weights_female, k_female), color='red', label=f'Female Fit (k={k_female:.2f})')

plt.scatter(weights_male, lifts_male, label='Male Actual Data', color='skyblue', s=80)
plt.plot(weights_male, model(weights_male, k_male), color='blue', label=f'Male Fit (k={k_male:.2f})')

plt.xlabel('Body Weight (kg)', fontsize=14)
plt.ylabel('Lift (kg)', fontsize=14)
plt.title('Winning Lifts vs. Body Weight (Male and Female)', fontsize=16)
plt.grid(True)
plt.legend(fontsize=12)
plt.show()

# Predict lifts and show tables
predicted_female = model(weights_female, k_female)
predicted_male = model(weights_male, k_male)

comparison_female = pd.DataFrame({
    'Body Weight (kg)': weights_female,
    'Actual Lift (kg)': lifts_female,
    'Predicted Lift (kg)': predicted_female.round(1),
    'Error (kg)': (lifts_female - predicted_female).round(1)
})

comparison_male = pd.DataFrame({
    'Body Weight (kg)': weights_male,
    'Actual Lift (kg)': lifts_male,
    'Predicted Lift (kg)': predicted_male.round(1),
    'Error (kg)': (lifts_male - predicted_male).round(1)
})

print("\n")
print("=== FEMALE LIFTERS COMPARISON ===")
print(comparison_female.to_string(index=False))
print("\n")
print("=== MALE LIFTERS COMPARISON ===")
print(comparison_male.to_string(index=False))
print("\n")

# Find Best Performer
performance_index_female = lifts_female / weights_female**(2/3)
performance_index_male = lifts_male / weights_male**(2/3)

best_female_idx = np.argmax(performance_index_female)
best_male_idx = np.argmax(performance_index_male)

print(f"Best Female Lifter: {lifts_female[best_female_idx]} kg at {weights_female[best_female_idx]} kg body weight")
print(f"Best Male Lifter: {lifts_male[best_male_idx]} kg at {weights_male[best_male_idx]} kg body weight")

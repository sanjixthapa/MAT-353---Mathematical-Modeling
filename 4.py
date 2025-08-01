import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

# Raw data
height = np.array([60, 62, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                   71, 72, 73, 74, 75, 76, 77, 78, 79, 80])
weight = np.array([135, 138, 142, 146, 152, 154, 159, 164, 168, 173,
                   179, 183, 188, 193, 200, 207, 211, 217, 222, 227, 235])

# Reshape for scikit-learn
X = height.reshape(-1, 1)
y = weight

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Get coefficients
slope = model.coef_[0]
intercept = model.intercept_
print(f"Regression equation: y = {slope:.2f}x + {intercept:.2f}")

# Calculate predicted values
predicted = model.predict(X)

# Create results DataFrame
results = pd.DataFrame({
    'Height': height,
    'Actual_Weight': weight,
    'Predicted_Weight': predicted,
    'Residual': weight - predicted
})

# Calculate metrics
y_mean = np.mean(weight)
sst = np.sum((weight - y_mean)**2)
ssr = np.sum((predicted - y_mean)**2)
sse = np.sum((weight - predicted)**2)
r_squared = ssr / sst

# Print results
print("\n=== Regression Metrics ===")
print(f"SST (Total Sum of Squares): {sst:.2f}")
print(f"SSE (Sum of Squared Errors): {sse:.2f}")
print(f"SSR (Sum of Squares Regression): {ssr:.2f}")
print(f"R-squared: {r_squared:.4f} ({r_squared*100:.1f}%)")

# Show first few predictions
print("\n=== First 5 Predictions ===")
print(results.head())

# Plot actual vs predicted
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(height, weight, color='blue', label='Actual')
plt.plot(height, predicted, color='red', label='Predicted')
plt.title('Actual vs Predicted Weights')
plt.xlabel('Height (inches)')
plt.ylabel('Weight (lbs)')
plt.legend()

# Residual plot
plt.subplot(1, 2, 2)
plt.scatter(height, results['Residual'], color='green')
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Height (inches)')
plt.ylabel('Residuals')
plt.tight_layout()
plt.show()
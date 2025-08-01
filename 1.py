import numpy as np
import matplotlib.pyplot as plt

# Given data points
X = np.array([17, 19, 20, 22, 23, 25, 31, 32, 33, 36, 37, 38, 39, 41])
Y = np.array([19, 25, 32, 51, 57, 71, 141, 123, 187, 192, 205, 252, 248, 294])  # Corrected Y values

# Fit a 13th-degree polynomial
coefficients = np.polyfit(X, Y, 3)
poly_func = np.poly1d(coefficients)

# Generate smooth X values for plotting
X_smooth = np.linspace(min(X), max(X), 200)
Y_smooth = poly_func(X_smooth)

# Plot data and polynomial fit
plt.scatter(X, Y, color='red', label='Data points')
plt.plot(X_smooth, Y_smooth, color='blue', label='13th-degree Polynomial Fit')
plt.xlabel("X (Diameter of tree)")
plt.ylabel("Y (Board feet volume)")
plt.legend()
plt.title("13th-degree Polynomial Fit to Data")
plt.show()

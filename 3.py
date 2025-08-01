import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Given data points
x = np.array([3, 5, 9])
y = np.array([4, 12, 18])

# Compute the natural cubic spline
cs = CubicSpline(x, y, bc_type='natural')

# Generate smooth x values for plotting
x_smooth = np.linspace(min(x), max(x), 100)
y_smooth = cs(x_smooth)

# Plot data points and spline curve
plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x_smooth, y_smooth, color='green', label='Natural Cubic Spline')
plt.legend()
plt.title(" Cubic Spline Interpolation")
plt.grid(True)
plt.show()

# Extract coefficients for each spline segment
coefficients = cs.c
coefficients

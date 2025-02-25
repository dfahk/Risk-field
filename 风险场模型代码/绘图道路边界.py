import numpy as np
import matplotlib.pyplot as plt

def piecewise_function(x, gamma1, gamma2, k, w):
    if w / 2 < x < (2 * k-1) * w / 2:
        return gamma1 * np.cos(2 * np.pi * x / w) + gamma1
    elif 0 <= x <= w / 2:
        return gamma2 / x - 2 * gamma2 / w if x != 0 else np.inf  # Avoid division by zero
    elif (2 * k - 1) * w / 2 < x < k * w:
        return -gamma2 / (x - k * w) - 2 * gamma2 / w
    else:
        return None

# Parameters
gamma1 = 15  # Adjust as needed
gamma2 = 2  # Adjust as needed
k = 4       # Adjust as needed
w = 3       # Adjust as needed

# Generate x values and compute y values
x_vals = np.linspace(0, k * w, 1000)
y_vals = []

for x in x_vals:
    y = piecewise_function(x, gamma1, gamma2, k, w)
    if y is not None:
        y_vals.append(y)
    else:
        y_vals.append(np.nan)  # For undefined regions

# Plot the function
plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label="Piecewise Function")
plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
plt.axvline(0, color="black", linewidth=0.5, linestyle="--")
plt.title("Piecewise Function Plot")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()

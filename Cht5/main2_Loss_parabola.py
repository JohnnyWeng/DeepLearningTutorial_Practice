import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-10, 10, 400)
y = np.sin(x) + 0.5 * np.sin(2 * x) + 0.3 * np.cos(3 * x) + 0.1 * x

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Complex Loss Landscape', color='blue', linewidth=2)
plt.title('Illustration of a Complex Loss Landscape')
plt.xlabel('Parameter Space')
plt.ylabel('Loss')
plt.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)

# Marking local minima
local_min_indices = [50, 150, 250]
for i in local_min_indices:
    plt.plot(x[i], y[i], 'ro')
    plt.text(x[i], y[i] + 0.5, 'Local Minima', color='red', ha='center')

# Highlighting the global minimum
global_min_index = np.argmin(y)
plt.plot(x[global_min_index], y[global_min_index], 'go', label="Global Minimum")
plt.text(x[global_min_index], y[global_min_index] - 0.5, 'Global Minimum', color='green', ha='center')

plt.legend()
plt.grid(True)
plt.show()
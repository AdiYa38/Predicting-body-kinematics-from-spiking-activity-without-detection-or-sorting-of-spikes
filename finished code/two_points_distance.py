import numpy as np
import matplotlib.pyplot as plt

def analytical_pdf(r, D):
    """
    The theoretical PDF of distances between
    tworandomly distributed points in a circle with radius r
    """
    epsilon = 1e-10
    r = np.clip(r, epsilon, D - epsilon)
    
    x = r / D
    
    term1 = (16 * r / (np.pi * D**2)) * np.arccos(x)
    term2 = (16 * r**2 / (np.pi * D**3)) * np.sqrt(1 - x**2)
    
    return term1 - term2

# Constants
D = 80.0
R = D / 2.0
N = 500000 # Pairs to generate

# Run simulation
theta1 = np.random.uniform(0, 2 * np.pi, N)
r1 = R * np.sqrt(np.random.uniform(0, 1, N))
x1 = r1 * np.cos(theta1)
y1 = r1 * np.sin(theta1)

theta2 = np.random.uniform(0, 2 * np.pi, N)
r2 = R * np.sqrt(np.random.uniform(0, 1, N))
x2 = r2 * np.cos(theta2)
y2 = r2 * np.sin(theta2)

distances = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# Plot
plt.figure(figsize=(12, 7))

# Plot histogram
plt.hist(distances, bins=100, density=True, alpha=0.7, 
         label='simulation result')

# Plot theoretical PDF
r_values = np.linspace(0, D, 300)

plt.plot(r_values, pdf_values, color='red', linewidth=3, 
         label='theoretical solution')

# Add the mean
analytical_mean = (64 * D) / (45 * np.pi)
plt.axvline(analytical_mean, color='k', linestyle='--', linewidth=2, 
            label=f'theoretical mean {analytical_mean:.4f}')

plt.title(f"Distance ditribution of distances in a circle (D={D})", fontsize=16)
plt.xlabel("Distance (r)", fontsize=12)
plt.ylabel("(p(r))", fontsize=12)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.ylim(bottom=0)
plt.xlim(left=0, right=D)
plt.show()
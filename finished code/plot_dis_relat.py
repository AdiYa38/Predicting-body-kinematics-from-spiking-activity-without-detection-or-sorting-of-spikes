import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Constants
BIN_SIZE = 1
ARENA_DIAMETER = 80 
MIN_NRS = 0.4
sessions = ["mP79_11", "mP79_12", "mP79_13", "mP79_14", "mP79_15", "mP79_16", "mP79_17", "mP31_18", "mP31_19", "mP31_20"]
# --------------

def analytical_pdf(r, D):
    epsilon = 1e-10
    r = np.clip(r, epsilon, D - epsilon)
    x = r / D 
    term1 = (16 * r / (np.pi * D**2)) * np.arccos(x)
    term2 = (16 * r**2 / (np.pi * D**3)) * np.sqrt(1 - x**2)
    return term1 - term2

session_means = []
session_pair_counts = [] 
valid_session_names = []
all_pairwise_distances = []

print(f"Processing {len(sessions)} sessions...")

for sess_id in sessions:
    filename = f"data/{sess_id}/Centers_{sess_id}_bin{BIN_SIZE}cm.csv"
    try:
        df = pd.read_csv(filename)
        df_filtered = df[df['n_r_squared'] > MIN_NRS].copy()
        num_units = len(df_filtered)
        
        if num_units < 2:
            continue

        coordinates = df_filtered[['center_x_cm', 'center_y_cm']].values
        current_session_distances = []
        for i in range(len(coordinates)):
            for j in range(i + 1, len(coordinates)):
                dist = np.sqrt(np.sum((coordinates[i] - coordinates[j])**2))
                current_session_distances.append(dist)
                all_pairwise_distances.append(dist)
        
        session_means.append(np.mean(current_session_distances))
        session_pair_counts.append(len(current_session_distances)) 
        valid_session_names.append(sess_id)

    except FileNotFoundError:
        print(f"Warning: File for {sess_id} not found.")

# Calculate statistics
distances_array = np.array(all_pairwise_distances)
theoretical_mean = (128 * (ARENA_DIAMETER / 2)) / (45 * np.pi)
mean_of_means = np.mean(session_means)
weighted_mean = np.average(session_means, weights=session_pair_counts)

# ---------------------------------------------------
# --- Plot 1: Hitogram of distances distributions---
# ---------------------------------------------------
plt.figure(figsize=(12, 7))

# Plot histogram
plt.hist(distances_array, bins=50, edgecolor='black', alpha=0.6, 
         density=True, label='Empirical Combined Data')

# Add theoretical PDF
r_values = np.linspace(0, ARENA_DIAMETER, 500)
pdf_values = analytical_pdf(r_values, ARENA_DIAMETER)
plt.plot(r_values, pdf_values, color='orange', linestyle='-', linewidth=3, 
         label='Theoretical PDF (Random points in circle)')

# Mean lines
plt.axvline(weighted_mean, color='red', linestyle='--', linewidth=2, 
            label=f'Weighted Mean: {weighted_mean:.2f} cm') # Overall Empirical Mean
plt.axvline(theoretical_mean, color='green', linestyle=':', linewidth=3, 
            label=f'Theoretical Mean: {theoretical_mean:.2f} cm')

plt.title(f'Distribution of Pairwise Distances (unbiased $R^2$ > {MIN_NRS})\n(Number of Sessions: {len(valid_session_names)}, Total pairs: {len(distances_array)})', fontsize=16)
plt.xlabel('Pairwise Euclidean Distance (cm)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.xlim(0, ARENA_DIAMETER) 
plt.tight_layout()

plt.show()

# -----------------------------------------------
# --- Plot 2 : means by session---
# -----------------------------------------------
plt.figure(figsize=(12, 7))

# Circle size for session set by its size by units
sizes = np.array(session_pair_counts) / np.max(session_pair_counts) * 500 + 50

# Plot
plt.scatter(valid_session_names, session_means, 
            s=sizes, color='royalblue', 
            alpha=0.7, edgecolors='black', label='Session Means', zorder=3)

plt.axhline(theoretical_mean, color='green', linestyle=':', linewidth=3, 
            label=f'Theoretical Mean ({theoretical_mean:.2f})', zorder=2)
plt.axhline(mean_of_means, color='red', linestyle='--', linewidth=2, 
            label=f'Simple Mean of Means ({mean_of_means:.2f})', zorder=2)
plt.axhline(weighted_mean, color='purple', linestyle='-.', linewidth=2, 
            label=f'Weighted Mean ({weighted_mean:.2f})', zorder=2)

plt.title(f'Session Means vs. Theoretical Expectation (unbiased $R^2$ > {MIN_NRS})', fontsize=16)
plt.ylabel('Mean Distance (cm)', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)

plt.legend(loc='best')

plt.tight_layout()
plt.show()

print(f"\nTheoretical Mean: {theoretical_mean:.4f}")
print(f"Simple Mean of Means: {mean_of_means:.4f}")
print(f"Weighted Mean (Overall): {weighted_mean:.4f}")
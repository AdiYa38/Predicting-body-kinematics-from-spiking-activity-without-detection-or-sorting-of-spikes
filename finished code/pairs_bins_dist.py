import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
r_squared_threshold = 0.4
NUM_BINS = 7 
ONLY_HIGH_QUALITY = True
THEORETICAL_MEAN = 36.22

# Load file
filename = f"Centers_pairs_Nth_{r_squared_threshold}.csv"

try:
    df = pd.read_csv(filename)
    print(f"Successfully loaded data from: {filename}")
    print(f"Total pairs in file: {len(df)}")
except FileNotFoundError:
    print(f"Error: The file '{filename}' was not found.")
    exit()

# Filter pairs
if ONLY_HIGH_QUALITY:
    df_clean = df[df['is_high_high'] == True].copy()
    print(f"Analyzing only HIGH quality pairs. n = {len(df_clean)}")
else:
    df_clean = df.copy()
    print(f"Analyzing ALL pairs (including low quality). n = {len(df_clean)}")

if len(df_clean) < NUM_BINS:
    print("Not enough data points to divide into bins.")
    exit()

# Split to bins
df_clean['range_bin'] = pd.qcut(df_clean['anatomical_dist_um'], q=NUM_BINS)

# Set labels
def format_interval(interval):
    return f"{interval.left:.0f}-{interval.right:.0f}\nµm"

df_clean = df_clean.sort_values('anatomical_dist_um')
df_clean['bin_label'] = df_clean['range_bin'].apply(format_interval)

# Plot in a box plot
plt.figure(figsize=(12, 8))

ax = sns.boxplot(data=df_clean, x='bin_label', y='field_euclidean_dist_cm', 
                 palette='viridis', showmeans=True, 
                 meanprops={"marker":"^", "markerfacecolor":"white", "markeredgecolor":"black"})

plt.title(f'Distribution of Place Field Distances by Anatomical Range\n(Divided into {NUM_BINS} equal-sized groups)', fontsize=16)
plt.xlabel('Anatomical Distance Range [µm]', fontsize=14)
plt.ylabel('Place Field Euclidean Distance [cm]', fontsize=14)

# Add mean line
overall_mean = df_clean['field_euclidean_dist_cm'].mean()
plt.axhline(y=overall_mean, color='red', linestyle='-.', linewidth=2, 
            label=f'Overall Empirical Mean ({overall_mean:.2f} cm)')

# Add theoretical mean line
plt.axhline(y=THEORETICAL_MEAN, color='green', linestyle='--', linewidth=2, 
            label=f'Theoretical Mean ({THEORETICAL_MEAN} cm)')

# Add statistics
means = df_clean.groupby('bin_label', sort=False)['field_euclidean_dist_cm'].mean()
counts = df_clean.groupby('bin_label', sort=False)['field_euclidean_dist_cm'].count()

xtick_labels = [label.get_text() for label in ax.get_xticklabels()]
for i, label in enumerate(xtick_labels):
    if label in means:
        val = means[label]
        n = counts[label]
        ax.text(i, val, f'μ={val:.1f}\n(n={n})', 
                ha='center', va='bottom', fontsize=9, color='black', fontweight='bold')

plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(loc='upper right', framealpha=0.9)
plt.tight_layout()
plt.show()
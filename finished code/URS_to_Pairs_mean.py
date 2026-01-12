import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Constants
BIN_SIZE = 1
SESSIONS = ["mP79_11", "mP79_12", "mP79_13", "mP79_14", "mP79_15", "mP79_16", "mP79_17", "mP31_18", "mP31_19", "mP31_20"]
THEORETICAL_MEAN = 36.22

# A list forthe pairs from all sessions
all_pairs_data = []

print(f"Loading data from sessions: {SESSIONS}...")

# Load Data and calculate anatomical distances
for SESSION in SESSIONS:
    # Load file of anatomical data
    loc_dict = {} 
    try:
        loc_file = f'data/{SESSION}/{SESSION}_unit_loc.csv'
        df_loc = pd.read_csv(loc_file, skipinitialspace=True, encoding='latin1')
        cols_to_clean = ['x_diff', 'y']
        for col in cols_to_clean:
            df_loc[col] = df_loc[col].astype(str).str.replace(r'\s+', '', regex=True)
            df_loc[col] = pd.to_numeric(df_loc[col])
        
        loc_dict = {(row.shank, row.unit): (row.x_diff, row.y) 
                    for row in df_loc.itertuples(index=False)}

    except FileNotFoundError:
        print(f"Warning: Unit location file for {SESSION} not found. Skipping session.")
        continue

    # Load file of data centers
    center_file = f"data/{SESSION}/Centers_{SESSION}_bin{BIN_SIZE}cm.csv"
    try:
        df_centers = pd.read_csv(center_file)
        fits_data = df_centers.to_dict('records')
    except FileNotFoundError:
        print(f"Warning: Center file '{center_file}' not found. Skipping session.")
        continue

    # Calculate distances
    for i in range(len(fits_data)): 
        for j in range(i + 1, len(fits_data)):
            r1 = fits_data[i]
            r2 = fits_data[j]

            # Euclidean Distance
            dx = r1['center_x_cm'] - r2['center_x_cm']
            dy = r1['center_y_cm'] - r2['center_y_cm']
            eucl_dist = np.sqrt(dx**2 + dy**2)
            
            # Save pair's data
            pair_info = {
                'field_euclidean_dist_cm': eucl_dist,
                'n_r_sq_1': r1['n_r_squared'],
                'n_r_sq_2': r2['n_r_squared']
            }
            all_pairs_data.append(pair_info)


df_all = pd.DataFrame(all_pairs_data)
print(f"Total pairs loaded: {len(df_all)}")

# Threshold Analysis

# X-axis linespace
thresholds = np.linspace(0, 0.9, 50)

mean_diffs = []
counts = []

print("Analyzing thresholds...")

for thresh in thresholds:
    # Plot only pairs with both centers above threshold
    mask = (df_all['n_r_sq_1'] > thresh) & (df_all['n_r_sq_2'] > thresh)
    df_filtered = df_all[mask]
    
    count = len(df_filtered)
    
    if count > 0:
        overall_mean = df_filtered['field_euclidean_dist_cm'].mean()
        
        diff = overall_mean - THEORETICAL_MEAN
    else:
        diff = np.nan
        
    mean_diffs.append(diff)
    counts.append(count)

# Plot

fig, ax1 = plt.subplots(figsize=(10, 6))

# Y-axis 1: diff of means
color = 'tab:blue'
ax1.set_xlabel('Unbiased R² Threshold', fontsize=12)
ax1.set_ylabel('Empirical - Theoretical Mean [cm]', color=color, fontsize=12)
ax1.plot(thresholds, mean_diffs, color=color, marker='o', markersize=4, linestyle='-', label='Mean Difference')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle=':', alpha=0.6)

# Add the zero line
ax1.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

# Y-axis 2: number of pairs
ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Number of Pairs (Log Scale)', color=color, fontsize=12)
ax2.plot(thresholds, counts, color=color, marker='x', markersize=4, linestyle='--', label='Pair Count')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_yscale('log')

plt.title(f'Effect of R² Threshold on Place Field Distance Statistics', fontsize=14)

plt.tight_layout()
plt.show()
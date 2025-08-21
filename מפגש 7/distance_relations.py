import numpy as np
import matplotlib.pyplot as plt
import heatmaps
import data
import scipy.optimize
import sim_gaus

r_squared_threshold = 0.7 

# Create fits
gaussian_fits = sim_gaus.fit_gaussians(False)

# --- Plotting Distances ---

# Prepare lists to hold the data points for each category
# Each item will be a tuple: (anatomical_distance, euclidean_distance)
low_low_fits = []   # Both fits have R² < threshold
low_high_fits = []  # One fit is below, one is above
high_high_fits = [] # Both fits have R² > threshold

# Iterate through all unique pairs of fits
for i in range(len(gaussian_fits)):
    for j in range(i + 1, len(gaussian_fits)):
        fit1 = gaussian_fits[i]
        fit2 = gaussian_fits[j]

        # Proceed only if both fits were successful (have parameters)
        if fit1['params'] is not None and fit2['params'] is not None:
            
            # --- Calculate Distances ---
            
            # 1. Euclidean distance between place field centers
            euclidean_dist = np.sqrt((fit1['params'][1] - fit2['params'][1])**2 + 
                                     (fit1['params'][2] - fit2['params'][2])**2)
            
            # 2. Anatomical distance
            # Assuming cell_id can be used to calculate anatomical distance
            # NOTE: You might need to adjust this if cell_id is not the correct identifier for your function
            id1 = fit1['cell_id'] - 2
            id2 = fit2['cell_id'] - 2
            # This is your function to calculate anatomical distance
            def calc_anatomical_distance (a, b):
                if ((a <= 6 and b <= 6) or (a >= 7 and b >= 7)):
                    return (np.abs(a - b) * 200)
                else: # min(a,b) < 7, max(a,b) > 7
                    l_diff = np.abs((min(a, b)-1) - (12-max(a, b)))
                    return np.sqrt(30**2 + l_diff**2)
            
            anatomical_dist = calc_anatomical_distance(id1, id2)

            # --- Categorize by R-squared value ---
            
            is_fit1_good = fit1.get('r_squared', -1) > r_squared_threshold
            is_fit2_good = fit2.get('r_squared', -1) > r_squared_threshold

            point_data = (anatomical_dist, euclidean_dist)

            if is_fit1_good and is_fit2_good:
                high_high_fits.append(point_data)
            elif is_fit1_good or is_fit2_good:
                low_high_fits.append(point_data)
            else:
                low_low_fits.append(point_data)

# --- Create the Scatter Plot ---

plt.figure(figsize=(10, 8))

# Plot each category with its specific color and label
# Use np.array to easily separate x and y coordinates for plotting
if low_low_fits:
    low_low_arr = np.array(low_low_fits)
    plt.scatter(low_low_arr[:, 0], low_low_arr[:, 1], color='lightgray', label=f'Both R² < {r_squared_threshold}')

if low_high_fits:
    low_high_arr = np.array(low_high_fits)
    plt.scatter(low_high_arr[:, 0], low_high_arr[:, 1], color='dimgray', label='One R² > threshold')

if high_high_fits:
    high_high_arr = np.array(high_high_fits)
    plt.scatter(high_high_arr[:, 0], high_high_arr[:, 1], color='black', label=f'Both R² > {r_squared_threshold}')

# Add plot titles and labels
plt.title('Anatomical vs. Euclidean Distance of Place Fields', fontsize=16)
plt.xlabel('Anatomical Distance (µm)', fontsize=12)
plt.ylabel('Euclidean Distance (bins)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Show the final plot
plt.show()
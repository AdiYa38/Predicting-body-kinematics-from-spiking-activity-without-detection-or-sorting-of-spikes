import numpy as np
import matplotlib.pyplot as plt
import heatmaps
import data
import scipy.optimize
import sim_gaus
import pandas as pd

BIN_SIZE = 1
r_squared_threshold = 0.5
SESSION = "mP79_16"

# Load unit locations from file into a dictionary
df = pd.read_csv(f'data/{SESSION}/{SESSION}_unit_loc.csv', skipinitialspace=True, encoding='latin1')
cols_to_clean = ['x_diff', 'y']
for col in cols_to_clean:
    df[col] = df[col].astype(str).str.replace(r'\s+', '', regex=True)
    df[col] = pd.to_numeric(df[col])
loc_dict = {(row.shank, row.unit): (row.x_diff, row.y) 
                for row in df.itertuples(index=False)}

df_indexed = df.set_index(['shank', 'unit'])

# This is a function to calculate anatomical distance in micron
def calc_anatomical_distance (shank1, unit1, shank2, unit2):
    
    # Seperate by is the shanks are in the same side or not
    if ((shank1 <= 6 and shank2 <= 6) or (shank1 >= 7 and shank2 >= 7)):
        dx = np.abs(shank1 - shank2) * 200 
        dz = 0
    else: # min(a,b) < 7, max(a,b) > 7
        dx = np.abs((min(shank1, shank2)-1) - (12-max(shank1, shank2)))
        dz = 30 

    # Add units differences
    x1, y1 = loc_dict[(shank1, unit1)]
    x2, y2 = loc_dict[(shank2, unit2)]
    dx = dx + np.abs(x1 - x2) * 15
    dy = np.abs(y1 - y2) * 15

    return np.sqrt(dx**2 + dy**2 + dz**2)

# Load centers file
filename = f"data/{SESSION}/Centers_{SESSION}_bin{BIN_SIZE}cm.csv"

try:
    df = pd.read_csv(filename)
    print(f"Successfully loaded data from: {filename}")
    print(f"Found {len(df)} successful fits to analyze.")
    results = df[['shank', 'unit','n_r_squared','center_x_cm', 'center_y_cm']].values

except FileNotFoundError:
    print(f"Error: The file '{filename}' was not found.")
    print("Please make sure the file is in the same directory, or provide the full path.")
    exit()

# Calculate Distances
for i in range(len(results)): 
    for j in range(i + 1, len(results)):
        r1 = results[i]
        r2 = results[j]

        dx = r1[1] - r2[1]
        dy = r1[2] - r2[2]
        euclidean_dist = np.sqrt(((dx * BIN_SIZE)**2) + ((dy * BIN_SIZE)**2))

        shank1 = r1[]
        unit1 = r1[]
        shank2 = r2[]
        unit2 = r2[]
        if (euclidean_dist > 100):
            print(f"unit {shank1}.{unit1} to {shank2}.{unit2} has distance {euclidean_dist}")
        
        anatomical_dist = calc_anatomical_distance(shank1, unit1, shank2, unit2)

    

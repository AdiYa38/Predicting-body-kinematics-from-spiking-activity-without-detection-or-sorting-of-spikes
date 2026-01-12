import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import heatmaps
import data

# --- Constants ---
ARENA_DIAMETER = 80
RES_SAMPLING_RATE = 20000
POS_SAMPLING_RATE = 1250
X_CHANNEL = 124
Y_CHANNEL = 125
N_CHANNELS = 136
SESSION_NUM = range(1, 8) 
DTYPE = np.int16
KERNEL_SIZE = 7
show = False
BIN_SIZE = 1
PERMUTATIONS_N = 20  # מספר הפרמוטציות

# --- Helper Functions ---

def create_permutations(map_matrix, N=20):
    """Creates N permutations of a matrix, shuffling only values != -1."""
    shuffle_mask = map_matrix != -1
    values_to_shuffle = map_matrix[shuffle_mask]
    permutations_list = []

    for _ in range(N):
        new_map = map_matrix.copy()
        shuffled_values = np.random.permutation(values_to_shuffle)
        new_map[shuffle_mask] = shuffled_values
        permutations_list.append(new_map)

    return np.array(permutations_list)

def calc_mutual_information(vec_x, vec_y, bins=20):
    """Calculates MI (bits) between two vectors."""
    c_xy, _, _ = np.histogram2d(vec_x, vec_y, bins=bins)
    p_xy = c_xy / np.sum(c_xy)
    p_x = np.sum(p_xy, axis=1)
    p_y = np.sum(p_xy, axis=0)
    
    mi = 0.0
    for i in range(p_xy.shape[0]):
        for j in range(p_xy.shape[1]):
            if p_xy[i, j] > 0:
                mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))
    return mi

def create_centers_density_map(df, bin_size, arena_diameter):
    """Creates a 2D histogram of centers."""
    x = df['center_x_cm'].values
    y = df['center_y_cm'].values
    num_bins = int(arena_diameter / bin_size)
    bins = np.linspace(0, arena_diameter, num_bins + 1)
    density_map, _, _ = np.histogram2d(x, y, bins=bins)
    return density_map.T

# --- Main Logic ---

# רשימות לשמירת התוצאות
results_observed = []
results_chance = []
results_diff = []
session_names = []

print("Starting analysis with permutations...")

for sn in SESSION_NUM:
    SESSION = f"mP79_1{sn}"
    session_names.append(SESSION)
    
    # 1. Get occupancy map
    occupancy_map = heatmaps.occupancy_map(SESSION, BIN_SIZE)
    
    # 2. Get centers locations
    filename = f"data/{SESSION}/Centers_{SESSION}_bin{BIN_SIZE}cm.csv"
    
    try:
        df_centers = pd.read_csv(filename)
        
        # 3. Create Centers Density Map
        centers_map = create_centers_density_map(df_centers, BIN_SIZE, ARENA_DIAMETER)
        
        # --- Prepare Masks ---
        rows, cols = occupancy_map.shape
        y_grid, x_grid = np.ogrid[:rows, :cols]
        center_y, center_x = rows / 2, cols / 2
        radius_bins = (ARENA_DIAMETER / 2) / BIN_SIZE
        
        # מסכה גאומטרית (בתוך העיגול)
        mask_inside = (x_grid - center_x)**2 + (y_grid - center_y)**2 <= radius_bins**2
        
        # מסכה לנתונים תקינים (גם בתוך העיגול וגם עם מידע תקין ב-occupancy)
        valid_indices = mask_inside & (occupancy_map != -1) & (~np.isnan(occupancy_map))
        
        if np.sum(valid_indices) == 0:
            print(f"Warning: No valid bins for {SESSION}.")
            results_observed.append(0); results_chance.append(0); results_diff.append(0)
            continue

        # --- A. Calculate Observed MI ---
        vec_occupancy = occupancy_map[valid_indices]
        vec_centers = centers_map[valid_indices]
        
        obs_mi = calc_mutual_information(vec_centers, vec_occupancy, bins=10)
        
        # --- B. Calculate Chance MI (Permutations) ---
        
        # הכנת מפת המרכזים לפרמוטציה:
        # הפונקציה create_permutations צריכה לדעת מה "בחוץ" כדי לא לערבב לשם נתונים.
        # נסמן את כל מה שמחוץ למסכה הגאומטרית כ 1-
        centers_map_for_perm = centers_map.copy()
        centers_map_for_perm[~mask_inside] = -1
        
        # יצירת פרמוטציות
        perm_maps = create_permutations(centers_map_for_perm, N=PERMUTATIONS_N)
        
        perm_mi_values = []
        for p_map in perm_maps:
            # חילוץ הוקטור המעורבב באותם אינדקסים תקינים בדיוק כמו המקורי
            vec_p = p_map[valid_indices]
            # חישוב MI מול ה-Occupancy המקורי
            perm_mi_values.append(calc_mutual_information(vec_p, vec_occupancy, bins=10))
            
        avg_chance_mi = np.mean(perm_mi_values)
        diff_mi = obs_mi - avg_chance_mi
        
        # שמירת תוצאות
        results_observed.append(obs_mi)
        results_chance.append(avg_chance_mi)
        results_diff.append(diff_mi)
        
        print(f"{SESSION}: Obs={obs_mi:.4f}, Chance={avg_chance_mi:.4f}, Diff={diff_mi:.4f}")

    except FileNotFoundError:
        print(f"Error: File not found for {SESSION}.")
        results_observed.append(0); results_chance.append(0); results_diff.append(0)
    except Exception as e:
        print(f"Error in {SESSION}: {e}")
        results_observed.append(0); results_chance.append(0); results_diff.append(0)

# --- Visualization ---

x = np.arange(len(session_names))  # מיקומי ה-X
width = 0.25  # רוחב העמודות

fig, ax = plt.subplots(figsize=(12, 7))

# יצירת העמודות
rects1 = ax.bar(x - width, results_observed, width, label='Observed MI', color='cornflowerblue')
rects2 = ax.bar(x, results_chance, width, label=f'Chance MI (Avg {PERMUTATIONS_N} Perms)', color='lightgray')
rects3 = ax.bar(x + width, results_diff, width, label='Difference (Obs - Chance)', color='mediumseagreen')

# עיצוב הגרף
ax.set_ylabel('Mutual Information (bits)', fontsize=12)
ax.set_xlabel('Session', fontsize=12)
ax.set_title(f'Mutual Information Analysis: Centers vs. Occupancy\n(Bin Size: {BIN_SIZE}cm)', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(session_names)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)

# פונקציה להוספת תוויות מעל העמודות
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, rotation=90)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()
plt.show()
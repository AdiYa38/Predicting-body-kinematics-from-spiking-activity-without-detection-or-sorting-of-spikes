import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- הגדרות ---
BIN_SIZE = 1
r_squared_threshold = 0.4
# רשימת הסשנים לעיבוד
SESSIONS = ["mP79_11", "mP79_12", "mP79_13", "mP79_14","mP79_15", "mP79_16", "mP79_17", "mP31_18", "mP31_19", "mP31_20"] 
# -------------

# רשימות גלובליות לאיסוף הזוגות לגרף
all_low_low_pairs = []   
all_mixed_pairs = []     
all_high_high_pairs = [] 

# --- הוספה: רשימה לאיסוף נתונים לשמירה ב-CSV ---
all_pairs_data = []

print(f"Starting analysis for sessions: {SESSIONS}")

# לולאה על כל סשן בנפרד
for SESSION in SESSIONS:
    print(f"\nProcessing Session: {SESSION}...")
    
    # 1. טעינת מיקומי יחידות (Anatomical Locations)
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

    # פונקציה פנימית לחישוב מרחק אנטומי
    def calc_anatomical_distance(shank1, unit1, shank2, unit2):
        if ((shank1 <= 6 and shank2 <= 6) or (shank1 >= 7 and shank2 >= 7)):
            dx_shank = np.abs(shank1 - shank2) * 200 
            dz_shank = 0
        else: 
            dx_shank = np.abs((min(shank1, shank2)-1) - (12-max(shank1, shank2))) * 200 
            dz_shank = 30 

        if (shank1, unit1) in loc_dict and (shank2, unit2) in loc_dict:
            x1, y1 = loc_dict[(shank1, unit1)]
            x2, y2 = loc_dict[(shank2, unit2)]
            
            dx_total = dx_shank + np.abs(x1 - x2) * 15 
            dy_total = np.abs(y1 - y2) * 15
            
            return np.sqrt(dx_total**2 + dy_total**2 + dz_shank**2)
        else:
            return np.nan

    # 2. טעינת נתוני מרכזי השדות
    center_file = f"data/{SESSION}/Centers_{SESSION}_bin{BIN_SIZE}cm.csv"
    fits_data = []
    
    try:
        df_centers = pd.read_csv(center_file)
        print(f"  Loaded center data: {len(df_centers)} cells.")
        fits_data = df_centers.to_dict('records')

    except FileNotFoundError:
        print(f"  Warning: Center file '{center_file}' not found. Skipping session.")
        continue

    # 3. חישוב מרחקים
    pairs_count = 0
    for i in range(len(fits_data)): 
        for j in range(i + 1, len(fits_data)):
            r1 = fits_data[i]
            r2 = fits_data[j]

            shank1, unit1 = int(r1['shank']), int(r1['unit'])
            shank2, unit2 = int(r2['shank']), int(r2['unit'])

            anat_dist = calc_anatomical_distance(shank1, unit1, shank2, unit2)
            
            if np.isnan(anat_dist):
                continue 

            # חישוב מרחק אוקלידי
            dx = r1['center_x_cm'] - r2['center_x_cm']
            dy = r1['center_y_cm'] - r2['center_y_cm']
            eucl_dist = np.sqrt(dx**2 + dy**2)

            # סיווג
            nrs1 = r1['n_r_squared']
            nrs2 = r2['n_r_squared']
            
            is_good1 = nrs1 > r_squared_threshold
            is_good2 = nrs2 > r_squared_threshold

            point_data = (anat_dist, eucl_dist)
            
            # קביעת תווית האיכות עבור ה-CSV
            quality_label = "Low_Low"
            
            # הוספה לרשימות הגלובליות לגרף
            if is_good1 and is_good2:
                all_high_high_pairs.append(point_data)
                quality_label = "High_High"
            elif is_good1 or is_good2:
                all_mixed_pairs.append(point_data)
                quality_label = "Mixed"
            else:
                all_low_low_pairs.append(point_data)
                quality_label = "Low_Low"
            
            # --- הוספה: שמירת הנתונים ל-CSV ---
            pair_info = {
                'session': SESSION,
                'shank1': shank1,
                'unit1': unit1,
                'shank2': shank2,
                'unit2': unit2,
                'n_r_sq_1': nrs1,
                'n_r_sq_2': nrs2,
                'anatomical_dist_um': anat_dist,
                'field_euclidean_dist_cm': eucl_dist,
                'quality_category': quality_label,
                'is_high_high': is_good1 and is_good2
            }
            all_pairs_data.append(pair_info)
            # ----------------------------------

            pairs_count += 1
    print(f"  Calculated {pairs_count} pairs for {SESSION}.")

# --- הוספה: שמירה לקובץ CSV ---
if all_pairs_data:
    output_filename = f"Centers_pairs_Nth_{r_squared_threshold}.csv"
    print(f"\nSaving all pairs data to: {output_filename}...")
    df_pairs = pd.DataFrame(all_pairs_data)
    df_pairs.to_csv(output_filename, index=False)
    print("Save complete.")
else:
    print("\nNo pairs found to save.")
# -----------------------------

# 4. יצירת הגרף המאוחד
print("\nPlotting combined data...")
plt.figure(figsize=(12, 10))

# הגדרות עיצוב
SMALL_SIZE = 1.5   
MAIN_SIZE = 2.5    
ALPHA_BG = 0.3     
ALPHA_MAIN = 0.5   

if all_low_low_pairs:
    arr = np.array(all_low_low_pairs)
    plt.scatter(arr[:, 0], arr[:, 1], c='#d3d3d3', alpha=ALPHA_BG, s=SMALL_SIZE, 
                label=f'Both R² < {r_squared_threshold}', marker='.')

if all_mixed_pairs:
    arr = np.array(all_mixed_pairs)
    plt.scatter(arr[:, 0], arr[:, 1], c='gray', alpha=ALPHA_BG, s=SMALL_SIZE, 
                label='Mixed Quality', marker='.')

if all_high_high_pairs:
    arr = np.array(all_high_high_pairs)
    plt.scatter(arr[:, 0], arr[:, 1], c='black', alpha=ALPHA_MAIN, s=MAIN_SIZE, 
                label=f'Both R² > {r_squared_threshold}', marker='o')

plt.xlabel('Anatomical Distance [µm]', fontsize=14)
plt.ylabel('Place Field Euclidean Distance [cm]', fontsize=14)
plt.title(f'Place Field Distance vs. Anatomical Distance\nCombined Sessions ({len(SESSIONS)} sessions)', fontsize=16)

lgnd = plt.legend(fontsize=12, markerscale=4) 

plt.grid(True, linestyle=':', alpha=0.4)
plt.tight_layout()
plt.show()
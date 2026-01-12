import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Definitions ---
ARENA_DIAMETER = 80
BIN_SIZE = 1
r_squared_threshold = 0.01
# יצירת רשימת הסשנים
sessions = ["mP79_11", "mP79_12", "mP79_13", "mP79_14", "mP79_15", "mP79_16", "mP79_17", "mP31_18", "mP31_19", "mP31_20"]
# -----------------

all_data_list = []

# לולאה שעוברת על כל הסשנים ואוספת נתונים
for session in sessions:
    filename = f"data/{session}/Centers_{session}_bin{BIN_SIZE}cm.csv"
    try:
        df = pd.read_csv(filename)
        # הוספת עמודה כדי שנדע מאיזה סשן הגיעה כל נקודה (אופציונלי)
        df['session_id'] = session
        all_data_list.append(df)
        print(f"Successfully loaded {len(df)} fits from: {session}")
    except FileNotFoundError:
        print(f"Warning: The file for '{session}' was not found. Skipping.")

if not all_data_list:
    print("No data files were found. Exiting.")
    exit()

# איחוד כל הטבלאות לטבלה אחת גדולה
df_combined = pd.concat(all_data_list, ignore_index=True)

# --- FILTERING STEP ---
df_filtered = df_combined[df_combined['n_r_squared'] > r_squared_threshold]
print(f"\nTotal filtered data across all sessions: {len(df_filtered)} fits have R^2 > {r_squared_threshold}")

if len(df_filtered) == 0:
    print("No data points passed the threshold filter. Exiting.")
    exit()
# ----------------------

# הצגת המרכזים (Centers Logic)
# 2. Extract coordinates and calculate statistics (using filtered data)
x_centers = df_filtered['center_x_cm'].values
y_centers = df_filtered['center_y_cm'].values

# Calculate mean and standard deviation
mean_x = np.mean(x_centers)
mean_y = np.mean(y_centers)
standard_deviation_x = np.sqrt(np.var(x_centers))
standard_deviation_y = np.sqrt(np.var(y_centers))

print("\n--- Combined Statistics for All Sessions (11-17) ---")
print(f"X Coordinate Mean: {mean_x:.4f} cm")
print(f"Y Coordinate Mean: {mean_y:.4f} cm")
print(f"X Std Dev:         {standard_deviation_x:.4f} cm")
print(f"Y Std Dev:         {standard_deviation_y:.4f} cm")

# 3. Create 2D Scatter Plot
plt.figure(figsize=(9, 9))
ax = plt.gca()

# Add circle representing the arena
arena_center = ARENA_DIAMETER / 2
arena_radius = ARENA_DIAMETER / 2
arena_circle = plt.Circle((arena_center, arena_center), arena_radius, 
                            color='black', fill=False, linestyle='--', label='Arena Boundary')
ax.add_artist(arena_circle)

# Scatter plot of cell centers - משתמש בצבעים שונים לפי סשן אם תרצה, או צבע אחיד
plt.scatter(x_centers, y_centers, alpha=0.5, s=20,
            label=f'Place Field Centers (n={len(x_centers)})')

# Mark the overall mean point
plt.plot(mean_x, mean_y, 'X', color='red', markersize=15, markeredgewidth=2,
        label=f'Combined Mean ({mean_x:.2f}, {mean_y:.2f})')

# Add titles and info
plt.title(f'Distribution of Place Field Centers\nBin Size: {BIN_SIZE}cm, R^2 > {r_squared_threshold}', fontsize=16)
plt.xlabel('X Position (cm)', fontsize=12)
plt.ylabel('Y Position (cm)', fontsize=12)

# Set axis limits and ensure equal aspect ratio
plt.xlim(0, ARENA_DIAMETER)
plt.ylim(0, ARENA_DIAMETER)
ax.set_aspect('equal', adjustable='box')

plt.legend(loc='upper right', fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)

# Show plot
plt.show()
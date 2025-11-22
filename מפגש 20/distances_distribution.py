import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- הגדרות ---
# עליך לעדכן את המשתנים האלה כך שיתאימו לקובץ שאתה רוצה לנתח
ARENA_DIAMETER = 80
session = "mP79_16"
BIN_SIZE = 1
# -----------------

# 1. בניית שם הקובץ וטעינת הנתונים
filename = f"data/{session}/Centers_{session}_bin{BIN_SIZE}cm.csv"

try:
    df = pd.read_csv(filename)
    print(f"Successfully loaded data from: {filename}")
    print(f"Found {len(df)} successful fits to analyze.")
except FileNotFoundError:
    print(f"Error: The file '{filename}' was not found.")
    print("Please make sure the file is in the same directory, or provide the full path.")
    exit()

pairs = False
if (pairs):
    # 2. חישוב המרחקים האוקלידיים בין כל זוגות התאים
    pairwise_distances = []
    # שלוף את הקואורדינטות כמערך NumPy לחישוב מהיר
    coordinates = df[['center_x_cm', 'center_y_cm']].values

    # לולאה על כל הזוגות הייחודיים
    for i in range(len(coordinates)):
        for j in range(i + 1, len(coordinates)):
            # קח שתי נקודות
            p1 = coordinates[i]
            p2 = coordinates[j]
            
            # חשב מרחק אוקלידי
            distance = np.sqrt(np.sum((p1 - p2)**2))
            pairwise_distances.append(distance)

    # המר את הרשימה למערך NumPy לניתוח סטטיסטי
    distances_array = np.array(pairwise_distances)

    # 3. חישוב תוחלת ושונות
    mean_distance = np.mean(distances_array)
    variance_distance = np.var(distances_array)

    print("\n--- Statistics for Pairwise Euclidean Distances ---")
    print(f"Mean (תוחלת):     {mean_distance:.4f} cm")
    print(f"Variance (שונות): {variance_distance:.4f} cm")

    # 4. הצגת ההתפלגות בגרף (היסטוגרמה)
    plt.figure(figsize=(12, 7))
    # 'bins=50' הוא בחירה טובה להתחלה, אפשר לשנות לפי הצורך
    plt.hist(distances_array, bins=50, edgecolor='black', alpha=0.7)

    # הוספת קו שמציין את התוחלת
    plt.axvline(mean_distance, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_distance:.2f} cm')

    # הוספת כותרות ומידע
    plt.title(f'Distribution of Pairwise Distances between Place Fields\n(Session: {session}, Bin Size: {BIN_SIZE}cm)', fontsize=16)
    plt.xlabel('Pairwise Euclidean Distance (cm)', fontsize=12)
    plt.ylabel('Frequency (Count)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    # הצג את הגרף
    plt.show()

centers = True
if (centers):
    # 2. חילוץ הקואורדינטות וחישוב סטטיסטיקות
    # שלוף את כל הקואורדינטות כמערכי NumPy
    x_centers = df['center_x_cm'].values
    y_centers = df['center_y_cm'].values

    # חישוב תוחלת ושונות
    mean_x = np.mean(x_centers)
    mean_y = np.mean(y_centers)
    standard_deviation_x = np.sqrt(np.var(x_centers))
    standard_deviation_y = np.sqrt(np.var(y_centers))

    print("\n--- Statistics for Place Field Centers ---")
    print(f"X Coordinate Mean {mean_x:.4f} cm")
    print(f"Y Coordinate Mean {mean_y:.4f} cm")
    print(f"X Coordinate standard deviation {standard_deviation_x:.4f} cm²")
    print(f"Y Coordinate standard deviation_y {standard_deviation_y:.4f} cm²")

    # 3. יצירת גרף פיזור דו-ממדי
    plt.figure(figsize=(9, 9))
    ax = plt.gca() # קבלת הצירים הנוכחיים

    # הוספת עיגול המייצג את הזירה
    # מרכז הזירה הוא ברדיוס (D/2)
    arena_center = ARENA_DIAMETER / 2
    arena_radius = ARENA_DIAMETER / 2
    arena_circle = plt.Circle((arena_center, arena_center), arena_radius, 
                            color='black', fill=False, linestyle='--', label='Arena Boundary')
    ax.add_artist(arena_circle)

    # פיזור הנקודות של מרכזי התאים
    plt.scatter(x_centers, y_centers, alpha=0.6, 
                label=f'Place Field Centers (n={len(x_centers)})')

    # סימון נקודת התוחלת
    plt.plot(mean_x, mean_y, 'X', color='red', markersize=15, markeredgewidth=2,
            label=f'Overall Mean ({mean_x:.2f}, {mean_y:.2f})')

    # הוספת כותרות ומידע
    plt.title(f'Distribution of Place Field Centers\n(Session: {session}, Bin Size: {BIN_SIZE}cm)', fontsize=16)
    plt.xlabel('X Position (cm)', fontsize=12)
    plt.ylabel('Y Position (cm)', fontsize=12)

    # קביעת גבולות הצירים והבטחת יחס גובה-רוחב שווה
    plt.xlim(0, ARENA_DIAMETER)
    plt.ylim(0, ARENA_DIAMETER)
    ax.set_aspect('equal', adjustable='box')

    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    # הצג את הגרף
    plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- הגדרות ---
# עליך לעדכן את המשתנים האלה כך שיתאימו לקובץ שאתה רוצה לנתח
session = "mP79_14"
BIN_SIZE = 1
ARENA_DIAMETER = 80 # קבוע שנדרש עבור הפונקציות התיאורטיות
MIN_NRS = 0.4
# -----------------
def analytical_pdf(r, D):
    """
    מחשבת את פונקציית צפיפות ההסתברות (PDF) התיאורטית
    של המרחק r בין שתי נקודות אקראיות בעיגול בקוטר D.
    """
    # מונע חלוקה ב-0 או שורש של מספר שלילי בקצוות
    epsilon = 1e-10
    r = np.clip(r, epsilon, D - epsilon)
    
    x = r / D 
    
    term1 = (16 * r / (np.pi * D**2)) * np.arccos(x)
    
    term2 = (16 * r**2 / (np.pi * D**3)) * np.sqrt(1 - x**2)
    
    return term1 - term2

# 1. בניית שם הקובץ וטעינת הנתונים
filename = f"data/{session}/Centers_{session}_bin{BIN_SIZE}cm.csv" # --- תיקנתי את שם הקובץ שיתאים לקוד הקודם ---

try:
    df = pd.read_csv(filename)
    print(f"Successfully loaded data from: {filename}")
    print(f"Found {len(df)} total successful fits.")
except FileNotFoundError:
    print(f"Error: The file '{filename}' was not found.")
    print("Please make sure the file is in the same directory, or provide the full path.")
    exit()

# --- הוספה: סינון הנתונים לפי סף R² ---
print(f"Filtering fits with n_r_squared > {MIN_NRS}...")
df_filtered = df[df['n_r_squared'] > MIN_NRS].copy()
print(f"Found {len(df_filtered)} fits passing the filter.")

# בדיקה אם נשארו מספיק נתונים לחישוב זוגות
if len(df_filtered) < 2:
    print("Not enough data points (< 2) after filtering to calculate pairwise distances.")
    exit()
# --- סוף ההוספה ---


# 2. חישוב המרחקים האוקלידיים בין כל זוגות התאים שעברו סינון
pairwise_distances = []
# --- שינוי: שימוש ב-df_filtered במקום ב-df ---
coordinates = df_filtered[['center_x_cm', 'center_y_cm']].values

for i in range(len(coordinates)):
    for j in range(i + 1, len(coordinates)):
        p1 = coordinates[i]
        p2 = coordinates[j]
        distance = np.sqrt(np.sum((p1 - p2)**2))
        pairwise_distances.append(distance)

distances_array = np.array(pairwise_distances)

# 3. חישוב תוחלת ושונות (מהנתונים המסוננים)
mean_distance = np.mean(distances_array)
variance_distance = np.var(distances_array)

# --- חישובים תיאורטיים ---
# 3b. חישוב תוחלת תיאורטית (עם הרדיוס, כפי שתיקנו)
theoretical_mean = (128 * (ARENA_DIAMETER / 2)) / (45 * np.pi)

# 3c. יצירת נתונים לגרף ה-PDF התיאורטי
r_values = np.linspace(0, ARENA_DIAMETER, 500) # ציר X רציף
pdf_values = analytical_pdf(r_values, ARENA_DIAMETER) # ציר Y התיאורטי

print("\n--- Statistics for Pairwise Euclidean Distances (Filtered Data) ---")
print(f"Empirical Mean (תוחלת מהנתונים): {mean_distance:.4f} cm")
print(f"Theoretical Mean (תוחלת תיאורטית): {theoretical_mean:.4f} cm")
print(f"Empirical Variance (שונות מהנתונים): {variance_distance:.4f} cm")

# 4. הצגת ההתפלגות בגרף (היסטוגרמה)
plt.figure(figsize=(12, 7))

plt.hist(distances_array, bins=50, edgecolor='black', alpha=0.6, 
         density=True, label='Empirical Data Distribution (Filtered)')

# הוספת הקו התיאורטי (PDF)
plt.plot(r_values, pdf_values, color='orange', linestyle='-', linewidth=3, 
         label='Theoretical PDF (All random points)')

# הוספת קו שמציין את התוחלת מהנתונים
plt.axvline(mean_distance, color='red', linestyle='--', linewidth=2, 
            label=f'Empirical Mean (Filtered): {mean_distance:.2f} cm')

# הוספת קו שמציין את התוחלת התיאורטית
plt.axvline(theoretical_mean, color='green', linestyle=':', linewidth=3, 
            label=f'Theoretical Mean (Random): {theoretical_mean:.2f} cm')

# --- שינוי: הוספת מידע על הסינון לכותרת ---
plt.title(f'Distribution of Pairwise Distances (n_r_squared > {MIN_NRS})\n(Session: {session}, Bin Size: {BIN_SIZE}cm)', fontsize=16)
plt.xlabel('Pairwise Euclidean Distance (cm)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.xlim(0, ARENA_DIAMETER) 

# הצג את הגרף
plt.show()
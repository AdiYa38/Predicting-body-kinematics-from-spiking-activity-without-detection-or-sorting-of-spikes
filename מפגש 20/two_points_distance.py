import numpy as np
import matplotlib.pyplot as plt

def analytical_pdf(r, D):
    """
    מחשבת את פונקציית צפיפות ההסתברות (PDF) התיאורטית
    של המרחק r בין שתי נקודות אקראיות בעיגול בקוטר D.
    """
    # מונע שגיאות חישוב בקצוות (r=0, r=D)
    epsilon = 1e-10
    r = np.clip(r, epsilon, D - epsilon)
    
    x = r / D  # מרחק מנורמל
    
    # החלק הראשון היה נכון
    term1 = (16 * r / (np.pi * D**2)) * np.arccos(x)
    
    # החלק השני - כאן היה התיקון (16 במקום 8)
    term2 = (16 * r**2 / (np.pi * D**3)) * np.sqrt(1 - x**2)
    
    return term1 - term2

# --- המשך הקוד שלך (הרצת הסימולציה) ---
# ... (כל הקוד של יצירת הנקודות והמרחקים נשאר זהה) ...

# --- הגדרות (לדוגמה כמו בתמונה) ---
D = 80.0
R = D / 2.0
N = 500000 # או כמה שהגדרת

# --- הרצת הסימולציה ---
theta1 = np.random.uniform(0, 2 * np.pi, N)
r1 = R * np.sqrt(np.random.uniform(0, 1, N))
x1 = r1 * np.cos(theta1)
y1 = r1 * np.sin(theta1)

theta2 = np.random.uniform(0, 2 * np.pi, N)
r2 = R * np.sqrt(np.random.uniform(0, 1, N))
x2 = r2 * np.cos(theta2)
y2 = r2 * np.sin(theta2)

distances = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# --- יצירת הגרף ---
plt.figure(figsize=(12, 7))

# א. שרטוט ההיסטוגרמה
plt.hist(distances, bins=100, density=True, alpha=0.7, 
         label='simulation result')

# ב. שרטוט הפונקציה האנליטית המתוקנת
r_values = np.linspace(0, D, 300)
pdf_values = analytical_pdf(r_values, D) # <-- שימוש בפונקציה המתוקנת

plt.plot(r_values, pdf_values, color='red', linewidth=3, 
         label='theoretical solution')

# ג. הוספת התוחלת
analytical_mean = (64 * D) / (45 * np.pi)
plt.axvline(analytical_mean, color='k', linestyle='--', linewidth=2, 
            label=f'theoretical mean {analytical_mean:.4f}')

# ... (המשך הגדרות הגרף: כותרות, צירים וכו') ...
plt.title(f"Distance ditribution of distances in a circle (D={D})", fontsize=16)
plt.xlabel("Distance (r)", fontsize=12)
plt.ylabel("(p(r))", fontsize=12)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.ylim(bottom=0)
plt.xlim(left=0, right=D)
plt.show()
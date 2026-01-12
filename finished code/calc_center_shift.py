import numpy as np
import matplotlib.pyplot as plt
import data

# Constants
session = "mP31_18"
X_CHANNEL = 59 #124
Y_CHANNEL = 60 #125
N_CHANNELS = 71 #136
DTYPE = np.int16
EEG_FILE = f"data/{session}/{session}.eeg"
eeg_data = data.get_eeg_data(EEG_FILE, DTYPE, N_CHANNELS)
ARENA_DIAMETER = 80


def get_eeg_channels(eeg_data, channel):
    """
    Extracts a specific channel from the EEG data.
    
    Parameters:
    - eeg_data: numpy array, EEG data with shape (n_samples, n_channel).
    - channel: int, index of the channel to extract.
    
    Returns:
    - channel_data: numpy array, data for the specified channel.
    """
    return eeg_data[:, channel]
    
def convert_to_cm(x_raw):
     # Calibration constants
    vRange = 10.0  # Volts
    vDAC = [-3.8e-3, 4.64]  # Volts
    maxBins = 640.0  # pixels
    pix2cm = 0.294  # cm
    nBits = 16.0
    
    x_raw = np.asarray(x_raw).astype(np.uint16)

    # The original conversion logic.
    xhat = (x_raw / (2**nBits) * vRange - vDAC[0]) / (vDAC[1] - vDAC[0])

    return xhat * maxBins * pix2cm  # in cm

# Calculate func
def calculate_arena_shift(eeg_data, x_chan, y_chan, arena_diameter, session, debug_plot=True):
    """
    מחשבת את ההזזה הנדרשת כדי למרכז את הזירה סביב (0,0).
    האלגוריתם מחפש את מרכז המעגל (ברדיוס הנתון) שמכיל את מספר הנקודות הגדול ביותר.
    """
    
    # Get location data
    raw_x = get_eeg_channels(eeg_data, x_chan)
    raw_y = get_eeg_channels(eeg_data, y_chan)
    
    x_cm = data.convert_to_cm(raw_x)
    y_cm = data.convert_to_cm(raw_y)
    
    # Remoce beggining and end of the session
    total_samples = len(x_cm)
    cull_percentage = 0.2 # אחוז הדגימות להסרה מכל קצה
    cull_samples = int(total_samples * cull_percentage)
    
    # קביעת הגבולות: הסר את cull_samples הראשונים והאחרונים
    start_index = cull_samples
    end_index = total_samples - cull_samples
    
    # סינון הנתונים
    x_cm_trimmed = x_cm[start_index:end_index]
    y_cm_trimmed = y_cm[start_index:end_index]
    
    # אם הוקטור ריק (קורה רק אם cull_percentage > 0.5), נשתמש בנתונים המקוריים
    if len(x_cm_trimmed) == 0:
        x_cm_trimmed, y_cm_trimmed = x_cm, y_cm

    # --- סוף התיקון ---
    
    # Reduce data to simple calculation (משתמש כעת בנתונים המסוננים)
    skip_step = 4
    x_sample = x_cm_trimmed[::skip_step]
    y_sample = y_cm_trimmed[::skip_step]
    
    # Reduce data and limit center poissible location
    search_margin = 10 # cm
    x_min, x_max = np.percentile(x_sample, 5), np.percentile(x_sample, 95)
    y_min, y_max = np.percentile(y_sample, 5), np.percentile(y_sample, 95)
    
    # Create net of possible centers
    x_centers = np.arange(x_min - search_margin, x_max + search_margin, 1)
    y_centers = np.arange(y_min - search_margin, y_max + search_margin, 1)
    
    arena_radius = arena_diameter / 2
    best_center = (0, 0)
    max_points_inside = -1

    # earc
    print("Searching for optimal arena center...")
    
    # Go over x axis, y axis
    for cx in x_centers:
        for cy in y_centers:
            # Calc the sum of square distances
            dist_sq = (x_sample - cx)**2 + (y_sample - cy)**2
            
            # Count points within the current arena
            count = np.sum(dist_sq <= arena_radius**2)
            
            if count > max_points_inside:
                max_points_inside = count
                best_center = (cx, cy)

    optimal_x_center, optimal_y_center = best_center
    
    # Calc shifts
    suggested_x_shift = -optimal_x_center
    suggested_y_shift = -optimal_y_center

    print(f"Found Center at: ({optimal_x_center:.2f}, {optimal_y_center:.2f})")
    print(f"Suggested Shifts -> x_shift: {suggested_x_shift:.2f}, y_shift: {suggested_y_shift:.2f}")

    # Plot
    if debug_plot:
        plt.figure(figsize=(6,6))
        plt.scatter(x_sample, y_sample, s=1, c='gray', alpha=0.5, label='Position Data')
        
        circle = plt.Circle(best_center, arena_radius, color='r', fill=False, linewidth=2, label='Detected Arena')
        plt.gca().add_patch(circle)
        plt.plot(optimal_x_center, optimal_y_center, 'rx', markersize=10)
        
        plt.title(f'Arena Detection for {session}\nShift: X={suggested_x_shift:.1f}, Y={suggested_y_shift:.1f}')
        plt.axis('equal')
        plt.legend()
        plt.show()

    return suggested_x_shift, suggested_y_shift

# Run
x_shift, y_shift = calculate_arena_shift(eeg_data, X_CHANNEL, Y_CHANNEL, ARENA_DIAMETER, session)

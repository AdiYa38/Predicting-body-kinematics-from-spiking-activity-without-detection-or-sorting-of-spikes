import numpy as np
import matplotlib.pyplot as plt
import heatmaps
import data
import prediction
import MUA
from scipy import signal
import matplotlib.patches as patches
from scipy.io import loadmat
import os


'''
This script processes multi-unit activity (MUA) data by removing periods around detected ripple events.
The signals numbers from matlab with these commands:
>> load( [ filebase '.sps' ], '-mat' )
>> vote 

'''
signals = np.array(['008' ,'016' ,'026' ,'036' ,'046' ,'055' ,'066','075', '087', '097', '111','118'])

ripple_samples = []

for sig in signals:
    file_path = f'mP79_17_sps_spw/mP79_17.spw.{sig}'
    file = loadmat(file_path)

    rips_struct = file['rips']
    rips_seg_data = rips_struct['seg'][0, 0]

    clean_data = rips_seg_data.reshape(-1, 2)
    data_as_list = clean_data.tolist()
    ripple_samples.extend(data_as_list)


def merge_overlapping_intervals(intervals):
    """
    Merges a list of event intervals (beginnings and endings) that overlap.

    Args:
        intervals (list of tuples/lists): A list where each element is
                                           a pair (start, end) representing an interval.

    Returns:
        list of tuples: A new list of non-overlapping, merged intervals.
    """
    if not intervals or len(intervals) < 2:
        return intervals

    # 2. Sort the intervals based on their starting times.
    intervals.sort(key=lambda x: x[0])

    # 3. Initialize the result list with the first sorted interval.
    merged = []
    # Use the first interval as the starting point for the comparison loop.
    current_start, current_end = intervals[0]

    # 4. Iterate through the rest of the sorted intervals.
    for next_start, next_end in intervals[1:]:
        if next_start <= current_end:
            current_end = max(current_end, next_end)
        else:
            merged.append((current_start, current_end))
            
            current_start, current_end = next_start, next_end

    merged.append((current_start, current_end))

    return merged


merged_events = merge_overlapping_intervals(ripple_samples)



#delete 100ms before and after each ripple
def delete_ripple_periods(x_smooth, y_smooth, mua, ripple_samples, fs=1250, pre_ripple_ms=100, post_ripple_ms=100):
    pre_ripple_samples = int((pre_ripple_ms / 1000) * fs)
    post_ripple_samples = int((post_ripple_ms / 1000) * fs)
    
    mask = np.ones(len(x_smooth), dtype=bool)
    
    for ripple_sample in ripple_samples:
        start = max(ripple_sample[0] - pre_ripple_samples, 0)
        end = min(ripple_sample[1] + post_ripple_samples, len(x_smooth))
        mask[start:end] = False
    
    return x_smooth[mask], y_smooth[mask], mua[mask]

ARENA_DIAMETER = 80
BIN_SIZE = 1
RES_SAMPLING_RATE = 20000
POS_SAMPLING_RATE = 1250
SESSION = "mP79_17"
DAT_file = f"data/{SESSION}/{SESSION}.dat"
EEG_FILE = f"data/{SESSION}/{SESSION}.eeg"
MUA_file = f"mua_{SESSION}_output.dat"
CHANNEL =  120
START_SAMPLE = 90*60*RES_SAMPLING_RATE
DURATION = 1 # Duration in seconds
KERNEL_SIZE = 7
DTYPE = np.int16

X_CHANNEL, Y_CHANNEL, N_CHANNELS = data.Channels(SESSION)
# --- Simulation Parameters ---

# Load EEG data
dat_data = data.get_eeg_data(DAT_file, np.int16, 136)
eeg_data = data.get_eeg_data(EEG_FILE, DTYPE, N_CHANNELS)
MUA_data = data.get_eeg_data(MUA_file, np.float32, N_CHANNELS)

# Create MUA signal
mua_signal, _ = MUA.create_mua_signal(dat_data, CHANNEL)
og_sig = data.get_eeg_channels(dat_data, CHANNEL)

# Load Data
x_values, y_values, x_in, y_in = data.import_position_data(eeg_data, X_CHANNEL, Y_CHANNEL, ARENA_DIAMETER, SESSION)
# 3. Create smoothing kernel
gaussian_kernel = heatmaps.create_gaussian_kernel(size=KERNEL_SIZE)

x_smooth, y_smooth = data.smooth_location(x_values, y_values)

#map mid ripple positions
ripple_samples = [((start + end)//2) for start, end in merged_events]

x_ripple = x_smooth[ripple_samples]
y_ripple = y_smooth[ripple_samples]

#plot x,y of middle ripple positions
plt.figure(figsize=(8, 6))
plt.scatter(x_ripple, y_ripple, s=1, alpha=0.6)  # s = dot size

ax = plt.gca()
circle = patches.Circle(
    (0, 0),
    40,
    facecolor='none',     # No fill color
    edgecolor='red',      # Set a visible color for the boundary
    linestyle='--',       # Use a dashed line
    linewidth=2,
    label='40 cm Radius'  # Label for the legend
)
ax.add_patch(circle)

plt.xlabel("X[cm]")
plt.ylabel("Y [cm]")
plt.title("XY Scatter Plot ( Ripple Positions)")
plt.axis("equal")
plt.grid(True)
plt.show()

bins_grid = heatmaps.create_bins(BIN_SIZE,ARENA_DIAMETER)

x_clean, y_clean, mua_clean = delete_ripple_periods(x_smooth, y_smooth, mua_signal, merged_events)

time_map_smoothed, new_vacants, bins_grid, vacants = heatmaps.smooth_map(
    data_array=None,
    x_values=x_clean,
    y_values=y_clean,
    vacants=None,
    BIN_SIZE=BIN_SIZE,
    time_data=True
)
mua_map_smoothed,_,_,_ = heatmaps.smooth_map(mua_clean,x_clean,y_clean,vacants,BIN_SIZE,time_data=False,from_mua=True)


# Final rates map
rates_map = mua_map_smoothed / time_map_smoothed
final_rates_map = heatmaps.remove_vacants(rates_map, new_vacants,True)
final_rates_map = heatmaps.remove_background(final_rates_map, bins_grid)
time_map_smoothed = heatmaps.remove_vacants(time_map_smoothed, new_vacants,True)
time_map_smoothed = heatmaps.remove_background(time_map_smoothed, bins_grid)

#plot final rates map
plt.figure(figsize=(8, 8))
plt.imshow(final_rates_map, origin='lower', cmap='jet',vmin = 1.5, vmax = 3.5)
plt.colorbar(label='MUA rate')
plt.title(f'Rate Map {CHANNEL} with Ripple Periods Removed')
plt.xlabel('X [cm]')
plt.ylabel('Y [cm]')
plt.show()

#plot occupancy map
plt.figure(figsize=(8, 8))
plt.imshow(time_map_smoothed, origin='lower', cmap='jet')
plt.colorbar(label='Occupancy Time')
plt.title(f'Occupancy Map with Ripple Periods Removed')
plt.xlabel('X [cm]')
plt.ylabel('Y [cm]')
plt.show()


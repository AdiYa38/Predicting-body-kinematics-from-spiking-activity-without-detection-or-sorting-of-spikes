import numpy as np
import matplotlib.pyplot as plt
import heatmaps
import data
import prediction
# --- Simulation Parameters ---
ARENA_DIAMETER = 100
BIN_SIZE = 50
RES_SAMPLING_RATE = 20000
POS_SAMPLING_RATE = 1250
X_CHANNEL = 124
Y_CHANNEL = 125
N_CHANNELS = 136
CELL_ID =  np.array([7,11,11, 4])#, 1, 2, 3, 4,1,2,3,4,1,2,3,4,1,2,3,4, 5,6,7,8, 6,7,8,5])
TETRODE_ID = np.array([1,8,2, 3])#, 1, 1, 1, 1,2,2,2,2,3,3,3,3,4,4,4,4, 5,5,5,5, 6,6,6,6])
KERNEL_SIZE = 7
EEG_FILE = "mp79_17/mP79_17.eeg"
DTYPE = np.int16

# Load Data
eeg_data = data.get_eeg_data(EEG_FILE, DTYPE, N_CHANNELS)
x_values, y_values, x_in, y_in = data.import_position_data(eeg_data, X_CHANNEL, Y_CHANNEL, ARENA_DIAMETER)
# 3. Create smoothing kernel
gaussian_kernel = heatmaps.create_gaussian_kernel(size=KERNEL_SIZE)

x_smooth, y_smooth = data.smooth_location(x_values, y_values)
# data.plot_mouse_animation(
#         x=x_smooth,
#         y=y_smooth,
#         start_time_minutes=90,
#         duration_seconds=100
#     )
# data.plot_mouse_animation(
#         x=x_values,
#         y=y_values,
#         start_time_minutes=90,
#         duration_seconds=100
#     )


# --- Plot  y location in time---
start_time_seconds = 90 * 60  # 90 minutes in seconds
duration_seconds = 5
num_points = int(duration_seconds * POS_SAMPLING_RATE)
time_axis = np.arange(num_points) / POS_SAMPLING_RATE
start_index = int(start_time_seconds * POS_SAMPLING_RATE)
y_raw_plot = y_values[start_index:start_index+num_points]
y_smooth_plot = y_smooth[start_index:start_index+num_points]
plt.figure(figsize=(12, 6))
plt.plot(time_axis, y_raw_plot, label='Raw Y-Position', color='lightgray', linewidth=2)
plt.plot(time_axis, y_smooth_plot, label='Smoothed Y-Position', color='blue', linewidth=2)
plt.title(f'Y-Position Data over {duration_seconds} Seconds (Sampling Rate: {POS_SAMPLING_RATE} Hz)', fontsize=16)
plt.xlabel('Time (seconds)', fontsize=12)
plt.ylabel('Y-Position', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# 1. Create the base grid
bins_grid = heatmaps.create_bins(BIN_SIZE,ARENA_DIAMETER)
occupancy_map_raw, vacants = heatmaps.calculate_time_in_bin(bins_grid, x_smooth, y_smooth, BIN_SIZE, ARENA_DIAMETER, POS_SAMPLING_RATE)
occupancy_map_smoothed, new_vacanta, bins_grid, vacants = heatmaps.smooth_map(None, x_smooth, y_smooth, None, BIN_SIZE,True )

res_list = []
final_rates_map = []

for tet, cell in zip(TETRODE_ID, CELL_ID):
    
    tet_res, clu = data.get_tetrode_spike_times("mp79_17/mP79_17.clu.", "mp79_17/mP79_17.res.", tet, POS_SAMPLING_RATE, RES_SAMPLING_RATE)
    res = data.get_cell_spike_times(clu, tet_res, cell)
    res_list.append(res)
   
 
    # --- Run Analysis Pipeline ---

    # 2. Calculate spike and time maps
    spike_map_raw = heatmaps.bins_spikes_count(bins_grid, res, x_smooth, y_smooth, BIN_SIZE, ARENA_DIAMETER)
    
    # 4. Perform smoothing
    spike_map_smoothed,_,_,_ =heatmaps.smooth_map(res,x_smooth, y_smooth, vacants, BIN_SIZE)

    # Final rates map
    rates_map = spike_map_raw / occupancy_map_raw
    final_rates_map.append(heatmaps.remove_vacants(rates_map, new_vacanta))


#=== test prediction success ===
# Recording parameters
recording_duration_sec = 10800  # 3 hours
min_start = 1800   # after 30 min
max_start = 9000   # before last 30 min
n_windows = 1000

# Generate random start times (in seconds)
np.random.seed(42)  # for reproducibility
start_times = np.random.randint(min_start, max_start, size=n_windows)
test_duration = 40.0

#lambda_map = prediction.lambda_rate_per_bin(spike_map_smoothed, occupancy_map_smoothed)
prior_map = prediction.bins_prior(occupancy_map_raw)

prediction_bins = []
actual_bins = []

for start in start_times:
  actual_bin = prediction.get_actual_bin(x_smooth, y_smooth, start, test_duration, BIN_SIZE, ARENA_DIAMETER, POS_SAMPLING_RATE)
  if actual_bin is None:
        continue
  
  actual_bins.append(actual_bin)


  log_poiss = prediction.MB_PBR(res_list, start, test_duration, final_rates_map)
  predicted_bin = prediction.MB_MAP_estimator(log_poiss, prior_map)
  prediction_bins.append(predicted_bin)
  
  
accuracy = prediction.prediction_quality(prediction_bins, actual_bins)
print(f"\nPrediction Accuracy: {accuracy:.2f}% over {len(start_times)} windows")

# === plot preditions vs time graph
durations = np.arange(1, 100, 0.5)
accuracies = []
chance_levels = []
errors = []

for duration in durations:
    
    prediction_bins = []
    actual_bins = []

    for start in start_times:
        log_poiss = prediction.MB_PBR(res_list, start, duration, final_rates_map)
        predicted_bin = prediction.MB_MAP_estimator(log_poiss, prior_map)
        prediction_bins.append(predicted_bin)

        actual_bin = prediction.get_actual_bin(
            x_smooth, y_smooth, start, duration,
            BIN_SIZE, ARENA_DIAMETER, POS_SAMPLING_RATE
        )
        if actual_bin is None:
            continue
        actual_bins.append(actual_bin)

    # build bins for current BIN_SIZE
    bins_grid = heatmaps.create_bins(BIN_SIZE, ARENA_DIAMETER)

    # accuracy
    acc = prediction.prediction_quality(prediction_bins, actual_bins)
    accuracies.append(acc)

    # chance (only inside bins)
    n_inside_bins = np.sum(bins_grid == heatmaps.INSIDE_FLAG)
    chance_levels.append(100 / n_inside_bins)

    # error
    err = data.average_error(prediction_bins, actual_bins, BIN_SIZE, ARENA_DIAMETER, bins_grid)
    errors.append(err)
    
# Corrected plotting code for the first graph
fig, ax1 = plt.subplots(figsize=(8, 5))

# Plot accuracy and chance on the primary y-axis (ax1)
ax1.plot(durations, accuracies, marker='o', color='b', label="Accuracy")
ax1.plot(durations, chance_levels, linestyle='--', color='r', label="Chance")
ax1.set_xlabel('Test Duration (s)')
ax1.set_ylabel('Prediction Accuracy (%)')
ax1.set_title('Prediction Accuracy vs Test Duration')
ax1.grid(True)

# Create a secondary y-axis that shares the same x-axis
ax2 = ax1.twinx()  
ax2.plot(durations, errors, linestyle='-', color='g', label="Error (cm)")
ax2.set_ylabel('Error (cm)')

# Combine legends from both axes
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='best')

plt.tight_layout()
plt.show()



# === plot preditions vs bin size 
# === plot predictions vs bin size 
bin_sizes = [1, 2, 4, 5, 10, 20, 25, 50, 100]
accuracies = []
chance_levels = []
errors = []


for bin_size in bin_sizes:
    bins_grid = heatmaps.create_bins(bin_size,ARENA_DIAMETER)
    occupancy_map_raw, vacants = heatmaps.calculate_time_in_bin(bins_grid, x_smooth, y_smooth, bin_size, ARENA_DIAMETER, POS_SAMPLING_RATE)
    occupancy_map_smoothed, new_vacants, bins_grid, vacants = heatmaps.smooth_map(None, x_smooth, y_smooth, None, BIN_SIZE,True )
    prior_map = prediction.bins_prior(occupancy_map_smoothed)
    res_list = []
    final_rates_map = []

    for tet, cell in zip(TETRODE_ID, CELL_ID):
        
        tet_res, clu = data.get_tetrode_spike_times("mp79_17/mP79_17.clu.", "mp79_17/mP79_17.res.", tet, POS_SAMPLING_RATE, RES_SAMPLING_RATE)
        res = data.get_cell_spike_times(clu, tet_res, cell)
        res_list.append(res)
    
    
        # --- Run Analysis Pipeline ---

        # 2. Calculate spike and time maps
        spike_map_raw = heatmaps.bins_spikes_count(bins_grid, res, x_smooth, y_smooth, bin_size, ARENA_DIAMETER)
        
        # 4. Perform smoothing
        spike_map_smoothed,_,_,_ =heatmaps.smooth_map(res,x_smooth, y_smooth, vacants, BIN_SIZE)

        # Final rates map
        rates_map = spike_map_smoothed / occupancy_map_smoothed
        final_rates_map.append(heatmaps.remove_vacants(rates_map, new_vacants))

    prediction_bins = []
    actual_bins = []

    for start in start_times:
        log_poiss = prediction.MB_PBR(res_list, start, test_duration, final_rates_map)
        predicted_bin = prediction.MB_MAP_estimator(log_poiss, prior_map)
        prediction_bins.append(predicted_bin)

        actual_bin = prediction.get_actual_bin(
            x_smooth, y_smooth, start, test_duration,
            bin_size, ARENA_DIAMETER, POS_SAMPLING_RATE
        )
        if actual_bin is None:
            continue
        actual_bins.append(actual_bin)

    # build bins for current bin size
    bins_grid = heatmaps.create_bins(bin_size, ARENA_DIAMETER)

    # accuracy
    acc = prediction.prediction_quality(prediction_bins, actual_bins)
    accuracies.append(acc)

    # chance
    n_inside_bins = np.sum(bins_grid == heatmaps.INSIDE_FLAG)
    chance_levels.append(100 / n_inside_bins)

    # error
    err = data.average_error(prediction_bins, actual_bins, bin_size, ARENA_DIAMETER, bins_grid)
    errors.append(err)

    

# Corrected plotting code for the second graph
fig, ax1 = plt.subplots(figsize=(8, 5))

# Plot accuracy and chance on the primary y-axis (ax1)
ax1.plot(bin_sizes, accuracies, marker='o', color='b', label="Accuracy")
ax1.plot(bin_sizes, chance_levels, linestyle='--', color='r', label="Chance")
ax1.set_xlabel('Bin Size (cm)')
ax1.set_ylabel('Prediction Accuracy (%)')
ax1.set_title('Prediction Accuracy vs Bin Size')
ax1.grid(True)

# Create a secondary y-axis that shares the same x-axis
ax2 = ax1.twinx()  
ax2.plot(bin_sizes, errors, linestyle='-', color='g', label="Error (cm)")
ax2.set_ylabel('Error (cm)')

# Combine legends from both axes
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='best')

plt.tight_layout()
plt.show()


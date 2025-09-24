import numpy as np
import matplotlib.pyplot as plt
import heatmaps
import data
import prediction
import MUA
from scipy import signal
# --- Simulation Parameters ---
ARENA_DIAMETER = 100
BIN_SIZE = 2
RES_SAMPLING_RATE = 20000
POS_SAMPLING_RATE = 1250
X_CHANNEL = 124
Y_CHANNEL = 125
N_CHANNELS = 136
DAT_file = "dat/mP79_17.dat"
EEG_FILE = "mp79_17/mP79_17.eeg"
CHANNEL = 50
START_SAMPLE = 57*60*RES_SAMPLING_RATE

DURATION = 1.0 # Duration in seconds
KERNEL_SIZE = 7
DTYPE = np.int16
init_bin_size = 1 


# --- Simulation Parameters ---

# Load EEG data
dat_data = data.get_eeg_data(DAT_file, np.int16, 136)
eeg_data = data.get_eeg_data(EEG_FILE, DTYPE, N_CHANNELS)

# Create MUA signal
mua_signal, og_sig = MUA.create_mua_signal(dat_data, CHANNEL, START_SAMPLE, DURATION, RES_SAMPLING_RATE)


# --- Plotting the signals in time ---
# Create a time array for the x-axis
show_mua = mua_signal[int(START_SAMPLE/16):int(START_SAMPLE/16) + int(DURATION * POS_SAMPLING_RATE)]
time_axis_mua = np.linspace(0, DURATION, len(show_mua), endpoint=False)
show_sig = og_sig[START_SAMPLE:START_SAMPLE + int(DURATION * RES_SAMPLING_RATE)]
time_axis_og = np.linspace(0, DURATION, len(show_sig), endpoint=False)

y_min = np.min(og_sig)
y_max = np.max(og_sig)

plt.figure(figsize=(12, 6))

# Plot the original signal with fixed y-axis limits
plt.subplot(2, 1, 1)
plt.plot(time_axis_og, show_sig)
plt.title(f'Original Signal for Channel {CHANNEL}')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude[microvolts]')
plt.grid()
# plt.ylim(y_min, y_max) # Set the y-axis limits

# Plot the MUA signal with the same fixed y-axis limits
plt.subplot(2, 1, 2)
plt.plot(time_axis_mua, show_mua, color='orange')
plt.title(f'MUA Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude[microvolts]')
plt.grid()
#plt.ylim(y_min, y_max) # Set the same y-axis limits

plt.tight_layout()
plt.show()


# Load Data

x_values, y_values, x_in, y_in = data.import_position_data(eeg_data, X_CHANNEL, Y_CHANNEL, ARENA_DIAMETER)
# 3. Create smoothing kernel
gaussian_kernel = heatmaps.create_gaussian_kernel(size=KERNEL_SIZE)

x_smooth, y_smooth = data.smooth_location(x_values, y_values)



bins_grid_1cm = heatmaps.create_bins(init_bin_size, arena_diameter_cm=ARENA_DIAMETER)
bins_grid = heatmaps.create_bins(BIN_SIZE,ARENA_DIAMETER)
occupancy_map_raw, vacants = heatmaps.calculate_time_in_bin(bins_grid_1cm, x_smooth, y_smooth, init_bin_size, ARENA_DIAMETER, POS_SAMPLING_RATE)
occupancy_map_raw_binsize, vacants_binsize = heatmaps.calculate_time_in_bin(bins_grid, x_smooth, y_smooth, BIN_SIZE, ARENA_DIAMETER, POS_SAMPLING_RATE)


mua_map_raw = MUA.bin_mua_count(bins_grid_1cm, mua_signal, x_smooth, y_smooth, init_bin_size, ARENA_DIAMETER)
mua_map_raw_binsize = MUA.bin_mua_count(bins_grid, mua_signal, x_smooth, y_smooth, BIN_SIZE, ARENA_DIAMETER)
mua_map_covered = heatmaps.cover_vacants(bins_grid_1cm, mua_map_raw, vacants)
occupancy_map_covered = heatmaps.cover_vacants(bins_grid_1cm, occupancy_map_raw, vacants)
mua_map_smoothed = heatmaps.smooth(mua_map_covered, gaussian_kernel, bins_grid_1cm)
occupancy_map_smoothed = heatmaps.smooth(occupancy_map_covered, gaussian_kernel, bins_grid_1cm)


mua_map_sized, new_vacants, bins_grid = heatmaps.change_grid(mua_map_smoothed, BIN_SIZE, ARENA_DIAMETER, vacants, init_bin_size)
occupancy_map_sized, new_vacants, bins_grid = heatmaps.change_grid(occupancy_map_smoothed, BIN_SIZE, ARENA_DIAMETER, vacants, init_bin_size)
# Final rates map
mua_rates_map = mua_map_sized / occupancy_map_sized
mua_rates_map = heatmaps.remove_vacants(mua_rates_map, new_vacants, True)

# --- Plotting spike_map_raw as a heatmap ---
plt.figure(figsize=(8, 8))
# Use imshow to create the heatmap from the 2D array
# 'origin="lower"' ensures the (0,0) index is at the bottom-left, matching a typical spatial grid
# 'cmap="jet"' sets the color map for the heatmap
plt.imshow(mua_map_raw_binsize, origin='lower', cmap='jet')

# Add a color bar to show the scale of the spike counts
plt.colorbar(label='Spike Count')

plt.title(f'Raw MUA Map of channel {CHANNEL} ')
plt.xlabel('X [cm]')
plt.ylabel('Y [cm]')
plt.show()

# ===================================================================
# Visualization
# ===================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle(f"Spatial Analysis Results of channel {CHANNEL}", fontsize=20)

# Helper function for plotting
def plot_map(ax, data, title, cmap='jet'):
    data_to_plot = np.copy(data).astype(float)
    if -1 in data:
      data_to_plot[data == -1] = np.nan
    
    max_val = heatmaps.max_val_to_show(data_to_plot)
    im = ax.imshow(data_to_plot, cmap=cmap, origin='lower', interpolation='nearest', vmax=max_val)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("X[cm]")
    ax.set_ylabel("Y[cm]")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    

# Plot the four main results
plot_map(axes[0, 0], mua_map_raw_binsize, "Raw MUA Map")
plot_map(axes[0, 1], occupancy_map_raw_binsize, "Raw Time Map")
plot_map(axes[1, 0], mua_map_sized, "Smoothed MUA Map")
plot_map(axes[1, 1], occupancy_map_sized, "Smoothed Time Map")#

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.subplots_adjust(hspace=0.3, wspace=0.3)
#plt.show()

# Optional: Plot the kernel separately
# fig_kernel, ax_kernel = plt.subplots(figsize=(5, 5))
# plot_map(ax_kernel, gaussian_kernel, "Gaussian Kernel", cmap='viridis')
#plt.show()

# Plot final retes map
fig_rts, rts_ax = plt.subplots(figsize=(5, 5))
plot_title = f"Final Rates Map (channel {CHANNEL})"
plot_map(rts_ax, mua_rates_map,plot_title, cmap='jet')
plt.show()



# --- Simulation Parameters ---
ARENA_DIAMETER = 100
BIN_SIZE = 50
RES_SAMPLING_RATE = 20000
POS_SAMPLING_RATE = 1250
X_CHANNEL = 124
Y_CHANNEL = 125 
N_CHANNELS = 136
CHANNELS = np.array([1,50])
KERNEL_SIZE = 7
EEG_FILE = "mp79_17/mP79_17.eeg"
DTYPE = np.int16


# 1. Create the base grid
time_map_smoothed,vacants,bins_grid = MUA.occupancy_map(x_values, y_values, init_bin_size,KERNEL_SIZE,ARENA_DIAMETER, POS_SAMPLING_RATE)


res_list = []
final_rates_map = []

for channel in CHANNELS:
    
    mua_signal, og_sig = MUA.create_mua_signal(dat_data, channel, START_SAMPLE, DURATION, RES_SAMPLING_RATE)
    og_sig = signal.resample_poly(og_sig, 1250, RES_SAMPLING_RATE)
    res_list.append(mua_signal)
 
    spike_map_raw = MUA.bin_mua_count(bins_grid, mua_signal, x_smooth, y_smooth, BIN_SIZE, ARENA_DIAMETER)
    rates_map = MUA.MUA_rate_map(mua_signal, x_values, y_values, occupancy_map_sized, vacants ,bins_grid, BIN_SIZE,KERNEL_SIZE,ARENA_DIAMETER, POS_SAMPLING_RATE)
    final_rates_map.append(heatmaps.remove_vacants(rates_map, vacants))


#=== test prediction success ===
# Recording parameters
recording_duration_sec = 10800  # 3 hours
min_start = 1800   # after 30 min
max_start = 9000   # before last 30 min
n_windows = 100

# Generate random start times (in seconds)
np.random.seed(42)  # for reproducibility
start_times = np.random.randint(min_start, max_start, size=n_windows)
test_duration = 6.0

# lambda_map = prediction.lambda_rate_per_bin(spike_map_smoothed, time_map_smoothed)
prior_map = prediction.bins_prior(time_map_smoothed)

prediction_bins = []
actual_bins = []

for start in start_times:
  actual_bin = prediction.get_actual_bin(x_smooth, y_smooth, start, test_duration, BIN_SIZE, ARENA_DIAMETER, POS_SAMPLING_RATE)
  if actual_bin is None:
        continue
  
  actual_bins.append(actual_bin)


  log_poiss = prediction.MUA_PBR(res_list, start, test_duration, final_rates_map)
  predicted_bin = prediction.MB_MAP_estimator(log_poiss, prior_map)
  prediction_bins.append(predicted_bin)
  
  
accuracy = prediction.prediction_quality(prediction_bins, actual_bins)
print(f"\nPrediction Accuracy: {accuracy:.2f}% over {len(start_times)} windows")

# === plot preditions vs time graph
durations = np.arange(1, 50, 1)
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
bin_sizes = [1, 2, 4, 5, 10,20, 25, 50, 100]
accuracies = []
chance_levels = []
errors = []

for bin_size in bin_sizes:
    time_map_smoothed,vacants,bins_grid = MUA.occupancy_map(x_values, y_values, bin_size,KERNEL_SIZE,ARENA_DIAMETER, POS_SAMPLING_RATE)
    prior_map = prediction.bins_prior(time_map_smoothed)
    final_rates_map = []
    prediction_bins = []
    actual_bins = []
    for channel in CHANNELS:
        rates_map = MUA.MUA_rate_map(res_list[channel], x_values, y_values, time_map_smoothed, vacants ,bins_grid, bin_size,KERNEL_SIZE,ARENA_DIAMETER, POS_SAMPLING_RATE)
        final_rates_map.append(heatmaps.remove_vacants(rates_map, vacants))


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


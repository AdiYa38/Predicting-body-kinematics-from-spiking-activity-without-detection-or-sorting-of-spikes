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
CHANNEL = 110
START_SAMPLE = 2253000
DURATION = 0.1 # Duration in seconds
SAMPLE_RATE = 20000  # Original sampling rate of the EEG data
KERNEL_SIZE = 7
DTYPE = np.int16


# --- Simulation Parameters ---

# Load EEG data
dat_data = data.get_eeg_data(DAT_file, np.int16, 136)
eeg_data = data.get_eeg_data(EEG_FILE, DTYPE, N_CHANNELS)

# Create MUA signal
mua_signal, og_sig = MUA.create_mua_signal(dat_data, CHANNEL, START_SAMPLE, DURATION, SAMPLE_RATE)
og_sig = signal.resample_poly(og_sig, 1250, SAMPLE_RATE)

# --- Plotting the signals in time ---
# Create a time array for the x-axis
show_mua = mua_signal[START_SAMPLE:START_SAMPLE + int(DURATION * SAMPLE_RATE)]
time_axis = np.linspace(0, DURATION, len(show_mua), endpoint=False)
show_sig = og_sig[START_SAMPLE:START_SAMPLE + int(DURATION * SAMPLE_RATE)]
y_min = np.min(og_sig)
y_max = np.max(og_sig)

plt.figure(figsize=(12, 6))

# Plot the original signal with fixed y-axis limits
plt.subplot(2, 1, 1)
plt.plot(time_axis, show_sig)
plt.title(f'Original Signal for Channel {CHANNEL}')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude[microvolts]')
plt.grid()
# plt.ylim(y_min, y_max) # Set the y-axis limits

# Plot the MUA signal with the same fixed y-axis limits
plt.subplot(2, 1, 2)
plt.plot(time_axis, show_mua, color='orange')
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




bins_grid = heatmaps.create_bins(BIN_SIZE,ARENA_DIAMETER)
time_map_raw = heatmaps.calculate_time_in_bin(bins_grid, x_values, y_values, BIN_SIZE, ARENA_DIAMETER, POS_SAMPLING_RATE)



spike_map_raw, vacants = MUA.bin_mua_count(bins_grid, mua_signal, x_smooth, y_smooth, BIN_SIZE, ARENA_DIAMETER)



spike_map_smoothed = heatmaps.smooth(spike_map_raw, gaussian_kernel, bins_grid)
time_map_smoothed = heatmaps.smooth(time_map_raw, gaussian_kernel, bins_grid)

# Final rates map
rates_map = spike_map_smoothed / time_map_smoothed
final_rates_map = heatmaps.remove_vacants(rates_map, vacants)
spike_map_raw[spike_map_raw == 0] = np.nan


# --- Plotting spike_map_raw as a heatmap ---
plt.figure(figsize=(8, 8))
# Use imshow to create the heatmap from the 2D array
# 'origin="lower"' ensures the (0,0) index is at the bottom-left, matching a typical spatial grid
# 'cmap="jet"' sets the color map for the heatmap
plt.imshow(spike_map_raw, origin='lower', cmap='jet')

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
fig.suptitle("Spatial Analysis Results", fontsize=20)

# Helper function for plotting
def plot_map(ax, data, title, cmap='jet'):
    data_to_plot = np.copy(data).astype(float)
    if -1 in data:
      data_to_plot[data == -1] = np.nan
    
    im = ax.imshow(data_to_plot, cmap=cmap, origin='lower', interpolation='nearest')
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("X[cm]")
    ax.set_ylabel("Y[cm]")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    

# Plot the four main results
plot_map(axes[0, 0], spike_map_raw, "Raw MUA Map")
plot_map(axes[0, 1], time_map_raw, "Raw Time Map")
plot_map(axes[1, 0], spike_map_smoothed, "Smoothed MUA Map")
plot_map(axes[1, 1], time_map_smoothed, "Smoothed Time Map")#

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
plot_map(rts_ax, final_rates_map,plot_title, cmap='jet')
plt.show()



# --- Simulation Parameters ---
ARENA_DIAMETER = 100
BIN_SIZE = 50
RES_SAMPLING_RATE = 20000
POS_SAMPLING_RATE = 1250
X_CHANNEL = 124
Y_CHANNEL = 125 
N_CHANNELS = 136
CHANNELS =  np.arange(0, 10, 1)
KERNEL_SIZE = 7
EEG_FILE = "mp79_17/mP79_17.eeg"
DTYPE = np.int16


# 1. Create the base grid
bins_grid = heatmaps.create_bins(BIN_SIZE,ARENA_DIAMETER)
time_map_raw = heatmaps.calculate_time_in_bin(bins_grid, x_smooth, y_smooth, BIN_SIZE, ARENA_DIAMETER, POS_SAMPLING_RATE)
time_map_smoothed = heatmaps.smooth(time_map_raw, gaussian_kernel, bins_grid)

res_list = []
final_rates_map = []

for channel in CHANNELS:
    
    mua_signal, og_sig = MUA.create_mua_signal(dat_data, channel, START_SAMPLE, DURATION, SAMPLE_RATE)
    og_sig = signal.resample_poly(og_sig, 1250, SAMPLE_RATE)
    res_list.append(mua_signal)
 
    # --- Run Analysis Pipeline ---

    # 2. Calculate spike and time maps
    spike_map_raw, vacants = MUA.bin_mua_count(bins_grid, mua_signal, x_smooth, y_smooth, BIN_SIZE, ARENA_DIAMETER)    
    # 4. Perform smoothing
    spike_map_smoothed =heatmaps.smooth(spike_map_raw, gaussian_kernel, bins_grid)

    # Final rates map
    rates_map = spike_map_smoothed / time_map_smoothed
    final_rates_map.append(heatmaps.remove_vacants(rates_map, vacants))


#=== test prediction success ===
# Recording parameters
recording_duration_sec = 10800  # 3 hours
min_start = 1800   # after 30 min
max_start = 9000   # before last 30 min
n_windows = 50

# Generate random start times (in seconds)
np.random.seed(42)  # for reproducibility
start_times = np.random.randint(min_start, max_start, size=n_windows)
test_duration = 6.0

lambda_map = prediction.lambda_rate_per_bin(spike_map_smoothed, time_map_smoothed)
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
bin_sizes = [1, 2, 3, 4, 5, 6, 8, 10, 15,20, 25, 33, 42, 50, 60, 75, 100]
accuracies = []
chance_levels = []
errors = []

for bin_size in bin_sizes:
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


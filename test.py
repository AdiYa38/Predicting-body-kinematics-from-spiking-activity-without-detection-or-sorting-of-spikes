import numpy as np
import matplotlib.pyplot as plt
import heatmaps
import data
import prediction
# --- Simulation Parameters ---
ARENA_DIAMETER = 100
BIN_SIZE = 2
RES_SAMPLING_RATE = 20000
POS_SAMPLING_RATE = 1250
X_CHANNEL = 124
Y_CHANNEL = 125
N_CHANNELS = 136
CELL_ID = 11
TETRODE_ID = 8
KERNEL_SIZE =7 
EEG_FILE = "mp79_17/mP79_17.eeg"
DTYPE = np.int16



# Load Data
eeg_data = data.get_eeg_data(EEG_FILE, DTYPE, N_CHANNELS)
tet_res, clu = data.get_tetrode_spike_times("mp79_17/mP79_17.clu.", "mp79_17/mP79_17.res.", TETRODE_ID, POS_SAMPLING_RATE, RES_SAMPLING_RATE)

x_values, y_values, x_in, y_in = data.import_position_data(eeg_data, X_CHANNEL, Y_CHANNEL, ARENA_DIAMETER)

#plot x,y 
# plt.figure(figsize=(8, 6))
# plt.scatter(x_values, y_values, s=1, alpha=0.6)  # s = dot size
# plt.xlabel("X [cm]")
# plt.ylabel("Y [cm]")
# plt.title("XY Scatter Plot (Position)")
# plt.axis("equal")
# plt.grid(True)
# plt.show()
# plt.figure(figsize=(8, 6))
# plt.scatter(x_in, y_in, s=1, alpha=0.6)  # s = dot size
# plt.xlabel("X [cm]")
# plt.ylabel("Y [cm]")
# plt.title("XY Scatter Plot (Position)")
# plt.axis("equal")
# plt.grid(True)
# plt.show()

# --- Run Analysis Pipeline ---
# 1. Create the base grid

res = data.get_cell_spike_times(clu, tet_res, CELL_ID)
bins_grid = heatmaps.create_bins(BIN_SIZE,ARENA_DIAMETER)

# 2. Calculate spike and time maps
spike_map_raw = heatmaps.bins_spikes_count(bins_grid, res, x_values, y_values, BIN_SIZE, ARENA_DIAMETER)
time_map_raw, vacants = heatmaps.calculate_time_in_bin(bins_grid, x_values, y_values, BIN_SIZE, ARENA_DIAMETER, POS_SAMPLING_RATE)

# 3. Create smoothing kernel
gaussian_kernel = heatmaps.create_gaussian_kernel(size=KERNEL_SIZE)

# 4. Perform smoothing
occupancy_map_smoothed,new_vacants, bins_grid,vacants  = heatmaps.smooth_map(None,x_values, y_values, None, BIN_SIZE,True)
spike_map_smoothed,_,_,_ = heatmaps.smooth_map(res, x_values, y_values,vacants, BIN_SIZE)

# Final rates map
rates_map = spike_map_smoothed / occupancy_map_smoothed
final_rates_map = heatmaps.remove_vacants(rates_map, new_vacants, True)
final_rates_map = heatmaps.remove_background(final_rates_map, bins_grid)

spike_map_plot = heatmaps.remove_vacants(spike_map_raw, new_vacants, True)
time_map_plot = heatmaps.remove_vacants(time_map_raw, new_vacants, True)
spike_map_smoothed_plot = heatmaps.remove_vacants(spike_map_smoothed, new_vacants, True)
occupancy_map_smoothed_plot = heatmaps.remove_vacants(occupancy_map_smoothed, new_vacants, True)
# ===================================================================
# Visualization
# ===================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle(f"Spatial Analysis Results for Cell {TETRODE_ID}.{CELL_ID} ", fontsize=20)

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
plot_map(axes[0, 0], spike_map_plot, "Spike Count Map")
plot_map(axes[0, 1], time_map_plot, "Occupancy Map")
plot_map(axes[1, 0], spike_map_smoothed_plot, "Smoothed Spike Count Map")
plot_map(axes[1, 1], occupancy_map_smoothed_plot, "Smoothed Occupancy Map")#

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.subplots_adjust(hspace=0.3, wspace=0.3)
#plt.show()

# Optional: Plot the kernel separately
# fig_kernel, ax_kernel = plt.subplots(figsize=(5, 5))
# plot_map(ax_kernel, gaussian_kernel, "Gaussian Kernel", cmap='viridis')
#plt.show()

# Plot final retes map
fig_rts, rts_ax = plt.subplots(figsize=(5, 5))
plot_title = f"Final Rates Map (Cell {TETRODE_ID}.{CELL_ID})"
plot_map(rts_ax, final_rates_map,plot_title, cmap='jet')
plt.show()


#=== test predictions ===
# i ran it on bin_size =20
 
# Set test window parameters
test_start_time = 1000  # in seconds
test_duration = 6.0    # window duration in seconds

# 1. Compute λ (spikes/sec) and prior
#lambda_map = prediction.lambda_rate_per_bin(spike_map_smoothed, occupancy_map_smoothed)
prior_map = prediction.bins_prior(occupancy_map_smoothed)

# 2. Predict bin using PBR and MAP estimator
log_poiss = prediction.PBR(res, test_start_time, test_duration, final_rates_map)
predicted_bin = prediction.MAP_estimator(log_poiss, prior_map)

# 3. Get the actual bin
actual_bin = prediction.get_actual_bin(x_values, y_values, test_start_time, test_duration, BIN_SIZE, ARENA_DIAMETER, POS_SAMPLING_RATE)

# 4. Print and compare results
print(f"\nPrediction Results (Window: {test_start_time}s to {test_start_time + test_duration}s)")
print(f"Predicted Bin (MAP): {predicted_bin}")
print(f"Actual Bin (from avg position): {actual_bin}")

if predicted_bin == actual_bin:
    print("✅ Prediction matched the actual bin.")
elif predicted_bin is None or actual_bin is None:
    print("⚠️ One of the bins is undefined (outside arena or invalid).")
else:
    print("❌ Prediction did NOT match the actual bin.")


#=== test prediction success ===
# Recording parameters
recording_duration_sec = 10800  # 3 hours
min_start = 1800   # after 30 min
max_start = 9000   # before last 30 min
n_windows = 1000

# Generate random start times (in seconds)
np.random.seed(42)  # for reproducibility
start_times = np.random.randint(min_start, max_start, size=n_windows)
test_duration = 5.0

#lambda_map = prediction.lambda_rate_per_bin(spike_map_smoothed, occupancy_map_smoothed)
prior_map = prediction.bins_prior(occupancy_map_smoothed)

prediction_bins = []
actual_bins = []

for start in start_times:

  log_poiss = prediction.PBR(res, start, test_duration, final_rates_map)
  predicted_bin = prediction.MAP_estimator(log_poiss, prior_map)
  prediction_bins.append(predicted_bin)
  
  actual_bin = prediction.get_actual_bin(x_values, y_values, start, test_duration, BIN_SIZE, ARENA_DIAMETER, POS_SAMPLING_RATE)
  actual_bins.append(actual_bin)

accuracy = prediction.prediction_quality(prediction_bins, actual_bins)
print(f"\nPrediction Accuracy: {accuracy:.2f}% over {len(start_times)} windows")

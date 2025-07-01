import numpy as np
import matplotlib.pyplot as plt
import heatmaps
import data

# --- Simulation Parameters ---
ARENA_DIAMETER = 100
BIN_SIZE = 2
RES_SAMPLING_RATE = 20000
POS_SAMPLING_RATE = 1250
X_CHANNEL = 124
Y_CHANNEL = 125
N_CHANNELS = 136
CELL_ID = 7
TETRODE_ID = 1
KERNEL_SIZE = 7
EEG_FILE = "mp79_17/mP79_17.eeg"
DTYPE = np.int16

# Load Data
eeg_data = data.get_eeg_data(EEG_FILE, DTYPE, N_CHANNELS)
tet_res, clu = data.get_tetrode_spike_times("mp79_17/mP79_17.clu.", "mp79_17/mP79_17.res.", TETRODE_ID, POS_SAMPLING_RATE, RES_SAMPLING_RATE)
res = data.get_cell_spike_times(clu, tet_res, CELL_ID)
x_values, y_values, _, _ = data.import_position_data(eeg_data, X_CHANNEL, Y_CHANNEL, ARENA_DIAMETER)
# DEBUG
print(tet_res[-1])
print(res[-1])

# --- Run Analysis Pipeline ---
# 1. Create the base grid
bins_grid = heatmaps.create_bins(BIN_SIZE,ARENA_DIAMETER)

# 2. Calculate spike and time maps
spike_map_raw, _ = heatmaps.bins_spikes_count(bins_grid, res, x_values, y_values, BIN_SIZE, ARENA_DIAMETER)
time_map_raw = heatmaps.calculate_time_in_bin(bins_grid, x_values, y_values, BIN_SIZE, ARENA_DIAMETER, POS_SAMPLING_RATE)

# 3. Create smoothing kernel
gaussian_kernel = heatmaps.create_gaussian_kernel(size=KERNEL_SIZE)

# 4. Perform smoothing
spike_map_smoothed = heatmaps.convolve(spike_map_raw, gaussian_kernel)
time_map_smoothed = heatmaps.convolve(time_map_raw, gaussian_kernel)

# ===================================================================
# Visualization
# ===================================================================
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
fig.suptitle("Spatial Analysis Results", fontsize=20)

# Helper function for plotting
def plot_map(ax, data, title, cmap='jet'):
    data_to_plot = np.copy(data).astype(float)
    if -1 in data:
      data_to_plot[data == -1] = np.nan
    
    im = ax.imshow(data_to_plot, cmap=cmap, origin='lower', interpolation='nearest')
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Bin X")
    ax.set_ylabel("Bin Y")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Plot the four main results
plot_map(axes[0, 0], spike_map_raw, "Raw Spike Map")
plot_map(axes[0, 1], time_map_raw, "Raw Time Map")
plot_map(axes[1, 0], spike_map_smoothed, "Smoothed Spike Map")
plot_map(axes[1, 1], time_map_smoothed, "Smoothed Time Map")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Optional: Plot the kernel separately
fig_kernel, ax_kernel = plt.subplots(figsize=(5, 5))
plot_map(ax_kernel, gaussian_kernel, "Gaussian Kernel", cmap='viridis')
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import heatmaps

# --- Simulation Parameters ---
ARENA_DIAMETER = 10.0
BIN_SIZE = 1.0
SAMPLING_RATE = 100
DURATION = 60
NUM_SAMPLES = int(DURATION * SAMPLING_RATE)
NUM_SPIKES = 250
KERNEL_SIZE = 5

# --- Generate Synthetic Data (without path calculation) ---
# Generate points clustered around the center using a multivariate normal distribution
# and filter to keep only those inside the circular arena.
radius = heatmaps.arena_curr_diam / 2
mean = [0, 0]
cov = [[(radius/2)**2, 0], [0, (radius/2)**2]] # Covariance for spread
points = np.random.multivariate_normal(mean, cov, NUM_SAMPLES)
dist_from_center = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
in_arena_mask = dist_from_center < radius
x_values = points[in_arena_mask, 0]
y_values = points[in_arena_mask, 1]
num_valid_samples = len(x_values)

# Select random spike times from the valid samples
res = np.random.choice(num_valid_samples, NUM_SPIKES, replace=False)
res.sort()

# --- Run Analysis Pipeline ---
# 1. Create the base grid
bins_grid = heatmaps.create_bins()

# 2. Calculate spike and time maps
spike_map_raw, _ = heatmaps.bins_spikes_count(bins_grid, res, x_values, y_values, BIN_SIZE, ARENA_DIAMETER)
time_map_raw = heatmaps.calculate_time_in_bin(bins_grid, x_values, y_values, SAMPLING_RATE, BIN_SIZE, ARENA_DIAMETER)

# 3. Create smoothing kernel
gaussian_kernel = heatmaps.create_gaussian_kernel(size=KERNEL_SIZE)

# 4. Perform smoothing
spike_map_smoothed = heatmaps.convolve(spike_map_raw, gaussian_kernel)
time_map_smoothed = heatmaps.convolve(time_map_raw, gaussian_kernel)

# ===================================================================
# Visualization
# ===================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 11))
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
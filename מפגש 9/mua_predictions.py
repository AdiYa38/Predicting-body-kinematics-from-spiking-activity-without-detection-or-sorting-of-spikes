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
CHANNEL = 90
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

# plt.figure(figsize=(12, 6))

# # Plot the original signal with fixed y-axis limits
# plt.subplot(2, 1, 1)
# plt.plot(time_axis, show_sig)
# plt.title(f'Original Signal for Channel {CHANNEL}')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude[microvolts]')
# plt.grid()
# # plt.ylim(y_min, y_max) # Set the y-axis limits

# # Plot the MUA signal with the same fixed y-axis limits
# plt.subplot(2, 1, 2)
# plt.plot(time_axis, show_mua, color='orange')
# plt.title(f'MUA Signal')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude[microvolts]')
# plt.grid()
# #plt.ylim(y_min, y_max) # Set the same y-axis limits

# plt.tight_layout()
# plt.show()


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



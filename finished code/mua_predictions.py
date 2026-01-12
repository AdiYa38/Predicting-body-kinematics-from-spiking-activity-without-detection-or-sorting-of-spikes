import numpy as np
import matplotlib.pyplot as plt
import heatmaps
import data
import prediction
import MUA
from scipy import signal

'''
This script performs MUA signal creation and spatial analysis on the MUA data.
It also tests prediction accuracy based on MUA rate maps.
while running this code you need to put in comment "[start_sample:start_sample + int(duration * sample_rate)] " in line 24 in MUA.py file.
'''
# --- Simulation Parameters ---
ARENA_DIAMETER = 80
BIN_SIZE = 1
RES_SAMPLING_RATE = 20000
POS_SAMPLING_RATE = 1250
SESSION = "mP79_17"
DAT_file = f"data/{SESSION}/{SESSION}.dat"
EEG_FILE = f"data/{SESSION}/{SESSION}.eeg"
MUA_file = f"mua_{SESSION}_output.dat"
CHANNEL =  1
START_SAMPLE = 90*60*RES_SAMPLING_RATE
DURATION = 1 # Duration in seconds
KERNEL_SIZE = 7
DTYPE = np.int16

X_CHANNEL, Y_CHANNEL, N_CHANNELS = data.Channels(SESSION)
# --- Simulation Parameters ---

# Load EEG data
dat_data = data.get_eeg_data(DAT_file, np.int16, N_CHANNELS)
eeg_data = data.get_eeg_data(EEG_FILE, DTYPE, N_CHANNELS)
#MUA_data = data.get_eeg_data(MUA_file, np.float16, N_CHANNELS)

# Create MUA signal
mua_signal, _ = MUA.create_mua_signal(dat_data, CHANNEL)
og_sig = data.get_eeg_channels(dat_data, CHANNEL)



# --- Plotting the signals in time ---  
# Create a time array for the x-axis
show_mua = mua_signal[int(START_SAMPLE/16):int(START_SAMPLE/16) + int(DURATION * POS_SAMPLING_RATE)]
time_axis_mua = np.linspace(0, DURATION, len(show_mua), endpoint=False)
show_sig = og_sig[START_SAMPLE:START_SAMPLE + int(DURATION * RES_SAMPLING_RATE)]
time_axis_og = np.linspace(0, DURATION, len(show_sig), endpoint=False)

y_min = np.min(show_sig)
y_max = np.max(show_sig)

plt.figure(figsize=(12, 6))

# Plot the original signal 
plt.subplot(2, 1, 1)
plt.plot(time_axis_og, show_sig)
plt.title(f'Original Signal for Channel {CHANNEL}')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude[milivolts]')
plt.grid()
# plt.ylim(y_min, y_max) # Set the y-axis limits

# Plot the MUA signal with the same fixed y-axis limits
plt.subplot(2, 1, 2)
plt.plot(time_axis_mua, show_mua, color='orange')
plt.title(f'MUA Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude[milivolts]')
plt.grid()
#plt.ylim(y_min, y_max) # Set the same y-axis limits

plt.tight_layout()
plt.show()


# Load Data
x_values, y_values, x_in, y_in = data.import_position_data(eeg_data, X_CHANNEL, Y_CHANNEL, ARENA_DIAMETER, SESSION)
# 3. Create smoothing kernel
gaussian_kernel = heatmaps.create_gaussian_kernel(size=KERNEL_SIZE)

x_smooth, y_smooth = data.smooth_location(x_values, y_values)


bins_grid = heatmaps.create_bins(BIN_SIZE,ARENA_DIAMETER)
time_map_raw, vacants = heatmaps.calculate_time_in_bin(bins_grid, x_smooth, y_smooth, BIN_SIZE, ARENA_DIAMETER, POS_SAMPLING_RATE)



spike_map_raw = MUA.bin_mua_count(bins_grid, mua_signal, x_smooth, y_smooth, BIN_SIZE, ARENA_DIAMETER)


time_map_smoothed, new_vacants, bins_grid, vacants = heatmaps.smooth_map(
    data_array=None,
    x_values=x_values,
    y_values=y_values,
    vacants=None,
    BIN_SIZE=BIN_SIZE,
    time_data=True
)
mua_map_smoothed,_,_,_ = heatmaps.smooth_map(mua_signal,x_values,y_values,vacants,BIN_SIZE,time_data=False,from_mua=True)

# Final rates map
rates_map = mua_map_smoothed / time_map_smoothed
final_rates_map = heatmaps.remove_vacants(rates_map, new_vacants)
final_rates_map = heatmaps.remove_background(final_rates_map, bins_grid)
spike_map_raw = heatmaps.remove_background(spike_map_raw, bins_grid)


# --- Plotting spike_map_raw as a heatmap ---
plt.figure(figsize=(8, 8))
# Use imshow to create the heatmap from the 2D array
# 'origin="lower"' ensures the (0,0) index is at the bottom-left, matching a typical spatial grid
# 'cmap="jet"' sets the color map for the heatmap
plt.imshow(spike_map_raw, origin='lower', cmap='jet')

# Add a color bar to show the scale of the spike counts
plt.colorbar(label='MUA Count')

plt.title(f'Raw MUA Map of channel {CHANNEL}, Bin size {BIN_SIZE}cm')
plt.xlabel('X [cm]')
plt.ylabel('Y [cm]')
plt.show()

# ===================================================================
# Visualization
# ===================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle(f"Spatial Analysis Results of channel {CHANNEL}", fontsize=20)

# Helper function for plotting
def plot_map(ax, data, title, name, cmap='jet'):
    data_to_plot = np.copy(data).astype(float)
    if -1 in data:
      data_to_plot[data == -1] = np.nan
    
    max_val = heatmaps.max_val_to_show(data_to_plot)
    im = ax.imshow(data_to_plot, cmap=cmap, origin='lower', interpolation='nearest')#, vmax=max_val, vmin=1.5)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("X[cm]")
    ax.set_ylabel("Y[cm]")
    fig.colorbar(im,label=name, ax=ax, fraction=0.046, pad=0.04)
   

# Plot the four main results
plot_map(axes[0, 0], spike_map_raw, f"Raw MUA Map of channel {CHANNEL}, Bin size {BIN_SIZE}cm",'MUA Count')
plot_map(axes[0, 1], time_map_raw, f"Raw Occupancy Map, Bin size {BIN_SIZE}cm",'time [s]')
plot_map(axes[1, 0], mua_map_smoothed,f"Smoothed MUA Map of channel {CHANNEL}, Bin size {BIN_SIZE}cm",'MUA Count')
plot_map(axes[1, 1], time_map_smoothed, f"Smoothed Occupancy Map, Bin size {BIN_SIZE}cm",'time [s]')

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.subplots_adjust(hspace=0.3, wspace=0.3)
plt.show()

# Optional: Plot the kernel separately
# fig_kernel, ax_kernel = plt.subplots(figsize=(5, 5))
# plot_map(ax_kernel, gaussian_kernel, "Gaussian Kernel", cmap='viridis')
#plt.show()

# Plot final retes map
fig_rts, rts_ax = plt.subplots(figsize=(5, 5))
plot_title = f"MUA Rate Map (channel {CHANNEL}), Bin size {BIN_SIZE}cm"
plot_map(rts_ax, final_rates_map,plot_title,' MUA rate', cmap='jet')
plt.show()


#--- this part is for multiple channels MUA analysis and prediction ---#
# --- Simulation Parameters ---


BIN_SIZE = 40
CHANNELS = np.array([45, 17, 86])#([7,17,25,33,45,54,66,73,86,98,110,120])



# 1. Create the base grid
bins_grid = heatmaps.create_bins(BIN_SIZE,ARENA_DIAMETER)
time_map_raw, vacants = heatmaps.calculate_time_in_bin(bins_grid, x_smooth, y_smooth, BIN_SIZE, ARENA_DIAMETER, POS_SAMPLING_RATE)
time_map_smoothed, new_vacants, bins_grid, vacants = heatmaps.smooth_map(None,x_smooth, y_smooth,None,BIN_SIZE,time_data=True)

res_list = []
final_rates_map = []
mua_maps = []
mua_min= -1
for channel in CHANNELS:
    
    mua_signal, _ = MUA.create_mua_signal(dat_data, channel)
    og_sig = data.get_eeg_channels(dat_data, channel)
    #og_sig = signal.resample_poly(og_sig, 1250, RES_SAMPLING_RATE)
    res_list.append(mua_signal)
 
    # --- Run Analysis Pipeline ---

    # 2. Calculate spike and time maps
    spike_map_raw = MUA.bin_mua_count(bins_grid, mua_signal, x_smooth, y_smooth, BIN_SIZE, ARENA_DIAMETER)    
    # 4. Perform smoothing
    mua_map_smoothed,_,_,_ = heatmaps.smooth_map(mua_signal,x_smooth,y_smooth,vacants,BIN_SIZE,time_data=False,from_mua=True)
    mua_maps.append(mua_map_smoothed)
    # Final rates map
    rates_map = mua_map_smoothed / time_map_smoothed
    if np.min(rates_map[rates_map!=-1])<mua_min or mua_min==-1:
        mua_min = np.min(rates_map[rates_map!=-1])
    rates_map = heatmaps.remove_background(rates_map, bins_grid)
    final_rates_map.append(heatmaps.remove_vacants(rates_map, new_vacants))
    rates_map = heatmaps.remove_vacants(rates_map, new_vacants, True)
#     plt.figure(figsize=(8, 8))
#     plt.imshow(rates_map, origin='lower', cmap='jet')#, vmin = 1.3, vmax = 4.5)
#     plt.colorbar(label='MUA rate')
#     plt.title(f'Rate Map {channel}, Bin size {BIN_SIZE}cm')
#     plt.xlabel('X [cm]')
#     plt.ylabel('Y [cm]')


# plt.show()
print("MUA min across channels:", mua_min) 

# mua_min_value = np.min(np.array(final_rates_map[final_rates_map!=None]))
# print("MUA map min:", mua_min_value)

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

prior_map = prediction.bins_prior(time_map_smoothed)


# === plot preditions vs time graph
durations = np.arange(1, 20, 0.5)
accuracies = []
chance_levels = []
errors = []
SEM = []

for duration in durations:
    prediction_bins = []
    actual_bins = []

    for start in start_times:

        actual_bin = prediction.get_actual_bin(
            x_smooth, y_smooth, start, duration,
            BIN_SIZE, ARENA_DIAMETER, POS_SAMPLING_RATE
        )
        if actual_bin is None:
            continue
        actual_bins.append(actual_bin)


        mean_maps = np.array(final_rates_map)
        cov_matrix = np.cov(np.array(res_list))
        predicted_bin = MUA.get_predicted_bin(
        mua_signals=res_list, 
        start_time=start, 
        duration=duration, 
        mean_maps=mean_maps, 
        cov_matrix=cov_matrix, 
        bins_prior=prior_map
    )
        prediction_bins.append(predicted_bin)

        # visualize Poisson probabilities
        # for poiss_map in poiss:
        #     plt.figure()
        #     plt.imshow(poiss_map, origin='lower', cmap='jet')
        #     plt.colorbar(label='Poisson Probability')
        #     plt.title(f'Poisson Probability Map for Start Time {start}s and Duration {duration}s')
        #     plt.xlabel('X [cm]')
        #     plt.ylabel('Y [cm]')
        #     plt.show()

        
    # build bins for current BIN_SIZE
    bins_grid = heatmaps.create_bins(BIN_SIZE, ARENA_DIAMETER)

    # accuracy
    acc = prediction.prediction_quality(prediction_bins, actual_bins)
    accuracies.append(acc)
    matches = (np.array(prediction_bins) == np.array(actual_bins))
    match_values = matches.astype(float)
    sem = data.SEM(match_values)*100

    # chance (only inside bins)
    n_inside_bins = np.sum(bins_grid == heatmaps.INSIDE_FLAG)
    chance_levels.append(100 / n_inside_bins)

    # error
    err = data.average_error(prediction_bins, actual_bins, BIN_SIZE, ARENA_DIAMETER, bins_grid)
    errors.append(err)
    SEM.append(sem)

top = np.array(accuracies) + np.array(SEM)
bottom = np.array(accuracies) - np.array(SEM)   
# Corrected plotting code for the first graph
fig, ax1 = plt.subplots(figsize=(8, 5))

# Plot accuracy and chance on the primary y-axis (ax1)
ax1.plot(durations, accuracies, marker='o', color='b', label="Accuracy")
ax1.plot(durations, chance_levels, linestyle='--', color='r', label="Chance")
ax1.fill_between(durations,  bottom,  top,  color='red',  alpha=0.3, zorder=1, label='SEM')
ax1.set_xlabel('Test Duration (s)')
ax1.set_ylabel('Prediction Accuracy (%)')
ax1.set_title(f'Prediction Accuracy vs Test Duration using {len(CHANNELS)} channels, Bin size {BIN_SIZE}cm ')
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
bin_sizes = [1, 2, 4, 5, 10, 20, 40, 80]
accuracies = []
chance_levels = []
errors = []
SEM = []

for bin_size in bin_sizes:
    bins_grid = heatmaps.create_bins(bin_size,ARENA_DIAMETER)
    time_map_raw, vacants = heatmaps.calculate_time_in_bin(bins_grid, x_smooth, y_smooth, bin_size, ARENA_DIAMETER, POS_SAMPLING_RATE)
    time_map_smoothed, new_vacants, bins_grid, vacants = heatmaps.smooth_map(None,x_smooth, y_smooth,None,bin_size,time_data=True)
    prior_map = prediction.bins_prior(time_map_smoothed)

    final_rates_map = []
    prediction_bins = []
    actual_bins = []

    for channel in CHANNELS:
        # 2. Calculate spike and time maps
        spike_map_raw = MUA.bin_mua_count(bins_grid, mua_signal, x_smooth, y_smooth, bin_size, ARENA_DIAMETER)    
        # 4. Perform smoothing
        mua_map_smoothed,_,_,_ = heatmaps.smooth_map(mua_signal,x_values,y_values,vacants,bin_size,time_data=False,from_mua=True)

        # Final rates map
        rates_map = mua_map_smoothed / time_map_smoothed
        final_rates_map.append(heatmaps.remove_vacants(rates_map, new_vacants))


    for start in start_times:
        actual_bin = prediction.get_actual_bin(
            x_smooth, y_smooth, start, test_duration,
            bin_size, ARENA_DIAMETER, POS_SAMPLING_RATE
        )
        if actual_bin is None:
            continue
        actual_bins.append(actual_bin)

        mean_maps = np.array(final_rates_map)
        cov_matrix = np.cov(np.array(res_list))
        predicted_bin = MUA.get_predicted_bin(
        mua_signals=res_list, 
        start_time=start, 
        duration=duration, 
        mean_maps=mean_maps, 
        cov_matrix=cov_matrix, 
        bins_prior=prior_map
    )
        prediction_bins.append(predicted_bin)

        

    # build bins for current bin size
    bins_grid = heatmaps.create_bins(bin_size, ARENA_DIAMETER)

    # accuracy
    acc = prediction.prediction_quality(prediction_bins, actual_bins)
    accuracies.append(acc)
    matches = (np.array(prediction_bins) == np.array(actual_bins))
    match_values = matches.astype(float)
    sem = data.SEM(match_values)*100

    # chance
    n_inside_bins = np.sum(bins_grid == heatmaps.INSIDE_FLAG)
    chance_levels.append(100 / n_inside_bins)

    # error
    err = data.average_error(prediction_bins, actual_bins, bin_size, ARENA_DIAMETER, bins_grid)
    errors.append(err)
    SEM.append(sem)

    
top = np.array(accuracies) + np.array(SEM)
bottom = np.array(accuracies) - np.array(SEM)
# Corrected plotting code for the second graph
fig, ax1 = plt.subplots(figsize=(8, 5))

# Plot accuracy and chance on the primary y-axis (ax1)
ax1.plot(bin_sizes, accuracies, marker='o', color='b', label="Accuracy")
ax1.plot(bin_sizes, chance_levels, linestyle='--', color='r', label="Chance")
ax1.fill_between(chance_levels,  bottom,  top,  color='red',  alpha=0.3, zorder=1, label='SEM')
ax1.set_xlabel('Bin Size (cm)')
ax1.set_ylabel('Prediction Accuracy (%)')
ax1.set_title(f'Prediction Accuracy vs Bin Size with {len(CHANNELS)} channels, ({ARENA_DIAMETER/BIN_SIZE}x{ARENA_DIAMETER/BIN_SIZE})')
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


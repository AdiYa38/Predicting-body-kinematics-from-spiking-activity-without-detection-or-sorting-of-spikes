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
CELL_ID =  np.array([7,11,11, 4])
TETRODE_ID = np.array([1,8,2, 3])
KERNEL_SIZE = 7
EEG_FILE = "mp79_17/mP79_17.eeg"
DTYPE = np.int16

# Load Data
eeg_data = data.get_eeg_data(EEG_FILE, DTYPE, N_CHANNELS)
x_values, y_values, x_in, y_in = data.import_position_data(eeg_data, X_CHANNEL, Y_CHANNEL, ARENA_DIAMETER)
# 3. Create smoothing kernel
gaussian_kernel = heatmaps.create_gaussian_kernel(size=KERNEL_SIZE)


# 1. Create the base grid
bins_grid = heatmaps.create_bins(BIN_SIZE,ARENA_DIAMETER)
time_map_raw = heatmaps.calculate_time_in_bin(bins_grid, x_values, y_values, BIN_SIZE, ARENA_DIAMETER, POS_SAMPLING_RATE)
time_map_smoothed = heatmaps.smooth(time_map_raw, gaussian_kernel, bins_grid)

res_list = []
final_rates_map = []

for tet, cell in zip(TETRODE_ID, CELL_ID):
    
    tet_res, clu = data.get_tetrode_spike_times("mp79_17/mP79_17.clu.", "mp79_17/mP79_17.res.", tet, POS_SAMPLING_RATE, RES_SAMPLING_RATE)
    res = data.get_cell_spike_times(clu, tet_res, cell)
    res_list.append(res)
   
 
    # --- Run Analysis Pipeline ---

    # 2. Calculate spike and time maps
    spike_map_raw, vacants = heatmaps.bins_spikes_count(bins_grid, res, x_values, y_values, BIN_SIZE, ARENA_DIAMETER)
    
    # 4. Perform smoothing
    spike_map_smoothed =heatmaps.smooth(spike_map_raw, gaussian_kernel, bins_grid)

    # Final rates map
    rates_map = spike_map_smoothed / time_map_smoothed
    final_rates_map.append(heatmaps.remove_vacants(rates_map, vacants))


#=== test prediction success ===
start_times =  [952,1000,1113, 1247, 1355,1490, 1586,1661,1782,1817, 1957, 2030,2115, 2239,2353, 2490,2553,3000,3542,4470, 5042,6080,6943,6998, 7050,7452, 7685, 7899, 8021, 9010]   
test_duration = 6.0

#lambda_map = prediction.lambda_rate_per_bin(spike_map_smoothed, time_map_smoothed)
prior_map = prediction.bins_prior(time_map_smoothed)

prediction_bins = []
actual_bins = []

for start in start_times:

  log_poiss = prediction.MB_PBR(res_list, start, test_duration, final_rates_map)
  predicted_bin = prediction.MB_MAP_estimator(log_poiss, prior_map)
  prediction_bins.append(predicted_bin)
  
  actual_bin = prediction.get_actual_bin(x_values, y_values, start, test_duration, BIN_SIZE, ARENA_DIAMETER, POS_SAMPLING_RATE)
  actual_bins.append(actual_bin)

accuracy = prediction.prediction_quality(prediction_bins, actual_bins)
print(f"\nPrediction Accuracy: {accuracy:.2f}% over {len(start_times)} windows")

# === plot preditions vs time graph
durations =  np.arange(1, 50, 0.5)
accuracies = [] 
for duration in durations:
    prediction_bins = []
    actual_bins = []

    for start in start_times:
        log_poiss = prediction.MB_PBR(res_list, start, duration, final_rates_map)
        predicted_bin = prediction.MB_MAP_estimator(log_poiss, prior_map)
        prediction_bins.append(predicted_bin)

        actual_bin = prediction.get_actual_bin(x_values, y_values, start, duration, BIN_SIZE, ARENA_DIAMETER, POS_SAMPLING_RATE)
        actual_bins.append(actual_bin)

    accuracy = prediction.prediction_quality(prediction_bins, actual_bins)
    accuracies.append(accuracy)

plt.figure(figsize=(8, 5))
plt.plot(durations, accuracies, marker='o', linestyle='-', color='b')
plt.xlabel('Test Duration (s)')
plt.ylabel('Prediction Accuracy (%)')
plt.title('Prediction Accuracy vs Test Duration')
plt.grid(True)
plt.tight_layout()
plt.show()

# === plot preditions vs bin size 
bin_sizes = [2, 4, 8,15, 25, 33,42, 50]
accuracies = []
for bin_size in bin_sizes:
    prediction_bins = []
    actual_bins = []

    for start in start_times:
        log_poiss = prediction.MB_PBR(res_list, start, test_duration, final_rates_map)
        predicted_bin = prediction.MB_MAP_estimator(log_poiss, prior_map)
        prediction_bins.append(predicted_bin)

        actual_bin = prediction.get_actual_bin(x_values, y_values, start, test_duration, bin_size, ARENA_DIAMETER, POS_SAMPLING_RATE)
        actual_bins.append(actual_bin)

    accuracy = prediction.prediction_quality(prediction_bins, actual_bins)
    accuracies.append(accuracy)

plt.figure(figsize=(8, 5))
plt.plot(bin_sizes, accuracies, marker='o', linestyle='-', color='b')
plt.xlabel('Bin Size (cm)')
plt.ylabel('Prediction Accuracy (%)')
plt.title('Prediction Accuracy vs bin Size')
plt.grid(True)
plt.tight_layout()
plt.show()



  


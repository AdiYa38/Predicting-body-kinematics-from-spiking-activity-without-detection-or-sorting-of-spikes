import numpy as np
import matplotlib.pyplot as plt
import heatmaps
import data
import prediction

# --- Simulation Parameters ---
ARENA_DIAMETER = 100
BIN_SIZE = 10
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
x_values, y_values, x_in, y_in = data.import_position_data(eeg_data, X_CHANNEL, Y_CHANNEL, ARENA_DIAMETER)



final_rates_map, bins_grid = heatmaps.rates_map(BIN_SIZE, CELL_ID, x_values, y_values, tet_res, clu)
occupancy_map_raw, vacants = heatmaps.calculate_time_in_bin(bins_grid, x_values, y_values, BIN_SIZE, ARENA_DIAMETER, POS_SAMPLING_RATE)
occupancy_map_raw = heatmaps.remove_background(occupancy_map_raw, bins_grid)
occupancy_map_raw = np.copy(occupancy_map_raw).astype(float)
spike_map_raw = heatmaps.bins_spikes_count(bins_grid, res, x_values, y_values, BIN_SIZE, ARENA_DIAMETER)
spike_map_raw = heatmaps.remove_background(spike_map_raw, bins_grid)
spike_map_raw = heatmaps.remove_vacants(spike_map_raw, vacants, True)


print(f"DEBUG: Type after change_grid: {type(final_rates_map)}")
# ===================================================================
# Visualization
# ===================================================================

bins = round(ARENA_DIAMETER/BIN_SIZE)
max_val = heatmaps.max_val_to_show(spike_map_raw)
plt.figure()
plt.imshow(spike_map_raw, cmap='jet', origin='lower', vmax=max_val)
plt.colorbar(label='Spikes Count')
plt.title(f"Spike count map for unit {CELL_ID}.{TETRODE_ID}, ({bins}x{bins})")
plt.xlabel("X [cm]")
plt.ylabel("Y [cm]")

plt.show()



if -1 in occupancy_map_raw:
      occupancy_map_raw[data == -1] = np.nan
plt.figure(figsize=(8, 8))

max_val = np.nanmax(occupancy_map_raw) if np.nanmax(occupancy_map_raw) > 0 else 1
im = plt.imshow(occupancy_map_raw, cmap='jet', origin='lower', interpolation='nearest', vmax=max_val)
    
    # Add a color bar
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.title(f'Occupancy Map ({bins}x{bins})', fontsize=16)
plt.xlabel('X [cm]', fontsize=12)
plt.ylabel('Y [cm]', fontsize=12)

plt.show()
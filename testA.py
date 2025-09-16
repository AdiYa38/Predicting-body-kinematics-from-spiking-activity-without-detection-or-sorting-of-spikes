import numpy as np
import matplotlib.pyplot as plt
import heatmaps
import data
import prediction

# --- Simulation Parameters ---
ARENA_DIAMETER = 100
BIN_SIZE = 1
RES_SAMPLING_RATE = 20000
POS_SAMPLING_RATE = 1250
X_CHANNEL = 124
Y_CHANNEL = 125
N_CHANNELS = 136
CELL_ID = 13
TETRODE_ID = 2
KERNEL_SIZE = 7
EEG_FILE = "mp79_17/mP79_17.eeg"
DTYPE = np.int16

# Load Data
eeg_data = data.get_eeg_data(EEG_FILE, DTYPE, N_CHANNELS)
tet_res, clu = data.get_tetrode_spike_times("mp79_17/mP79_17.clu.", "mp79_17/mP79_17.res.", TETRODE_ID, POS_SAMPLING_RATE, RES_SAMPLING_RATE)
res = data.get_cell_spike_times(clu, tet_res, CELL_ID)
x_values, y_values, x_in, y_in = data.import_position_data(eeg_data, X_CHANNEL, Y_CHANNEL, ARENA_DIAMETER)

final_rates_map, bins_grid = heatmaps.rates_map(BIN_SIZE, CELL_ID, x_values, y_values, tet_res, clu)
print(f"DEBUG: Type after change_grid: {type(final_rates_map)}")
# ===================================================================
# Visualization
# ===================================================================

bins = round(ARENA_DIAMETER/BIN_SIZE)
max_val = heatmaps.max_val_to_show(final_rates_map)
plt.figure()
plt.imshow(final_rates_map, cmap='jet', origin='lower', vmax=max_val)
plt.colorbar(label='Spikes/s')
plt.title(f"Rates map for unit {CELL_ID}, shank {TETRODE_ID}, {bins}x{bins} bins")
plt.xlabel("X Bins")
plt.ylabel("Y Bins")

plt.show()
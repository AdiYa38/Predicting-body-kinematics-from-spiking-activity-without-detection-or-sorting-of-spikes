import numpy as np
import matplotlib.pyplot as plt
import heatmaps
import data
import prediction

# --- Simulation Parameters ---
ARENA_DIAMETER = 80
BIN_SIZE = 1
RES_SAMPLING_RATE = 20000
POS_SAMPLING_RATE = 1250
"""X_CHANNEL = 124
Y_CHANNEL = 125
N_CHANNELS = 136"""
X_CHANNEL = 59
Y_CHANNEL = 60
N_CHANNELS = 71

CELL_ID = 28
TETRODE_ID = 4
KERNEL_SIZE = 7
SESSION = "mP31_20"
EEG_FILE = f"data\{SESSION}\{SESSION}.eeg"
CLU_FILE = f"data\{SESSION}\{SESSION}.clu"
RES_FILE = f"data\{SESSION}\{SESSION}.res"
DTYPE = np.int16

# Load Data
eeg_data = data.get_eeg_data(EEG_FILE, DTYPE, N_CHANNELS)
tet_res, clu = data.get_tetrode_spike_times(CLU_FILE, RES_FILE, TETRODE_ID, POS_SAMPLING_RATE, RES_SAMPLING_RATE)
res = data.get_cell_spike_times(clu, tet_res, CELL_ID)
x_values, y_values, x_in, y_in = data.import_position_data(eeg_data, X_CHANNEL, Y_CHANNEL, ARENA_DIAMETER, SESSION)


# Plot x,y 
plt.figure(figsize=(8, 6))
plt.scatter(x_values, y_values, s=1, alpha=0.6)  # s = dot size
plt.xlabel("X [cm]")
plt.ylabel("Y [cm]")
plt.title(f"XY Scatter Plot (Position) for session {SESSION}")
plt.axis("equal")
plt.grid(True)
plt.show()
plt.figure(figsize=(8, 6))
plt.scatter(x_in, y_in, s=1, alpha=0.6)  # s = dot size
plt.xlabel("X [cm]")
plt.ylabel("Y [cm]")
plt.title("Cleaned XY Scatter Plot (Position) for session mP79_16")
plt.axis("equal")
plt.grid(True)
plt.show()

final_rates_map, bins_grid = heatmaps.rates_map(BIN_SIZE, CELL_ID, x_values, y_values, tet_res, clu, SESSION)

# Plot rates_map
bins = round(ARENA_DIAMETER/BIN_SIZE)

plt.figure()
max_val = heatmaps.max_val_to_show(final_rates_map)
extent_in_cm = [0, ARENA_DIAMETER, 0, ARENA_DIAMETER]
plt.imshow(final_rates_map, cmap='jet', origin='lower', vmax=max_val, extent=extent_in_cm)
plt.colorbar(label='spikes/s')
plt.title(f"Rates map for unit {SESSION}.{TETRODE_ID}.{CELL_ID} ({bins}x{bins})")
plt.xlabel("X [cm]")
plt.ylabel("Y [cm]")        

plt.show()

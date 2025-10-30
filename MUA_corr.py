import numpy as np
import matplotlib.pyplot as plt
import heatmaps
import data
import MUA
from scipy.stats import pearsonr

ARENA_DIAMETER = 80
BIN_SIZE = 20
RES_SAMPLING_RATE = 20000
POS_SAMPLING_RATE = 1250
X_CHANNEL = 124
Y_CHANNEL = 125 
N_CHANNELS = 136
CHANNELS = np.array([7,17,25,33,45,54,66,73,86,98,110,120])
KERNEL_SIZE = 7
START_SAMPLE = 90*60*RES_SAMPLING_RATE
DURATION = 1 # Duration in seconds
DAT_file = "dat/mP79_17.dat"
EEG_FILE = "mp79_17/mP79_17.eeg"
DTYPE = np.int16
MUA_file = "mua_output.dat"

# Load EEG data
dat_data = data.get_eeg_data(DAT_file, np.int16, 136)
eeg_data = data.get_eeg_data(EEG_FILE, DTYPE, N_CHANNELS)
MUA_data = data.get_eeg_data(MUA_file, np.float32, N_CHANNELS)

# MUA.create_mua_file(dat_data, START_SAMPLE,DURATION, RES_SAMPLING_RATE, N_CHANNELS)
# Load Data
x_values, y_values, x_in, y_in = data.import_position_data(eeg_data, X_CHANNEL, Y_CHANNEL, ARENA_DIAMETER)
# 3. Create smoothing kernel
gaussian_kernel = heatmaps.create_gaussian_kernel(size=KERNEL_SIZE)

x_smooth, y_smooth = data.smooth_location(x_values, y_values)


bins_grid = heatmaps.create_bins(BIN_SIZE,ARENA_DIAMETER)
occupancy_map_smoothed, new_vacants, bins_grid, vacants = heatmaps.smooth_map(
    data_array=None,
    x_values=x_values,
    y_values=y_values,
    vacants=None,
    BIN_SIZE=BIN_SIZE,
    time_data=True
)

mua_signals = []
final_rates_maps = []

# For each channel
for channel in CHANNELS:
    
    mua_signal = data.get_eeg_channels(MUA_data, channel)
    
    spike_map_raw = MUA.bin_mua_count(bins_grid, mua_signal, x_smooth, y_smooth, BIN_SIZE, ARENA_DIAMETER)    
    # 4. Perform smoothing

    mua_map_smoothed,_,_,_ = heatmaps.smooth_map(mua_signal,x_values,y_values,vacants,BIN_SIZE,time_data=False,from_mua=True)

    mua_signals.append(mua_signal)
    
    
    # Final rates map
    rates_map = mua_map_smoothed / occupancy_map_smoothed
    final_rates_map = heatmaps.remove_vacants(rates_map, new_vacants)
    final_rates_map = heatmaps.remove_background(final_rates_map, bins_grid)

    final_rates_maps.append(final_rates_map)


# calculate correlation between MUA maps
correlation_mat = np.zeros((len(CHANNELS), len(CHANNELS)), dtype=object)
for i in range(len(CHANNELS)):
    for j in range(len(CHANNELS)):
        if i <= j:
            channel_i_map = final_rates_maps[i]
            channel_j_map = final_rates_maps[j]
            flat_m1 = channel_i_map.flatten()
            flat_m2 = channel_j_map.flatten() 
            valid_mask = np.isfinite(flat_m1) & np.isfinite(flat_m2)
            flat_m1 = flat_m1[valid_mask]
            flat_m2 = flat_m2[valid_mask]
            r,p = pearsonr(flat_m1, flat_m2)
            
            correlation_mat[i,j] =(r,p)
            correlation_mat[j,i] = (r,p)

print("Correlation Matrix between MUA maps:")
print(correlation_mat)



import pandas as pd
import numpy as np
# Assuming the rest of your calculation has run and correlation_mat is populated

# --- 1. Prepare Data for Excel ---
# Separate the (r, p) tuples into two distinct NumPy arrays
# One for the correlation coefficients (r) and one for the p-values (p).

# Get the 'r' values (first element of each tuple)
correlation_r = np.array([[t[0] for t in row] for row in correlation_mat], dtype=float)

# Get the 'p' values (second element of each tuple)
correlation_p = np.array([[t[1] for t in row] for row in correlation_mat], dtype=float)

# --- 2. Convert to Pandas DataFrames ---
# Create DataFrames. You can optionally add channel names as row/column labels
# Assuming CHANNELS is a list of channel names/labels
channel_labels = [f'Channel {i+1}' for i in range(len(CHANNELS))] # Use CHANNELS if it contains names

df_r = pd.DataFrame(correlation_r, index=channel_labels, columns=channel_labels)
df_p = pd.DataFrame(correlation_p, index=channel_labels, columns=channel_labels)

# --- 3. Export to Excel File (Multiple Sheets) ---
excel_filename = 'MUA_Correlation_Matrix.xlsx'

with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
    # Save the correlation coefficients to one sheet
    df_r.to_excel(writer, sheet_name='Correlation_Coefficients (r)')
    
    # Save the p-values to another sheet
    df_p.to_excel(writer, sheet_name='P_Values (p)')

print(f"\n✅ Successfully saved correlation data to '{excel_filename}' in two sheets.")


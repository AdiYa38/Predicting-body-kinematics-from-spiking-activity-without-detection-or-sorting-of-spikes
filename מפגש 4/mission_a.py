import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample_poly
import importlib.resources
from scipy.ndimage import gaussian_filter
import mission_b as mb
# === PARAMETERS ===
filename = "mp79_17/mP79_17.eeg"
dtype = np.int16
n_channels = 136
x_chan = 124
y_chan = 125
arena_diameter = 0.4 # in meters
arena_diameter_cm = arena_diameter * 100  
bin_size_cm = 0.5  # in centimeters
res_sample_rate = 20000  # 20kHz spike sampling
pos_sample_rate = 1250   # 1.25kHz position sampling


# Downsample by a factor of 16 (up=1, down=16)
#     data_downsampled[i] = resample_poly(data[i], up=1, down=16)


# === READ BINARY FILE ===
def import_data(filename, dtype, n_channels, x_chan, y_chan, arena_diameter):
  
    raw = np.fromfile(filename, dtype=dtype)
    data = raw.reshape(-1, n_channels)

    # === EXTRACT X AND Y ===
    x = data[:, x_chan]
    y = data[:, y_chan]

    x= x/32767 - 0.49 # Normalize to [-1, 1] range
    y = y/32767 -0.29  # Normalize to [-1, 1] range

    r = np.sqrt(x**2 + y**2)
    inside_mask = r <= (arena_diameter / 2)

    # Define quadrants
    q1 = (x >= 0) &(y >= 0) & inside_mask
    q2 = (x < 0) & (y >= 0) & inside_mask
    q3 = (x < 0) & (y < 0) & inside_mask
    q4 = (x >= 0) &(y < 0) & inside_mask
    quadrants = {'Q1': q1, 'Q2': q2, 'Q3': q3, 'Q4': q4}

    x_in = x[inside_mask]
    y_in = y[inside_mask]
  
    return x, y, x_in, y_in, quadrants
    # ===visualcheck of position data===
    # plt.figure(figsize=(8, 6))
    # plt.scatter(x, y, s=1, alpha=0.6)  # s = dot size
    # plt.xlabel("X (V)")
    # plt.ylabel("Y (V)")
    # plt.title("XY Scatter Plot (Position)")
    # plt.axis("equal")
    # plt.grid(True)
    # plt.show()

    # plt.figure(figsize=(8, 6))
    # plt.scatter(x_in, y_in, s=1, alpha=0.6)  # s = dot size
    # plt.xlabel("X (V)")
    # plt.ylabel("Y (V)")
    # plt.title("XY Scatter Plot (Position)")
    # plt.axis("equal")
    # plt.grid(True)
    # plt.show()

def read_spike_data( x, res_sample_rate, pos_sample_rate,tetrode_id, target_cell):
    

    res_file = f"mp79_17/mP79_17.res.{tetrode_id}"
    clu_file = f"mp79_17/mP79_17.clu.{tetrode_id}"

    with open(res_file, 'r') as f:
        spike_times = np.array([int(line.strip()) for line in f])

    with open(clu_file, 'r') as f:
        lines = f.readlines()
    clu_labels = np.array([int(line.strip()) for line in lines[1:]])  # skip first line

    # === SANITY CHECK ===
    assert len(spike_times) == len(clu_labels), "Mismatch between .res and .clu spike counts!"

    # === FILTER SPIKES FOR CLUSTER target_cell ===
    cluster_mask = clu_labels == target_cell
    filtered_spike_times = spike_times[cluster_mask]

    spike_pos_idx = ((filtered_spike_times / res_sample_rate) * pos_sample_rate).astype(int)

    valid_idx = (spike_pos_idx >= 0) & (spike_pos_idx < len(x))
    spike_pos_idx = spike_pos_idx[valid_idx]

    return spike_pos_idx, filtered_spike_times

def plot_heatmap(x, y, spike_pos_idx, filtered_spike_times, arena_diameter, bin_size_cm, pos_sample_rate, quadrants, tetrode_id, target_cell):
    
    x_spike = x[spike_pos_idx]
    y_spike = y[spike_pos_idx]
    

    # === CREATE HEATMAP ===
    arena_diameter_cm = arena_diameter * 100
    bins = int(arena_diameter_cm / bin_size_cm)

    spike_heatmap, xedges, yedges = np.histogram2d(
        x_spike, y_spike, bins=bins,
        range=[[-arena_diameter/2, arena_diameter/2], [-arena_diameter/2, arena_diameter/2]]
    )

    pos_heatmap, _, _ = np.histogram2d(
        x, y, bins=bins,
        range=[[-arena_diameter/2, arena_diameter/2], [-arena_diameter/2, arena_diameter/2]]
    )
    time_per_bin = pos_heatmap / pos_sample_rate  # seconds

    # === COMPUTE SMOOTHED RATE MAP ===
    smooth_time = gaussian_filter(time_per_bin, sigma=1.0)
    smooth_spikes = gaussian_filter(spike_heatmap, sigma=1.0)

    with np.errstate(divide='ignore', invalid='ignore'):
        rate_map = np.where(smooth_time > 0, smooth_spikes / smooth_time, 0)

    # === PLOT ===
    plt.imshow(
        np.rot90(rate_map), cmap='coolwarm',
        extent=[-arena_diameter/2, arena_diameter/2, -arena_diameter/2, arena_diameter/2]
    )
    plt.title(f"Firing Rate Heatmap (Cluster {target_cell}, Tetrode {tetrode_id})")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.colorbar(label="Firing Rate (Hz)")
    plt.gca().set_aspect('equal')
    plt.show()
    return 1

def calc_spike_rates( spike_pos_idx, pos_sample_rate, quadrants):
    
    rates = []
    time_in_quardinate = [] 
    for q_name, mask in quadrants.items():
        time_in_q = np.sum(mask) / pos_sample_rate  # in seconds
        time_in_quardinate.append(time_in_q)
        # Find spike indices that fall in this quadrant
        spike_in_q = mask[spike_pos_idx]
        spikes_in_q = np.sum(spike_in_q)

        rate = spikes_in_q / time_in_q if time_in_q > 0 else 0
        rates.append(rate)
        print(f"{q_name}: {spikes_in_q} spikes, {time_in_q:.1f} sec, rate = {rate:.2f} Hz")

    return rates, time_in_quardinate


def quarter_predictor(filtered_spike_times,rates,time_in_quardinate):
    '''
    Predicts the quarter based on the firing rates and time spent in each quadrant.
    Args:
        filtered_spike_times (np.array): Spike times of the target cell for 5 seconsd.
        rates (list): Firing rates for each quadrant.
        time_in_quardinate (list): Time spent in each quadrant.
    Returns:
        int: Index of the predicted quadrant (0-3).
    '''

    T_total = np.sum(time_in_quardinate)
    priors = [T_i / T_total for T_i in time_in_quardinate]
    log_posteriors = []
    for i in range(4):
        rate = rates[i]
        prior = priors[i]
        spikes_num = filtered_spike_times.size
        if rate > 0 and prior > 0:
            likelihood = spikes_num * np.log(rate) - rate
            log_posteriors.append(likelihood + np.log(prior))
        else:
            log_posteriors.append(-np.inf)
    return np.argmax(log_posteriors)


def get_spikes_in_window(filtered_spike_times, start_time_sec, window_duration_sec = 5, res_sample_rate=20000):
    """
    Returns the spike times within a given time window.

    Args:
        filtered_spike_times (np.array): spike times in samples (20kHz).
        start_time_sec (float): start of the window in seconds.
        window_duration_sec (float): duration of the window in seconds.
        res_sample_rate (int): spike sample rate in Hz (default = 20kHz).

    Returns:
        np.array: spike times within the window (in samples).
    """
    start_sample = int(start_time_sec * res_sample_rate)
    end_sample = int((start_time_sec + window_duration_sec) * res_sample_rate)

    # Keep only spikes within that sample range
    window_spikes = filtered_spike_times[
        (filtered_spike_times >= start_sample) & (filtered_spike_times < end_sample)
    ]
    return window_spikes

def get_actual_quadrant(x, y, start_time_sec,quadrants, window_duration_sec = 5, pos_sample_rate=1250):
    """
    Determines which quadrant the mouse was in most of the time during a window.

    Args:
        x (np.array): x-position time series (voltage-normalized).
        y (np.array): y-position time series (voltage-normalized).
        start_time_sec (float): start time of window.
        window_duration_sec (float): window length in seconds.
        pos_sample_rate (int): position sample rate (e.g., 1250 Hz).
        quadrants (dict): dictionary with 'Q1', 'Q2', 'Q3', 'Q4' masks over full x/y arrays.

    Returns:
        str: quadrant where the mouse spent the most time in the window ('Q1'...'Q4' or 'Outside').
    """
    start_idx = int(start_time_sec * pos_sample_rate)
    end_idx = int((start_time_sec + window_duration_sec) * pos_sample_rate)

    if end_idx > len(x):
        print("Warning: Window exceeds position data length.")
        end_idx = len(x)

    win_mask = np.zeros_like(x, dtype=bool)
    win_mask[start_idx:end_idx] = True

    max_overlap = 0
    actual_quadrant = "Outside"

    for q_name, q_mask in quadrants.items():
        overlap = np.sum(q_mask & win_mask)
        if overlap > max_overlap:
            max_overlap = overlap
            actual_quadrant = q_name

    return actual_quadrant

def compute_smoothed_firing_rate_map(occupancy_map, spike_map, sigma=1.0):
    """
    Smooth the occupancy and spike maps and compute the firing rate.

    Returns:
        rate_map (2D array): smoothed firing rate in Hz
    """
    smooth_time = gaussian_filter(occupancy_map, sigma=sigma)
    smooth_spikes = gaussian_filter(spike_map, sigma=sigma)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rate_map = np.where(smooth_time > 0, smooth_spikes / smooth_time, 0)

    return rate_map



# === MAIN FUNCTION ===
tetrode_id = np.array([1,5,5, 3])
target_cell = np.array([7,5,14, 3])

x, y, x_in, y_in, quadrants = import_data(
    filename, dtype, n_channels, x_chan, y_chan,
    arena_diameter
)

for i in range(len(tetrode_id)):
    print(f"Processing Tetrode {tetrode_id[i]}, Target Cell {target_cell[i]}")
    spike_pos_idx, filtered_spike_times = read_spike_data(
        x_in, res_sample_rate, pos_sample_rate,
        tetrode_id[i], target_cell[i]
    )
    # Plot heatmap
    plot_heatmap(x_in, y_in, spike_pos_idx, filtered_spike_times, arena_diameter, bin_size_cm, pos_sample_rate, quadrants, tetrode_id[i], target_cell[i])
    # Calculate firing rates
    rates, time_in_quardinate = calc_spike_rates(spike_pos_idx, pos_sample_rate, quadrants)

    print(f"Rates for Tetrode {tetrode_id[i]}, Cell {target_cell[i]}: {rates}")

    # get 5 seconds of filtered spike times
    start_time = 10000
    end_time = start_time + 5
    filtered_spike_times = get_spikes_in_window(filtered_spike_times, start_time)
    quater = quarter_predictor(filtered_spike_times, rates, time_in_quardinate)
    print(f"quater prediction for time {start_time} - {end_time} seconds is:", quater )
    actual_quarter = get_actual_quadrant(x, y, start_time, quadrants)
    print(f"Actual quarter for time {start_time} - {end_time} seconds is:", actual_quarter)


   
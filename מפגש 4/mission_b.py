
from scipy.fft import rfft, rfftfreq
from scipy.ndimage import gaussian_filter
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
# === CONSTANTS ===
vRange = 10  # Volts
vDAC = [-3.8e-3, 4.64]  # Volts
maxBins = 640  # pixels
pix2cm = 0.366  # cm
nBits = 16
# === POSITION TO CM CONVERSION ===
def convert_to_cm(x_raw):
    xhat = (x_raw / (2**nBits) * vRange - vDAC[0]) / (vDAC[1] - vDAC[0])
    return xhat * maxBins * pix2cm  # in cm

arena_diameter = convert_to_cm(arena_diameter*32767)
# === READ BINARY FILE ===
def import_data(filename, dtype, n_channels, x_chan, y_chan, arena_diameter):
  
    raw = np.fromfile(filename, dtype=dtype)
    data = raw.reshape(-1, n_channels)

    # === EXTRACT X AND Y ===
    x = data[:, x_chan]
    y = data[:, y_chan]

    x_cm = convert_to_cm(x - 0.49*32767) 
    y_cm = convert_to_cm(y - 0.27*32767)

    return x_cm, y_cm




# === COMPUTE VELOCITY ===
def compute_velocity(x_cm, y_cm, pos_sample_rate):
    dx = np.diff(x_cm)
    dy = np.diff(y_cm)
    dt = 1.0 / pos_sample_rate
    vx = dx / dt
    vy = dy / dt
    speed = np.sqrt(vx**2 + vy**2)
    return vx, vy, speed

# === COMPUTE VELOCITY SPECTRUM ===
def compute_velocity_spectrum(speed, pos_sample_rate):
    speed = speed - np.mean(speed)
    yf = np.abs(rfft(speed))
    xf = rfftfreq(len(speed), 1 / pos_sample_rate)
    return xf, yf

# === FIND DOMINANT FREQUENCY ===
def dominant_frequency_duration(xf, yf):
    dominant_idx = np.argmax(yf[1:]) + 1  # ignore DC
    dominant_freq = xf[dominant_idx]
    T = 1 / dominant_freq if dominant_freq > 0 else np.inf
    return dominant_freq, T

# === ESTIMATE BIN SIZE ===
def estimate_bin_size_cm(avg_speed_cm_per_sec, window_duration_sec):
    return avg_speed_cm_per_sec * window_duration_sec

# === BIN INDICES ===
def compute_bin_indices(x_cm, y_cm, bin_size_cm, arena_diameter_cm):
    x_bins = np.floor((x_cm + arena_diameter_cm / 2) / bin_size_cm).astype(int)
    y_bins = np.floor((y_cm + arena_diameter_cm / 2) / bin_size_cm).astype(int)
    return x_bins, y_bins

def build_bin_masks(x_bins, y_bins):
    masks = {}
    for i in np.unique(x_bins):
        for j in np.unique(y_bins):
            key = (i, j)
            masks[key] = (x_bins == i) & (y_bins == j)
    return masks

def compute_bins(x_cm, y_cm, masks, arena_diameter_cm):
    r = np.sqrt(x_cm**2 + y_cm**2)
    inside_mask = r <= (arena_diameter / 2)
    x_bins, y_bins = compute_bin_indices(x_cm[inside_mask], y_cm[inside_mask], bin_size_cm, arena_diameter_cm)
    
    x_cm_in = x_cm[inside_mask]
    y_cm_in = y_cm[inside_mask]

    x_bins = np.floor((x_cm + arena_diameter_cm / 2) / bin_size_cm).astype(int)
    y_bins = np.floor((y_cm + arena_diameter_cm / 2) / bin_size_cm).astype(int)

    bin_masks = {}
    for i in np.unique(x_bins):
        for j in np.unique(y_bins):
            key = (i, j)
            bin_mask = (x_bins == i) & (y_bins == j) & inside_mask
            bin_masks[key] = bin_mask

    return x_cm_in, y_cm_in, bin_masks
        

# === SMOOTHED FIRING RATE MAP ===
def compute_smoothed_firing_rate_map(occupancy_map, spike_map, sigma=1.0):
    smooth_time = gaussian_filter(occupancy_map, sigma=sigma)
    smooth_spikes = gaussian_filter(spike_map, sigma=sigma)
    with np.errstate(divide='ignore', invalid='ignore'):
        rate_map = np.where(smooth_time > 0, smooth_spikes / smooth_time, 0)
    return rate_map



def bin_predictor(filtered_spike_times, rates, time_per_bin):
    T_total = np.sum(time_per_bin)
    priors = [T_i / T_total for T_i in time_per_bin]
    spikes_num = filtered_spike_times.size
    log_posteriors = []

    for i in range(len(rates)):
        rate = rates[i]
        prior = priors[i]
        if rate > 0 and prior > 0:
            likelihood = spikes_num * np.log(rate) - rate
            log_posteriors.append(likelihood + np.log(prior))
        else:
            log_posteriors.append(-np.inf)

    return np.argmax(log_posteriors)  # Index of most likely bin

def get_actual_bin(x_bins, y_bins, start_time_sec, duration_sec, pos_sample_rate):
    start_idx = int(start_time_sec * pos_sample_rate)
    end_idx = int((start_time_sec + duration_sec) * pos_sample_rate)

    x_win = x_bins[start_idx:end_idx]
    y_win = y_bins[start_idx:end_idx]

    # Count most frequent (x_bin, y_bin) tuple
    coords, counts = np.unique(list(zip(x_win, y_win)), return_counts=True, axis=0)
    most_common = coords[np.argmax(counts)]
    return tuple(most_common)

def bin_predictor(filtered_spike_times, rates, time_in_bin):
    T_total = np.sum(time_in_bin)
    priors = [T_i / T_total for T_i in time_in_bin]
    spikes_num = filtered_spike_times.size
    log_posteriors = []

    for rate, prior in zip(rates, priors):
        if rate > 0 and prior > 0:
            likelihood = spikes_num * np.log(rate) - rate
            log_posteriors.append(likelihood + np.log(prior))
        else:
            log_posteriors.append(-np.inf)

    return np.argmax(log_posteriors)

def calc_bin_spike_rates(spike_pos_idx, pos_sample_rate, bin_masks):
    """
    Compute spike rate per bin using bin masks.
    
    Returns:
        rates: list of firing rates per bin
        times: list of total time spent in each bin
        keys:  list of bin keys (i, j)
    """
    rates = []
    time_in_bin = []
    keys = list(bin_masks.keys())

    for key in keys:
        mask = bin_masks[key]
        time_in = np.sum(mask) / pos_sample_rate  # sec
        spike_in = np.sum(mask[spike_pos_idx])   # number of spikes in bin

        rate = spike_in / time_in if time_in > 0 else 0

        time_in_bin.append(time_in)
        rates.append(rate)

    return rates, time_in_bin, keys


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

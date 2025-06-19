import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample_poly
import importlib.resources
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

#plt.figure(figsize=(8, 6))
plt.scatter(x, y, s=1, alpha=0.6)  # s = dot size
plt.xlabel("X (V)")
plt.ylabel("Y (V)")
plt.title("XY Scatter Plot (Position)")
plt.axis("equal")
plt.grid(True)
plt.show()

#plt.figure(figsize=(8, 6))
plt.scatter(x_in, y_in, s=1, alpha=0.6)  # s = dot size
plt.xlabel("X (V)")
plt.ylabel("Y (V)")
plt.title("XY Scatter Plot (Position)")
plt.axis("equal")
plt.grid(True)
plt.show()


spike_data={}
tetrode_id = np.array([1,5,5, 3])
target_cell = np.array([7,5,14, 3])
for i in range(len(tetrode_id)):
    res_file = f"mp79_17/mP79_17.res.{tetrode_id[i]}"
    clu_file = f"mp79_17/mP79_17.clu.{tetrode_id[i]}"

    with open(res_file, 'r') as f:
        spike_times = np.array([int(line.strip()) for line in f])

    with open(clu_file, 'r') as f:
        lines = f.readlines()
    clu_labels = np.array([int(line.strip()) for line in lines[1:]])  # skip first line

    # === SANITY CHECK ===
    assert len(spike_times) == len(clu_labels), "Mismatch between .res and .clu spike counts!"

    # === FILTER SPIKES FOR CLUSTER 7 ===
    cluster_mask = clu_labels == target_cell[i]
    filtered_spike_times = spike_times[cluster_mask]

    spike_pos_idx = ((filtered_spike_times / res_sample_rate) * pos_sample_rate).astype(int)

    valid_idx = (spike_pos_idx >= 0) & (spike_pos_idx < len(x))
    spike_pos_idx = spike_pos_idx[valid_idx]

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

    # === COMPUTE RATE MAP ===
    with np.errstate(divide='ignore', invalid='ignore'):
        rate_map = np.where(time_per_bin > 0, spike_heatmap / time_per_bin, 0)

    for q_name, mask in quadrants.items():
        time_in_q = np.sum(mask) / pos_sample_rate  # in seconds

        # Find spike indices that fall in this quadrant
        spike_in_q = mask[spike_pos_idx]
        spikes_in_q = np.sum(spike_in_q)

        rate = spikes_in_q / time_in_q if time_in_q > 0 else 0
        print(f"{q_name}: {spikes_in_q} spikes, {time_in_q:.1f} sec, rate = {rate:.2f} Hz")

    # === PLOT ===
    plt.imshow(
        np.rot90(rate_map), cmap='coolwarm',
        extent=[-arena_diameter/2, arena_diameter/2, -arena_diameter/2, arena_diameter/2]
    )
    plt.title(f"Firing Rate Heatmap (Cluster {target_cell[i]}, Tetrode {tetrode_id[i]})")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.colorbar(label="Firing Rate (Hz)")
    plt.gca().set_aspect('equal')
    plt.show()


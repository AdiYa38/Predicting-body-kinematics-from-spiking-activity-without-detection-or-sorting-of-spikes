import numpy as np
import matplotlib.pyplot as plt
import heatmaps
import data
from scipy import signal



def create_mua_signal(DAT, channel,start_sample = 144000000, duration = 300,sample_rate = 20000):
    '''
    Create a Multi-Unit Activity (MUA) signal from the spike data.
    Parameters:
    - DAT:  numpy array, DAT data with shape (n_samples, n_channel).
    - channel: int, index of the channel to extract.
    - start_sample: int, starting sample index for the MUA signal.
    - duration: int, duration in seconds for the MUA signal.

    Returns:
    - mua_signal: numpy array, MUA signal for the specified channel.
    - sig: numpy array, original signal for the specified channel.
    '''
    raw_siganl = data.get_eeg_channels(DAT, channel)
    # Extract the relevant segment of the signal if needed
    sig = raw_siganl#[start_sample:start_sample + int(duration * sample_rate)]  
    # Normalize the signal
    conversion_factor = (245.0 / 192.0) / (2**16)
    sig = (sig * conversion_factor).astype(np.float32)
    #apply bandpass filter
    low_cutoff_hz = 300.0  
    high_cutoff_hz = 6000.0 
    nyquist = 1 * sample_rate
    low_norm = low_cutoff_hz / nyquist
    high_norm = high_cutoff_hz / nyquist
    order = 2
    sos_bp = signal.bessel(order, [low_norm, high_norm], btype='band', output='sos')  
    bpf_filtered_sig = signal.sosfiltfilt(sos_bp, sig)
    #RMS 
    bpf_filtered_sig = bpf_filtered_sig**2
    cutoff_hz = 100.0
    norm_cutoff = cutoff_hz / nyquist
    sos_lp = signal.bessel(order, norm_cutoff, btype='low', output='sos')
    lpf_filtered_sig = signal.sosfiltfilt(sos_lp, bpf_filtered_sig)
    lpf_filtered_sig = np.sqrt(np.clip(lpf_filtered_sig, 0, None))
    # Resample to 1250 Hz
    MUA_signal = signal.resample_poly(lpf_filtered_sig, 1250, sample_rate)
    MUA_signal = np.clip(MUA_signal, 0, None)
    return MUA_signal, sig

def create_mua_file(DAT,SESSION,start_sample = 144000000, duration = 300,sample_rate = 20000, n_channels=124):
    """
    Create a Multi-Unit Activity (MUA) file for all channels in the DAT data.
    Parameters:
    - DAT: numpy array, DAT data with shape (n_samples, n_channel).
    - start_sample: int, starting sample index for the MUA signal.
    - duration: int, duration in seconds for the MUA signal.
    - sample_rate: int, original sampling rate of the EEG data.
    - n_channels: int, number of channels in the DAT data.
    Returns:
    - None, but creates a file "mua_output.dat" containing the MUA signals for all channels.
    """
    # Run first channel to get output shape
    mua0, _ = create_mua_signal(DAT, 0, start_sample, duration, sample_rate)
    n_out_samples = mua0.shape[0]
    output = np.memmap(f"mua_{SESSION}_output.dat", dtype=np.float16, mode="w+",
                   shape=(n_out_samples, n_channels))

    output[:, 0] = mua0
    for ch in range(1, n_channels):
        print(f"Processing channel {ch}...")
        mua_signal, _ = create_mua_signal(DAT, ch, start_sample, duration, sample_rate)
        output[:, ch] = mua_signal

    output.flush()



import numpy as np

def bin_mua_count(bins_matrix, mua_signal, x_values, y_values, bin_size_cm, arena_diameter_cm):
    """
    Calculates the mean MUA count per bin and identifies vacant bins using a 2D bin matrix.

    Args:
        bins_matrix (np.ndarray): 2D matrix representing the arena grid.
                                  Cells are flagged as INSIDE_FLAG or OUTSIDE_FLAG.
        mua_signal (np.ndarray): 1D array of MUA counts for each time point.
        x_values (np.ndarray): 1D array of x-coordinates for each time point.
        y_values (np.ndarray): 1D array of y-coordinates for each time point.
        bin_size_cm (float): The side length of each square bin in cm.
        arena_diameter_cm (float): The diameter of the circular arena in cm.

    Returns:
        tuple: A tuple containing:
            - mua_rate_matrix (np.ndarray): A 2D matrix of mean MUA rates per bin.
            - vacant_matrix (np.ndarray): A 2D boolean matrix indicating unvisited or zero-MUA bins.
    """
    
    # Define flags
    INSIDE_FLAG = 0
    OUTSIDE_FLAG = -1

    # Get the dimensions of the bin matrix
    num_bins_per_axis = bins_matrix.shape[0]

    # Initialize matrices to store MUA sums and visit counts for each bin
    mua_sum_matrix = np.zeros_like(bins_matrix, dtype=float)
    visit_count_matrix = np.zeros_like(bins_matrix, dtype=int)
    
    # Convert coordinates from arena space (centered at 0,0) to bin matrix indices
    arena_radius_cm = arena_diameter_cm / 2.0
    x_indices = np.floor((x_values + arena_radius_cm) / bin_size_cm).astype(int)
    y_indices = np.floor((y_values + arena_radius_cm) / bin_size_cm).astype(int)

    # Filter out data points that are outside the bin matrix boundaries
    valid_indices = (x_indices >= 0) & (x_indices < num_bins_per_axis) & \
                    (y_indices >= 0) & (y_indices < num_bins_per_axis)

    x_indices_valid = x_indices[valid_indices]
    y_indices_valid = y_indices[valid_indices]
    mua_signal_valid = mua_signal[valid_indices]

    # Iterate through the valid data points to populate the matrices
    for i in range(len(x_indices_valid)):
        y_idx = y_indices_valid[i]
        x_idx = x_indices_valid[i]
        
        # Check if the bin is inside the arena circle
        if bins_matrix[y_idx, x_idx] == INSIDE_FLAG:
            mua_sum_matrix[y_idx, x_idx] += mua_signal_valid[i]
            visit_count_matrix[y_idx, x_idx] += 1

    mua_sum_matrix[mua_sum_matrix == 0] = -1
    # Calculate the mean MUA rate for each bin
    # mua_rate_matrix = np.divide(mua_sum_matrix, visit_count_matrix,
    #                             out=np.zeros_like(mua_sum_matrix),
    #                             where=visit_count_matrix != 0)
    
    # Create the vacant matrix
    # A bin is 'vacant' if it was not visited OR if it's outside the arena circle
    # vacant_matrix = (visit_count_matrix == 0) | (bins_matrix == OUTSIDE_FLAG)
    
    return mua_sum_matrix#, vacant_matrix

def occupancy_map(x_values, y_values, BIN_SIZE,KERNEL_SIZE=7,ARENA_DIAMETER=100, POS_SAMPLING_RATE = 1250):
    init_bin_size = 1 
    bins_grid_1cm = heatmaps.create_bins(init_bin_size, arena_diameter_cm=ARENA_DIAMETER)
    x_smooth, y_smooth = data.smooth_location(x_values, y_values)
    occupancy_map_raw, vacants = heatmaps.calculate_time_in_bin(bins_grid_1cm, x_smooth, y_smooth, init_bin_size, ARENA_DIAMETER, POS_SAMPLING_RATE)
    gaussian_kernel = heatmaps.create_gaussian_kernel(size=KERNEL_SIZE)
    occupancy_map_covered = heatmaps.cover_vacants(bins_grid_1cm, occupancy_map_raw, vacants)
    occupancy_map_smoothed = heatmaps.smooth(occupancy_map_covered, gaussian_kernel, bins_grid_1cm)
    occupancy_map_sized, new_vacants, bins_grid = heatmaps.change_grid(occupancy_map_smoothed, BIN_SIZE, ARENA_DIAMETER, vacants, init_bin_size)

    return occupancy_map_sized, new_vacants, bins_grid

def MUA_rate_map(mua_signal, x_values, y_values, occupancy_map_sized, vacants ,bins_grid, BIN_SIZE, KERNEL_SIZE=7, ARENA_DIAMETER=100, POS_SAMPLING_RATE = 1250):

    # 1. Create the base grid
    init_bin_size = 1 
    bins_grid_1cm = heatmaps.create_bins(init_bin_size, arena_diameter_cm=ARENA_DIAMETER)

    # 2. Calculate MUA and time maps
    # TODO: remove vacants returning from bin_mua_count and the accepting from here
    x_smooth, y_smooth = data.smooth_location(x_values, y_values)
    mua_map_raw = bin_mua_count(bins_grid, mua_signal, x_smooth, y_smooth, BIN_SIZE, ARENA_DIAMETER)
    

    # 3. Create smoothing kernel
    gaussian_kernel = heatmaps.create_gaussian_kernel(size=KERNEL_SIZE)

        # 4. Cover vacants
    mua_map_covered = heatmaps.cover_vacants(bins_grid_1cm, mua_map_raw, vacants)

    # 5. Perform smoothing
    mua_map_smoothed = heatmaps.smooth(mua_map_covered, gaussian_kernel, bins_grid_1cm)
    

    # 6. Change bins size
    mua_map_sized, new_vacants, bins_grid = heatmaps.change_grid(mua_map_smoothed, BIN_SIZE, ARENA_DIAMETER, vacants, init_bin_size)
    

    # Final rates map
    mua_rates_map = mua_map_sized / occupancy_map_sized
    mua_rates_map = heatmaps.remove_vacants(mua_rates_map, new_vacants)

    # TODO: Is this one necessary?
    mua_map_sized[mua_map_sized == 0] = np.nan

    return mua_rates_map



def create_simulated_raw_signal(duration_sec=300, sample_rate=20000):
    """
    Creates a simulated raw neural signal (single channel) for testing MUA.
    
    The signal is structured in three parts:
    1. Zeroed signal (100 microseconds).
    2. Zeroed signal with occasional random spikes (first half of the remaining duration).
    3. Zeroed signal with high-frequency, high-amplitude spikes (second half of the remaining duration).

    Parameters:
    - duration_sec: int, total duration of the signal in seconds.
    - sample_rate: int, sampling rate in Hz.

    Returns:
    - simulated_sig: numpy array, the single-channel raw signal.
    """
    
    n_samples = duration_sec * sample_rate
    simulated_sig = np.zeros(n_samples, dtype=np.float64)
    
    # --- Part 1: Zeroed Signal (100 microseconds) ---
    # 100 microseconds = 0.0001 seconds
    microsec_duration = 0.0001
    n_samples_microsec = int(microsec_duration * sample_rate)
    # The signal is already initialized to zero, but we define the end point
    part1_end_idx = n_samples_microsec
    # simulated_sig[0:part1_end_idx] is all zeros
    
    # --- Remaining duration (for Parts 2 and 3) ---
    remaining_samples = n_samples - part1_end_idx
    half_remaining = remaining_samples // 2
    
    # --- Part 2: Occasional Spikes (First half of remaining) ---
    # Low spike density: e.g., 50 spikes total in this segment
    
    part2_start_idx = part1_end_idx
    part2_end_idx = part1_end_idx + half_remaining
    
    num_spikes_low = 50
    spike_amplitude = 500.0  # arbitrary amplitude
    spike_width = 10         # width of the spike in samples (0.5ms at 20kHz)
    
    # Choose random indices for the spikes in this segment
    spike_indices_low = np.random.choice(
        np.arange(part2_start_idx, part2_end_idx - spike_width), 
        size=num_spikes_low, 
        replace=False
    )
    
    # Generate the spikes (e.g., a simple inverted triangle or Gaussian)
    for start_idx in spike_indices_low:
        end_idx = start_idx + spike_width
        # Simple negative spike (mimicking extracellular spike)
        simulated_sig[start_idx] = -spike_amplitude * np.random.uniform(0.8, 1.2)
        # Taper off over the width
        for i in range(1, spike_width):
             simulated_sig[start_idx + i] = simulated_sig[start_idx] * (1 - i / spike_width)


    # --- Part 3: Plenty of Spikes (Second half of remaining) ---
    # High spike density: e.g., 1000 spikes total in this segment
    
    part3_start_idx = part2_end_idx
    part3_end_idx = n_samples
    
    num_spikes_high = 1000
    spike_amplitude_high = 1500.0 # Higher amplitude
    spike_width_high = 15         # Slightly wider spike
    
    # Choose random indices for the spikes in this segment
    # Allow for overlapping spikes here for the "plenty of spikes" effect
    spike_indices_high = np.random.choice(
        np.arange(part3_start_idx, part3_end_idx - spike_width_high), 
        size=num_spikes_high, 
        replace=True # Allow repetition for high density
    )
    
    # Generate the spikes
    for start_idx in spike_indices_high:
        end_idx = start_idx + spike_width_high
        
        # Add the spike to the signal (instead of setting it, to allow overlap)
        simulated_sig[start_idx] += -spike_amplitude_high * np.random.uniform(0.8, 1.2)
        
        # Taper off
        for i in range(1, spike_width_high):
             simulated_sig[start_idx + i] += -spike_amplitude_high * np.random.uniform(0.8, 1.2) * (1 - i / spike_width_high)

    return simulated_sig


import numpy as np

def get_predicted_bin(mua_signals, start_time, duration, mean_maps, cov_matrix, bins_prior, pos_sample_rate=1250):
    """
    Implements the Multivariate Gaussian MAP estimator equation from the image.
    
    Args:
        mua_signals: List of arrays (one per channel).
        start_time: Current time window start (seconds).
        duration: Window length (seconds).
        mean_maps: np.array of shape (Channels, Rows, Cols) containing mean rates.
        cov_matrix: np.array of shape (Channels, Channels) - The Sigma matrix.
        bins_prior: np.array of shape (Rows, Cols) containing P(Yi).
    """
    # 1. Extract the current activity vector X (one rate per channel)
    start_sample = int(start_time * pos_sample_rate)
    end_sample = int((start_time + duration) * pos_sample_rate)
    
    # X is the vector of spike rates across all channels for this window
    X = np.array([np.sum(res[start_sample:end_sample]) / duration for res in mua_signals])

    n_channels, rows, cols = mean_maps.shape
    
    # 2. Sigma calculations (Inverse and Log-Determinant)
    # We add a tiny 'epsilon' to the diagonal to ensure the matrix is invertible
    reg_cov = cov_matrix + np.eye(n_channels) * 1e-6
    inv_cov = np.linalg.inv(reg_cov)
    _, log_det_cov = np.linalg.slogdet(reg_cov)
    
    # 3. Log-Prior calculation
    with np.errstate(divide='ignore'):
        log_prior = np.full(bins_prior.shape, -np.inf)
        valid_mask = (bins_prior > 0) & (bins_prior != -1)
        log_prior[valid_mask] = np.log(bins_prior[valid_mask])

    # 4. Compute Log-Posterior for every bin (The 'argmax' logic)
    log_posterior = np.full((rows, cols), -np.inf)

    # Constant term: -0.5 * log|Sigma|
    const_term = -0.5 * log_det_cov 

    for r in range(rows):
        for c in range(cols):
            if not valid_mask[r, c]:
                continue
            
            # Mean vector mu for this specific bin (Yi)
            mu_yi = mean_maps[:, r, c]
            diff = X - mu_yi
            
            # Quadratic term: -0.5 * (X - mu)^T * Sigma^-1 * (X - mu)
            exponent = -0.5 * np.dot(diff.T, np.dot(inv_cov, diff))
            
            # Total log-probability: Likelihood + Prior
            log_posterior[r, c] = const_term + exponent + log_prior[r, c]

    # 5. Find the bin index that maximizes the posterior
    return np.unravel_index(np.argmax(log_posterior), log_posterior.shape)








import numpy as np
import matplotlib.pyplot as plt
import heatmaps
import data
from scipy import signal
import prediction

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
    # Extract the relevant segment of the signal
    sig = raw_siganl#[start_sample:start_sample + duration * sample_rate]  
    # Normalize the signal
    sig = (sig/(2**16))*(245/192)
    #apply bandpass filter
    low_cutoff_hz = 300.0  
    high_cutoff_hz = 6000.0 
    nyquist = 1 * sample_rate
    low_norm = low_cutoff_hz / nyquist
    high_norm = high_cutoff_hz / nyquist
    order = 5
    b, a = signal.butter(order, [low_norm, high_norm], btype='band')
    bpf_filtered_sig = signal.filtfilt(b, a, sig)
    #RMS 
    bpf_filtered_sig = bpf_filtered_sig**2
    cutoff_hz = 100.0
    norm_cutoff = cutoff_hz / nyquist
    b, a = signal.butter(order, norm_cutoff, btype='low')
    lpf_filtered_sig = signal.filtfilt(b, a, bpf_filtered_sig)
    lpf_filtered_sig = np.sqrt(np.clip(lpf_filtered_sig, 0, None))
    # Resample to 1250 Hz
    MUA_signal = signal.resample_poly(lpf_filtered_sig, 1250, sample_rate)
    return MUA_signal, sig

def create_mua_file(DAT,start_sample = 144000000, duration = 300,sample_rate = 20000, n_channels=124):
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
    output = np.memmap("mua_output.dat", dtype=np.float32, mode="w+",
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

    # Calculate the mean MUA rate for each bin
    # mua_rate_matrix = np.divide(mua_sum_matrix, visit_count_matrix,
    #                             out=np.zeros_like(mua_sum_matrix),
    #                             where=visit_count_matrix != 0)
    
    # Create the vacant matrix
    # A bin is 'vacant' if it was not visited OR if it's outside the arena circle
    vacant_matrix = (visit_count_matrix == 0) | (bins_matrix == OUTSIDE_FLAG)
    
    return mua_sum_matrix, vacant_matrix

    
   
    
# #main function to run the MUA signal creation
# def main():
#     # --- Simulation Parameters ---
#     DAT_file = "dat/mP79_17.dat"
#     CHANNEL = 5
#     START_SAMPLE = 2253000
#     DURATION = 10 # Duration in seconds
#     SAMPLE_RATE = 20000  # Original sampling rate of the EEG data

#     # Load EEG data
#     dat_data = data.get_eeg_data(DAT_file, np.int16, 136)

#     # Create MUA signal
#     mua_signal, og_sig = create_mua_signal(dat_data, CHANNEL, START_SAMPLE, DURATION, SAMPLE_RATE)
#     og_sig = signal.resample_poly(og_sig, 1250, SAMPLE_RATE)

#     # --- Plotting the signals in time ---
#     # Create a time array for the x-axis
#     time_axis = np.linspace(0, DURATION, len(mua_signal), endpoint=False)

#     y_min = np.min(og_sig)
#     y_max = np.max(og_sig)

#     plt.figure(figsize=(12, 6))

#     # Plot the original signal with fixed y-axis limits
#     plt.subplot(2, 1, 1)
#     plt.plot(time_axis, og_sig)
#     plt.title(f'Original Signal for Channel {CHANNEL}')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Amplitude[microvolts]')
#     plt.grid()
#     # plt.ylim(y_min, y_max) # Set the y-axis limits

#     # Plot the MUA signal with the same fixed y-axis limits
#     plt.subplot(2, 1, 2)
#     plt.plot(time_axis, mua_signal, color='orange')
#     plt.title(f'MUA Signal')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Amplitude[microvolts]')
#     plt.grid()
#     #plt.ylim(y_min, y_max) # Set the same y-axis limits

#     plt.tight_layout()
#     plt.show()









# main()







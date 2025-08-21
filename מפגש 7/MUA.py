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
    sig = raw_siganl[start_sample:start_sample + duration * sample_rate]  
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



#main function to run the MUA signal creation
def main():
    # --- Simulation Parameters ---
    DAT_file = "dat/mP79_17.dat"
    CHANNEL = 19 
    START_SAMPLE = 72000000
    DURATION = 10  # Duration in seconds
    SAMPLE_RATE = 20000  # Original sampling rate of the EEG data

    # Load EEG data
    dat_data = data.get_eeg_data(DAT_file, np.int16, 136)

    # Create MUA signal
    mua_signal, og_sig = create_mua_signal(dat_data, CHANNEL, START_SAMPLE, DURATION, SAMPLE_RATE)
    og_sig = signal.resample_poly(og_sig, 1250, SAMPLE_RATE)

    # --- Plotting the signals in time ---
    # Create a time array for the x-axis
    time_axis = np.linspace(0, DURATION, len(mua_signal), endpoint=False)

    y_min = np.min(og_sig)
    y_max = np.max(og_sig)

    plt.figure(figsize=(12, 6))

    # Plot the original signal with fixed y-axis limits
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, og_sig)
    plt.title(f'Original Signal for Channel {CHANNEL}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude[microvolts]')
    plt.grid()
    # plt.ylim(y_min, y_max) # Set the y-axis limits

    # Plot the MUA signal with the same fixed y-axis limits
    plt.subplot(2, 1, 2)
    plt.plot(time_axis, mua_signal, color='orange')
    plt.title(f'MUA Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude[microvolts]')
    plt.grid()
    #plt.ylim(y_min, y_max) # Set the same y-axis limits

    plt.tight_layout()
    plt.show()
    create_mua_file(dat_data, start_sample=START_SAMPLE, duration=DURATION, sample_rate=SAMPLE_RATE, n_channels=124)
    mua_file = data.get_eeg_data("mua_output.dat", np.float32, 124)
    mua_from_file = mua_file[:, CHANNEL]

    # Plot the MUA signal from the file and the original MUA signal
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, mua_signal, label='MUA Signal (in memory)', color='orange')
    plt.plot(time_axis, mua_from_file, label='MUA Signal (from file)',
                color='green', linestyle='--')
    plt.title(f'MUA Signal Comparison for Channel {CHANNEL}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()







main()







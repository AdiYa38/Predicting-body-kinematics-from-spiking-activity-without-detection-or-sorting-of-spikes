import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample_poly
import importlib.resources
from scipy.ndimage import gaussian_filter

def get_eeg_data(filename, dtype, n_channels):
    """
    Reads EEG data from a binary file and returns the data as a numpy array.
    
    Parameters:
    - filename: str, path to the binary file containing EEG data.
    - dtype: numpy dtype, data type of the EEG data in the file.
    - n_channel: int, number of channels in the EEG data.
    
    Returns:
    - eeg_data: numpy array, reshaped EEG data with shape (n_samples, n_channel).
    """
    raw = np.fromfile(filename, dtype=dtype)
    eeg_data = raw.reshape(-1, n_channels)
    return eeg_data

def get_eeg_channels(eeg_data, channel):
    """
    Extracts a specific channel from the EEG data.
    
    Parameters:
    - eeg_data: numpy array, EEG data with shape (n_samples, n_channel).
    - channel: int, index of the channel to extract.
    
    Returns:
    - channel_data: numpy array, data for the specified channel.
    """
    return eeg_data[:, channel]
    
def convert_to_cm(x_raw):
    vRange = 10  # Volts
    vDAC = [-3.8e-3, 4.64]  # Volts
    maxBins = 640  # pixels
    pix2cm = 0.366  # cm
    nBits = 16
    xhat = (x_raw / (2**nBits) * vRange - vDAC[0]) / (vDAC[1] - vDAC[0])
    return xhat * maxBins * pix2cm  # in cm


def import_position_data(eeg_data, x_chan, y_chan, arena_diameter): 
  
    # === EXTRACT X AND Y ===
    x = get_eeg_channels(eeg_data, x_chan)
    y = get_eeg_channels(eeg_data, y_chan)
   
    #convert to cm and center around 0
    x = convert_to_cm(x)-124
    y = convert_to_cm(y)-72.5

    # filter data outside the arena
    r = np.sqrt(x**2 + y**2)
    inside_mask = r <= (arena_diameter / 2)

    x_in = x[inside_mask]
    y_in = y[inside_mask]
  
    return x, y, x_in, y_in

def get_tetrode_spike_times(clu_file_name, res_file_name, tetrode_id, pos_sample_rate, res_sample_rate):
    res_file =  res_file_name + (str)(tetrode_id)
    clu_file = clu_file_name + (str)(tetrode_id)

    with open(res_file, 'r') as f:
        spike_times = np.array([int(line.strip()) for line in f])

    with open(clu_file, 'r') as f:
        lines = f.readlines()
    clu_labels = np.array([int(line.strip()) for line in lines[1:]])  # skip first line

    #remove from re and clu files every 0 or 1 spike
    mask = np.isin(clu_labels, [0, 1], invert=True)
    spike_times = spike_times[mask]
    clu_labels = clu_labels[mask]
    
    return (spike_times*pos_sample_rate/res_sample_rate), (clu_labels*pos_sample_rate/res_sample_rate)


def get_cell_spike_times(clu_labels, spike_times, cell_id, pos_sample_rate, res_sample_rate):
    """
    Extracts spike times for a specific cell ID from the spike times and cluster labels.
    
    Parameters:
    - clu_labels: numpy array, cluster labels for each spike.
    - spike_times: numpy array, spike times corresponding to the cluster labels.
    - cell_id: int, ID of the cell to extract spikes for.
    
    Returns:
    - cell_spike_times: numpy array, spike times for the specified cell ID.
    """
    mask = clu_labels == cell_id
    return (spike_times[mask]*pos_sample_rate/res_sample_rate) if np.any(mask) else np.array([])  # return empty array if no spikes found 

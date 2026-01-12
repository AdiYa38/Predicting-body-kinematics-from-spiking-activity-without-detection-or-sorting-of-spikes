import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample_poly
from matplotlib.animation import FuncAnimation
import importlib.resources
from scipy.ndimage import gaussian_filter

def Channels(session):
    '''
    Returns the channel mapping for a given mouse identifier
    '''

    SESSION_CHANNELS = {
    "mP79_11": (124, 125, 136), 
    "mP79_12": (124, 125, 136),
    "mP79_13": (124, 125, 136),
    "mP79_14": (124, 125, 136),
    "mP79_15": (124, 125, 136),
    "mP79_16": (124, 125, 136),
    "mP79_17": (124, 125, 136), 
    "mP79_18":(124, 125, 136),
    "mP79_19":(124, 125, 136),
    "mP31_18": (59, 60, 71),
    "mP31_19": (59, 60, 71),
    "mP31_20": (59, 60, 71)   

    }
    if session in SESSION_CHANNELS:
        x_chan, y_chan, n_chan = SESSION_CHANNELS[session]
        return x_chan, y_chan, n_chan
    
    else:
        raise ValueError(f"Session '{session}' not found in CHANNELS mapping.")

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
    raw = np.memmap(filename, dtype=dtype, mode='r')
    n_samples = raw.size // n_channels
    eeg_data = raw.reshape(n_samples, n_channels)
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
     # Calibration constants
    vRange = 10.0  # Volts
    vDAC = [-3.8e-3, 4.64]  # Volts
    maxBins = 640.0  # pixels
    pix2cm = 0.294  # cm
    nBits = 16.0

    
    x_raw = np.asarray(x_raw).astype(np.uint16)

    # The original conversion logic.
    xhat = (x_raw / (2**nBits) * vRange - vDAC[0]) / (vDAC[1] - vDAC[0])

    return xhat * maxBins * pix2cm  # in cm

def import_position_data(eeg_data, x_chan, y_chan, arena_diameter, session): 
  
    SESSION_SHIFTS = {
    "mP79_11": (-102.8, -70.9), 
    "mP79_12": (-105.6, -65.4),
    "mP79_13": (-98.5, -67.1),
    "mP79_14": (-98.7, -67.1),
    "mP79_15": (-97.9, -67.1),
    "mP79_16": (-98.5, -64.1),
    "mP79_17": (-99.1, -60.0),
    "mP79_18": (-394.9, -9.7),
    "mP79_19": (-0, -0),
    "mP31_18": (-106.6, -66.9),
    "mP31_19": (-91.7, -61.7),
    "mP31_20": (-96.3, -74.5)   

    }
    
    # Set shift according to session
    if session in SESSION_SHIFTS:
        x_shift, y_shift = SESSION_SHIFTS[session]
        print(f"Coordinates shifted by ({x_shift},{y_shift})")
    else:
        # אם שם הסשן לא נמצא, נזרק אזהרה ונשתמש בהזזה 0
        print(f"ERROR: Session '{session}' not found in ARENA_SHIFTS")

    # === EXTRACT X AND Y ===
    x = get_eeg_channels(eeg_data, x_chan)
    y = get_eeg_channels(eeg_data, y_chan)
   
    # Convert to cm and center around 0
    x = convert_to_cm(x) + x_shift
    y = convert_to_cm(y) + y_shift

    # filter data outside the arena
    r = np.sqrt(x**2 + y**2)
    inside_mask = r <= (arena_diameter / 2)

    x_in = x[inside_mask]
    y_in = y[inside_mask]
  
    return x, y, x_in, y_in

import numpy as np

def get_tetrode_spike_times(clu_file_name, res_file_name, tetrode_id, pos_sample_rate, res_sample_rate):
    res_file = res_file_name + "." + str(tetrode_id)
    clu_file = clu_file_name + "." + str(tetrode_id)

    with open(res_file, 'r') as f:
        # Read the raw, possibly wrapped, timestamps
        raw_spike_times = np.array([int(line.strip()) for line in f])

    with open(clu_file, 'r') as f:
        lines = f.readlines()
    clu_labels = np.array([int(line.strip()) for line in lines[1:]])

    mask = np.isin(clu_labels, [0, 1], invert=True)
    raw_spike_times = raw_spike_times[mask]
    clu_labels = clu_labels[mask]

    # Unwrapping the timestamps
    counter_max = 10300  # Based on your observation
    unwrapped_times = np.copy(raw_spike_times).astype(np.int64)

    # Find where the counter jumps back to 0
    resets = np.where(np.diff(unwrapped_times) < 0)[0] + 1
    
    # Add the counter max to all subsequent segments
    offset = 0
    for reset_idx in resets:
        offset += counter_max
        unwrapped_times[reset_idx:] += offset

    # Now, convert the unwrapped timestamps to the new sample rate
    scaled_times = unwrapped_times * pos_sample_rate / res_sample_rate
    
    # Ensure no negative values from potential precision errors and cast to int
    return np.maximum(np.round(scaled_times), 0).astype(np.int64), clu_labels

def get_cell_spike_times(clu_labels, spike_times, cell_id):
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
    return spike_times[mask] if np.any(mask) else np.array([])  # return empty array if no spikes found 



def average_error(predicted_bins, actual_bins, bin_size, arena_diameter, bins_grid, inside_flag=0):
    """
    Compute mean Euclidean distance (cm) between predicted and actual bin centers,
    ignoring invalid/outside bins.
    """
   
    errors = []

    for p, a in zip(predicted_bins, actual_bins):
    
        errors.append(np.sqrt((p[0] - a[0])**2 + (p[1] - a[1])**2))

    return np.mean(errors) if errors else None 

def smooth_location(x_cm, y_cm):
    '''
    smooth the location of the animal using fourorier transform and low pass filter
    Args:
        x_cm(array): x location in cm
        y_cm(array): y location in cm
    Returns:
        (array,array): smoothed x and y location in cm
    '''
    n = len(x_cm)
    freq = np.fft.rfftfreq(n, d=1/1250)  # Frequency bins
    fft_x = np.fft.rfft(x_cm)
    fft_y = np.fft.rfft(y_cm)
    cutoff = 6  # Cutoff frequency in Hz
    fft_x[np.abs(freq) > cutoff] = 0.0
    fft_y[np.abs(freq) > cutoff] = 0.0
    x = np.abs(fft_x)
    y = np.abs(fft_y)
    smoothed_x = np.fft.irfft(fft_x, n=n)
    smoothed_y = np.fft.irfft(fft_y, n=n)
    return smoothed_x, smoothed_y


def plot_mouse_animation(x, y, start_time_minutes, duration_seconds,title, sample_rate=1250):
    """
    Plots an animated visualization of mouse movement data.

    Args:
        x (list or np.array): A list or NumPy array of x-coordinates.
        y (list or np.array): A list or NumPy array of y-coordinates.
        start_time_minutes (float): The starting time of the animation in minutes.
        duration_seconds (float): The duration of the animation in seconds.
        sample_rate (int): The sampling frequency in Hertz (Hz).
    """

    # Constants
    total_data_points = len(x)
    total_duration_hours = total_data_points / (sample_rate * 3600)

    # Convert start time to seconds
    start_time_seconds = start_time_minutes * 60
    end_time_seconds = start_time_seconds + duration_seconds

    # Validate inputs to ensure they are within the data bounds
    if start_time_seconds < 0 or end_time_seconds > total_duration_hours * 3600:
        print(f"Error: The specified time range ({start_time_minutes} min to {end_time_seconds/60:.2f} min) is outside the data's total duration of {total_duration_hours:.2f} hours.")
        return

    # Calculate the start and end indices for the data arrays
    start_index = int(start_time_seconds * sample_rate)
    end_index = int(end_time_seconds * sample_rate)

    if start_index >= end_index:
        print("Error: No data to plot for the specified time range. Please check your start time and duration.")
        return

    # Select the subset of data to be animated
    x_plot = x[start_index:end_index]
    y_plot = y[start_index:end_index]

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(f'Mouse {title} Movement Animation from {start_time_minutes} min')
    ax.set_xlabel('X-coordinate')
    ax.set_ylabel('Y-coordinate')
    ax.set_facecolor('#f3f4f6') # Light gray background
    
    # Set plot limits to be consistent with the entire dataset to prevent resizing
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y), np.max(y))
    ax.set_aspect('equal', adjustable='box')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Initialize a line object for the path and a dot for the current position
    line, = ax.plot([], [], lw=2, color='#3b82f6') # Blue line for the path
    dot, = ax.plot([], [], 'o', color='#ef4444', markersize=8) # Red dot for the current position

    def update(frame):
        """
        Update function for the animation.

        This function is called for each frame and updates the plot to show
        the path and the current position of the mouse.
        """
        # Get the subset of data up to the current frame
        path_x = x_plot[:frame]
        path_y = y_plot[:frame]

        # Update the line object with the new path
        line.set_data(path_x, path_y)

        # Update the position of the red dot
        dot.set_data(path_x[-1:], path_y[-1:])

        return line, dot

    # Create the animation
    # The interval determines the speed of the animation. Here it's set to 1/1250 * 1000 = 0.8 ms
    # This might be too fast, so let's use a more reasonable interval for visualization
    # A skip factor of 10 means we update every 10 samples, or 125 frames per second
    skip_factor = 10
    anim = FuncAnimation(
        fig,
        update,
        frames=np.arange(0, len(x_plot), skip_factor),
        interval=skip_factor / sample_rate * 1000,
        blit=True
    )

    plt.show()

    


def SEM (data):
    '''
    Calculate standard Error of the Mean of a signal
    Args:
        signal(array of arrays): input signals in an array
    Returns:
       SEM(array): mean SEM of the signal
      
    '''
   
    var = np.var(data)
    SEM = np.sqrt(var)/ np.sqrt(len(data))

    return SEM
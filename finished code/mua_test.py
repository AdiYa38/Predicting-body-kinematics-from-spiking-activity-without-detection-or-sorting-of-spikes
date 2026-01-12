import numpy as np
import matplotlib.pyplot as plt
import heatmaps
import data
import prediction
import MUA
from scipy import signal

'''
This script tests the MUA signal creation process.
It also checks signal behavior at "stain" entry.
While using this code you need to uncomment "[start_sample:start_sample + int(duration * sample_rate)] " in line 24 in MUA.py file.
'''
# --- Simulation Parameters ---
ARENA_DIAMETER = 80
BIN_SIZE = 20
RES_SAMPLING_RATE = 20000
POS_SAMPLING_RATE = 1250
SESSION = "mP79_17"
DAT_file = f"data/{SESSION}/{SESSION}.dat"
EEG_FILE = f"data/{SESSION}/{SESSION}.eeg"
CHANNELS = np.array([7,17,25,33,45,54,66,73,86,98,110,120])
START_SAMPLE = int((2500.65-0.3)*RES_SAMPLING_RATE)

#---this part is to test MUA creation on made up sig---
made_up_sig = np.zeros(20000*1) # 1 seconds of data at 20kHz

# Create a made up pulse sig 

made_up_sig[10000] = 2000

sig = made_up_sig 

time_axis_og = np.linspace(0, len(sig)/RES_SAMPLING_RATE, len(sig) , endpoint=False)

 # Normalize the signal
sig = (sig/(2**16))*(245/192)
#apply bandpass filter
low_cutoff_hz = 300.0  
high_cutoff_hz = 6000.0 
nyquist = 1 * RES_SAMPLING_RATE
low_norm = low_cutoff_hz / nyquist
high_norm = high_cutoff_hz / nyquist
order = 2
b, a = signal.bessel(order, [low_norm, high_norm], btype='band')
bpf_filtered_sig = signal.filtfilt(b, a, sig)
#RMS 
bpf_filtered_sig_squared = bpf_filtered_sig**2
cutoff_hz = 100.0
norm_cutoff = cutoff_hz / nyquist
b, a = signal.bessel(order, norm_cutoff, btype='low') #bessel
lpf_filtered_sig = signal.filtfilt(b, a, bpf_filtered_sig_squared)
lpf_filtered_sig_sqrt = np.sqrt(np.clip(lpf_filtered_sig, 0, None))
# Resample to 1250 Hz
MUA_signal = signal.resample_poly(lpf_filtered_sig_sqrt, 1250, RES_SAMPLING_RATE)
MUA_signal = np.clip(MUA_signal, 0, None)


time_axis_mua = np.linspace(0, len(MUA_signal)/POS_SAMPLING_RATE, len(MUA_signal))

plt.figure(figsize=(12, 8))

# Plot the signal processing steps
plt.subplot(6, 1, 1)
plt.plot(time_axis_og, sig, color='black')
plt.title(f' Made up signal')
plt.ylabel('Amplitude')
plt.grid()
#plt.ylim(plot_ylim) # Use consistent y-limits

plt.subplot(6, 1, 2)
plt.plot(time_axis_og, bpf_filtered_sig, color='black')
plt.title(f' BPF signal ')
plt.ylabel('Amplitude')
plt.grid()

plt.subplot(6, 1, 3)
plt.plot(time_axis_og, bpf_filtered_sig_squared, color='black')
plt.title(f' BPF signal squared ')
plt.ylabel('Amplitude')
plt.grid()

plt.subplot(6, 1, 4)
plt.plot(time_axis_og, lpf_filtered_sig_sqrt, color='black')
plt.title(f' LPF signal ')
plt.ylabel('Amplitude')
plt.grid()

plt.subplot(6, 1, 5)
plt.plot(time_axis_og, lpf_filtered_sig_sqrt, color='black')
plt.title(f' LPF signal sqrt ')
plt.ylabel('Amplitude')
plt.grid()

# Plot the single channel signal
plt.subplot(6, 1, 6)
plt.plot(time_axis_mua, MUA_signal, color='orange')
plt.title(f' MUA Signal from made up signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid()



plt.tight_layout()
plt.show()

X_CHANNEL, Y_CHANNEL, N_CHANNELS = data.Channels(SESSION)

DURATION = 2 # Duration in seconds

DTYPE = np.int16


# --- Simulation Parameters ---

# Load EEG data
dat_data = data.get_eeg_data(DAT_file, DTYPE, N_CHANNELS)
eeg_data = data.get_eeg_data(EEG_FILE, DTYPE, N_CHANNELS)



x_values, y_values, x_in, y_in = data.import_position_data(eeg_data, X_CHANNEL, Y_CHANNEL, ARENA_DIAMETER,SESSION)
x_smooth, y_smooth = data.smooth_location(x_values, y_values)

num_samples = len(x_smooth)
timestamps = np.arange(num_samples) / POS_SAMPLING_RATE

signals = []
#plots all channels signal
for channel in CHANNELS:
   sig = data.get_eeg_channels(dat_data, channel)
   sig = sig[START_SAMPLE:START_SAMPLE + int(DURATION * RES_SAMPLING_RATE)]
   signals.append(sig)

fig, axes = plt.subplots(len(CHANNELS), 1, figsize=(15, 10), sharex=True)

# Set the overall title
fig.suptitle(f'Signals for channels {CHANNELS[0]}-{CHANNELS[-1]}', fontsize=16)
time_axis = np.linspace(0, DURATION, len(signals[0]))
for i, mua_signal in enumerate(signals):
    # Use the axes object for plotting
    line, =  axes[i].plot(time_axis, mua_signal, zorder=2, label='MUA Average')
    line_color = line.get_color()

    axes[i].plot(time_axis, mua_signal, color=line_color, zorder=2)        
    
    # Set the y-axis label to identify the channel
    axes[i].set_ylabel(f'Ch {CHANNELS[i]}', rotation=0, labelpad=30, ha='right')
    axes[i].spines['top'].set_visible(False)
    axes[i].spines['right'].set_visible(False)
    axes[i].spines['bottom'].set_visible(False)

    # Hide x-axis ticks and labels for all but the last plot
    if i < len(CHANNELS) - 1:
        # Hide the labels/ticks
        plt.setp(axes[i].get_xticklabels(), visible=False)
        axes[i].tick_params(axis='x', which='both', bottom=False)
    else:
        # Set the common x-axis label on the last plot
        axes[i].set_xlabel('Time (s)')

# Add a common label for the Y-axis
fig.text(0.04, 0.5, 'Amplitude [volt]', va='center', rotation='vertical', fontsize=12)
handles, labels = axes[0].get_legend_handles_labels()
#MUA.create_mua_file(dat_data, SESSION, START_SAMPLE, DURATION, RES_SAMPLING_RATE, N_CHANNELS)



#--- Find Entry Events from the Left Side of the Box ---

SHIFT = ARENA_DIAMETER / 2.0
#box edges (from the image)
X_BIN_MIN_EDGE = 56.5
X_BIN_MAX_EDGE = 63.5
Y_BIN_MIN_EDGE = 41.5
Y_BIN_MAX_EDGE = 49.5

# --- 1. Define Area Boundaries (from the image) ---
X_MIN = X_BIN_MIN_EDGE - SHIFT
X_MAX = X_BIN_MAX_EDGE - SHIFT
Y_MIN = Y_BIN_MIN_EDGE - SHIFT
Y_MAX = Y_BIN_MAX_EDGE - SHIFT

# --- 2. Create Boolean Masks (using your smoothed data) ---

# Check if mouse is inside the box for ALL time points
is_inside = (x_smooth >= X_MIN) & (x_smooth <= X_MAX) & \
            (y_smooth >= Y_MIN) & (y_smooth <= Y_MAX)
inside = np.where(is_inside)[0]
# Check if mouse was to the left of the box for ALL time points
is_left_of_box = (x_smooth < X_MIN)

# --- 3. Find Entry Events ---

# Shift 'is_inside' to find where the mouse was 1 sample *ago*
was_inside = np.roll(is_inside, 1)
was_inside[0] = False  # The first sample can't be an "entry"

# Shift 'is_left_of_box' to find where the mouse was 1 sample *ago*
was_left = np.roll(is_left_of_box, 1)
was_left[0] = False

# Find indices where ALL three conditions are true:
# 1. is_inside: Is in the box NOW
# 2. ~was_inside: Was NOT in the box just before
# 3. was_left: Was to the LEFT of the box just before
entry_indices = np.where(is_inside& ~was_inside & was_left)[0]

# --- 4. Get the Final Timestamps ---
event_timestamps = timestamps[entry_indices]

print(f"Found {len(event_timestamps)} entry events from the left.")
print(event_timestamps)

PADDING_SEC = 0.3
DURATION_WITH_PADDING = DURATION + 2 * PADDING_SEC
expected_final_len = int(DURATION * POS_SAMPLING_RATE)
padding_samples_mua = int(PADDING_SEC * POS_SAMPLING_RATE)

mua_average =  []
og_sigs = []
event_timestamps_fixed = event_timestamps
#--- this part calculates event triggered average of each MUA channel for all events---
for CHANNEL in CHANNELS:
    mua_signals = []
    for timestamp in event_timestamps_fixed:
        start_time_sec = timestamp - 1.0 - PADDING_SEC
        START_SAMPLE = int(start_time_sec * RES_SAMPLING_RATE)
        
        mua_signal_padded, og_sig = MUA.create_mua_signal(
            dat_data, 
            CHANNEL, 
            START_SAMPLE, 
            DURATION_WITH_PADDING, 
            RES_SAMPLING_RATE
        )

        # plt.figure(figsize=(12, 4))
        # time_axis_og = np.linspace(-1, 1, len(og_sig), endpoint=False)
        # plt.plot(time_axis_og, og_sig, color='orange')
        # plt.title(f'signal around  {timestamp} sec for Channel {CHANNEL}')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Amplitude[microvolts]')
        # plt.grid()
        # #plt.ylim(global_ylim) # Set the same y-axis limits
        # plt.show()
        
        mua_signal_clean = mua_signal_padded[padding_samples_mua : -padding_samples_mua]
        
        if len(mua_signal_clean) == expected_final_len:
            mua_signals.append(mua_signal_clean)

    if mua_signals:
        
        mua_average.append(np.mean(np.stack(mua_signals, axis=0), axis=0))
    else:
        mua_average.append(np.full(expected_final_len, np.nan))

mua_average = np.array(mua_average)
SEM = []
for i in range (len(mua_signals[1])):
    SEM.append(data.SEM([mua_signals[j][i] for j in range(len(mua_signals))]))

top = mua_average + SEM
bottom = mua_average - SEM

# Plot MUA signals average
time_axis = np.linspace(-1.0, 1.0, len(mua_average[0]))

# 1. Determine the global min and max amplitude across all MUA signals
all_amplitudes = np.concatenate(mua_average)
y_min = all_amplitudes.min()
y_max = all_amplitudes.max()
y_range = y_max - y_min
# Add a small buffer to the y-limits for better visualization
y_buffer = y_range * 0.05
global_ylim = (y_min - y_buffer, y_max + y_buffer)


# 2. Create the figure and subplots with shared x-axis
fig, axes = plt.subplots(len(CHANNELS), 1, figsize=(15, 10), sharex=True)

# Set the overall title
fig.suptitle(f'MUA Signals average over {len(event_timestamps_fixed)} events', fontsize=16)

# 3. Loop through and plot the signals
for i, mua_signal in enumerate(mua_average):
    # Use the axes object for plotting
    line, =  axes[i].plot(time_axis, mua_signal, zorder=2, label='MUA Average')
    line_color = line.get_color()

    axes[i].fill_between(time_axis, 
                         bottom[i],        
                         top[i],           
                         color='red', 
                         alpha=0.3,      
                         zorder=1,label = 'SEM')         
    # Set the global y-limits for amplitude scaling
    axes[i].set_ylim(global_ylim)
    
    # Set the y-axis label to identify the channel
    axes[i].set_ylabel(f'Ch {CHANNELS[i]}', rotation=0, labelpad=30, ha='right')
    axes[i].spines['top'].set_visible(False)
    axes[i].spines['right'].set_visible(False)
    axes[i].spines['bottom'].set_visible(False)

    # Hide x-axis ticks and labels for all but the last plot
    if i < len(CHANNELS) - 1:
        # Hide the labels/ticks
        plt.setp(axes[i].get_xticklabels(), visible=False)
        axes[i].tick_params(axis='x', which='both', bottom=False)
    else:
        # Set the common x-axis label on the last plot
        axes[i].set_xlabel('Time (s)')

fig.text(0.04, 0.5, 'Amplitude [volt]', va='center', rotation='vertical', fontsize=12)
handles, labels = axes[0].get_legend_handles_labels()

# Create the legend on the Figure (fig), not the axes
fig.legend(handles, labels, 
           loc='upper right',          # Position: Top Right
           bbox_to_anchor=(0.95, 0.90), # Fine-tune position (x, y) coordinates
           ncol=1,                     # Number of columns in the legend
           fontsize=12)
    
# Adjust layout to make space for suptitle and the common y-axis label
plt.tight_layout(rect=[0.05, 0.03, 1, 0.95])
plt.show()

# Plot MUA signals
time_axis = np.arange(0, DURATION, 1/RES_SAMPLING_RATE)
# for i, mua_signal in enumerate(mua_signals):
#     plt.figure(figsize=(12, 4))
#     plt.plot(time_axis, mua_signal, color='orange')
#     plt.title(f'MUA Signal for Channel {CHANNELS[i]}')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Amplitude[microvolts]')
#     plt.grid()
#     plt.ylim(global_ylim) # Set the same y-axis limits
#     plt.show()



#plot original signal of channel 45 at chosen timestamps
CHANNEL = 45
event_timestamps = [325.29, 375.94, 1007.43, 1239.51,3127.32, 4890.79,  5056.78, 5072.27, 5079.03,  5080.3, 6935.77, 11205.6, 11938.92]


fig, axes = plt.subplots(len(event_timestamps), 1, figsize=(15, 10), sharex=True)

# Set the overall title
fig.suptitle(f'Channel 45 signals at events {len(event_timestamps)}', fontsize=16)
DURATION = 0.6
for i,timestamp in enumerate(event_timestamps):
     
    start_time_sec = timestamp - 0.3 
    START_SAMPLE = int(start_time_sec * RES_SAMPLING_RATE)
    mua_signal, og_sig = MUA.create_mua_signal(dat_data, CHANNEL, START_SAMPLE, DURATION, RES_SAMPLING_RATE)
    #og_sig = signal.resample_poly(og_sig, 1250, RES_SAMPLING_RATE)
    # --- Plotting the signals in time ---
    # Create a time array for the x-axis
    time_axis_og = np.linspace(-0.3, 0.3, len(og_sig), endpoint=False)
    time_axis_mua = np.linspace(-0.3, 0.3, len(mua_signal))
    y_min = np.min(og_sig)
    y_max = np.max(og_sig)

    line, =  axes[i].plot(time_axis_og, og_sig, zorder=2)
    line_color = line.get_color()  
    # Set the global y-limits for amplitude scaling
    
    # Set the y-axis label to identify the channel
    axes[i].set_ylabel(f'Time {timestamp}[s]', rotation=0, labelpad=30, ha='right')
    axes[i].spines['top'].set_visible(False)
    axes[i].spines['right'].set_visible(False)
    axes[i].spines['bottom'].set_visible(False)

    # Hide x-axis ticks and labels for all but the last plot
    if i < len(event_timestamps) - 1:
        # Hide the labels/ticks
        plt.setp(axes[i].get_xticklabels(), visible=False)
        axes[i].tick_params(axis='x', which='both', bottom=False)
    else:
        # Set the common x-axis label on the last plot
        axes[i].set_xlabel('Time (s)')

# Add a common label for the Y-axis
fig.text(0.04, 0.5, 'Amplitude [volt]', va='center', rotation='vertical', fontsize=12)
    
# Adjust layout to make space for suptitle and the common y-axis label
plt.tight_layout(rect=[0.05, 0.03, 1, 0.95])
plt.show()
    
# --- Calculate the average MUA signal ---
# Stack the list of 1D arrays into a 2D matrix (channels x time)
mua_matrix = np.stack(mua_signals, axis=0)

# Calculate the mean across channels (axis=0)
mua_average = np.mean(mua_matrix, axis=0)

# Get the single channel signal you already calculated
CHANNEL = 45
time_stamp = 3127.32
start_time_sec = time_stamp - 0.3 
START_SAMPLE = int(start_time_sec * RES_SAMPLING_RATE)
mua_signal_single, og_sig = MUA.create_mua_signal(dat_data, CHANNEL, START_SAMPLE, DURATION, RES_SAMPLING_RATE)
# --- Plot Average MUA vs. Single Channel MUA ---

# Create the time axis for the MUA signal (at 1250 Hz)
time_axis_og = np.linspace(-0.3, 0.3, len(og_sig), endpoint=False)
time_axis_mua = np.linspace(-0.3, 0.3, len(mua_signal))
# Get the y-limits from the single channel MUA for a fair comparison
y_min = np.min(mua_average)
y_max = np.max(mua_average)
y_buffer = (y_max - y_min) * 0.1
plot_ylim = (y_min - y_buffer, y_max + y_buffer)

x_min = np.min(mua_signal)
x_max = np.max(mua_signal)
x_buffer = (x_max - x_min) * 0.1
plot_xlim = (x_min - x_buffer, x_max + x_buffer)

plt.figure(figsize=(12, 8))

# Plot the MUA signal vs the original channel signal at ripple band
plt.subplot(2, 1, 1)
plt.plot(time_axis_og, og_sig, color='black')
plt.title(f' OG Signal channel {CHANNEL} around {time_stamp}[s]')
plt.ylabel('Amplitude')
plt.grid()
#plt.ylim(plot_ylim) # Use consistent y-limits

# Plot the single channel signal
plt.subplot(2, 1, 2)
plt.plot(time_axis_mua, mua_signal_single, color='orange')
plt.title(f'Single MUA Signal (Channel {CHANNEL}) around {time_stamp}[s]')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid()
plt.ylim(plot_xlim) # Use consistent y-limits

plt.tight_layout()
plt.show()

#--- Plot MUA signals at chosen timestamps for all channels ---
fig, axes = plt.subplots(len(event_timestamps), 1, figsize=(15, 10), sharex=True)

# Set the overall title
fig.suptitle(f'MUA, 300-6000 Hz @ 1250 Hz', fontsize=16)

for i,channel in enumerate(CHANNELS):
    mua_signal, og_sig = MUA.create_mua_signal(dat_data, channel, START_SAMPLE, DURATION, RES_SAMPLING_RATE)
    #og_sig = signal.resample_poly(og_sig, 1250, RES_SAMPLING_RATE)
    # --- Plotting the signals in time ---
    # Create a time array for the x-axis
    time_axis_og = np.linspace(-0.3, 0.3, len(og_sig), endpoint=False)
    time_axis_mua = np.linspace(-0.3, 0.3, len(mua_signal))
    y_min = np.min(og_sig)
    y_max = np.max(og_sig)

    line, =  axes[i].plot(time_axis_mua, mua_signal, zorder=2)
    line_color = line.get_color()  
    # Set the global y-limits for amplitude scaling
    
    # Set the y-axis label to identify the channel
    axes[i].set_ylabel(f'{channel}', rotation=0, labelpad=30, ha='right')
    axes[i].spines['top'].set_visible(False)
    axes[i].spines['right'].set_visible(False)
    axes[i].spines['bottom'].set_visible(False)

    # Hide x-axis ticks and labels for all but the last plot
    if i < len(event_timestamps) - 1:
        # Hide the labels/ticks
        plt.setp(axes[i].get_xticklabels(), visible=False)
        axes[i].tick_params(axis='x', which='both', bottom=False)
    else:
        # Set the common x-axis label on the last plot
        axes[i].set_xlabel('Time (s)')

# Add a common label for the Y-axis
fig.text(0.04, 0.5, 'Amplitude', va='center', rotation='vertical', fontsize=12)
    
# Adjust layout to make space for suptitle and the common y-axis label
plt.tight_layout(rect=[0.05, 0.03, 1, 0.95])
plt.show()


#--- Plot original signal of a channel at many timestamps in chunks to avoid crowding ---
# Parameters for plotting
CHUNKS_SIZE = 10 
all_events = event_timestamps_fixed
num_total_events = len(all_events)

# Calculate the number of figures needed
num_figures = int(np.ceil(num_total_events / CHUNKS_SIZE))

# Determine ideal figure height to prevent crowding (using ~0.7 inches per subplot)
BASE_WIDTH = 15
HEIGHT_PER_SUBPLOT = 0.7
FIGURE_HEIGHT = CHUNKS_SIZE * HEIGHT_PER_SUBPLOT + 2 # Add buffer for titles/labels

# Iterate through the list in chunks
for chunk_index in range(num_figures):
    # Determine the start and end indices for the current chunk
    start_idx = chunk_index * CHUNKS_SIZE
    end_idx = min((chunk_index + 1) * CHUNKS_SIZE, num_total_events)
    
    current_chunk = all_events[start_idx:end_idx]
    
    # Create a new figure and axes for this chunk
    # The number of subplots is dynamic based on the actual size of the last chunk
    num_subplots = len(current_chunk)
    fig, axes = plt.subplots(num_subplots, 1, figsize=(BASE_WIDTH, FIGURE_HEIGHT), sharex=True)
    
    # Ensure axes is iterable even if only one subplot is created
    if num_subplots == 1:
        axes = [axes]
        
    # Set the overall title for the current figure
    fig.suptitle(f'Channel {CHANNEL} signals (Events {start_idx+1} to {end_idx})', fontsize=16)

    # Loop through the timestamps in the current chunk
    for i, timestamp in enumerate(current_chunk):
        # Calculate sample indices
        start_time_sec = timestamp - 0.3
        START_SAMPLE = int(start_time_sec * RES_SAMPLING_RATE)
        
        # Get the signals (using the standalone function)
        # NOTE: You must replace 'create_mua_signal' with your actual 'MUA.create_mua_signal'
        # when running with your real environment and data.
        mua_signal, og_sig = MUA.create_mua_signal(dat_data, CHANNEL, START_SAMPLE, DURATION, RES_SAMPLING_RATE)
        
        # --- Plotting the signals in time ---
        # Create a time array for the x-axis, centered at 0.0 s
        time_axis_og = np.linspace(-0.3, 0.3, len(og_sig), endpoint=False)
        
        # Plot the signal
        line, = axes[i].plot(time_axis_og, og_sig, zorder=2)
        
        # --- Styling ---
        
        # Set the y-axis label to identify the time
        axes[i].set_ylabel(f'Time {timestamp:.2f}[s]', rotation=0, labelpad=30, ha='right')
        
        # Clean up the spines
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        
        # Set common y-limits for consistent scaling across the figure
        axes[i].set_ylim([-0.06, 0.06]) 

        # Hide x-axis ticks and labels for all but the last plot in the figure
        if i < num_subplots - 1:
            axes[i].spines['bottom'].set_visible(False)
            plt.setp(axes[i].get_xticklabels(), visible=False)
            axes[i].tick_params(axis='x', which='both', bottom=False)
        else:
            # Set the common x-axis label on the last plot
            axes[i].set_xlabel('Time (s)')
            
    # Add a common label for the Y-axis
    fig.text(0.04, 0.5, 'Amplitude [volt]', va='center', rotation='vertical', fontsize=12)
        
    # Adjust layout to make space for suptitle and the common y-axis label
    plt.tight_layout(rect=[0.05, 0.03, 1, 0.95])
    plt.show()

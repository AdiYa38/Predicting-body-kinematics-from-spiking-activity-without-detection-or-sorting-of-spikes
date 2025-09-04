import numpy as np
import matplotlib.pyplot as plt
import heatmaps
import data
import prediction
import MUA
from scipy import signal
# --- Simulation Parameters ---
ARENA_DIAMETER = 100
BIN_SIZE = 2
RES_SAMPLING_RATE = 20000
POS_SAMPLING_RATE = 1250
X_CHANNEL = 124
Y_CHANNEL = 125
N_CHANNELS = 136
DAT_file = "dat/mP79_17.dat"
EEG_FILE = "mp79_17/mP79_17.eeg"
CHANNELS = [ 9, 55]
START_SAMPLE = 2253000
DURATION = 0.1 # Duration in seconds
SAMPLE_RATE = 20000  # Original sampling rate of the EEG data
KERNEL_SIZE = 7
DTYPE = np.int16

# Load EEG data
dat_data = data.get_eeg_data(DAT_file, np.int16, 136)
eeg_data = data.get_eeg_data(EEG_FILE, DTYPE, N_CHANNELS)

mua_signals = []
signals = []
show_mua = []
show_sig = []

for CHANNEL in CHANNELS:
    # Create MUA signal
    mua_signal, og_sig = MUA.create_mua_signal(dat_data, CHANNEL, START_SAMPLE, DURATION, SAMPLE_RATE)
    og_sig = signal.resample_poly(og_sig, 1250, SAMPLE_RATE)
    mua_signals.append(mua_signal)
    signals.append(og_sig)
    show_mua.append(mua_signal[START_SAMPLE:START_SAMPLE + int(DURATION * SAMPLE_RATE)])
    show_sig.append(og_sig[START_SAMPLE:START_SAMPLE + int(DURATION * SAMPLE_RATE)])

for i, CHANNEL in enumerate(CHANNELS):
    # Plot Original Signal
    plt.subplot(len(CHANNELS), 2, 2*i + 1)
    time_axis_orig = np.linspace(0, DURATION, len(signals[i]), endpoint=False)
    plt.plot(time_axis_orig, signals[i])
    plt.title(f'Original Signal for Channel {CHANNEL}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude[microvolts]')
    plt.grid()

    # Plot MUA Signal
    plt.subplot(len(CHANNELS), 2, 2*i + 2)
    time_axis_mua = np.linspace(0, DURATION, len(mua_signals[i]), endpoint=False)
    plt.plot(time_axis_mua, mua_signals[i], color='orange')
    plt.title(f'MUA Signal for Channel {CHANNEL}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude[microvolts]')
    plt.grid()
    
plt.tight_layout()
plt.show()







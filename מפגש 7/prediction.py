import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample_poly
import importlib.resources
from scipy.ndimage import gaussian_filter

def spike_rate_in_bin(bin_spike_count, time_in_bin):
    '''
    calculates the spike rate in each bin
    Args:
        bin_spike_count(array): number of spikes counted in each bin
        time_in_bin(array): amount of time spent in the bin
    Returns:
        array: spike rate in each bin
    '''
    return (bin_spike_count/time_in_bin)

def bins_prior(time_in_bin):
    '''
    calculates the prior for each bin
    Args:
        time_in_bin(array): amount of time spent in each bin
    Returns:
        array: the probability to occupay each bin
    '''
    T_tot = np.sum(time_in_bin)
    return (time_in_bin/T_tot)

# === dont realy need ===
def lambda_rate_per_bin(spike_map, time_map):
    '''
        calculates the lambda of poisson probability in each bin
        Args:
            spike_map(matrix): num of spikes per bin
            time_map(matrix): time spent in each bin
        Returns:
            matrix: lambda values (spikes/time) in each bin,
                with -1 where spike_map == -1 or time_map == 0
    '''
    lambda_map = np.full(spike_map.shape, -1.0)
    valid_mask = (spike_map != -1) & (time_map != 0)
    lambda_map[valid_mask] = spike_map[valid_mask] / time_map[valid_mask]

    return lambda_map

def PBR(cell_spike_times, start_time, duration, lambda_map,pos_sample_rate=1250):
    '''
        calculates the poisson probability to get a specific spike rate on each bin
        Args:
            cell_spike_times(array): spikes times stemps for a specific cell
            start_time(float): start time in seconds of the section we want to evaluate in seconds
            duration(int): for how long we want to calculate in seconds
            lambda_map(matrix): spikes/time of each bin
            pos_sample_rate(int): helps translate spikes to seconds
        Returns:
            matrix: the log of the probability that the measured spike rate was measured in each bin,  with -inf where invalid 
    '''
    start_sample = int(start_time * pos_sample_rate)
    end_sample = int((start_time + duration) * pos_sample_rate)

    # Keep only spikes within that sample range
    window_spikes = cell_spike_times[
        (cell_spike_times >= start_sample) & (cell_spike_times < end_sample)
    ]

    spike_rate = len(window_spikes)/duration

    log_poiss_prob = np.full(lambda_map.shape, -np.inf)
    valid_mask = lambda_map >= 0

    log_poiss_prob[valid_mask] = spike_rate * np.log(lambda_map[valid_mask]) - lambda_map[valid_mask]

    return log_poiss_prob

def MAP_estimator(log_poiss_prob, bins_prior ):
    '''
        calculates the most likely bin to be in based on the the given spike rate
        Args:
            log_poiss_prob(matrix): the probability to get the spike rate based on location
            bins_prior(matrix): the probability to be in each bin
        
        Returns:
            (int,int): the most likley bin returns by the map estimator
    '''
   # Mask out bins with zero prior (we've never been there)
   # Mask out bins with zero prior (we've never been there)
    with np.errstate(divide='ignore'):
        log_prior = np.log(bins_prior)

    log_posterior = log_poiss_prob + log_prior

    max_idx = np.unravel_index(np.argmax(log_posterior), log_posterior.shape)

    return max_idx

def get_actual_bin (x, y, start_time, duration,bin_size,arena_diameter, pos_sample_rate=1250):
    '''
        calculates the actual bin location in time frame
        Args:
            x(array): x location in time
            y(array): y location in time
            start_time(float): start time in seconds of the section we want to evaluate in seconds
            duration(int): for how long we want to calculate in seconds
            bin_size(float):size of each bin
            arena_diameter(float): size of arena
            pos_sample_rate(int): helps translate spikes to seconds
        Returns:
            (int,int): the actual location in time frame
    '''
    arena_radius = arena_diameter / 2.0
    num_bins_per_axis = int(np.ceil(arena_diameter / bin_size))

    start_sample = int(start_time * pos_sample_rate)
    end_sample = int((start_time + duration) * pos_sample_rate)

    if start_sample >= len(x) or end_sample > len(x) or start_sample >= end_sample:
        return None  # invalid time window

    # Average position in window
    x_avg = np.mean(x[start_sample:end_sample])
    y_avg = np.mean(y[start_sample:end_sample])

    # Check if inside arena
    if np.sqrt(x_avg**2 + y_avg**2) > arena_radius:
        return None

    # Convert to bin indices
    col_idx = int((x_avg + arena_radius) / bin_size)  # x --> j
    row_idx = int((y_avg + arena_radius) / bin_size)  # y --> i

    if 0 <= row_idx < num_bins_per_axis and 0 <= col_idx < num_bins_per_axis:
        return (row_idx, col_idx)
    else:
        return None
    


def prediction_quality(prediction_bins, actual_bins):
    '''
        calculates the precentage of right predictions
        Args:
            prediction_bins(array): predicted bin 
            actual_bins(array):coresponding actual bin
    
        Returns:
           int: the precent of correct estimations
    '''
    correct = 0
    total = 0
    for pred, actual in zip(prediction_bins, actual_bins):
        if pred is not None and actual is not None:
            total += 1
            if pred == actual:
                correct += 1

    return (correct / total) * 100 if total > 0 else 0.0

# def compute_average_velocity(x_cm, y_cm, pos_sample_rate = 1250):
#     '''
#         calculates the mouce's average speed
#         Args:
#             x_cm(array): predicted bin 
#             y_cm(array):coresponding actual bin
#             pos_sample_rate(int): helps track locations times
    
#         Returns:
#            int: the average speed 
#     '''
#     dx = np.diff(x_cm)
#     dy = np.diff(y_cm)
#     dt = 1.0 / pos_sample_rate
#     num_large_dx = np.sum(np.abs(dx) > 1)
#     num_large_dy = np.sum(np.abs(dy) > 1)

#     print(f"dx > 1: {num_large_dx} times")
#     print(f"dy > 1: {num_large_dy} times")
#     vx = dx / dt
#     vy = dy / dt
#     speed = np.sqrt(vx**2 + vy**2)

#     plt.hist(speed, bins=1, color='skyblue')
#     plt.xlabel("Speed (cm/s)")
#     plt.ylabel("Count")
#     plt.title("Speed distribution")
#     plt.grid(True)
#     plt.show()

   
#     return np.median(speed) 

def MB_MAP_estimator(log_poiss_prob, bins_prior ):
    '''
        calculates the most likely bin to be in based on the the given spike rate
        Args:
            log_poiss_prob(matrix): the probability to get the spike rate based on location
            bins_prior(matrix): the probability to be in each bin
        
        Returns:
            (int,int): the most likley bin returns by the map estimator
    '''
     # Mask out bins with zero prior (we've never been there)
    with np.errstate(divide='ignore'):
        log_prior = np.log(bins_prior)
    log_sum = np.sum(log_poiss_prob, axis=0)
    log_posterior = log_sum + log_prior

    max_idx = np.unravel_index(np.argmax(log_posterior), log_posterior.shape)

    return max_idx

def MB_PBR(cell_spike_times, start_time, duration, lambda_map,pos_sample_rate=1250):
    '''
        calculates the poisson probability to get a specific spike rate on each bin
        Args:
            cell_spike_times(array): spikes times stemps for a specific cell
            start_time(float): start time in seconds of the section we want to evaluate in seconds
            duration(int): for how long we want to calculate in seconds
            lambda_map(matrix): spikes/time of each bin
            pos_sample_rate(int): helps translate spikes to seconds
        Returns:
            matrix: the log of the probability that the measured spike rate was measured in each bin,  with -inf where invalid 
    '''
    start_sample = int(start_time * pos_sample_rate)
    end_sample = int((start_time + duration) * pos_sample_rate)

    log_poiss_prob = [] 

    for i, res in enumerate(cell_spike_times):
        # Keep only spikes within that sample range
        window_spikes = res[
            (res >= start_sample) & (res < end_sample)
        ]

        spike_rate = len(window_spikes)/duration

        #log_poiss_prob = np.full(lambda_map[0].shape, -np.inf)
       # valid_mask = lambda_map >= 0
        this_lambda_map = lambda_map[i] if isinstance(lambda_map, list) else lambda_map

        
        log_poiss_prob.append( spike_rate * np.log(this_lambda_map) - this_lambda_map)

    return log_poiss_prob
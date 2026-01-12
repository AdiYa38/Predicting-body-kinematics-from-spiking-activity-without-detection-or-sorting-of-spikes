import numpy as np
import matplotlib.pyplot as plt
import heatmaps
import data
import scipy.optimize
import pandas as pd

# Constants
ARENA_DIAMETER = 80
RES_SAMPLING_RATE = 20000
POS_SAMPLING_RATE = 1250
"""X_CHANNEL = 124
Y_CHANNEL = 125
N_CHANNELS = 136"""
X_CHANNEL = 59
Y_CHANNEL = 60
N_CHANNELS = 71
SESSION = "mP31_18"
EEG_FILE = f"data\{SESSION}\{SESSION}.eeg"
CLU_FILE = f"data\{SESSION}\{SESSION}.clu"
RES_FILE = f"data\{SESSION}\{SESSION}.res"
DTYPE = np.int16
KERNEL_SIZE = 7
show = False
BIN_SIZE = 1

def gaussian_2d(coords, A, x0, y0, sigma_x, sigma_y, offset):
    """Calculates the value of a 2D Gaussian function at given coordinates."""
    x, y = coords
    exponent = -(((x - x0)**2 / (2 * sigma_x**2)) + ((y - y0)**2 / (2 * sigma_y**2)))
    return A * np.exp(exponent) + offset

def create_grid(N):
    """Creates a 2D coordinate grid of size N x N."""
    x = np.arange(0, N)
    y = np.arange(0, N)
    X, Y = np.meshgrid(x, y)
    return (X, Y)

def fit_gaussian_to_map(heatmap_data, BIN_SIZE):
    """
    Attempts to fit a 2D Gaussian model to a given heatmap.
    Returns the optimal parameters and the data used for fitting.
    """
    N = heatmap_data.shape[0]
    X, Y = create_grid(N)

    # Filter out invalid data points (NaN or -1)
    valid_mask = ~np.isnan(heatmap_data) & (heatmap_data != -1)
    
    if np.sum(valid_mask) < 6: # Need at least 6 points to fit 6 parameters
        return None, None, None, 'Not enough data points'

    z_data = heatmap_data[valid_mask]
    coords_data = np.vstack((X[valid_mask].ravel(), Y[valid_mask].ravel()))

    # Create an intelligent initial guess for the parameters
    initial_amplitude = np.nanmax(z_data)
    max_index = np.unravel_index(np.nanargmax(heatmap_data), heatmap_data.shape)
    initial_x0, initial_y0 = max_index[1], max_index[0]
    initial_sigma = N / 4
    initial_offset = np.nanmedian(z_data)
    initial_guess = [initial_amplitude, initial_x0, initial_y0, initial_sigma, initial_sigma, initial_offset]

    try:
        # Perform the curve fitting
        popt, pcov = scipy.optimize.curve_fit(gaussian_2d, coords_data, z_data, p0=initial_guess, maxfev=10000)
        # If the center is outside of the arena
        if ((np.abs(popt[1] * BIN_SIZE) > (ARENA_DIAMETER)) or (np.abs(popt[2] * BIN_SIZE) > (ARENA_DIAMETER))
        or (0 > popt[1] * BIN_SIZE) or (0 > popt[1] * BIN_SIZE)):
            return None, None, None, 'Center outside of the arena'
        
        return popt, z_data, coords_data, None # Success
    except RuntimeError:
        return None, None, None, 'Fit failed to converge' # Failure

def calculate_r_squared(popt, z_data, coords_data):
    # Predict values using the fitted model
    z_predicted = gaussian_2d(coords_data, *popt)
    
    # Calculate Sum of Squared Residuals
    ss_res = np.sum((z_data - z_predicted)**2)
    # Calculate Total Sum of Squares
    ss_tot = np.sum((z_data - np.mean(z_data))**2)
    
    # Calculate R-squared, handling the case where ss_tot is zero
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    return r_squared

def create_permutations(map_matrix, N = 20):
    # Create mask of the inside
    shuffle_mask = map_matrix != -1
    
    # Extract the values
    values_to_shuffle = map_matrix[shuffle_mask]

    # Create N permutations
    permutations_list = []

    for _ in range(N):
        new_map = map_matrix.copy()
        shuffled_values = np.random.permutation(values_to_shuffle)
        new_map[shuffle_mask] = shuffled_values
        permutations_list.append(new_map)

    return np.array(permutations_list)

def calc_avg_R_square(map_matrix, BIN_SIZE, N=20):
    R_squares = []
    perm_maps = create_permutations(map_matrix, N)

    for m in perm_maps:
        # Fit model for permutation
        popt_perm, z_data_perm, coords_data_perm, error_msg = fit_gaussian_to_map(m, BIN_SIZE)

        # For successful fitting, calculate R²
        if popt_perm is not None:
            r_squared_perm = calculate_r_squared(popt_perm, z_data_perm, coords_data_perm)
            R_squares.append(r_squared_perm)

    # Return results
    if not R_squares:
        return 0.0
    
    return np.average(R_squares)

# Read data of unit
gaussian_kernel = heatmaps.create_gaussian_kernel(size=KERNEL_SIZE)
eeg_data = data.get_eeg_data(EEG_FILE, DTYPE, N_CHANNELS)
x_values, y_values, x_in, y_in = data.import_position_data(eeg_data, X_CHANNEL, Y_CHANNEL, ARENA_DIAMETER, SESSION)
cell_maps = []

# Create rates maps
for shank in range(1,7):
    # --- תיקון: שימוש ב-f-string גם בקריאה לפונקציה ---
    tet_res, clu = data.get_tetrode_spike_times(CLU_FILE, RES_FILE, shank, POS_SAMPLING_RATE, RES_SAMPLING_RATE)
    for cell in range(2, 31):
        
            final_rates_map, grid = heatmaps.rates_map(BIN_SIZE, cell, x_values, y_values, tet_res, clu, SESSION)
            # If the unit does not exist in the shank, continue
            if (final_rates_map is None):
                print(f"unit {cell} does not exist in shank {shank}")
                continue

            cell_maps.append([final_rates_map, cell, shank, BIN_SIZE])
            print(f"Unit {shank}.{cell}: Rates map for bin size {BIN_SIZE} created")

# 2. Gaussian Fitting using for each map
gaussian_fits = []
for i, item in enumerate(cell_maps):
    heatmap_data, cell_id, shank_id, BIN_SIZE = item

    # Create fit
    popt, z_data, coords_data, error_msg = fit_gaussian_to_map(heatmap_data, BIN_SIZE)

    # Print progress and save result
    if popt is None:
        # If fitting failed, store the failure details
        print(f"Bin size {BIN_SIZE}, Clu: {cell_id} of shank {shank_id}: Fit FAILED ({error_msg})")
        gaussian_fits.append({'matrix_index': i, 'cell_id': cell_id, 'shank_id': shank_id, 'bin_size': BIN_SIZE, 'params': None, 'r_squared': -np.inf, 'error': error_msg})
    else:
        # If fitting succeeded calculate normalized R²
        r_squared = calculate_r_squared(popt, z_data, coords_data)
        print(f"r squared = {r_squared}. calc avg R squared")
        avg_r_squared = calc_avg_R_square(heatmap_data, BIN_SIZE)
        print(f"Avg R squared = {avg_r_squared}")
        n_r_squared = np.tanh(np.arctanh(r_squared) - np.arctanh(avg_r_squared))

        # Store the success details
        # --- תיקון: שומר את n_r_squared ---
        gaussian_fits.append({'matrix_index': i, 'cell_id': cell_id, 'shank_id': shank_id, 'bin_size': BIN_SIZE, 'params': popt, 'n_r_squared': n_r_squared})
        print(f"Bin size {BIN_SIZE}, Clu {cell_id} of shank {shank_id} has normalized R-squared: {n_r_squared:.4f}")

# Save results

print("\nProcessing results to save to file...")

# Create data list
results_to_save = []
for fit in gaussian_fits:
    # Save only successful fits
    if fit['params'] is not None:
        params = fit['params']
        
        # Create dictionary
        row_data = {
            'shank': fit['shank_id'],
            'unit': fit['cell_id'],
            'n_r_squared': fit['n_r_squared'],
            'center_x_cm': params[1] * fit['bin_size'],
            'center_y_cm': params[2] * fit['bin_size']
        }
        results_to_save.append(row_data)

# Create and save file
if results_to_save:
    df_results = pd.DataFrame(results_to_save)
    filename = f"Centers_{SESSION}_bin{BIN_SIZE}cm.csv"
    
    df_results.to_csv(filename, index=False)
    
    print(f"\nSuccessfully saved {len(df_results)} successful fits to: {filename}")
else:
    print("\nNo successful fits were found to save.")
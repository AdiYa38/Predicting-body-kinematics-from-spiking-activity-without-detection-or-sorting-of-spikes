import numpy as np
import matplotlib.pyplot as plt
import heatmaps
import data
import scipy.optimize

# Constants
ARENA_DIAMETER = 100
RES_SAMPLING_RATE = 20000
POS_SAMPLING_RATE = 1250
X_CHANNEL = 124
Y_CHANNEL = 125
N_CHANNELS = 136
EEG_FILE = "mp79_17/mP79_17.eeg"
DTYPE = np.int16
KERNEL_SIZE = 7

# --- Helper Functions ---
def gaussian_2d(coords, A, x0, y0, sigma_x, sigma_y, offset):
    x, y = coords
    exponent = -(((x - x0)**2 / (2 * sigma_x**2)) + ((y - y0)**2 / (2 * sigma_y**2)))
    return A * np.exp(exponent) + offset

def create_grid(N):
    """Creates a 2D coordinate grid of size N x N."""
    x = np.arange(0, N)
    y = np.arange(0, N)
    X, Y = np.meshgrid(x, y)
    return (X, Y)

def fit_gaussians(BIN_SIZE, show = False):
    # Create smoothing kernel
    gaussian_kernel = heatmaps.create_gaussian_kernel(size=KERNEL_SIZE)

    # Load Data
    eeg_data = data.get_eeg_data(EEG_FILE, DTYPE, N_CHANNELS)
    cell_maps = []

    for shank in range(1, 13):
        tet_res, clu = data.get_tetrode_spike_times("mp79_17/mP79_17.clu.", "mp79_17/mP79_17.res.", shank, POS_SAMPLING_RATE, RES_SAMPLING_RATE)
        for cell in [2]:# range(2, 18):
            res = data.get_cell_spike_times(clu, tet_res, cell)
            x_values, y_values, x_in, y_in = data.import_position_data(eeg_data, X_CHANNEL, Y_CHANNEL, ARENA_DIAMETER)

            # Create map for neuron
            bins_grid = heatmaps.create_bins(BIN_SIZE,ARENA_DIAMETER)
            spike_map_raw, vacants = heatmaps.bins_spikes_count(bins_grid, res, x_values, y_values, BIN_SIZE, ARENA_DIAMETER)
            time_map_raw = heatmaps.calculate_time_in_bin(bins_grid, x_values, y_values, BIN_SIZE, ARENA_DIAMETER, POS_SAMPLING_RATE)
            spike_map_covered  = heatmaps.cover_vacants(bins_grid, spike_map_raw, vacants)
            time_map_covered = heatmaps.cover_vacants(bins_grid, time_map_raw, vacants)
            spike_map_smoothed = heatmaps.smooth(spike_map_covered, gaussian_kernel, bins_grid)
            time_map_smoothed = heatmaps.smooth(time_map_covered, gaussian_kernel, bins_grid)
            
            # Handle division by zero safely DELETE?
            #with np.errstate(divide='ignore', invalid='ignore'):
            rates_map = spike_map_smoothed / time_map_smoothed
            
            # Replace NaNs or Infs that result from division by zero with 0
            rates_map[~np.isfinite(rates_map)] = 0
            
            final_rates_map = heatmaps.remove_vacants(rates_map, vacants)
            
            cell_maps.append([final_rates_map, cell, shank])
            print(f"Finished cell {cell} of shank {shank}")

        # --- Gaussian Fitting Loop ---
        gaussian_fits = []

    for i, item in enumerate(cell_maps):
        heatmap_data, cell_id, shank_id = item
        N = heatmap_data.shape[0]
        X, Y = create_grid(N)

        # Handle NaN values
        positive_mask = ~np.isnan(heatmap_data) & (heatmap_data != -1)
        
        if np.sum(positive_mask) < 6: 
            print(f"Clu {cell_id} in shank {shank_id} is damaged")
            gaussian_fits.append({'matrix_index': i, 'cell_id': cell_id, 'shank_id': shank_id, 'params': None, 'r_squared': -np.inf, 'error': 'Not enough data points'})
            continue

        z_data = heatmap_data[positive_mask]
        coords_data = np.vstack((X[positive_mask].ravel(), Y[positive_mask].ravel()))

        # Initial guess
        initial_amplitude = np.nanmax(z_data)
        max_index = np.unravel_index(np.nanargmax(heatmap_data), heatmap_data.shape)
        initial_x0, initial_y0 = max_index[1], max_index[0]
        initial_sigma = N / 4
        initial_offset = np.nanmedian(z_data)
        initial_guess = [initial_amplitude, initial_x0, initial_y0, initial_sigma, initial_sigma, initial_offset]

        try:
            popt, pcov = scipy.optimize.curve_fit(gaussian_2d, coords_data, z_data, p0=initial_guess, maxfev=10000)
            
            z_predicted = gaussian_2d(coords_data, *popt)
            
            # Calculate Sum of Squared Residuals
            ss_res = np.sum((z_data - z_predicted)**2)
            # Calculate Total Sum of Squares
            ss_tot = np.sum((z_data - np.mean(z_data))**2)
            # Calculate R-squared
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Store results with the new metric
            gaussian_fits.append({'matrix_index': i, 'cell_id': cell_id, 'shank_id': shank_id, 'params': popt, 'r_squared': r_squared})
            print(f"Clu {cell_id} of shank {shank_id} has R-squared: {r_squared:.4f}")

        except RuntimeError:
            print(f"Clu: {cell_id} of shank {shank_id}: Fit FAILED")
            gaussian_fits.append({'matrix_index': i, 'cell_id': cell_id, 'shank_id': shank_id, 'params': None, 'r_squared': -np.inf, 'error': 'Fit failed to converge'})

    print(f"show = {show}")
    if (show):
        for fit in gaussian_fits:
            matrix_index = fit['matrix_index']
            cell_id = fit['cell_id']
            shank_id = fit.get('shank_id', shank)
            original_map = cell_maps[matrix_index][0].copy()

            if fit['params'] is not None:
                fit_params = fit['params']
                
                # Extract center coordinates for plotting
                center_x = fit_params[1]
                center_y = fit_params[2]
                
                N = original_map.shape[0]
                X, Y = create_grid(N)
                fitted_gaussian_map = gaussian_2d((X, Y), *fit_params)
                
                valid_data = original_map[~np.isnan(original_map) & (original_map != -1)]
                vmin = valid_data.min() if len(valid_data) > 0 else 0
                vmax = valid_data.max() if len(valid_data) > 0 else 1
                
                # Probably can be better written with remove background
                vacant_mask = (original_map == -1) | np.isnan(original_map)
                original_map[vacant_mask] = np.nan
                fitted_gaussian_map[vacant_mask] = np.nan
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                fig.suptitle(f'Shank: {shank_id}, Clu: {cell_id}\nR²: {fit["r_squared"]:.4f}', fontsize=16)
                
                # Plot 1: Original Firing Rate Map
                im1 = ax1.imshow(original_map, cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
                ax1.set_title('Original Firing Rate Map')
                ax1.set_xlabel('X Bins')
                ax1.set_ylabel('Y Bins')
                # Mark the calculated center on the original map as well
                ax1.plot(center_x, center_y, '+', color='red', markersize=12, markeredgewidth=2)
                
                # Plot 2: Fitted Gaussian Model
                ax2.imshow(fitted_gaussian_map, cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
                # Update title to include center coordinates
                ax2.set_title(f'Fitted Gaussian Model\nCenter: ({center_x:.2f}, {center_y:.2f})')
                ax2.set_xlabel('X Bins')
                ax2.set_yticklabels([])
                # Add a marker for the Gaussian center
                ax2.plot(center_x, center_y, '+', color='red', markersize=12, markeredgewidth=2)
                
                fig.colorbar(im1, ax=[ax1, ax2], label='Firing Rate [spikes/sec]', shrink=0.7)
                
            else: # This block handles cases where the fit failed
                fig, ax = plt.subplots(1, 1, figsize=(7, 6))
                title_text = f'Shank: {shank_id}, Clu: {cell_id}\n(Fit Failed: {fit["error"]})'
                ax.set_title(title_text)
                
                vacant_mask = (original_map == -1) | np.isnan(original_map)
                original_map[vacant_mask] = np.nan
                im = ax.imshow(original_map, cmap='jet', origin='lower')
                ax.set_xlabel('X Bins')
                ax.set_ylabel('Y Bins')
                fig.colorbar(im, ax=ax, label='Firing Rate [spikes/sec]', shrink=0.8)

        plt.show()

    return gaussian_fits

#fit_gaussians(True)
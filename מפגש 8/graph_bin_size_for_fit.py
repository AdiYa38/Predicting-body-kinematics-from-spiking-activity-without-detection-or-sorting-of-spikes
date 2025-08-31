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
shank = 2
cell = 8
show = True

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

# Create smoothing kernel
gaussian_kernel = heatmaps.create_gaussian_kernel(size=KERNEL_SIZE)

# Load Data
eeg_data = data.get_eeg_data(EEG_FILE, DTYPE, N_CHANNELS)
cell_maps = []
tet_res, clu = data.get_tetrode_spike_times("mp79_17/mP79_17.clu.", "mp79_17/mP79_17.res.", shank, POS_SAMPLING_RATE, RES_SAMPLING_RATE)
res = data.get_cell_spike_times(clu, tet_res, cell)
x_values, y_values, x_in, y_in = data.import_position_data(eeg_data, X_CHANNEL, Y_CHANNEL, ARENA_DIAMETER)

for BIN_SIZE in [100, 50, 25, 20, 10, 5, 4, 2, 1]:
    # Create rates map for neuron
    bins_grid = heatmaps.create_bins(BIN_SIZE,ARENA_DIAMETER)
    spike_map_raw, vacants = heatmaps.bins_spikes_count(bins_grid, res, x_values, y_values, BIN_SIZE, ARENA_DIAMETER)
    time_map_raw = heatmaps.calculate_time_in_bin(bins_grid, x_values, y_values, BIN_SIZE, ARENA_DIAMETER, POS_SAMPLING_RATE)
    spike_map_covered  = heatmaps.cover_vacants(bins_grid, spike_map_raw, vacants)
    time_map_covered = heatmaps.cover_vacants(bins_grid, time_map_raw, vacants)
    spike_map_smoothed = heatmaps.smooth(spike_map_covered, gaussian_kernel, bins_grid)
    time_map_smoothed = heatmaps.smooth(time_map_covered, gaussian_kernel, bins_grid)
    rates_map = spike_map_smoothed / time_map_smoothed
    final_rates_map = heatmaps.remove_vacants(rates_map, vacants)
    cell_maps.append([final_rates_map, cell, shank, BIN_SIZE])
    print(f"Rates map for bin size {BIN_SIZE} created")

# Create gaussian fit
gaussian_fits = []

for i, item in enumerate(cell_maps):
    heatmap_data, cell_id, shank_id, BIN_SIZE = item
    N = heatmap_data.shape[0]
    X, Y = create_grid(N)

    # Handle NaN values
    positive_mask = ~np.isnan(heatmap_data) & (heatmap_data != -1)
    
    if np.sum(positive_mask) < 6: 
        print(f"Clu {cell_id} in shank {shank_id} is damaged")
        gaussian_fits.append({'matrix_index': i, 'cell_id': cell_id, 'shank_id': shank_id, 'bin_size':BIN_SIZE, 'params': None, 'r_squared': -np.inf, 'error': 'Not enough data points'})
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
        gaussian_fits.append({'matrix_index': i, 'cell_id': cell_id, 'shank_id': shank_id, 'bin_size':BIN_SIZE, 'params': popt, 'r_squared': r_squared})
        print(f"Bin size {BIN_SIZE}, Clu {cell_id} of shank {shank_id} has R-squared: {r_squared:.4f}")

    except RuntimeError:
        print(f"Bin size {BIN_SIZE}, Clu: {cell_id} of shank {shank_id}: Fit FAILED")
        gaussian_fits.append({'matrix_index': i, 'cell_id': cell_id, 'shank_id': shank_id, 'bin_size': BIN_SIZE, 'params': None, 'r_squared': -np.inf, 'error': 'Fit failed to converge'})

print(f"show = {show}")
if (show):
    # Loop through the fit results to plot them
    for fit in gaussian_fits:
        matrix_index = fit['matrix_index']
        cell_id = fit['cell_id']
        shank_id = fit.get('shank_id', shank)
        
        # --- FIX 1: Use a unique variable name for the bin size of the current plot ---
        # This prevents confusion with the global BIN_SIZE variable from the first loop.
        current_bin_size = fit['bin_size'] 
        
        original_map = cell_maps[matrix_index][0].copy()

        if fit['params'] is not None:
            fit_params = fit['params']
            
            center_x = fit_params[1]
            center_y = fit_params[2]
            
            N = original_map.shape[0]
            X, Y = create_grid(N)
            fitted_gaussian_map = gaussian_2d((X, Y), *fit_params)
            
            valid_data = original_map[~np.isnan(original_map) & (original_map != -1)]
            vmin = valid_data.min() if len(valid_data) > 0 else 0
            vmax = valid_data.max() if len(valid_data) > 0 else 1
            
            vacant_mask = (original_map == -1) | np.isnan(original_map)
            original_map[vacant_mask] = np.nan
            fitted_gaussian_map[vacant_mask] = np.nan
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            # --- FIX 2: Use the correct variable 'current_bin_size' in the title ---
            fig.suptitle(f'Bin size:{current_bin_size}cm, Shank: {shank_id}, Clu: {cell_id}\nR²: {fit["r_squared"]:.4f}', fontsize=16)
            
            # Plot 1: Original Firing Rate Map
            im1 = ax1.imshow(original_map, cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
            ax1.set_title('Original Firing Rate Map')
            ax1.set_xlabel('X Bins')
            ax1.set_ylabel('Y Bins')
            ax1.plot(center_x, center_y, '+', color='red', markersize=12, markeredgewidth=2)
            
            # Plot 2: Fitted Gaussian Model
            ax2.imshow(fitted_gaussian_map, cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
            ax2.set_title(f'Fitted Gaussian Model\nCenter: ({center_x:.2f}, {center_y:.2f})')
            ax2.set_xlabel('X Bins')
            ax2.set_yticklabels([])
            ax2.plot(center_x, center_y, '+', color='red', markersize=12, markeredgewidth=2)
            
            fig.colorbar(im1, ax=[ax1, ax2], label='Firing Rate [spikes/s]', shrink=0.7)
            
        else: # This block handles cases where the fit failed
            fig, ax = plt.subplots(1, 1, figsize=(7, 6))
            
            # --- FIX 3: Add bin size to the title for failed fits ---
            title_text = f'Bin size:{current_bin_size}cm, Shank: {shank_id}, Clu: {cell_id}\n(Fit Failed: {fit["error"]})'
            ax.set_title(title_text)
            
            vacant_mask = (original_map == -1) | np.isnan(original_map)
            original_map[vacant_mask] = np.nan
            im = ax.imshow(original_map, cmap='jet', origin='lower')
            ax.set_xlabel('X Bins')
            ax.set_ylabel('Y Bins')
            fig.colorbar(im, ax=ax, label='Firing Rate [spikes/s]', shrink=0.8)

    plt.show()

# Graph R-squared for bin size
# Extract the data for the plot from the results
# Use list comprehensions for a concise way to get the data
bin_sizes = [fit['bin_size'] for fit in gaussian_fits if fit['params'] is not None]
r_squared_values = [fit['r_squared'] for fit in gaussian_fits if fit['params'] is not None]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(bin_sizes, r_squared_values, marker='o', linestyle='-', color='b')

# Add titles and labels
plt.title(f'R² vs. Bin Size for Cell {cell}, Shank {shank}', fontsize=16)
plt.xlabel('Bin Size [cm]', fontsize=12)
plt.ylabel('R-squared [R²]', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Because your bin sizes are widely and non-linearly spaced,
# a logarithmic scale on the x-axis can make the plot easier to read.
plt.xscale('log')
# Ensure all bin size values appear as ticks on the x-axis
plt.xticks(ticks=bin_sizes, labels=bin_sizes)


# Add a horizontal line at R²=0 for reference
plt.axhline(0, color='grey', linestyle='--', linewidth=1)

# The final plt.show() will display this graph after the others are closed.
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import heatmaps
import data
import scipy.optimize

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

def fit_gaussians(BIN_SIZE, SESSION, show=False):

    EEG_FILE = f"data\{SESSION}\{SESSION}.eeg"
    CLU_FILE = f"data\{SESSION}\{SESSION}.clu"
    RES_FILE = f"data\{SESSION}\{SESSION}.res"
    # Load Data
    eeg_data = data.get_eeg_data(EEG_FILE, DTYPE, N_CHANNELS)
    x_values, y_values, x_in, y_in = data.import_position_data(eeg_data, X_CHANNEL, Y_CHANNEL, ARENA_DIAMETER, SESSION)
    cell_maps = []

    for shank in [1,2,3]: #range(1, 13):
        tet_res, clu = data.get_tetrode_spike_times(CLU_FILE, RES_FILE, shank, POS_SAMPLING_RATE, RES_SAMPLING_RATE)
        for cell in [1,2,3,10]:#range(2, 20):
            final_rates_map, bins_grid = heatmaps.rates_map(BIN_SIZE, cell, x_values, y_values, tet_res, clu, SESSION)
            # If the unit does not exist in the shank, continue
            if (final_rates_map is None):
                print(f"unit {cell} does not exist in shank {shank}")
                continue

            cell_maps.append([final_rates_map, cell, shank, bins_grid])
            print(f"Finished cell {cell} of shank {shank}")

    # --- Gaussian Fitting Loop ---
    gaussian_fits = []
    
    for i, item in enumerate(cell_maps):
        heatmap_data, cell_id, shank_id, bins_grid = item
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
            ss_res = np.sum((z_data - z_predicted)**2)
            ss_tot = np.sum((z_data - np.mean(z_data))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # If the center is outside of the arena
            if ((np.abs(popt[1] * BIN_SIZE) > (ARENA_DIAMETER)) or (np.abs(popt[2] * BIN_SIZE) > (ARENA_DIAMETER))
            or (0 > popt[1] * BIN_SIZE) or (0 > popt[1] * BIN_SIZE)):
                print(f"Clu {cell_id} of shank {shank_id} is centered outside of the arena")
                gaussian_fits.append({'matrix_index': i, 'cell_id': cell_id, 'shank_id': shank_id, 'params': None, 'r_squared': -np.inf, 'error': 'Center outside of the arena'})
            else:
                gaussian_fits.append({'matrix_index': i, 'cell_id': cell_id, 'shank_id': shank_id, 'params': popt, 'r_squared': r_squared})
                print(f"Clu {cell_id} of shank {shank_id} has R-squared: {r_squared:.4f}, center: ({popt[1]},{popt[2]})")

        except RuntimeError:
            print(f"Clu: {cell_id} of shank {shank_id}: Fit FAILED")
            gaussian_fits.append({'matrix_index': i, 'cell_id': cell_id, 'shank_id': shank_id, 'params': None, 'r_squared': -np.inf, 'error': 'Fit failed to converge'})

    print(f"show = {show}")
    if (show):
        extent_in_cm = [0, ARENA_DIAMETER, 0, ARENA_DIAMETER]

        for fit in gaussian_fits:
            matrix_index = fit['matrix_index']
            cell_id = fit['cell_id']
            shank_id = fit.get('shank_id', shank)
            original_map, _, _, bins_grid = cell_maps[matrix_index]
            original_map = original_map.copy()

            # --- שינוי 1: חישוב ערכי המינימום והמקסימום להצגה ---
            # נשתמש בערכים אלו בכל המפות כדי לשמור על סקאלה אחידה
            max_val = heatmaps.max_val_to_show(original_map)
            vmin = 0  # הנחה שקצב הירי אינו שלילי

            if fit['params'] is not None:
                fit_params = fit['params']
                
                center_x_bins = fit_params[1]
                center_y_bins = fit_params[2]
                center_x_cm = center_x_bins * BIN_SIZE
                center_y_cm = center_y_bins * BIN_SIZE
                
                N = original_map.shape[0]
                X, Y = create_grid(N)
                fitted_gaussian_map = gaussian_2d((X, Y), *fit_params)
                
                # --- שינוי 2: הסרת החישוב הישן של vmin/vmax ---
                # השורות שהיו כאן הוסרו כי הגדרנו vmin ו-max_val מחוץ לתנאי
                
                vacant_mask = (original_map == -1) | np.isnan(original_map)
                original_map[vacant_mask] = np.nan
                fitted_gaussian_map[vacant_mask] = np.nan
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                fig.suptitle(f'Shank: {shank_id}, Clu: {cell_id}\nR²: {fit["r_squared"]:.4f}', fontsize=16)
                
                # Plot 1: Original Firing Rate Map
                # --- שינוי 3: שימוש ב-max_val שהוגדר מראש ---
                im1 = ax1.imshow((heatmaps.remove_background(original_map, bins_grid)), cmap='jet', origin='lower', vmin=vmin, vmax=max_val, extent=extent_in_cm)
                ax1.set_title('Original Firing Rate Map')
                ax1.set_xlabel('X [cm]')
                ax1.set_ylabel('Y [cm]')
                ax1.plot(center_x_cm, center_y_cm, '+', color='red', markersize=12, markeredgewidth=2)
                
                # Plot 2: Fitted Gaussian Model
                # --- שינוי 3: שימוש ב-max_val שהוגדר מראש ---
                ax2.imshow((heatmaps.remove_background(fitted_gaussian_map, bins_grid)), cmap='jet', origin='lower', vmin=vmin, vmax=max_val, extent=extent_in_cm)
                ax2.set_title(f'Fitted Gaussian Model\nCenter: ({center_x_cm:.2f}, {center_y_cm:.2f}) cm')
                ax2.set_xlabel('X [cm]')
                ax2.set_yticklabels([])
                ax2.plot(center_x_cm, center_y_cm, '+', color='red', markersize=12, markeredgewidth=2)
                
                fig.colorbar(im1, ax=[ax1, ax2], label='Firing Rate [spikes/sec]', shrink=0.7)
                
            else: # This block handles cases where the fit failed
                fig, ax = plt.subplots(1, 1, figsize=(7, 6))
                title_text = f'Shank: {shank_id}, Clu: {cell_id}\n(Fit Failed: {fit["error"]})'
                ax.set_title(title_text)
                
                vacant_mask = (original_map == -1) | np.isnan(original_map)
                original_map[vacant_mask] = np.nan
                # --- שינוי 3: שימוש ב-max_val שהוגדר מראש ---
                im = ax.imshow(original_map, cmap='jet', origin='lower', extent=extent_in_cm, vmin=vmin, vmax=max_val)
                ax.set_xlabel('X [cm]')
                ax.set_ylabel('Y [cm]')
                fig.colorbar(im, ax=ax, label='Firing Rate [spikes/sec]', shrink=0.8)

        plt.show()

    return gaussian_fits

fit_gaussians(1, "mP31_19", True)

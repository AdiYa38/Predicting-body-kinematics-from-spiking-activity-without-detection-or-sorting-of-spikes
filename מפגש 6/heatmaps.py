import numpy as np
from scipy.ndimage import convolve, distance_transform_edt
 
temp_bin_size_cm =10
arena_curr_diam = 100

OUTSIDE_FLAG = -1
INSIDE_FLAG = 0

def create_bins(bin_size_cm = temp_bin_size_cm, arena_diameter_cm = arena_curr_diam):
    
    num_bins_per_axis = int(np.ceil(arena_diameter_cm / bin_size_cm))
    bins = np.full((num_bins_per_axis, num_bins_per_axis), fill_value=INSIDE_FLAG, dtype=np.int32)
    arena_radius_cm = arena_diameter_cm / 2.0
    
    for i in range(num_bins_per_axis):
        for j in range(num_bins_per_axis):
            y_cm = (i + 0.5) * bin_size_cm - arena_radius_cm
            x_cm = (j + 0.5) * bin_size_cm - arena_radius_cm
            
            if np.sqrt(x_cm**2 + y_cm**2) > arena_radius_cm:
                bins[i, j] = OUTSIDE_FLAG 
                
    return bins

def map_coords_to_bin_ids(x_cm, y_cm, bins, bin_size_cm = temp_bin_size_cm, arena_diameter_cm = arena_curr_diam):

    OUTSIDE_FLAG = -1

    is_single_input = np.isscalar(x_cm)
    x_cm = np.atleast_1d(x_cm)
    y_cm = np.atleast_1d(y_cm)

    arena_radius_cm = arena_diameter_cm / 2.0
    num_rows, num_cols = bins.shape

    # Pharse coordinates to indexes
    x_indices = np.floor((x_cm + arena_radius_cm) / bin_size_cm).astype(np.int32)
    y_indices = np.floor((y_cm + arena_radius_cm) / bin_size_cm).astype(np.int32)

    # Create array of results - x, y coordinates
    results = np.full((len(x_indices), 2), fill_value=OUTSIDE_FLAG, dtype=np.int32)

    # Filter coordinates outsite the matrix
    in_bounds_mask = (x_indices >= 0) & (x_indices < num_cols) & \
                     (y_indices >= 0) & (y_indices < num_rows)

    valid_y_indices = y_indices[in_bounds_mask]
    valid_x_indices = x_indices[in_bounds_mask]
    
    # Filter coordinatesin matrix, but ouutside of the arena
    bin_values = bins[valid_y_indices, valid_x_indices]
    is_inside_arena_mask = (bin_values != OUTSIDE_FLAG)

    final_y_indices = valid_y_indices[is_inside_arena_mask]
    final_x_indices = valid_x_indices[is_inside_arena_mask]
    
    # Pair x, y coordinates
    valid_index_pairs = np.stack((final_y_indices, final_x_indices), axis=1)

    final_placement_mask = np.zeros_like(in_bounds_mask)
    final_placement_mask[in_bounds_mask] = is_inside_arena_mask
    
    results[final_placement_mask] = valid_index_pairs

    # In case of a single pair of coordinates, return 1 bin indexes
    if is_single_input:
        return tuple(results[0])
        
    return results

def bins_spikes_count(bins, res, x_values, y_values, bin_len, arena_diameter):
    
    # Initial matrixes
    bins_spikes_count =  np.copy(bins)
    vacants = np.copy(bins)
    vacants[vacants == 0] = 1

    if len(res) == 0:
        return bins_spikes_count, vacants

    # Calculate spikes bins
    spike_x = x_values[res]
    spike_y = y_values[res]

    spike_bin_indices = map_coords_to_bin_ids(
        spike_x, spike_y, bins, bin_len, arena_diameter
    )

    valid_mask = spike_bin_indices[:, 0] != -1
    valid_indices = spike_bin_indices[valid_mask]

    # All spikes are outside of the arena
    if valid_indices.shape[0] == 0:
        return bins_spikes_count, vacants

    # Count into bins
    rows = valid_indices[:, 0]
    cols = valid_indices[:, 1]

    np.add.at(bins_spikes_count, (rows, cols), 1)

    # Create the vacants matrix
    unique_valid_indices = np.unique(valid_indices, axis=0)
    unique_rows = unique_valid_indices[:, 0]
    unique_cols = unique_valid_indices[:, 1]

    vacants[unique_rows, unique_cols] = 0

    return bins_spikes_count, vacants

def calculate_time_in_bin(bins, x_values, y_values, bin_size_cm, arena_diameter_cm, sampling_rate=1250):

    time_per_sample = 1.0 / sampling_rate

    # Map the coordinates to bins and filter the outsides
    all_bin_indices = map_coords_to_bin_ids(
        x_values, y_values, bins, bin_size_cm, arena_diameter_cm
    )

    valid_mask = all_bin_indices[:, 0] != -1
    valid_indices = all_bin_indices[valid_mask]

    # All coordinates are outside
    if valid_indices.shape[0] == 0:
        time_matrix = np.copy(bins).astype(float)
        time_matrix[time_matrix == 0] = 0.0 # אתחול תאים פנימיים ל-0.0
        return time_matrix

    # Count number of indices
    unique_indices, counts = np.unique(valid_indices, axis=0, return_counts=True)
    
    # Create the results matrix
    time_in_bin = np.copy(bins).astype(float)
    time_in_bin[time_in_bin == 0] = 0.0

    rows = unique_indices[:, 0]
    cols = unique_indices[:, 1]
    
    # Results in s
    time_values = counts * time_per_sample
    time_in_bin[rows, cols] = time_values
    
    return time_in_bin

def create_gaussian_kernel(size=7, sigma_cutoff=2.57):

    # Claculate sigma
    half_size = size // 2
    sigma = half_size / sigma_cutoff

    # Create grid
    ax = np.arange(-half_size, half_size + 1)
    xx, yy = np.meshgrid(ax, ax)

    # Calculate and ertuen a normal gaussian
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

    return kernel / np.sum(kernel)

import numpy as np

def create_reflection_padded_matrix(data_matrix, bins_grid, kernel_size):

    # Initial padded matrix
    padded_matrix = np.copy(data_matrix).astype(float)
    padded_matrix[bins_grid == -1] = 0
    
    rows, cols = data_matrix.shape
    pad_dist = kernel_size // 2

    # Find border pixels
    border_pixels = []
    inside_mask = (bins_grid != -1)
    
    # Border pixel have 2 pixels on their sides that outside of the arena
    for r in range(rows):
        for c in range(cols):
            if not inside_mask[r, c]:
                continue
            
            is_border = False
            # Check naighbors
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                
                if not (0 <= nr < rows and 0 <= nc < cols) or not inside_mask[nr, nc]:
                    is_border = True
                    break
            
            if is_border:
                border_pixels.append((r, c))

    # For outside bins, change to unside vakue
    for r_out in range(rows):
        for c_out in range(cols):
            if inside_mask[r_out, c_out]:
                continue

            # Find closest border cell
            min_dist = float('inf')
            closest_border_pixel = None
            for r_in, c_in in border_pixels:
                dist_sq = (r_in - r_out)**2 + (c_in - c_out)**2
                if dist_sq < min_dist:
                    min_dist = dist_sq
                    closest_border_pixel = (r_in, c_in)
            
            # Change only cells that efffect convolution result
            if np.sqrt(min_dist) > pad_dist + 1:
                continue

            # Find the cell to reflect
            r_border, c_border = closest_border_pixel
            dr = r_border - r_out
            dc = c_border - c_out
            
            r_source = r_border + dr
            c_source = c_border + dc
            
            if 0 <= r_source < rows and 0 <= c_source < cols:
                padded_matrix[r_out, c_out] = padded_matrix[r_source, c_source]

    # Safety check - keep inside cells the same
    padded_matrix[inside_mask] = data_matrix[inside_mask]
    
    return padded_matrix

def smooth(data_matrix, kernel, bins_grid):

    # create padded matrix
    padded_matrix = create_reflection_padded_matrix(data_matrix, bins_grid, kernel.shape[0])
    
    # Convolve with padded matrix
    smoothed_map = convolve(padded_matrix, kernel, mode='mirror')
    
    # Remove padding
    smoothed_map[bins_grid == -1] = -1
    
    return smoothed_map


def remove_vacants(mat, vacants):
    res_mat = mat
    mat[vacants == 1] = 0
    
    return mat
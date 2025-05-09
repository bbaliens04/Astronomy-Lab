import numpy as np
import numba
from astropy.io import fits
import sys
import time

# --------------------------------------------------------------------------------
#                        Numba-friendly helper functions
# --------------------------------------------------------------------------------

@numba.njit
def median_numba(arr):
    """
    Median of 1D array (no NaNs).
    If the array length is even, returns average of the two middle values.
    """
    n = arr.size
    sorted_arr = np.sort(arr)
    mid = n // 2
    if n % 2 == 1:
        return sorted_arr[mid]
    else:
        return 0.5 * (sorted_arr[mid - 1] + sorted_arr[mid])

@numba.njit
def median_ignore_nan(arr):
    """
    Median of 1D array ignoring NaNs.
    Returns NaN if the resulting subarray is empty.
    """
    # Count valid (non-NaN) elements
    count = 0
    for x in arr:
        if not np.isnan(x):
            count += 1
    if count == 0:
        return np.nan

    # Collect valid elements into a temp array
    valid = np.empty(count, dtype=np.float64)
    idx = 0
    for x in arr:
        if not np.isnan(x):
            valid[idx] = x
            idx += 1
    return median_numba(valid)

@numba.njit
def mean_numba(arr):
    """
    Mean of 1D array (no NaNs).
    """
    s = 0.0
    for x in arr:
        s += x
    return s / arr.size

@numba.njit
def std_numba(arr):
    """
    Standard deviation of 1D array (no NaNs).
    """
    m = mean_numba(arr)
    s = 0.0
    for x in arr:
        s += (x - m)**2
    return np.sqrt(s / arr.size)

@numba.njit
def std_ignore_nan(arr):
    """
    Standard deviation ignoring NaNs.
    Returns np.nan if all are NaN.
    """
    # First, count valid values
    count = 0
    for x in arr:
        if not np.isnan(x):
            count += 1
    if count == 0:
        return np.nan

    # Collect valid elements
    valid = np.empty(count, dtype=np.float64)
    idx = 0
    for x in arr:
        if not np.isnan(x):
            valid[idx] = x
            idx += 1

    return std_numba(valid)

@numba.njit
def filter_clip_numba(arr, center, clip):
    """
    Return a new array containing only values of arr
    within [center - clip, center + clip].
    Ignores NaNs (i.e., NaNs are skipped).
    """
    low = center - clip
    high = center + clip
    # first pass: count how many pass the filter
    count = 0
    for x in arr:
        if not np.isnan(x) and low < x < high:
            count += 1
    if count == 0:
        return np.empty(0, dtype=np.float64)

    # second pass: collect them
    out = np.empty(count, dtype=np.float64)
    idx = 0
    for x in arr:
        if not np.isnan(x) and low < x < high:
            out[idx] = x
            idx += 1
    return out

@numba.njit
def replace_nan_with_neighborhood_median(arr):
    """
    Replaces NaNs in a 2D array with the median of valid neighbors.
    """
    arr = arr.copy()  # work on a copy
    rows, cols = arr.shape
    for i in range(rows):
        for j in range(cols):
            if np.isnan(arr[i, j]):
                # Gather neighbors
                values = []
                for ii in range(max(0, i - 1), min(rows, i + 2)):
                    for jj in range(max(0, j - 1), min(cols, j + 2)):
                        if (ii != i or jj != j) and not np.isnan(arr[ii, jj]):
                            values.append(arr[ii, jj])
                if len(values) > 0:
                    values_arr = np.array(values, dtype=np.float64)
                    arr[i, j] = median_numba(values_arr)
    return arr

@numba.njit
def replace_nan_with_neighborhood_median_rgb(arr):
    """
    Replaces NaNs in a 3D array with the median of valid neighbors (per-channel).
    Shape expected: (channels, rows, cols).
    """
    arr = arr.copy()  # work on a copy
    channels, rows, cols = arr.shape
    for c in range(channels):
        for i in range(rows):
            for j in range(cols):
                if np.isnan(arr[c, i, j]):
                    values = []
                    for ii in range(max(0, i - 1), min(rows, i + 2)):
                        for jj in range(max(0, j - 1), min(cols, j + 2)):
                            if (ii != i or jj != j) and not np.isnan(arr[c, ii, jj]):
                                values.append(arr[c, ii, jj])
                    if len(values) > 0:
                        values_arr = np.array(values, dtype=np.float64)
                        arr[c, i, j] = median_numba(values_arr)
    return arr

# --------------------------------------------------------------------------------
#                  Master frames and other heavy-lifting in Numba
# --------------------------------------------------------------------------------

@numba.njit
def master_dark_array_numba(arr):
    """
    Create a master dark from a stack of 2D images: arr of shape (N, height, width).
    Each pixel:
      1) Compute median and std ignoring NaNs
      2) Clip outliers > 3 * std
      3) Take nanmedian of the remaining
      4) Round up with np.ceil, cast to int
    Then replace any NaNs with neighbor median.
    """
    N, height, width = arr.shape
    final_arr = np.zeros((height, width), dtype=np.float64)
    for j in range(height):
        for i in range(width):
            pixel_values = arr[:, j, i]
            # std and median ignoring NaNs
            pixel_med = median_ignore_nan(pixel_values)
            pixel_std = std_ignore_nan(pixel_values)
            if np.isnan(pixel_med) or np.isnan(pixel_std):
                # if all are NaN or something weird
                final_arr[j, i] = np.nan
                continue

            clip = 3.0 * pixel_std
            filtered = filter_clip_numba(pixel_values, pixel_med, clip)
            if filtered.size > 0:
                val = median_ignore_nan(filtered)
                if not np.isnan(val):
                    val_ceil = np.ceil(val)
                    final_arr[j, i] = val_ceil
                else:
                    final_arr[j, i] = np.nan
            else:
                final_arr[j, i] = np.nan
    # Now replace NaNs
    final_arr = replace_nan_with_neighborhood_median(final_arr)
    return final_arr

@numba.njit
def master_flat_array_numba(arr):
    """
    Same approach as master_dark_array_numba, but for flats.
    """
    N, height, width = arr.shape
    final_arr = np.zeros((height, width), dtype=np.float64)
    for j in range(height):
        for i in range(width):
            pixel_values = arr[:, j, i]
            pixel_med = median_ignore_nan(pixel_values)
            pixel_std = std_ignore_nan(pixel_values)
            if np.isnan(pixel_med) or np.isnan(pixel_std):
                final_arr[j, i] = np.nan
                continue

            clip = 3.0 * pixel_std
            filtered = filter_clip_numba(pixel_values, pixel_med, clip)
            if filtered.size > 0:
                val = median_ignore_nan(filtered)
                if not np.isnan(val):
                    val_ceil = np.ceil(val)
                    final_arr[j, i] = val_ceil
                else:
                    final_arr[j, i] = np.nan
            else:
                final_arr[j, i] = np.nan
    # Now replace NaNs
    final_arr = replace_nan_with_neighborhood_median(final_arr)
    return final_arr

@numba.njit
def master_flat_array_rgb_numba(arr):
    """
    Expects arr of shape (N, 3, height, width).
    For each channel, do the same logic.
    Returns final_arr of shape (3, height, width).
    """
    N, C, height, width = arr.shape
    out = np.zeros((C, height, width), dtype=np.float64)
    for k in range(C):
        for j in range(height):
            for i in range(width):
                pixel_values = arr[:, k, j, i]
                pixel_med = median_ignore_nan(pixel_values)
                pixel_std = std_ignore_nan(pixel_values)
                if np.isnan(pixel_med) or np.isnan(pixel_std):
                    out[k, j, i] = np.nan
                    continue

                clip = 3.0 * pixel_std
                filtered = filter_clip_numba(pixel_values, pixel_med, clip)
                if filtered.size > 0:
                    val = median_ignore_nan(filtered)
                    if not np.isnan(val):
                        val_ceil = np.ceil(val)
                        out[k, j, i] = val_ceil
                    else:
                        out[k, j, i] = np.nan
                else:
                    out[k, j, i] = np.nan

    # Replace NaNs per-channel
    # We'll do the replace_nan_with_neighborhood_median_rgb at the end
    out = replace_nan_with_neighborhood_median_rgb(out)
    return out

@numba.njit
def flat_weights_numba(flat):
    """
    Build a 'weights' array from a flat image:
      1) compute median and std of all pixels ignoring NaNs
      2) For each pixel: if it is within [median-3*std, median+3*std], weight = med / pixel
                         else weight = NaN
    """
    # flatten ignoring nans
    # first gather valid
    valid_list = []
    for x in flat.flatten():
        if not np.isnan(x):
            valid_list.append(x)
    if len(valid_list) == 0:
        # All NaNs => return an array of NaNs
        return np.full_like(flat, np.nan, dtype=np.float64)

    valid_arr = np.array(valid_list, dtype=np.float64)
    med = median_numba(np.sort(valid_arr))   # sort used in median_numba anyway
    std_ = std_numba(valid_arr)
    low = med - 3.0 * std_
    high = med + 3.0 * std_

    out = np.zeros_like(flat)
    rows, cols = flat.shape
    for r in range(rows):
        for c in range(cols):
            val = flat[r, c]
            if not np.isnan(val) and (low <= val <= high):
                out[r, c] = med / val
            else:
                out[r, c] = np.nan
    return out

@numba.njit
def flat_weights_rgb_numba(flat_rgb):
    """
    Expects shape (3, height, width).
    """
    C, H, W = flat_rgb.shape
    out = np.zeros_like(flat_rgb)
    for k in range(C):
        # gather valid for this channel
        valid_list = []
        for x in flat_rgb[k].flatten():
            if not np.isnan(x):
                valid_list.append(x)
        if len(valid_list) == 0:
            # fill entire channel with NaN
            out[k] = np.full((H, W), np.nan, dtype=np.float64)
            continue

        valid_arr = np.array(valid_list, dtype=np.float64)
        med = median_numba(np.sort(valid_arr))
        std_ = std_numba(valid_arr)
        low = med - 3.0 * std_
        high = med + 3.0 * std_

        for r in range(H):
            for c in range(W):
                val = flat_rgb[k, r, c]
                if not np.isnan(val) and (low <= val <= high):
                    out[k, r, c] = med / val
                else:
                    out[k, r, c] = np.nan
    return out

@numba.njit
def dark_reduction_numba(image_arr, dark_arr):
    """
    Subtract dark frame from image, set <=0 to NaN, then fill with neighbors.
    image_arr, dark_arr: 2D arrays of same shape.
    """
    subtracted = image_arr - dark_arr
    rows, cols = subtracted.shape
    for r in range(rows):
        for c in range(cols):
            if subtracted[r, c] <= 0:
                subtracted[r, c] = np.nan
    # Replace NaNs
    corrected_subtracted = replace_nan_with_neighborhood_median(subtracted)
    return corrected_subtracted

@numba.njit
def dark_reduction_rgb_numba(image_arr, dark_arr):
    """
    Subtract dark from each channel, set <=0 to NaN, then fill with neighbors.
    image_arr, dark_arr: shape (3, rows, cols).
    """
    subtracted = image_arr - dark_arr
    C, R, W = subtracted.shape
    for k in range(C):
        for r in range(R):
            for c in range(W):
                if subtracted[k, r, c] <= 0:
                    subtracted[k, r, c] = np.nan
    # Replace NaNs per channel
    corrected_subtracted = replace_nan_with_neighborhood_median_rgb(subtracted)
    return corrected_subtracted

@numba.njit
def nan_multiply_numba(arr1, arr2):
    """
    Multiply two arrays of the same shape, treat any NaN in either as NaN in result.
    """
    rows, cols = arr1.shape
    out = np.empty((rows, cols), dtype=np.float64)
    for r in range(rows):
        for c in range(cols):
            a = arr1[r, c]
            b = arr2[r, c]
            if np.isnan(a) or np.isnan(b):
                out[r, c] = np.nan
            else:
                out[r, c] = a * b
    return out

# --------------------------------------------------------------------------------
#                  High-level Python functions (I/O + calls)
# --------------------------------------------------------------------------------

def fits_to_arr_gray(files):
    """
    Load list of B&W FITS images into a numpy array [N, height, width].
    This part can't meaningfully be numba-jitted (I/O bound).
    """
    pics = []
    for f in files:
        with fits.open(f) as hdul:
            array = hdul[0].data.astype(np.float64)
        pics.append(array)
    arr = np.array(pics)
    print(f"Converted {len(files)} B&W files to arrays.")
    return arr

def fits_to_arr_rgb_rgb(files):
    """
    Load list of 3D (RGB) FITS images into a numpy array [N, channels, height, width].
    """
    pics = []
    for f in files:
        with fits.open(f) as hdul:
            array = hdul[0].data.astype(np.float64)  # shape should be (3, H, W)
        pics.append(array)
    arr = np.array(pics)  # shape = (N, 3, H, W)
    print(f"Converted {len(files)} RGB files to RGB arrays.")
    return arr

def fits_to_arr_rgb_gray(files):
    """
    Load list of 3D (RGB) FITS images, then convert each one to grayscale [N, H, W].
    """
    def grayscale(arr):
        # Instead of np.einsum(), do a direct weighted sum for Numba-compat if needed.
        # But here it's just standard Python anyway:
        # arr shape (3, H, W)
        return 0.299 * arr[0] + 0.587 * arr[1] + 0.114 * arr[2]

    pics = []
    for f in files:
        with fits.open(f) as hdul:
            array = hdul[0].data.astype(np.float64)  # shape (3, H, W)
        gray = grayscale(array)
        pics.append(gray)
    arr = np.array(pics)  # shape = (N, H, W)
    print(f"Converted {len(files)} RGB files to Gray arrays.")
    return arr

def image_output(arr, name, path):
    """
    Write a 2D or 3D array to a FITS file.
    """
    output_file = path + name
    hdu = fits.PrimaryHDU(arr)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(output_file, overwrite=True)
    print(f"Image {name} saved at {path}.")

# --------------------------------------------------------------------------------
#                  Wrappers that call the Numba-accelerated routines
# --------------------------------------------------------------------------------

def master_dark_array(arr):
    """
    arr shape [N, H, W]. Pure python wrapper around numba core.
    """
    start_time = time.time()
    final_arr = master_dark_array_numba(arr)
    elapsed_time = time.time() - start_time
    print(f"Master dark array created. Elapsed time: {elapsed_time:.2f}s")
    return final_arr

def master_flat_array(arr):
    """
    arr shape [N, H, W].
    """
    start_time = time.time()
    final_arr = master_flat_array_numba(arr)
    elapsed_time = time.time() - start_time
    print(f"Master flat array created. Elapsed time: {elapsed_time:.2f}s")
    return final_arr

def master_flat_array_rgb(arr):
    """
    arr shape [N, 3, H, W].
    """
    start_time = time.time()
    final_arr = master_flat_array_rgb_numba(arr)
    elapsed_time = time.time() - start_time
    print(f"Master flat array for RGB created. Elapsed time: {elapsed_time:.2f}s")
    return final_arr

def flat_weights(arr):
    """
    2D array => 2D weights array.
    """
    start_time = time.time()
    final_arr = flat_weights_numba(arr)
    elapsed_time = time.time() - start_time
    print(f"Flat weights array computed. Elapsed time: {elapsed_time:.2f}s")
    return final_arr

def flat_weights_rgb(arr):
    """
    3D array => 3D weights array (channels, H, W).
    """
    start_time = time.time()
    final_arr = flat_weights_rgb_numba(arr)
    elapsed_time = time.time() - start_time
    print(f"Flat weights array for RGB computed. Elapsed time: {elapsed_time:.2f}s")
    return final_arr

def dark_reduction(image_arr, dark_arr):
    """
    Subtract the dark frame, set <=0 to NaN, fill with neighbor median.
    """
    start_time = time.time()
    corrected = dark_reduction_numba(image_arr, dark_arr)
    elapsed_time = time.time() - start_time
    print(f"Dark reduction (gray) done. Elapsed time: {elapsed_time:.2f}s")
    return corrected

def dark_reduction_rgb(image_arr, dark_arr):
    """
    Subtract the dark frame for each channel, set <=0 to NaN, fill with neighbor median.
    """
    start_time = time.time()
    corrected = dark_reduction_rgb_numba(image_arr, dark_arr)
    elapsed_time = time.time() - start_time
    print(f"Dark reduction (RGB) done. Elapsed time: {elapsed_time:.2f}s")
    return corrected

def nan_multiply(arr1, arr2):
    """
    Multiply two arrays, setting result to NaN if either is NaN.
    """
    start_time = time.time()
    result = nan_multiply_numba(arr1, arr2)
    elapsed_time = time.time() - start_time
    print(f"NaN-multiply done. Elapsed time: {elapsed_time:.2f}s")
    return result


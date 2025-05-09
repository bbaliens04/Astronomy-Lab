import numpy as np
import numba
from astropy.io import fits
import sys
import time
from scipy.ndimage import center_of_mass as com

@numba.njit
def median_numba(arr):
    n = arr.size
    sorted_arr = np.sort(arr)
    if n % 2 == 1:
        return sorted_arr[n // 2]
    else:
        return 0.5 * (sorted_arr[n // 2 - 1] + sorted_arr[n // 2])

@numba.njit
def replace_nan_with_neighborhood_median(arr):
    arr = arr.copy()
    rows, cols = arr.shape
    for i in range(rows):
        for j in range(cols):
            if np.isnan(arr[i, j]):
                values = np.empty(8, dtype=arr.dtype)
                count = 0
                for ii in range(max(0, i - 1), min(rows, i + 2)):
                    for jj in range(max(0, j - 1), min(cols, j + 2)):
                        if (ii != i or jj != j) and not np.isnan(arr[ii, jj]):
                            values[count] = arr[ii, jj]
                            count += 1
                if count > 0:
                    arr[i, j] = median_numba(values[:count])
    return arr

@numba.njit
def replace_nan_with_neighborhood_median_rgb(arr):
    arr = arr.copy()
    channels, rows, cols = arr.shape
    for c in range(channels):
        for i in range(rows):
            for j in range(cols):
                if np.isnan(arr[c, i, j]):
                    values = np.empty(8, dtype=arr.dtype)
                    count = 0
                    for ii in range(max(0, i - 1), min(rows, i + 2)):
                        for jj in range(max(0, j - 1), min(cols, j + 2)):
                            if (ii != i or jj != j) and not np.isnan(arr[c, ii, jj]):
                                values[count] = arr[c, ii, jj]
                                count += 1
                    if count > 0:
                        arr[c, i, j] = median_numba(values[:count])
    return arr

def fits_to_arr_gray(file):
    pics = []
    for i in file:
        with fits.open(i) as hdul:
            array = hdul[0].data.astype(np.float64)
        pics.append(array)
    
    pics = np.array(pics)
    print(f"Converted {len(file)} B&W files to arrays.")
    
    return pics


def fits_to_arr_rgb_rgb(file):
    pics = []
    for i in file:
        with fits.open(i) as hdul:
            array = hdul[0].data.astype(np.float64)
        
        pics.append(array)
    
    pics = np.array(pics)
    print(f"Converted {len(file)} RGB files to RGB arrays.")
    
    return pics


def fits_to_arr_rgb_gray(file):
    def grayscale(arr):
        rgb_index = np.array([0.299, 0.587, 0.114])
        arr = np.array(arr, dtype=np.float64)
        grayscale_array = np.einsum('i,ijk->jk', rgb_index, arr)
        
        return grayscale_array

    pics = []
    for i in file:
        with fits.open(i) as hdul:
            array = hdul[0].data.astype(np.float64)
        
        grayscaled_arr = grayscale(array)
        pics.append(grayscaled_arr)
    
    pics = np.array(pics)
    print(f"Converted {len(file)} RGB files to Gray arrays.")
    
    return pics

def master_dark_array(arr):
    arr = np.array(arr, dtype=np.float64)
    final_arr = np.zeros((arr.shape[1], arr.shape[2]), dtype=np.float64)
    total_process = arr.shape[1] * arr.shape[2]
    x = 0
    start_time = time.time()
    for j in range(arr.shape[1]):
        for i in range(arr.shape[2]):
            pixel_values = arr[:, j, i]
            clip = 3 * np.std(pixel_values)
            median = np.median(pixel_values)
            filtered_values = pixel_values[(pixel_values > median - clip) & (pixel_values < median + clip)]
            if filtered_values.size > 0:
                final_value = np.ceil(np.nanmedian(filtered_values)).astype(int)
            else:
                final_value = np.nan  # Set NaN for problematic pixels
            final_arr[j][i] = final_value
            x += 1
            if x % 100000 == 0:
                elapsed_time = time.time() - start_time
                progress = x / total_process
                remaining_time = elapsed_time / progress * (1 - progress)
                progress_percent = round(progress * 100, ndigits=3)
                sys.stdout.write(f"\rProgress: {progress_percent}% - Time remaining: {int(remaining_time // 60)}m {int(remaining_time % 60)}s")
                sys.stdout.flush()
    # Replace NaN values with the median of their neighbors
    final_arr = replace_nan_with_neighborhood_median(final_arr)
    print("\nMaster dark array created.")
    return final_arr


def image_output(arr, name, path):
    output_file = path + name
    hdu = fits.PrimaryHDU(arr)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(output_file, overwrite=True)
    print(f"Image {name} saved at {path}.")

def master_flat_array(arr):
    arr = np.array(arr, dtype=np.float64)
    final_arr = np.zeros((arr.shape[1], arr.shape[2]), dtype=np.float64)
    total_process = arr.shape[1] * arr.shape[2]
    x = 0
    start_time = time.time()
    for j in range(arr.shape[1]):
        for i in range(arr.shape[2]):
            pixel_values = arr[:, j, i]
            clip = 3 * np.std(pixel_values)
            median = np.median(pixel_values)
            filtered_values = pixel_values[(pixel_values > median - clip) & (pixel_values < median + clip)]
            if filtered_values.size > 0:
                final_value = np.ceil(np.nanmedian(filtered_values)).astype(int)
            else:
                final_value = np.nan  # Set NaN for problematic pixels
            final_arr[j][i] = final_value
            x += 1
            if x % 100000 == 0:
                elapsed_time = time.time() - start_time
                progress = x / total_process
                remaining_time = elapsed_time / progress * (1 - progress)
                progress_percent = round(progress * 100, ndigits=3)
                sys.stdout.write(f"\rProgress: {progress_percent}% - Time remaining: {int(remaining_time // 60)}m {int(remaining_time % 60)}s")
                sys.stdout.flush()
    # Replace NaN values with the median of their neighbors
    final_arr = replace_nan_with_neighborhood_median(final_arr)
    print("\nMaster flat array created.")
    return final_arr

def master_flat_array_rgb(arr):
    final_arr_rgb = []
    rgb = ["R", "G", "B"]
    for k in range(3):  # Loop through each RGB channel
        final_arr = np.zeros((arr.shape[2], arr.shape[3]), dtype=np.float64)
        total_process = arr.shape[2] * arr.shape[3]
        x = 0
        start_time = time.time()
        for j in range(arr.shape[2]):  # Loop through height
            for i in range(arr.shape[3]):  # Loop through width
                pixel_values = arr[:, k, j, i]
                clip = 3 * np.std(pixel_values)
                median = np.median(pixel_values)
                filtered_values = pixel_values[(pixel_values > median - clip) & (pixel_values < median + clip)]
                if filtered_values.size > 0:
                    final_value = np.ceil(np.nanmedian(filtered_values)).astype(int)
                else:
                    final_value = np.nan  # Set NaN for problematic pixels
                final_arr[j][i] = final_value
                x += 1
                if x % 100000 == 0:
                    elapsed_time = time.time() - start_time
                    progress = x / total_process
                    remaining_time = elapsed_time / progress * (1 - progress)
                    progress_percent = round(progress * 100, ndigits=3)
                    sys.stdout.write(f"\rProgress for {rgb[k]}: {progress_percent}% - Time remaining: {int(remaining_time // 60)}m {int(remaining_time % 60)}s")
                    sys.stdout.flush()
        # Replace NaN values with the median of their neighbors for each channel
        final_arr = replace_nan_with_neighborhood_median(final_arr)
        final_arr_rgb.append(final_arr)
    final_arr_rgb = np.array(final_arr_rgb)
    print("\nMaster flat array for RGB created.")
    return final_arr_rgb

def flat_weights(arr):
    arr_flat = arr.flatten()
    med = np.median(arr_flat)
    sigma_clip = 3 * np.std(arr_flat)
    arr_with_nan = np.where((arr >= med - sigma_clip) & (arr <= med + sigma_clip), arr, np.nan)
    final_arr = np.zeros_like(arr_with_nan)
    final_arr[~np.isnan(arr_with_nan)] = med / arr_with_nan[~np.isnan(arr_with_nan)]
    print("Flat weights array computed.")
    return final_arr

def flat_weights_rgb(arr):
    final_arr_rgb = np.zeros_like(arr)
    for k in range(3):  # Loop through each RGB channel
        arr_flat = arr[k].flatten()
        med = np.median(arr_flat)
        sigma_clip = 3 * np.std(arr_flat)
        arr_with_nan = np.where((arr[k] >= med - sigma_clip) & (arr[k] <= med + sigma_clip), arr[k], np.nan)
        final_arr = np.zeros_like(arr_with_nan)
        final_arr[~np.isnan(arr_with_nan)] = med / arr_with_nan[~np.isnan(arr_with_nan)]
        final_arr_rgb[k] = final_arr
    print("Flat weights array for RGB computed.")
    return final_arr_rgb

def dark_reduction(image_arr, dark_arr):
    # Subtract the dark frame
    subtracted = image_arr - dark_arr
    # Identify negative or zero values
    negative_mask = subtracted <= 0
    # Set negative or zero values to NaN
    subtracted = subtracted.copy()
    subtracted[negative_mask] = np.nan
    # Replace NaNs with the median of neighboring pixels
    corrected_subtracted = replace_nan_with_neighborhood_median(subtracted)
    return corrected_subtracted

def dark_reduction_rgb(image_arr, dark_arr):
    # Subtract the dark frame
    subtracted = image_arr - dark_arr
    # Identify negative or zero values
    negative_mask = subtracted <= 0
    # Set negative or zero values to NaN
    subtracted = subtracted.copy()
    subtracted[negative_mask] = np.nan
    # Replace NaNs with the median of neighboring pixels for each channel
    corrected_subtracted = replace_nan_with_neighborhood_median_rgb(subtracted)
    return corrected_subtracted

def nan_multiply(arr1, arr2):
    arr1 = np.array(arr1, dtype=np.float64)
    arr2 = np.array(arr2, dtype=np.float64)
    nan_mask1 = np.isnan(arr1)
    nan_mask2 = np.isnan(arr2)
    arr1_no_nan = np.nan_to_num(arr1, nan=1.0)
    arr2_no_nan = np.nan_to_num(arr2, nan=1.0)
    result = arr1_no_nan * arr2_no_nan
    result[nan_mask1 | nan_mask2] = np.nan
    return result
'''
def center_of_mass(pic, star, num):
    init_cutout = pic []
'''

import astropy.io as ap
from astropy.io import fits
import numpy as np
from astropy.utils.data import get_pkg_data_filename as gpdf
import sys
sys.path.append('/Users/bardiya/Desktop/Astronomy_lab/Modules')
import Project_Mods as mod
import matplotlib.pyplot as plt
import scipy as sp
import copy
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D

numbers = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15"]
address = "/Users/bardiya/Desktop/Astronomy_lab/Opacity/Data/fits"
flat_adds = [f"{address}/flat_canon_d4000_iso400_130s_opacity_experiment_bardia_alian_reza_keyvanfar_ali_jorooghi_000{i}.fits" for i in numbers[:14]]
dark_flat_adds = [f"{address}/dark_flat_canon_d4000_iso400_130s_opacity_experiment_bardia_alian_reza_keyvanfar_ali_jorooghi_000{i}.fits" for i in numbers[:13]]
data_adds = [f"{address}/data_canon_d4000_iso400_10s_opacity_experiment_bardia_alian_reza_keyvanfar_ali_jorooghi_000{i}.fits" for i in numbers[:8]]
dark_data_adds = [f"{address}/dark_data_canon_d4000_iso400_10s_opacity_experiment_bardia_alian_reza_keyvanfar_ali_jorooghi_000{i}.fits" for i in numbers[:14]]

flat_arrs = mod.fits_to_arr_rgb(flat_adds)
dark_flat_arrs = mod.fits_to_arr_rgb(dark_flat_adds)
data_arrs = mod.fits_to_arr_rgb(data_adds)
dark_data_arrs = mod.fits_to_arr_rgb(dark_data_adds)

assert all(arr.size > 0 for arr in flat_arrs), "Some flat field arrays are empty."
assert all(arr.size > 0 for arr in dark_flat_arrs), "Some dark flat field arrays are empty."
assert all(arr.size > 0 for arr in data_arrs), "Some data arrays are empty."
assert all(arr.size > 0 for arr in dark_data_arrs), "Some dark data arrays are empty."

print("Loaded all arrays successfully.")

master_dark_flat_arr = mod.master_dark_array(dark_flat_arrs)
mod.image_output(master_dark_flat_arr, "Master_Dark_Flat.fits", "/Users/bardiya/Desktop/Astronomy_lab/Opacity/Final_data/")
print("Created master dark flat array.")

corrected_flat_arrs = [i - master_dark_flat_arr for i in flat_arrs]
print("Corrected flat arrays.")

master_flat_arr = mod.master_flat_array(corrected_flat_arrs)
mod.image_output(master_flat_arr, "Master_flat.fits", "/Users/bardiya/Desktop/Astronomy_lab/Opacity/Final_data/")
print("Created master flat array.")

master_dark_data_arr = mod.master_dark_array(dark_data_arrs)
mod.image_output(master_dark_data_arr, "Master_Dark_Data.fits", "/Users/bardiya/Desktop/Astronomy_lab/Opacity/Final_data/")
print("Created master dark data array.")

corrected_data_arrs = [i - master_dark_data_arr for i in data_arrs]
print("Corrected data arrays.")

flat_weights_arr = mod.flat_weights(master_flat_arr)
print("Computed flat weights array.")

def nan_multiply(arr1, arr2):
    # Ensure the arrays are numpy arrays
    arr1 = np.array(arr1, dtype=np.float64)
    arr2 = np.array(arr2, dtype=np.float64)
    
    # Create masks to identify NaNs
    nan_mask1 = np.isnan(arr1)
    nan_mask2 = np.isnan(arr2)
    
    # Replace NaNs with 1s in both arrays
    arr1_no_nan = np.nan_to_num(arr1, nan=1.0)
    arr2_no_nan = np.nan_to_num(arr2, nan=1.0)
    
    # Perform element-wise multiplication
    result = arr1_no_nan * arr2_no_nan
    
    # Set positions back to NaN where either original array had NaN
    result[nan_mask1 | nan_mask2] = np.nan
    
    return result

for i in range(len(corrected_data_arrs)):
    corrected_data_arrs[i] = nan_multiply(flat_weights_arr, corrected_data_arrs[i])

j = 1
for i in corrected_data_arrs:
    print(f"Processing corrected data array {j}")
    mod.image_output(i, f"corrected_data_{j}.fits", "/Users/bardiya/Desktop/Astronomy_lab/Opacity/Final_data/corrected_datas/")
    j += 1
print("All corrected data arrays processed and saved.")


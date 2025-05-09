mimport astropy.io as ap
from astropy.io import fits
import numpy as np
from astropy.utils.data import get_pkg_data_filename as gpdf
import sys
sys.path.append('/Users/bardiya/Desktop/Astronomy_lab/Modules')
import Project_Mods as mod
import matplotlib.pyplot as plt
import copy
import scipy.optimize as opt
from scipy import stats

with fits.open('/Users/bardiya/Desktop/Astronomy_lab/Sun_Experiment/Final_Data/Sun_Image_with_dark_correction.fits') as hdul:
    sun_array = hdul[0].data

height = sun_array.shape[0]
width = sun_array.shape[1]
one_arr = np.zeros((height, width))
length_hor = np.zeros((height))
for i in range(height):
    for j in range(width):
        if sun_array[i][j] > 1000:
            one_arr[i][j] = 1
            length_hor[i] += 1

diameter_hor = max((length_hor))

list_of_maxes_hor = []
for i in range(len(length_hor)):
    if length_hor[i] == diameter_hor:
        list_of_maxes_hor.append(i)



center_height = np.round(np.median(list_of_maxes_hor)).astype(np.int64)

length_ver = np.zeros((width))
for i in range(width):
    for j in range(height):
        if sun_array[j][i] > 1000:
            length_ver[i] += 1

diameter_ver = max((length_ver))

list_of_maxes_ver = []
for i in range(len(length_ver)):
    if length_ver[i] == diameter_ver:
        list_of_maxes_ver.append(i)

center_width = np.round(np.median(list_of_maxes_ver)).astype(np.int64)

diameter = (diameter_hor + diameter_ver) / 2
radius = diameter / 2

print(f'''Center's height is: {center_height}
Center's width is: {center_width}''')
print(f'The raduis is: {radius} pixels')

new_sun = copy.deepcopy(sun_array)

for i in range(8):
    for j in range(51):
        new_sun[center_height + j][center_width + i] = 0.0
        new_sun[center_height + j][center_width - i] = 0.0
        new_sun[center_height - j][center_width + i] = 0.0
        new_sun[center_height - j][center_width - i] = 0.0

for i in range(8):
    for j in range(51):
        new_sun[center_height + i][center_width + j] = 0.0
        new_sun[center_height + i][center_width - j] = 0.0
        new_sun[center_height - i][center_width + j] = 0.0
        new_sun[center_height - i][center_width - j] = 0.0

def intensity(array, bin):
    padded_array = np.pad(array, bin, mode='reflect')

    new_arr = np.zeros_like(array)

    i = center_height
    for j in range(array.shape[1]):
        neighborhood = padded_array[i: i + 2 * bin + 1, j: j + 2 * bin + 1]
        new_arr[i, j] = np.median(neighborhood)

    # Intensity = (count * planck_constant) / (coefficient * exposure_time * pixel_surface)
    coefficient = 0.3
    sensor_surface = 332.27 * 10 ** -9
    pixel_surface = sensor_surface / (array.shape[0] * array.shape[1])
    planck_constant = 6.6261 * 10 ** -34
    exposure_time = 1 / 30
    intensity_array = planck_constant / (coefficient * exposure_time * pixel_surface) * new_arr
    
    return intensity_array

def normalize(array):
    norm_coefficient = 1/max(array)
    normalized_arr = norm_coefficient * array
    return normalized_arr
    
def miu(array):
    mius = np.zeros(array.shape)
    for i in range(len(array)):
        if radius ** 2 - (i - center_width) ** 2 < 0:
            mius[i] = np.nan
        elif radius ** 2 - (i - center_width) ** 2 > 0 and (i - center_width) < 0 :
            mius[i] = ((radius ** 2 - (i - center_width) ** 2) / radius ** 2) ** (1 / 2)
        
        else:
            mius[i] = -((radius ** 2 - (i - center_width) ** 2) / radius ** 2) ** (1 / 2) + 2
    
    return mius

mod.image_output(new_sun, "Sun_with_marked_center.fits", "/Users/bardiya/Desktop/Astronomy_lab/Sun_Experiment/Final_Data/")

plt.figure()
plt.imshow(new_sun[::-1], interpolation ='nearest', cmap = 'gray')

norm_arr_no_bin = normalize(intensity(sun_array, 0)[center_height])
norm_arr_3x3_bin = normalize(intensity(sun_array, 1)[center_height])
norm_arr_5x5_bin = normalize(intensity(sun_array, 2)[center_height])
norm_arr_7x7_bin = normalize(intensity(sun_array, 3)[center_height])

def plotting(i, name):
    names = [norm_arr_no_bin, norm_arr_3x3_bin, norm_arr_5x5_bin, norm_arr_7x7_bin]
    plt.figure()
    plt.bar(range(sun_array.shape[1]), names[i])
    plt.xlabel("No. of pixel")
    plt.ylabel("Intensity per ferquency normalized")
    plt.title(f"Normalized Intensity Diagram of Pixels ({name} Convolution)")
    
    miu_array = miu(sun_array[center_height])
    plt.figure()
    plt.plot(miu_array, names[i], linestyle="-")
    plt.xlabel("Miu")
    plt.ylabel("Intensity per frequency normalized")
    plt.title(f"Scatter Plot of Normalized Intensity vs Miu ({name} Convolution)")
    miu_values = np.linspace(0, 2, 1000)
    line1 = (2 + 3 * miu_values) / 5
    line2 = -(3 * (miu_values)) / 5 + 8 / 5
    line1 = np.ma.masked_where((miu_values < 0) | (miu_values >= 1), line1)
    line2 = np.ma.masked_where((miu_values <= 1) | (miu_values >= 2), line2)
    plt.plot(miu_values, line1, color='red', label = "Eddington's model")
    plt.plot(miu_values, line2, color='red')
    
    valid_idx_0_1 = np.where((miu_array >= 0) & (miu_array < 1))[0]
    valid_idx_1_2 = np.where((miu_array >= 1) & (miu_array <= 2))[0]

    if len(valid_idx_0_1) > 1:
        coeffs_0_1 = np.polyfit(miu_array[valid_idx_0_1], names[i][valid_idx_0_1], 1)
        line_0_1 = np.poly1d(coeffs_0_1)
        plt.plot(miu_array[valid_idx_0_1], line_0_1(miu_array[valid_idx_0_1]), color='green', label=f'Line Fit 0-1, Slope: {coeffs_0_1[0]:.2f}')

    if len(valid_idx_1_2) > 1:
        coeffs_1_2 = np.polyfit(miu_array[valid_idx_1_2], names[i][valid_idx_1_2], 1)
        line_1_2 = np.poly1d(coeffs_1_2)
        plt.plot(miu_array[valid_idx_1_2], line_1_2(miu_array[valid_idx_1_2]), color='black', label=f'Line Fit 1-2, Slope: {coeffs_1_2[0]:.2f}')

    plt.legend()

    profile = names[i][:center_width + 1]
    profile = profile[::-1]
    profile_rad = range(len(profile)) / radius

    plt.figure()
    plt.plot(profile_rad, profile, linestyle="-")
    plt.xlabel("r / R_sun")
    plt.ylabel("Intensity per frequency normalized")
    plt.title(f"Intensity Radial Profile ({name} Convolution)")

plotting(0, "No")
plotting(1, "3x3")
plotting(2, "5x5")
plotting(3, "7x7")

plt.show()

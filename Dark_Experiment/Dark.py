import astropy.io as ap
from astropy.io import fits
import numpy as np
import matplotlib as mplib
import matplotlib.pyplot as plt
from astropy.utils.data import get_pkg_data_filename as gpdf
from scipy.stats import linregress

file_path = "./Final_Datas/"
num = ["01","02","03","04","05","06","07","08","09","10"]
data_14000sec = [f"./fits/dark_nikon_d_90_iso_200_14000sec_14021208_Bardia_Alian_Bardia_hassanpour_000{i}.fits" for i in num]
data_11000sec = [f"./fits/dark_nikon_d_90_iso_200_11000sec_14021208_Bardia_Alian_Bardia_hassanpour_000{i}.fits" for i in num]
data_1100sec = [f"./fits/dark_nikon_d_90_iso_200_1100sec_14021208_Bardia_Alian_Bardia_hassanpour_000{i}.fits" for i in num]
data_110sec = [f"./fits/dark_nikon_d_90_iso_200_110sec_14021208_Bardia_Alian_Bardia_hassanpour_000{i}.fits" for i in num]
data_1sec = [f"./fits/dark_nikon_d_90_iso_200_1sec_14021208_Bardia_Alian_Bardia_hassanpour_000{i}.fits" for i in num]
data_10sec = [f"./fits/dark_nikon_d_90_iso_200_10sec_14021208_Bardia_Alian_Bardia_hassanpour_000{i}.fits" for i in num]
data_30sec = [f"./fits/dark_nikon_d_90_iso_200_30sec_14021208_Bardia_Alian_Bardia_hassanpour_000{i}.fits" for i in num]

def fits_to_arr(file):

    def grayscale(arr):
        rgb_index=np.array([0.299, 0.587, 0.114])
        arr = np.array(arr, dtype=np.float64)
        grayscale_array = np.einsum('i,ijk->jk', rgb_index, arr)
        
        return grayscale_array

    pics = []
    for i in file:
        data_name = gpdf(i)
        rgb_array = fits.getdata(data_name).astype(np.float64)
        grayscaled_array = grayscale(rgb_array)
        pics.append(grayscaled_array)
    
    pics = np.array(pics)
    return pics

def master_dark_array(arr):
    arr = np.array(arr, dtype=np.float64)
    final_arr = np.zeros((arr.shape[1], arr.shape[2]), dtype=np.float64)
    total_process = arr.shape[1] * arr.shape[2]
    x = 0
    for j in range(arr.shape[1]):
        for i in range(arr.shape[2]):
            pixel_values = arr[:, j, i]
            clip = 3 * np.std(pixel_values)
            median = np.median(pixel_values)
            filtered_values = pixel_values[(pixel_values > median - clip) & (pixel_values < median + clip)]
            if filtered_values.size > 0:
                final_value = np.ceil(np.nanmedian(filtered_values)).astype(int)
            else:
                final_value = np.nan
            final_arr[j][i] = final_value
            
            x += 1
            
            if x % 1000000 == 0:
                print(f"Progress: {round(x / total_process * 100, ndigits = 3)}%")
            
    return final_arr

def master_dark_image(arr, name, path):
    output_file = path + name
    hdu = fits.PrimaryHDU(arr)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(output_file, overwrite=True)

def histogram(arr, name, show=False):
    plt.figure()
    flat_arr = arr.flatten()
    plt.hist(flat_arr, bins=1000, color='blue', alpha=0.7)
    plt.title(f'Histogram of {name} dark')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.grid(True)
    if show:
        plt.show()

def mean_count(arr, count, name):
    total_pixels = arr.shape[0] * arr.shape[1]
    mean = count / total_pixels
    
    print(f"Mean count in every pixel of {name} image is: {mean} count per pixel.")
 
data_14000sec_master = master_dark_array(fits_to_arr(data_14000sec))
#data_11000sec_master = master_dark_array(fits_to_arr(data_11000sec))
#data_1100sec_master = master_dark_array(fits_to_arr(data_1100sec))
#data_110sec_master = master_dark_array(fits_to_arr(data_110sec))
#data_1sec_master = master_dark_array(fits_to_arr(data_1sec))
#data_10sec_master = master_dark_array(fits_to_arr(data_10sec))
#data_30sec_master = master_dark_array(fits_to_arr(data_30sec))

#count_14000sec = np.nansum(data_14000sec_master)
#count_11000sec = np.nansum(data_11000sec_master)
#count_1100sec = np.nansum(data_1100sec_master)
#count_110sec = np.nansum(data_110sec_master)
#count_1sec = np.nansum(data_1sec_master)
#count_10sec = np.nansum(data_10sec_master)
#count_30sec = np.nansum(data_30sec_master)

#master_dark_image(data_14000sec_master,"master_dark_nikon_d_90_iso_200_14000sec_14021208_Bardia_Alian_Bardia_hassanpour.fits" , file_path)
#master_dark_image(data_11000sec_master,"master_dark_nikon_d_90_iso_200_11000sec_14021208_Bardia_Alian_Bardia_hassanpour.fits" , file_path)
#master_dark_image(data_1100sec_master,"master_dark_nikon_d_90_iso_200_1100sec_14021208_Bardia_Alian_Bardia_hassanpour.fits" , file_path)
#master_dark_image(data_110sec_master,"master_dark_nikon_d_90_iso_200_110sec_14021208_Bardia_Alian_Bardia_hassanpour.fits" , file_path)
#master_dark_image(data_1sec_master,"master_dark_nikon_d_90_iso_200_1sec_14021208_Bardia_Alian_Bardia_hassanpour.fits" , file_path)
#master_dark_image(data_10sec_master,"master_dark_nikon_d_90_iso_200_10sec_14021208_Bardia_Alian_Bardia_hassanpour.fits" , file_path)
#master_dark_image(data_30sec_master,"master_dark_nikon_d_90_iso_200_30sec_14021208_Bardia_Alian_Bardia_hassanpour.fits" , file_path)

#print(f"Total count of 1/4000sec is: {count_14000sec}")
#print(f"Total count of 1/1000sec is: {count_11000sec}")
#print(f"Total count of 1/100sec is: {count_1100sec}")
#print(f"Total count of 1/10sec is: {count_110sec}")
#print(f"Total count of 1sec is: {count_1sec}")
#print(f"Total count of 10sec is: {count_10sec}")
#print(f"Total count of 30sec is: {count_30sec}")

#mean_count(data_14000sec_master, count_14000sec, "1/4000sec")
#mean_count(data_11000sec_master, count_11000sec, "1/1000sec")
#mean_count(data_1100sec_master, count_1100sec, "1/100sec")
#mean_count(data_110sec_master, count_110sec, "1/10sec")
#mean_count(data_1sec_master, count_1sec, "1sec")
#mean_count(data_10sec_master, count_10sec, "10sec")
#mean_count(data_30sec_master, count_30sec, "30sec")

histogram(data_14000sec_master, 'dark 1/4000 sec', show=False)
#histogram(data_11000sec_master, 'dark 1/1000 sec', show=False)
#histogram(data_1100sec_master, 'dark 1/100 sec', show=False)
#histogram(data_110sec_master, 'dark 1/10 sec', show=False)
#histogram(data_1sec_master, 'dark 1 sec', show=False)
#histogram(data_10sec_master, 'dark 10 sec', show=False)
#histogram(data_30sec_master, 'dark 30 sec', show=False)

exposure_times = np.array([1/4000, 1/1000, 1/100, 1/10, 1, 10, 30])
total_counts = np.array([count_14000sec, count_11000sec, count_1100sec, count_110sec, count_1sec, count_10sec, count_30sec])

slope, intercept = np.polyfit(exposure_times, total_counts, 1)
regression_line = slope * exposure_times + intercept

#correlation_matrix = np.corrcoef(total_counts, exposure_times * slope + intercept)
#correlation_xy = correlation_matrix[0,1]
#r_squared = correlation_xy**2

#regression_results = linregress(exposure_times, total_counts)

#print(f"Linear Regression Fit: y = {slope:.2f}x + {intercept:.2f}")
#print(f"R-squared: {r_squared:.4f}")
#print(f"Slope standard error: {regression_results.stderr:.4f}")
#print(f"Intercept standard error: {regression_results.intercept_stderr:.4f}")

plt.figure(figsize=(10, 6))
plt.scatter(exposure_times, total_counts, color='blue', label='Data points')
plt.plot(exposure_times, regression_line, color='red', label=f'Fit line: y = {slope:.2f}x + {intercept:.2f}')
#plt.xlabel('Exposure Time (s)')
#plt.ylabel('Total Counts')
#plt.title('Total Counts vs. Exposure Time')
#plt.legend()
#plt.grid(True)

plt.show()

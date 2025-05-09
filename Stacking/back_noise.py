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
from scipy.stats import sigmaclip

def snr(pic_arr ,point):

    x = []
    y = []
    for rad in range(2, 31):
        N = 0
        coords = []
        coords_sky = []
        for i in range(101):
            for j in range(101):
                if (i - 51) ** 2 + (j - 51) ** 2 < rad ** 2:
                    coords.append(point - np.array([51,51]) + np.array([i, j]))
                    N += 1
        summ = 0
        for k in coords:
            summ += pic_arr[k[0], k[1]]

            
        for i in range(101):
            for j in range(101):
                if (rad * 2) ** 2 < (i - 51) ** 2 + (j - 51) ** 2 < (rad * 2.5) ** 2:
                    coords_sky.append(point - np.array([51,51]) + np.array([i, j]))
        
        data_sky = []
        for k in coords_sky:
            data_sky.append(pic_arr[k[0], k[1]])
            
        npixel = N
        noise = np.sqrt(summ + npixel)
        sky = np.median(data_sky)
        snr = (summ - sky * npixel) / noise
        sig = (summ - sky * npixel)
        x.append(rad)
        y.append(snr)
    
    index_max = y.index(max(y))
    
    plt.figure()
    plt.plot(x,y)
    plt.xlabel('radius')
    plt.ylabel('SNR')
    return x[index_max], max(y)

image_filepath = ['/Users/bardiya/Desktop/Astronomy_lab/Stacking/Final_data/Stacked_image.fits']
stars_filepath = [f'/Users/bardiya/Desktop/Astronomy_lab/Stacking/Final_data/Selected_Images/star_1_{i}.fits' for i in range(1, 18)]

bad_stars = []
bad_stars.append(stars_filepath.pop(7))
bad_stars.append(stars_filepath.pop(10))

coords = []
for i in stars_filepath:
    with fits.open(i) as hdul:
                coords.append(hdul[1].data.astype(np.int64)[0])

image_arr = mod.fits_to_arr_gray(image_filepath)[0]

for i in range(len(coords)):
    coords[i] = np.array([coords[i][1], coords[i][0]])

plt.figure()
plt.imshow(image_arr, interpolation = 'nearest', origin = 'lower')
for i in coords:
    plt.plot(i[1], i[0], 'o')

snrs = []

for i in coords:
    snrs.append(snr(image_arr, i))

radiuses = []

for i in snrs:
    radiuses.append(i[0])

radiuses = np.array(radiuses)

clipped_radiuses_data = sigmaclip(radiuses, low=3, high=3)[0]

radius = np.median(clipped_radiuses_data)

def noise(pic_arr, rad, point):
    N = 0
    for i in range(101):
        for j in range(101):
            if (2*rad)**2 <= (i-51)**2 + (j-51)**2 <= (3*rad)**2:
                coords.append(point - np.array([51,51]) + np.array([i, j]))
                N += 1
    
    values = np.zeros([N])
    for k in range(N):
        values[k] = pic_arr[coords[k][0], coords[k][1]]
    
    clipped_values_data = sigmaclip(values, low=3, high=3)[0]
    background_noise = np.median(clipped_values_data)
    
    return background_noise

back_noises = []

for i in range(len(coords)):
    back_noises.append(noise(image_arr, radiuses[i], coords[i]))
    
clipped_back_noises_data = sigmaclip(back_noises, low=3, high=3)[0]
background_noise = np.median(clipped_back_noises_data).astype(np.int64)
background = np.full([image_arr.shape[0], image_arr.shape[1]], background_noise)

cleaned_image = image_arr - background

cleaned_image_with_no_negative_value = np.where(cleaned_image < 0, 0, cleaned_image)

plt.imshow(cleaned_image_with_no_negative_value, interpolation = 'nearest', origin = 'lower')

mod.image_output(cleaned_image_with_no_negative_value, 'Cleaned_image.fits', '/Users/bardiya/Desktop/Astronomy_lab/Stacking/Final_data/')

plt.show()

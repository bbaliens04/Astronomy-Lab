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
from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation, label, find_objects, center_of_mass

data_adds = ["/Users/bardiya/Desktop/Astronomy_lab/PSF/Data/light-g_20C-2023_10_10-exp00.04.00.000-1x1_High_1.fit", "/Users/bardiya/Desktop/Astronomy_lab/PSF/Data/light_r_20C-2023_10_10-exp00.02.00.000-1x1_High_2.fit", "/Users/bardiya/Desktop/Astronomy_lab/PSF/Data/light-r_20C-2023_10_10-exp00.03.00.000-1x1_High_3.fit"]

for j in range(len(data_adds)):


    hdul = fits.open(data_adds[j])
    data = hdul[0].data

    smoothed_data = gaussian_filter(data, sigma=2)
    threshold = 1000
    binary_mask = (smoothed_data > threshold)
    binary_mask = binary_erosion(binary_mask, iterations=1)
    binary_mask = binary_dilation(binary_mask, iterations=1)

    labeled_stars, num_stars = label(binary_mask)
    regions = find_objects(labeled_stars)

    indicator = 0
    for i, region in enumerate(regions):
        y_slice, x_slice = region
        center_y, center_x = center_of_mass(data[y_slice, x_slice])
        center_x += x_slice.start
        center_y += y_slice.start


        y_min = int(center_y) - 50
        y_max = int(center_y) + 50
        x_min = int(center_x) - 50
        x_max = int(center_x) + 50

        cutout_data = data[y_min:y_max + 1, x_min:x_max + 1]

        primary_hdu = fits.PrimaryHDU(data=cutout_data)

        coord_data = np.zeros((1, 2))
        coord_data[0, 0] = center_x
        coord_data[0, 1] = center_y
        coord_header = fits.Header()
        coord_header['COORDSYS'] = ('PIXEL', 'Coordinate system is pixel indices')
        coords_hdu = fits.ImageHDU(data=coord_data, header=coord_header)

        new_hdul = fits.HDUList([primary_hdu, coords_hdu])
        new_hdul.writeto(f'/Users/bardiya/Desktop/Astronomy_lab/PSF/Final_Data/Selected_Images/star_{j+1}_{i+1}.fits', overwrite=True)
        indicator += 1
    coordinates = []
    for i in range(indicator):
        with fits.open(f'/Users/bardiya/Desktop/Astronomy_lab/PSF/Final_Data/Selected_Images/star_{j+1}_{i+1}.fits') as hdul:
            coordinates.append(np.round(hdul[1].data))

    x = []
    y = []

    for i in range(indicator):
        x.append(coordinates[i][0,0])
        y.append(coordinates[i][0,1])
    plt.figure()
    plt.title(f"Plot of image {j+1}")
    plt.imshow(data, interpolation = 'nearest', origin='lower')
    plt.plot(x, y, 'o')

plt.show()


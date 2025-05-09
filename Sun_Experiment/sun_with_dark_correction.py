import astropy.io as ap
from astropy.io import fits
import numpy as np
from astropy.utils.data import get_pkg_data_filename as gpdf
import sys
sys.path.append('/Users/bardiya/Desktop/Astronomy_lab/Modules')
import Project_Mods as mod
import matplotlib.pyplot as plt

sun_img = ["/Users/bardiya/Desktop/Astronomy_lab/Sun_Experiment/Data/Sun/fits/sun_sun_canon1200d_iso100_130sec_bardia_alian_bardia_hassanpour_00001.fits"]
numbers = ["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20"]
dark_img = [f"/Users/bardiya/Desktop/Astronomy_lab/Sun_Experiment/Data/Dark/fits/sun_dark_canon1200d_iso100_130sec_bardia_alian_bardia_hassanpour_000{i}.fits" for i in numbers]

sun_array = mod.fits_to_arr_rgb(sun_img)
dark_array = mod.fits_to_arr_rgb(dark_img)

master_dark_arr = mod.master_dark_array(dark_array)
master_dark_arr_rounded = np.round(master_dark_arr)

sun_final_arr_with_negative = sun_array[0] - master_dark_arr
sun_final_arr = np.where(sun_final_arr_with_negative < 0, 0, sun_final_arr_with_negative)
sun_final_arr_rounded = np.round(sun_final_arr)

plt.imshow(sun_final_arr_rounded, interpolation ='nearest')

mod.image_output(master_dark_arr_rounded, "Master_Dark_Image.fits", "/Users/bardiya/Desktop/Astronomy_lab/Sun_Experiment/Final_Data/")
mod.image_output(sun_final_arr_rounded, "Sun_Image_with_dark_correction.fits", "/Users/bardiya/Desktop/Astronomy_lab/Sun_Experiment/Final_Data/")

plt.show()

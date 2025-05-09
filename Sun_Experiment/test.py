from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
with fits.open('/Users/bardiya/Desktop/Astronomy_lab/Sun_Experiment/Final_Data/Sun_Image_with_dark_correction.fits') as hdul:
    data_from_fits = hdul[0].data

plt.imshow(data_from_fits, interpolation ='nearest')
data = np.random.randn(1000)
plt.figure()
plt.hist(data_from_fits.flatten(), bins = 1000, color = 'blue')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

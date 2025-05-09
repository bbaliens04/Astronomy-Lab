from astropy.io import fits

with fits.open("/Users/bardiya/Desktop/Astronomy_lab/PSF/Final_Data/star_0.fits") as hdul:
    print(hdul[1].data)

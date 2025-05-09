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

data_adds = ["/Users/bardiya/Desktop/Astronomy_lab/PSF/Data/light-g_20C-2023_10_10-exp00.04.00.000-1x1_High_1.fit", "/Users/bardiya/Desktop/Astronomy_lab/PSF/Data/light_r_20C-2023_10_10-exp00.02.00.000-1x1_High_2.fit", "/Users/bardiya/Desktop/Astronomy_lab/PSF/Data/light-r_20C-2023_10_10-exp00.03.00.000-1x1_High_3.fit"]

images = mod.fits_to_arr_gray(data_adds)

image_1_num = ["1","2","3","4","5","6","7","8","10","11","12","13","14","15","16","17"]
image_2_num = ["2","3","4","22","25","28","29","35","41","42","43","44","45"]
image_3_num = ["1","2","3","4","5","6","7","14","15","16","21","23","24","25","26","27","29"]
im_1_adds = [f"/Users/bardiya/Desktop/Astronomy_lab/PSF/Final_Data/Selected_Images/star_1_{i}.fits" for i in image_1_num]
im_2_adds = [f"/Users/bardiya/Desktop/Astronomy_lab/PSF/Final_Data/Selected_Images/star_2_{i}.fits" for i in image_2_num]
im_3_adds = [f"/Users/bardiya/Desktop/Astronomy_lab/PSF/Final_Data/Selected_Images/star_3_{i}.fits" for i in image_3_num]

im_adds = [im_1_adds, im_2_adds, im_3_adds]

im_1_arr = mod.fits_to_arr_gray(im_1_adds)
im_2_arr = mod.fits_to_arr_gray(im_2_adds)
im_3_arr = mod.fits_to_arr_gray(im_3_adds)

len1 = 8
len2 = 8
len3 = 7

ims = np.concatenate((im_1_arr, im_2_arr, im_3_arr), axis=0)

def is_clustered(image, x, y, saturation_threshold):
    x_min = max(x - 1, 0)
    x_max = min(x + 2, image.shape[1])
    y_min = max(y - 1, 0)
    y_max = min(y + 2, image.shape[0])
    neighborhood = image[y_min:y_max, x_min:x_max]
    return np.sum(neighborhood >= saturation_threshold) > 0

saturation_threshold = 3000
filtered_images = []
filtered_images_indices = []

for idx, image in enumerate(ims):
    saturated_pixels = np.argwhere(image >= saturation_threshold)
    clustered = any(is_clustered(image, x, y, saturation_threshold) for x, y in saturated_pixels)
    if not clustered:
        filtered_images.append(image)
        filtered_images_indices.append(idx)
        
ims = filtered_images

lengths = []
for j in range(3):
    coordinates = []
    relevant_indices = [index for index in filtered_images_indices if index in range(sum(len(im) for im in im_adds[:j]), sum(len(im) for im in im_adds[:j+1]))]
    lengths.append(len(relevant_indices))
    file_numbers = im_adds[j]

    for index in relevant_indices:
        file_path = file_numbers[index - sum(len(im) for im in im_adds[:j])]
        with fits.open(file_path) as hdul:
            data = hdul[0].data
            if len(hdul) > 1:
                coordinates.append(np.round(hdul[1].data))

    x = []
    y = []
    for coord in coordinates:
        x.append(coord[0,0])
        y.append(coord[0,1])
    
    plt.figure()
    plt.title(f"Plot of image {j+1}")
    plt.imshow(images[j], interpolation = 'nearest', origin='lower', cmap = 'hist')
    plt.plot(x, y, 'o')

len1 = lengths[0]
len2 = lengths[1]
len3 = lengths[2]

def hot_pixel_clip(image):

    padded_image = np.pad(image, pad_width=2, mode='edge')
    temp_image = np.copy(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
        
            local_patch = padded_image[i:i+5, j:j+5]
            median = np.median(local_patch)
            std = np.std(local_patch)
            clip_threshold = median + 3 * std
            if image[i][j] > clip_threshold:
                temp_image[i][j] = median

    np.copyto(image, temp_image)

for image in ims:
    hot_pixel_clip(image)


def gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude * np.exp(- (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()

Amplitudes = []
sigmas_x = []
sigmas_y = []
thetas = []
for i, image in enumerate(ims):
    x = np.linspace(0, image.shape[1] - 1, image.shape[1])
    y = np.linspace(0, image.shape[0] - 1, image.shape[0])
    x, y = np.meshgrid(x, y)
    xy = (x, y)

    initial_guess = (np.max(image), x.mean(), y.mean(), 10, 10, 0, np.min(image))
    
    popt, pcov = curve_fit(gaussian, xy, image.ravel(), p0=initial_guess)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(x, y, image, cmap='viridis')
    ax.set_title('Data')
    ax.set_xlabel('Pixel')
    ax.set_ylabel('Pixel')
    ax.set_zlabel('Count')

    ax = fig.add_subplot(122, projection='3d')
    z = gaussian((x, y), *popt).reshape(image.shape[0], image.shape[1])
    ax.plot_surface(x, y, z, cmap='viridis')
    ax.set_title('Fit')
    ax.set_xlabel('Pixel')
    ax.set_ylabel('Pixel')
    ax.set_zlabel('Count')
    if i < len1:
        print(f"Fitted parameters for star_1_{i+1}: Amplitude={popt[0]:.3f}, X0={popt[1]:.3f}, Y0={popt[2]:.3f}, SigmaX={popt[3]:.3f}, SigmaY={popt[4]:.3f}, Theta={popt[5]:.3f}, Offset={popt[6]}\n")
    elif len1 <= i < len1 + len2:
        print(f"Fitted parameters for star_2_{i-len1+1}: Amplitude={popt[0]:.3f}, X0={popt[1]:.3f}, Y0={popt[2]:.3f}, SigmaX={popt[3]:.3f}, SigmaY={popt[4]:.3f}, Theta={popt[5]:.3f}, Offset={popt[6]:.3f}\n")
    else:
        print(f"Fitted parameters for star_3_{i-len1-len2+1}: Amplitude={popt[0]:.3f}, X0={popt[1]:.3f}, Y0={popt[2]:.3f}, SigmaX={popt[3]:.3f}, SigmaY={popt[4]:.3f}, Theta={popt[5]:.3f}, Offset={popt[6]:.3f}\n")
        
    Amplitudes.append(popt[0])
    sigmas_x.append(popt[3])
    sigmas_y.append(popt[4])
    thetas.append(popt[5])
    
sigmas_x = np.array(sigmas_x)
sigmas_y = np.array(sigmas_y)
sigmas = (sigmas_x ** 2 + sigmas_y ** 2) ** (1 / 2)

def gaussian(x, mu, sigma, A):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

plt.figure()
plt.title('All Gaussian plots x (Image 1)')
plt.xlabel('pixel')
plt.ylabel('Count')
x = np.linspace(-100, 100, 400)

for i in range(len1):
    y = gaussian(x, np.mean(x), sigmas_x[i], Amplitudes[i])
    plt.plot(x, y, label=f'Image 1 Gaussian Curve {i+1}')
plt.legend()
plt.grid(True)

# Image set 2
plt.figure()
plt.title('All Gaussian plots x (Image 2)')
plt.xlabel('pixel')
plt.ylabel('Count')
for i in range(len1, len1 + len2):
    y = gaussian(x, np.mean(x), sigmas_x[i], Amplitudes[i])
    plt.plot(x, y, label=f'Image 2 Gaussian Curve {i - len1 + 1}')
plt.legend()
plt.grid(True)

# Image set 3
plt.figure()
plt.title('All Gaussian plots x (Image 3)')
plt.xlabel('pixel')
plt.ylabel('Count')
for i in range(len1 + len2, len1 + len2 + len3):
    y = gaussian(x, np.mean(x), sigmas_x[i], Amplitudes[i])
    plt.plot(x, y, label=f'Image 3 Gaussian Curve {i - len1 - len2 + 1}')
plt.legend()
plt.grid(True)

plt.figure()
plt.title('All Gaussian plots y (Image 1)')
plt.xlabel('pixel')
plt.ylabel('Count')
x = np.linspace(-100, 100, 400)

# You should loop over the range of indices for each set separately:
# Image set 1
for i in range(len1):
    y = gaussian(x, np.mean(x), sigmas_y[i], Amplitudes[i])
    plt.plot(x, y, label=f'Image 1 Gaussian Curve {i+1}')
plt.legend()
plt.grid(True)

# Image set 2
plt.figure()
plt.title('All Gaussian plots y (Image 2)')
plt.xlabel('pixel')
plt.ylabel('Count')
for i in range(len1, len1 + len2):
    y = gaussian(x, np.mean(x), sigmas_y[i], Amplitudes[i])
    plt.plot(x, y, label=f'Image 2 Gaussian Curve {i - len1 + 1}')
plt.legend()
plt.grid(True)

# Image set 3
plt.figure()
plt.title('All Gaussian plots y (Image 3)')
plt.xlabel('pixel')
plt.ylabel('Count')
for i in range(len1 + len2, len1 + len2 + len3):
    y = gaussian(x, np.mean(x), sigmas_y[i], Amplitudes[i])
    plt.plot(x, y, label=f'Image 3 Gaussian Curve {i - len1 - len2 + 1}')
plt.legend()
plt.grid(True)

plt.figure()
plt.title('Histogram of PSF x of Image 1')
plt.hist(sigmas_y[:len1])
plt.xlabel('Sigma')

plt.figure()
plt.title('Histogram of PSF x of Image 2')
plt.hist(sigmas_y[len1:len1+len2])
plt.xlabel('Sigma')

plt.figure()
plt.title('Histogram of PSF x of Image 3')
plt.hist(sigmas_y[len1+len2:len1+len2+len3])
plt.xlabel('Sigma')

plt.figure()
plt.title('Histogram of PSF y of Image 1')
plt.hist(sigmas_y[:len1])
plt.xlabel('Sigma')

plt.figure()
plt.title('Histogram of PSF y of Image 2')
plt.hist(sigmas_y[len1:len1+len2])
plt.xlabel('Sigma')

plt.figure()
plt.title('Histogram of PSF y of Image 3')
plt.hist(sigmas_y[len1+len2:len1+len2+len3])
plt.xlabel('Sigma')

sigma_x_1 = np.median(sigmas_x[:len1])
sigma_y_1 = np.median(sigmas_y[:len1])
sigma_x_2 = np.median(sigmas_x[len1:len1+len2])
sigma_y_2 = np.median(sigmas_y[len1:len1+len2])
sigma_x_3 = np.median(sigmas_x[len1+len2:len1+len2+len3])
sigma_y_3 = np.median(sigmas_y[len1+len2:len1+len2+len3])
theta_1 = np.median(thetas[:len1])
theta_2 = np.median(thetas[len1:len1+len2])
theta_3 = np.median(thetas[len1+len2:len1+len2+len3])


print(f"Sigma_x of Image 1: {sigma_x_1:.3f} Pixels")
print(f"Sigma_x of Image 2: {sigma_x_2:.3f} Pixels")
print(f"Sigma_x of Image 3: {sigma_x_3:.3f} Pixels")
print(f"Sigma_y of Image 1: {sigma_y_1:.3f} Pixels")
print(f"Sigma_y of Image 2: {sigma_y_2:.3f} Pixels")
print(f"Sigma_y of Image 3: {sigma_y_3:.3f} Pixels")
print(f"Stars ellipse's rotation of Image 1: {theta_1:.3f} radians")
print(f"Stars ellipse's rotation of Image 2: {theta_2:.3f} radians")
print(f"Stars ellipse's rotation of Image 3: {theta_3:.3f} radians")

plt.show()

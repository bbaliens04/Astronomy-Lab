import numpy as np
from astropy.io import fits
import math
import matplotlib.pyplot as plt

def rotate_image(image, angle):
    """
    Rotate a 2D image array by a given angle around its center.
    
    Parameters:
    image (2D array): The image to be rotated.
    angle (float): The angle in degrees to rotate the image.
    
    Returns:
    2D array: The rotated image.
    """
    # Convert angle from degrees to radians
    angle_rad = math.radians(angle)
    
    # Get image dimensions
    height, width = image.shape
    
    # Calculate the center of the image
    cx, cy = width // 2, height // 2
    
    # Create an output image
    rotated_image = np.zeros_like(image)
    
    # Rotation matrix components
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)
    
    # Rotate each pixel
    for y in range(height):
        for x in range(width):
            # Translate coordinates to origin (center of the image)
            x_translated = x - cx
            y_translated = y - cy
            
            # Apply rotation
            x_rotated = cos_theta * x_translated + sin_theta * y_translated
            y_rotated = -sin_theta * x_translated + cos_theta * y_translated
            
            # Translate coordinates back to the image space
            x_new = int(round(x_rotated + cx))
            y_new = int(round(y_rotated + cy))
            
            # Assign pixel value if the new coordinates are within bounds
            if 0 <= x_new < width and 0 <= y_new < height:
                rotated_image[y_new, x_new] = image[y, x]
    
    return rotated_image

# Read the .fits file
def read_fits_file(file_path):
    with fits.open(file_path) as hdul:
        image_data = hdul[0].data
        header = hdul[0].header
    return image_data, header

# Save the rotated image data back to a .fits file
def save_fits_file(file_path, image_data, header):
    hdu = fits.PrimaryHDU(data=image_data, header=header)
    hdu.writeto(file_path, overwrite=True)

# Main function to read, rotate, save, and visualize the image
def main(input_file, output_file, angle):
    image_data, header = read_fits_file(input_file)
    rotated_image_data = rotate_image(image_data, angle)
    save_fits_file(output_file, rotated_image_data, header)
    
    # Display the original and rotated images
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image_data, cmap='gray')
    plt.title('Original Image')
    
    plt.subplot(1, 2, 2)
    plt.imshow(rotated_image_data, cmap='gray')
    plt.title(f'Rotated Image by {angle} degrees')
    
    plt.show()

# Example usage
input_file = '/Users/bardiya/Desktop/Astronomy_lab/Opacity/Final_data/corrected_datas/corrected_data_1.fits'
output_file = 'rotated_image.fits'
angle = 30  # Rotation angle in degrees
main(input_file, output_file, angle)

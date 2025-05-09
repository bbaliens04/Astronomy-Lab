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
from scipy.ndimage import center_of_mass as com
from mpl_toolkits.mplot3d import Axes3D

location = ["/Users/bardiya/Desktop/Astronomy_lab/PSP/Data/Some_Random_Stars_2s_ISO_400.fits"]

a_init = np.array([1701, 1479])
b_init = np.array([2206, 1718])
c_init = np.array([2751, 1923])
d_init = np.array([3074, 2036])
e_init = np.array([3303, 2096])
or_init = np.array([2986, 1200])
q_init = np.array([2834, 564])

pic_arr = mod.fits_to_arr_gray(location)[0]


a_init_cutout = pic_arr[a_init[0] - 50:a_init[0] + 51, a_init[1] - 50:a_init[1] + 51]
b_init_cutout = pic_arr[b_init[0] - 50:b_init[0] + 51, b_init[1] - 50:b_init[1] + 51]
c_init_cutout = pic_arr[c_init[0] - 50:c_init[0] + 51, c_init[1] - 50:c_init[1] + 51]
d_init_cutout = pic_arr[d_init[0] - 50:d_init[0] + 51, d_init[1] - 50:d_init[1] + 51]
e_init_cutout = pic_arr[e_init[0] - 50:e_init[0] + 51, e_init[1] - 50:e_init[1] + 51]
or_init_cutout = pic_arr[or_init[0] - 50:or_init[0] + 51, or_init[1] - 50:or_init[1] + 51]
q_init_cutout = pic_arr[q_init[0] - 50:q_init[0] + 51, q_init[1] - 50:q_init[1] + 51]

a_center = np.round(a_init - np.array([51,51]) + np.array(com(a_init_cutout))).astype(np.int64)
b_center = np.round(b_init - np.array([51,51]) + np.array(com(b_init_cutout))).astype(np.int64)
c_center = np.round(c_init - np.array([51,51]) + np.array(com(c_init_cutout))).astype(np.int64)
d_center = np.round(d_init - np.array([51,51]) + np.array(com(d_init_cutout))).astype(np.int64)
e_center = np.round(e_init - np.array([51,51]) + np.array(com(e_init_cutout))).astype(np.int64)
or_center = np.round(or_init - np.array([51,51]) + np.array(com(or_init_cutout))).astype(np.int64)
q_center = np.round(q_init - np.array([51,51]) + np.array(com(q_init_cutout))).astype(np.int64)

a_center = np.array([3464 - a_center[1], a_center[0]])
b_center = np.array([3464 - b_center[1], b_center[0]])
c_center = np.array([3464 - c_center[1], c_center[0]])
d_center = np.array([3464 - d_center[1], d_center[0]])
e_center = np.array([3464 - e_center[1], e_center[0]])
or_center = np.array([3464 - or_center[1], or_center[0]])
q_center = np.array([3464 - q_center[1], q_center[0]])

def snr(point):

    x = []
    y = []
    for rad in range(2, 50):
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
    
    index_max = lst.index(max(y))
    
    plt.figure()
    plt.plot(x,y)
    plt.xlabel('radius')
    plt.ylabel('SNR')
    return max(y)

snr_a = snr(a_center, "A")
snr_b = snr(b_center, "B")
snr_c = snr(c_center, "C")
snr_d = snr(d_center, "D")
snr_e = snr(e_center, "E")
snr_or = snr(or_center, "Refrence")
snr_q = snr(q_center, "Question Mark")

def mag(point, rad, snr):
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
            if 25 ** 2 < (i - 51) ** 2 + (j - 51) ** 2 < 30 ** 2:
                coords_sky.append(point - np.array([51,51]) + np.array([i, j]))
        
    data_sky = []
    for k in coords_sky:
        data_sky.append(pic_arr[k[0], k[1]])
            
    npixel = N
    sky = np.median(data_sky)
    sig = (summ - sky * npixel)
    mag_inst = -2.5 * np.log10(sig)
    
    mag_err = 1.08/snr
    return mag_inst, mag_err
    
mag_or , mag_err_or = mag(or_center, 10, snr_or)
mag_a , mag_err_a = mag(a_center, 6, snr_a)
mag_b , mag_err_b = mag(b_center, 9, snr_b)
mag_c , mag_err_c = mag(c_center, 7, snr_c)
mag_d , mag_err_d = mag(d_center, 7, snr_d)
mag_e , mag_err_e = mag(e_center, 7, snr_e)
mag_q , mag_err_q = mag(q_center, 7, snr_q)
mag_a = mag_a - mag_or + 4.12
mag_b = mag_b - mag_or + 4.12
mag_c = mag_c - mag_or + 4.12
mag_d = mag_d - mag_or + 4.12
mag_e = mag_e - mag_or + 4.12
mag_q = mag_q - mag_or + 4.12
mag_err_a = mag_err_a + mag_err_or
mag_err_b = mag_err_b + mag_err_or
mag_err_c = mag_err_c + mag_err_or
mag_err_d = mag_err_d + mag_err_or
mag_err_e = mag_err_e + mag_err_or
mag_err_q = mag_err_q + mag_err_or

print(mag_a)
print(mag_b)
print(mag_c)
print(mag_d)
print(mag_e)
print(mag_q)
print(mag_err_a)
print(mag_err_b)
print(mag_err_c)
print(mag_err_d)
print(mag_err_e)
print(mag_err_q)

plt.show()
    

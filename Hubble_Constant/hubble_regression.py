import numpy as np
import matplotlib.pyplot as plt

app_mag = [25.2, 24.8, 25.6, 24.9, 25.5, 24.5, 24.1, 25.1, 25.1, 24.8]
period = [39.5, 38.7, 35.8, 41.7, 30.2, 39.2, 42.9, 41.4, 32.5, 42.1]
err_app_mag = [0.2, 0.2, 0.1, 0.1, 0.1, 0.2, 0.1, 0.2, 0.2, 0.1]
v = [547, 468, 625, 492, 510, 385, 343, 557, 454, 482]
err_v = [6, 12, 4, 13, 18, 14, 13, 3, 6, 3]

def distance_calculator(app_mag, period, err_app_mag):
    # M_v = alog(p) + b
    a = -3.81
    err_a = 0.00403
    b = 1.47
    err_b = 0.00752
    
    M_v = [a * np.log10(i) + b for i in period]
    err_M_v = [np.sqrt((err_a * np.log10(i)) ** 2 + (err_b) ** 2) for i in period]

    # calculating distance in Mpc
    distance = [10 ** ((m - M) / 5 - 5) for m, M in zip(app_mag, M_v)]
    err_distance = [np.log(10) * d / 5 * np.sqrt((err_m) ** 2 + (err_M) ** 2) for d, err_m, err_M in zip(distance, err_app_mag, err_M_v)]
    
    return distance, err_distance

def reg_xy_error(y, x, y_err, x_err):

    y = np.array(y)
    x = np.array(x)
    y_err = np.array(y_err)
    x_err = np.array(x_err)
    slope = 74
    tolerance = 0.001
    max_iterations = 10000000
    iteration = 0
    while True:
        combined_err = np.sqrt(y_err**2 + (slope * x_err)**2)
        w = 1 / combined_err**2

        x_wmean = np.average(x, weights=w)
        y_wmean = np.average(y, weights=w)

        covariance_w = np.average((x - x_wmean) * (y - y_wmean), weights=w)
        variance_w = np.average((x - x_wmean)**2, weights=w)

        slope_new = covariance_w / variance_w

        if np.abs(slope_new - slope) < tolerance or iteration > max_iterations:
            slope = slope_new
            break

        slope = slope_new
        iteration += 1
        
    intercept = y_wmean - slope * x_wmean
    
    y_pred = intercept + slope * x
    residuals = y - y_pred
    
    N = len(x)
    slope_error = np.sqrt(np.sum(w * residuals**2) / ((N - 2) * np.sum(w * (x - x_wmean)**2)))
    intercept_error = np.sqrt(np.sum(w * residuals**2) / (N - 2)) * np.sqrt((1/N) + (x_wmean**2) / np.sum(w * (x - x_wmean)**2))
    
    R_2 = 1 - np.sum(w * residuals ** 2) / np.sum(w * (y - y_wmean) ** 2)
    
    return slope, intercept, R_2, slope_error, intercept_error

def reg_xy_error_origin(y, x, y_err, x_err):
    y = np.array(y)
    x = np.array(x)
    y_err = np.array(y_err)
    x_err = np.array(x_err)
    slope = 74
    tolerance = 0.001
    max_iterations = 10000000
    iteration = 0
    while True:
        combined_err = np.sqrt(y_err**2 + (slope * x_err)**2)
        w = 1 / combined_err**2
        x_wmean = np.average(x, weights=w)
        y_wmean = np.average(y, weights=w)
        slope_new = np.sum(w * x * y) / np.sum(w * x**2)
        
        if np.abs(slope_new - slope) < tolerance or iteration > max_iterations:
            slope = slope_new
            break
        
        slope = slope_new
        iteration += 1
    
    y_pred = slope * x
    residuals = y - y_pred
    
    N = len(x)
    slope_error = np.sqrt(np.sum(w * residuals**2) / ((N - 1) * np.sum(w * (x - x_wmean)**2)))
    
    R_2 = 1 - np.sum(w * residuals ** 2) / np.sum(w * (y - y_wmean) ** 2)
    
    return slope, R_2, slope_error
    
distance, err_distance = distance_calculator(app_mag, period, err_app_mag)

slope_1, intercept_1, R_2_1, slope_error_1, intercept_error_1 = reg_xy_error(v, distance, err_v, err_distance)
slope_2 , R_2_2 , slope_error_2 = reg_xy_error_origin(v, distance, err_v, err_distance)

print(f"""For not through origin:
Slope : {slope_1}
intercept : {intercept_1}
R_2 : {R_2_1}
slope_error : {slope_error_1}
intercept_error : {intercept_error_1}""")

print(f"""For through origin:
Slope : {slope_2}
R_2 : {R_2_2}
slope_error : {slope_error_2}""")

if R_2_1 > R_2_2:
    line_of_best_fit = slope_1 * np.array(distance) + intercept_1
    print("Not through origin is better")

else:
    line_of_best_fit = slope_2 * np.array(distance)
    print("Through origin is better")
    
plt.plot(distance, line_of_best_fit, color='red', label='Line of Best Fit')
plt.errorbar(distance, v, xerr=err_distance, yerr=err_v, fmt='o', ecolor='r', capsize=5, label='Data with error bars')
plt.xlabel('Distance (Mpc)')
plt.ylabel('Radial Velocity (Km/s)')
plt.title('Radial Velocity vs Distance')
plt.grid(True)
plt.legend()
plt.show()

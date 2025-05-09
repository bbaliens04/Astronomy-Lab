import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt

app_mag = [21.54, 20.45, 21.25, 20.35, 22.37, 21.35, 22.09, 20.36, 20.98, 22.17, 20.17, 22.30, 20.34, 20.33, 21.18, 22.54]
period = [13.15, 25.40, 15.62, 26.98, 7.97, 14.78, 9.41, 26.79, 18.39, 8.96, 30.08, 8.27, 27.14, 27.32, 16.29, 7.17]
app_mag_err = [0.12, 0.17, 0.13, 0.18, 0.10, 0.13, 0.11, 0.18, 0.14, 0.11, 0.19, 0.11, 0.18, 0.18, 0.14, 0.16]

def app_to_abs_table(app_mag, period, app_mag_err):
    distance = 2400000 / 3.26
    abs_mag = [round(i + 5 - 5  * np.log10(distance), ndigits = 2) for i in app_mag]
    abs_mag_error =  [i for i in app_mag_err]
    data = [['Mv', 'Period', 'Mv Error']]
    for i in range(len(app_mag)):
        data.append([abs_mag[i], period[i], abs_mag_error[i]])
        
    print(tabulate(data, headers='firstrow', tablefmt='fancy_grid'))
    
    return abs_mag, abs_mag_error

abs_mag, abs_mag_err = app_to_abs_table(app_mag, period, app_mag_err)

def reg_y_error(y, x, y_err):

    y = np.array(y)
    x = np.array(x)
    y_err = np.array(y_err)
    
    w_y = 1 / y_err ** 2
    
    x_wmean = np.average(x, weights=w_y)
    y_wmean = np.average(y, weights=w_y)
    
    covariance_w = np.average((x - x_wmean) * (y - y_wmean), weights=w_y)
    variance_w = np.average((x - x_wmean)**2, weights=w_y)
    
    slope = covariance_w / variance_w
    intercept = y_wmean - slope * x_wmean
    
    y_pred = intercept + slope * x
    residuals = y - y_pred
    
    N = len(x)
    slope_error = np.sqrt(np.sum(w_y * residuals**2) / ((N - 2) * np.sum(w_y * (x - x_wmean)**2)))
    intercept_error = np.sqrt(np.sum(w_y * residuals**2) / (N - 2)) * np.sqrt((1/N) + (x_wmean**2) / np.sum(w_y * (x - x_wmean)**2))
    
    R_2 = 1 - np.sum(w_y * residuals ** 2) / np.sum(w_y * (y - y_wmean) ** 2)
    
    return slope, intercept, R_2, slope_error, intercept_error


log_period = [np.log10(i) for i in period]

slope, intercept, R_2, slope_error, intercept_error = reg_y_error(abs_mag, log_period, abs_mag_err)
print(f"""Slope : {slope}
intercept : {intercept}
R_2 : {R_2}
slope_error : {slope_error}
intercept_error : {intercept_error}""")

line_of_best_fit = slope * np.array(log_period) + intercept
plt.plot(log_period, line_of_best_fit, color='red', label='Line of Best Fit')
plt.errorbar(log_period, abs_mag, yerr=abs_mag_err, fmt='o', color='b', ecolor='r', capsize=5, label = 'Data with error bars')
plt.xlabel('Log(Period) (log(day))')
plt.ylabel('M_v')
plt.title('Absolute Magnitude vs Log Period')
plt.grid(True)
plt.legend()
plt.show()

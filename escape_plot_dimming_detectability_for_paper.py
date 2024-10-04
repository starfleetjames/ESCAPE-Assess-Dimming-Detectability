import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
from sklearn.metrics import r2_score
from scipy.interpolate import interp1d


def process_and_plot_data(data, spectral_integration_type=''):
    # Sort the data by the first column ("log10 F(X) (erg/cm2/s)")
    data_sorted = data.sort_values(by=data.columns[0])

    # Extracting x, y, and "column density" data for plotting and analysis
    x = data_sorted.iloc[:, 0]
    y = data_sorted.iloc[:, 2]
    column_density = data_sorted.iloc[:, 1]

    # Fit a 3rd order polynomial
    coefficients = np.polyfit(x, y, 3)
    p = np.poly1d(coefficients)
    formatted_equation = format_poly_equation(coefficients)

    # Print the equation of fit
    polynomial_equation = ' + '.join([f'{coeff:.2f}x^{i}' for i, coeff in enumerate(coefficients[::-1])])
    excel_polynomial = ' + '.join([f'{coeff} * POWER(K2, {i})' for i, coeff in enumerate(coefficients[::-1])])
    print(f'For {spectral_integration_type}, the fit equation is:')
    print(polynomial_equation)
    print(excel_polynomial)

    r_squared = r2_score(y, p(x))
    print(f'R-squared: {r_squared}')

    # Take the last x-value and associated y-values and column density values
    last_x = x.iloc[-1]
    last_y_values = y[x == last_x]
    last_column_density_values = column_density[x == last_x]

    # Calculate the fitted y-value for the last x
    fitted_y_last_x = p(last_x)

    # Interpolate to find the corresponding column density value for the fitted y-value
    interpolation_function = interp1d(last_y_values, last_column_density_values, kind='linear', fill_value="extrapolate")
    interpolated_column_density = interpolation_function(fitted_y_last_x)

    plt.figure(figsize=(10, 6))
    plt.axhspan(0, 3, color='grey', alpha=0.5, zorder=0)
    ax = plt.gca()  # Get the current axis
    ax.set_axisbelow(True)
    plt.grid(True)
    scatter = plt.scatter(x, y, c=column_density, cmap='viridis')
    plt.plot(x, p(x), color='black', label=f'Fit: {formatted_equation} | R$^2$ = {r_squared:.2f}')
    plt.xlabel('X-ray flux | log$_{10}$ F(X) [erg cm$^{-2}$ s$^{-1}$]')
    plt.ylabel('significance of detection [$\sigma$]')
    plt.ylim(bottom=0)
    plt.yticks(np.arange(0, plt.ylim()[1] + 1, 1))
    plt.title('Spectrally Integrated ' + spectral_integration_type)
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1))
    cbar = plt.colorbar(scatter, label='log$_{10}$ N(HI) [cm$^{-2}$]')
    cbar.ax.invert_yaxis()

    spectral_integration_type_underscores = spectral_integration_type.replace(" ", "_").lower()
    filename = f"{save_path}escape_dimming_detection_significance_{spectral_integration_type_underscores}"
    plt.savefig(f"{filename}.png", bbox_inches='tight', dpi=300)
    plt.savefig(f"{filename}.pdf", bbox_inches='tight')
    plt.show()

    return interpolated_column_density


def format_poly_equation(coefficients):
    terms = []
    for i, coeff in enumerate(coefficients[::-1]):
        exponent = len(coefficients) - 1 - i
        term = f"{coeff:0.3g}" if exponent == 0 else f"{coeff:0.3g}x$^{exponent}$"
        terms.append(term)
    return " + ".join(terms)


save_path = '/Users/masonjp2/Library/CloudStorage/GoogleDrive-jmason86@gmail.com/.shortcut-targets-by-id/1aM0cJ5QKqP52iZb4GeBxx032vFk_c9CW/ESCAPE Initial Groundwork/Dimming Sensitivity Study/Significance of Detection/'
data_path = '/Users/masonjp2/Dropbox/Research/Data/ESCAPE/escape_dimming_detectability_exploration/'
file_path_combo = data_path + 'escape dimming parameter exploration 3 line combo solid gold.csv'
file_path_bands = data_path + 'escape dimming parameter exploration line combo solid gold 17.5-19 ism 600s exposure.csv'
#data_combo = pd.read_csv(file_path_combo)
data_bands = pd.read_csv(file_path_bands)

# Process the data and get the best fit column density
best_fit_column_density = process_and_plot_data(data_bands, spectral_integration_type='Solid Gold Line Combo 17.5-19 N(HI) 600s Exposure')

# Print the best fit column density
print("Best fit column density values:")
print(best_fit_column_density)

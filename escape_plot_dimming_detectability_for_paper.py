import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
from sklearn.metrics import r2_score
from scipy.interpolate import interp1d

# Set font sizes for publication
plt.rc('font', size=16)          # controls default text sizes
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=18)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
plt.rc('ytick', labelsize=16)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)    # legend fontsize

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
    plt.title('Spectrally-Integrated ' + spectral_integration_type)
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1))
    cbar = plt.colorbar(scatter, label='log$_{10}$ N(HI) [cm$^{-2}$]')
    cbar.ax.invert_yaxis()

    # Shade the regions based on detection significance threshold
    shade_regions(ax, x, y, threshold=3)

    spectral_integration_type_underscores = spectral_integration_type.replace(" ", "_").lower()
    if "escape" in spectral_integration_type.lower():
        prefix = "escape"
    elif "euve" in spectral_integration_type.lower():
        prefix = "euve"
    else:
        prefix = "other"

    png_filename = f"{save_path}png/{prefix}_dimming_detection_significance_{spectral_integration_type_underscores}"
    pdf_filename = f"{save_path}{prefix}_dimming_detection_significance_{spectral_integration_type_underscores}"
    plt.savefig(f"{png_filename}.png", bbox_inches='tight', dpi=300)
    plt.savefig(f"{pdf_filename}.pdf", bbox_inches='tight')
    plt.show()

    return interpolated_column_density


def format_poly_equation(coefficients):
    terms = []
    for i, coeff in enumerate(coefficients[::-1]):
        exponent = len(coefficients) - 1 - i
        term = f"{coeff:0.3g}" if exponent == 0 else f"{coeff:0.3g}x$^{exponent}$"
        terms.append(term)
    return " + ".join(terms)


def shade_regions(ax, flux, detection_significance, threshold=3):
    """
    Shade regions in the plot based on detection significance threshold.

    Parameters:
    ax (matplotlib.axes.Axes): The axes object of the plot.
    flux (pd.Series): The flux values.
    detection_significance (pd.Series): The detection significance values.
    threshold (float): The threshold value for detection significance. Default is 3.
    """
    # Define the vertical limits for the shaded regions
    y_min, y_max = ax.get_ylim()
    shade_height = 0.03 * (y_max - y_min) 
    y_bottom = y_min
    y_top = y_bottom + shade_height

    # Group the data by flux values
    grouped = detection_significance.groupby(flux)

    # Find the flux values where all detection significance values are below the threshold
    original_xlim = ax.get_xlim()

    valid_flux_values_below = grouped.filter(lambda group: all(group <= threshold)).index.unique()
    if not valid_flux_values_below.empty:
        ax.axvspan(original_xlim[0], flux[valid_flux_values_below.max()], ymin=(y_bottom - y_min) / (y_max - y_min), ymax=(y_top - y_min) / (y_max - y_min), color='tomato', alpha=0.3, zorder=0)

    # Find the flux values where all detection significance values are above the threshold
    valid_flux_values_above = grouped.filter(lambda group: all(group > threshold)).index.unique()
    if not valid_flux_values_above.empty:
        ax.axvspan(flux[valid_flux_values_above.min()], original_xlim[1], ymin=(y_bottom - y_min) / (y_max - y_min), ymax=(y_top - y_min) / (y_max - y_min), color='limegreen', alpha=0.3, zorder=0)

    # Shade the region between the red and green regions as yellow
    if not valid_flux_values_below.empty and not valid_flux_values_above.empty:
        ax.axvspan(flux[valid_flux_values_below.max()], flux[valid_flux_values_above.min()], ymin=(y_bottom - y_min) / (y_max - y_min), ymax=(y_top - y_min) / (y_max - y_min), color='yellow', alpha=0.3, zorder=0)

    ax.set_xlim(original_xlim)


save_path = '/Users/masonjp2/Dropbox/Apps/Overleaf/ESCAPE Dimming Detectability Paper/figures/'
data_path = '/Users/masonjp2/Dropbox/Research/Data/ESCAPE/escape_dimming_detectability_exploration/'

file_path_combo_escape = data_path + 'escape dimming parameter exploration 5-line combo solid gold 17.5-19 ism 600s exposure.csv'
file_path_bands_escape = data_path + 'escape dimming parameter exploration bands solid gold 17.5-19 ism 600s exposure.csv'
file_path_combo_euve = data_path + 'euve dimming parameter exploration 5-line combo solid gold 17.5-19 ism 600s exposure.csv'
file_path_bands_euve = data_path + 'euve dimming parameter exploration bands solid gold 17.5-19 ism 600s exposure.csv'
                                
data_combo_escape = pd.read_csv(file_path_combo_escape)
data_band_escape = pd.read_csv(file_path_bands_escape)
data_combo_euve = pd.read_csv(file_path_combo_euve)
data_band_euve = pd.read_csv(file_path_bands_euve)

# Process the data and get the best fit column density
best_fit_column_density_line_combo = process_and_plot_data(data_combo_escape, spectral_integration_type='ESCAPE 5-line Combo 600s Exposure')
best_fit_column_density_bands = process_and_plot_data(data_band_escape, spectral_integration_type='ESCAPE Bands 600s Exposure')
best_fit_column_density_line_combo_euve = process_and_plot_data(data_combo_euve, spectral_integration_type='EUVE 5-line Combo 600s Exposure')
best_fit_column_density_bands_euve = process_and_plot_data(data_band_euve, spectral_integration_type='EUVE Bands 600s Exposure')

# Print the best fit column density
print(f"Best fit column density values for line combo: {best_fit_column_density_line_combo}")
print(f"Best fit column density values for bands: {best_fit_column_density_bands}")
print(f"Best fit column density values for line combo EUVE: {best_fit_column_density_line_combo_euve}")
print(f"Best fit column density values for bands EUVE: {best_fit_column_density_bands_euve}")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.io import readsav
import re

# Set font sizes for publication
plt.rc('font', size=18)          # controls default text sizes
plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=20)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
plt.rc('legend', fontsize=16)    # legend fontsize

def extract_values(filename, instrument):
    # Define a regex pattern to match the required values, using the instrument parameter
    pattern = fr"light_curve_{instrument}_([-+]?\d*\.\d+|\d+)_xray_([-+]?\d*\.\d+|\d+)_ism"
    
    # Search for the pattern in the filename
    match = re.search(pattern, filename)
    
    if match:
        # Extract the values and convert them to appropriate types
        xray_flux = float(match.group(1))
        ism = float(match.group(2))
        
        return xray_flux, ism
    else:
        raise ValueError(f"Filename format is incorrect or values not found for instrument {instrument}")


def get_depth_window_indices(dimming, best_detection):
    padded_indices = np.reshape(dimming[0].depth_time_range_indices[:, best_detection.index[0]], -1)
    return padded_indices[np.where(padded_indices != 0)]


def get_instrument_name(instrument):
    return instrument.name[0].decode('utf-8')

def compute_noise(light_curve, instrument, num_lines_to_combine, data_group, wavelengths):
    instrument_name = get_instrument_name(instrument)
    if instrument_name == 'ESCAPE Solid Gold':
        if data_group == 'single_lines':
            background_noise = (instrument.size_of_resel * instrument.background_rate_per_resel * instrument.exposure_time_sec)
        elif data_group == 'combo_lines':
            background_noise = (instrument.size_of_resel * instrument.background_rate_per_resel * instrument.exposure_time_sec) * num_lines_to_combine
        elif data_group == 'bands':
            bandpass_width = wavelengths[1] - wavelengths[0]
            background_noise = (instrument.size_of_resel * instrument.background_rate_per_resel * instrument.exposure_time_sec) * bandpass_width
    elif instrument_name == 'EUVE':
        if data_group == 'single_lines':
            background_rate = get_euve_spectrometer_background_rate(wavelengths)
            width_of_emission_line_bin = 2.0
            background_noise = (background_rate * instrument.exposure_time_sec) * width_of_emission_line_bin
        elif data_group == 'combo_lines':
            background_rate = get_euve_spectrometer_background_rate(wavelengths)
            width_of_emission_line_bin = 2.0
            background_noise = np.sum(background_rate * instrument.exposure_time_sec) * width_of_emission_line_bin * num_lines_to_combine
        elif data_group == 'bands':
            bandpass_width = wavelengths[1] - wavelengths[0]
            background_noise = (instrument.size_of_resel_deep * instrument.background_rate_per_resel_deep * instrument.exposure_time_sec) * bandpass_width
    else:
        raise ValueError(f"Instrument {instrument_name} not supported")
    
    return np.sqrt(light_curve + background_noise)


def plot_light_curve(ax, dimming, lines, preflare_baselines, best_detection, instrument, num_lines_to_combine, xray, ism, index, color, data_group):
    # Find the light curve with the specified index and store it for plotting
    light_curve = np.reshape(lines.intensity[:, index], -1)
    preflare_baseline = preflare_baselines.intensity[0][index]
    time_hours = (lines.jd - lines.jd[0]) * 24.

    if data_group == 'single_lines':
        wavelengths = lines.wave[index]
    else:
        wavelengths = lines.wave[:, index]



    noise = compute_noise(light_curve, instrument, num_lines_to_combine, data_group, wavelengths)

    # Plotting
    #ax.plot(time_hours, light_curve, color=color, linewidth=2, label=f'Light Curve ISM={ism}')
    ax.errorbar(time_hours, light_curve, yerr=noise, fmt='-', linewidth=2,
                   ecolor=(*plt.matplotlib.colors.to_rgba(color)[:3], 0.2), capsize=5, elinewidth=1.0,
                   color=color, label=f'Light Curve ISM={ism}')
    ax.plot(ax.get_xlim(), [preflare_baseline, preflare_baseline], '--', linewidth=1, color=color, label=f'Baseline ISM={ism}')

    ax.set_xlim(0, 24)
    best_detection_wavelength_combo = best_detection.best_detection_wavelength_combo[0].decode('utf-8')


def get_euve_spectrometer_background_rate(wavelengths):
    """
    Calculate EUVE spectrometer background rates based on wavelength ranges.
    
    Parameters:
    wavelengths : numpy.ndarray
        Array of wavelengths in Angstroms
        
    Returns:
    numpy.ndarray
        Array of background rates in counts/Å/sec
    """ 
    result = np.zeros_like(wavelengths, dtype=float)
    
    # Apply background rates based on wavelength ranges
    mask_lt_190 = wavelengths < 190
    mask_190_to_370 = (wavelengths >= 190) & (wavelengths < 370)
    mask_ge_370 = wavelengths >= 370
    
    result[mask_lt_190] = 5.4e-4
    result[mask_190_to_370] = 2.9e-4
    result[mask_ge_370] = 1.7e-4
    
    return result  # [counts/Å/sec]


# Plot for either escape or euve
instrument_name = 'escape'  

save_path = '/Users/masonjp2/Dropbox/Apps/Overleaf/ESCAPE Dimming Detectability Paper/figures/'
data_path = '/Users/masonjp2/Dropbox/Research/Data/ESCAPE/escape_dimming_detectability_exploration/' #euve line combo/'
filenames = [
    f'light_curve_{instrument_name}_-10.5_xray_17.5_ism.sav',
    f'light_curve_{instrument_name}_-10.5_xray_18.2_ism.sav',
    f'light_curve_{instrument_name}_-10.5_xray_19.0_ism.sav'
]

# Load data from all files
data_list = []
xray_list = []
ism_list = []
for file in filenames:
    data = readsav(data_path + file)
    xray, ism = extract_values(file, instrument_name)
    data_list.append(data)
    xray_list.append(xray)
    ism_list.append(ism)

# Need to rename some keys for easy looping further down
for data in data_list:
    data['single_lines'] = data.pop('emission_lines')
    data['best_single_lines'] = data.pop('best_single')
    data['combo_lines'] = data.pop('combined_lines')
    data['best_combo_lines'] = data.pop('best_combo')

# Colors for different traces
colors = ['tomato', 'limegreen','dodgerblue']
isms = [17.5, 18.25, 19.0]

# Create a 3x3 subplot structure
fig, axes = plt.subplots(3, 3, figsize=(16, 9))

# Define data groups for each column
data_groups = ['single_lines', 'combo_lines', 'bands']
column_titles = ['Single Line', 'Combined Lines', 'Band']

# Loop through each column and plot the corresponding data
for col, data_group in enumerate(data_groups):
    for row in range(3):
        ax = axes[row, col]
        for i in range(3):
            plot_light_curve(ax, data_list[i][f'dimming_{data_group}'], data_list[i][data_group][0], data_list[i][f'preflare_baselines_{data_group}'], data_list[i][f'best_{data_group}'], data_list[i]['instrument'], data_list[i]['num_lines_to_combine'], xray_list[i], ism_list[i], row, colors[i], data_group)
        ax.grid(True)
        if row == 2:
            ax.set_xlabel('hours since start')
        # Remove individual y-axis labels
        ax.set_ylabel('')
        
        # Set the title using the wave variable
        if col == 0:
            wave_value = round(data_list[i][data_group][0].wave[row])
            ax.set_title(f'{column_titles[col]}: {wave_value} Å')
        elif col == 1:
            wave_value = data_list[i][data_group][0].wave[:, row]
            wave_values_rounded = [f"{int(round(val))}" for val in wave_value]
            wave_values_str = '+'.join(wave_values_rounded)
            ax.set_title(f'{column_titles[col]}: {wave_values_str} Å')
        elif col == 2:
            wave_value = data_list[i][data_group][0].wave[:, row]
            wave_values_rounded = [f"{int(round(val))}" for val in wave_value[:2]]
            wave_values_str = '-'.join(wave_values_rounded)
            ax.set_title(f'{column_titles[col]}: {wave_values_str} Å')

# Add a single y-axis label that spans the rows
fig.text(-0.05, 0.5, 'intensity [counts]', va='center', rotation='vertical', fontsize=20)

handles = [mpatches.Patch(color=colors[i], label=rf'log$_{{10}}$N(HI) [cm$^{{-2}}$]={isms[i]}') for i in range(len(isms))]
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=len(isms))

plt.tight_layout(rect=[0, 0, 1, 0.92])
filename_png = f"{save_path}png/{instrument_name}_dimming_light_curve_examples"
filename_pdf = f"{save_path}{instrument_name}_dimming_light_curve_examples"
plt.savefig(f"{filename_png}.png", bbox_inches='tight', dpi=300)
plt.savefig(f"{filename_pdf}.pdf", bbox_inches='tight')
plt.show()

pass
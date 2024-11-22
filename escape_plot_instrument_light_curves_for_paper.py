import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.io import readsav
import re

def extract_values(filename):
    # Define a regex pattern to match the required values
    pattern = r"light_curve_instrument_([-+]?\d*\.\d+|\d+)_xray_([-+]?\d*\.\d+|\d+)_ism"
    
    # Search for the pattern in the filename
    match = re.search(pattern, filename)
    
    if match:
        # Extract the values and convert them to appropriate types
        xray_flux = float(match.group(1))
        ism = float(match.group(2))
        
        return xray_flux, ism
    else:
        raise ValueError("Filename format is incorrect or values not found")


def get_depth_window_indices(dimming, best_detection):
    padded_indices = np.reshape(dimming[0].depth_time_range_indices[:, best_detection.index[0]], -1)
    return padded_indices[np.where(padded_indices != 0)]


def plot_light_curve(ax, dimming, lines, preflare_baselines, best_detection, instrument, num_lines_to_combine, xray, ism, index, color):
    # Find the light curve with the specified index and store it for plotting
    light_curve = np.reshape(lines.intensity[:, index], -1)
    preflare_baseline = preflare_baselines.intensity[0][index]
    time_hours = (lines.jd - lines.jd[0]) * 24.

    # Plotting
    ax.plot(time_hours, light_curve, color=color, linewidth=2, label=f'Light Curve ISM={ism}')
    ax.plot(ax.get_xlim(), [preflare_baseline, preflare_baseline], '--', linewidth=1, color=color, label=f'Baseline ISM={ism}')

    ax.set_xlim(0, 24)
    best_detection_wavelength_combo = best_detection.best_detection_wavelength_combo[0].decode('utf-8')


# Plot for either escape or euve
instrument = 'escape'  

save_path = '/Users/masonjp2/Dropbox/Apps/Overleaf/ESCAPE Dimming Detectability Paper/figures/'
data_path = '/Users/masonjp2/Dropbox/Research/Data/ESCAPE/escape_dimming_detectability_exploration/'
filenames = [
    f'light_curve_{instrument}_-10.5_xray_17.5_ism.sav',
    f'light_curve_{instrument}_-10.5_xray_18.2_ism.sav',
    f'light_curve_{instrument}_-10.5_xray_19.0_ism.sav'
]

# Load data from all files
data_list = []
xray_list = []
ism_list = []
for file in filenames:
    data = readsav(data_path + file)
    xray, ism = extract_values(file)
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
            plot_light_curve(ax, data_list[i][f'dimming_{data_group}'], data_list[i][data_group][0], data_list[i][f'preflare_baselines_{data_group}'], data_list[i][f'best_{data_group}'], data_list[i]['instrument'], data_list[i]['num_lines_to_combine'], xray_list[i], ism_list[i], row, colors[i])
        ax.grid(True)
        if row == 2:
            ax.set_xlabel('hours since start')
        if col == 0:
            ax.set_ylabel(f'intensity [counts]')
        
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

handles = [mpatches.Patch(color=colors[i], label=rf'log$_{{10}}$N(HI) [cm$^{{-2}}$]={isms[i]}') for i in range(len(isms))]
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=len(isms))

plt.tight_layout(rect=[0, 0, 1, 0.92])
filename_png = f"{save_path}png/{instrument}_dimming_light_curve_examples"
filename_pdf = f"{save_path}{instrument}_dimming_light_curve_examples"
plt.savefig(f"{filename_png}.png", bbox_inches='tight', dpi=300)
plt.savefig(f"{filename_pdf}.pdf", bbox_inches='tight')
plt.show()

pass
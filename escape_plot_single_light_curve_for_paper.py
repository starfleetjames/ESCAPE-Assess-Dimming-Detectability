from scipy.io import readsav
import matplotlib.pyplot as plt

# Set font sizes for publication
plt.rc('font', size=20)          # controls default text sizes
plt.rc('axes', titlesize=22)     # fontsize of the axes title
plt.rc('axes', labelsize=22)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
plt.rc('ytick', labelsize=20)    # fontsize of the tick labels
plt.rc('legend', fontsize=20)    # legend fontsize

# Read in the file
file_path = '/Users/masonjp2/Dropbox/Research/Data/ESCAPE/escape_dimming_detectability_exploration/ESCAPE Solid Gold Best Detection Light Curve.sav'
data = readsav(file_path)

time_hours = data['time_hours']
light_curve = data['light_curve']
noise = data['noise']
best_detection = data['best_detection']
depth_window_indices = data['depth_window_indices']
preflare_window_indices = data['preflare_window_indices']
preflare_baseline = data['preflare_baseline']

# Extract and decode the best detection wavelength combo
best_detection_wavelength_combo = best_detection['best_detection_wavelength_combo'][0].decode('utf-8')

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the light curve with noise
ax.errorbar(time_hours, light_curve, yerr=noise, fmt='-', ecolor='grey', capsize=5, elinewidth=1.0, linewidth=3.1, color='black', label='')

# Highlight the preflare window in dodgerblue
preflare_times = time_hours[preflare_window_indices[:-1]]
preflare_values = light_curve[preflare_window_indices[:-1]]
ax.plot(preflare_times, preflare_values, color='dodgerblue', linewidth=3.0, label='used for baseline calculation', zorder=5)

# Add a thick dashed dodgerblue horizontal line for the baseline
ax.axhline(preflare_baseline, color='dodgerblue', linestyle='--', linewidth=3.5, label='baseline')

# Highlight the depth window in tomato
depth_times = time_hours[depth_window_indices[:-1]]
depth_values = light_curve[depth_window_indices[:-1]]
ax.plot(depth_times, depth_values, color='tomato', linewidth=3.1, label='used for depth calculation', zorder=5)

# Add labels
ax.set_xlabel('hours since start', fontsize=22)
ax.set_ylabel(f'{best_detection_wavelength_combo}\nintensity [counts]', fontsize=22)
ax.legend()

# Tick formatting
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.minorticks_on()
ax.tick_params(axis='both', which='major', labelsize=20)
ax.tick_params(axis='both', which='minor', labelsize=18)
plt.tight_layout()

# Save
save_path = '/Users/masonjp2/Dropbox/Apps/Overleaf/ESCAPE Dimming Detectability Paper/figures/'
plt.savefig(f"{save_path}png/single_light_curve_with_noise.png", bbox_inches='tight', dpi=300)
plt.savefig(f"{save_path}single_light_curve_with_noise.pdf", bbox_inches='tight')
plt.show()

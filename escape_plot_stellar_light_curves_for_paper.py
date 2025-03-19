import numpy as np
import astropy.units as u
from astropy.time import Time
from scipy.io import readsav
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

save_path = '/Users/masonjp2/Dropbox/Apps/Overleaf/ESCAPE Dimming Detectability Paper/figures/'
data_path = '/Users/masonjp2/Dropbox/Research/Data/ESCAPE/escape_dimming_detectability_exploration/'
filenames = [
    'stellar_spectrum_-10.5_xray_17.5_ism.sav',
    'stellar_spectrum_-10.5_xray_18.2_ism.sav',
    'stellar_spectrum_-10.5_xray_19.0_ism.sav'
]

colors = ['tomato', 'dodgerblue', 'lime']
isms = [17.5, 18.25, 19.0]

def process_file(file_path):
    tmp = readsav(file_path)
    eve = tmp['eve_stellar']

    # Break out the relevant parts
    photon_flux = eve['irrad'][0] # [photons/s/cm2/Å]
    wavelength = eve['wave'][0] * u.angstrom
    jd = eve['jd'][0]

    # Convert time
    hours = (jd - jd[0]) * 24
    time_iso = Time(jd, format='jd').iso

    return hours, time_iso, photon_flux, wavelength

def integrate_photon_flux_over_wavelength(photon_flux, wavelength, start_wavelength, stop_wavelength):
    """
    Integrate photon flux over a given wavelength range.

    Parameters:
    photon_flux (list of Quantity arrays): Photon flux values for each time step.
    wavelength (Quantity array): Wavelength values corresponding to the photon flux.
    start_wavelength (Quantity): Start wavelength for integration.
    stop_wavelength (Quantity): Stop wavelength for integration.

    Returns:
    integrated_flux (Quantity array): Integrated photon flux over the specified wavelength range for each time step.
    """
    # Ensure the start and stop wavelengths are in the same units as the wavelength array
    start_wavelength = start_wavelength.to(wavelength.unit)
    stop_wavelength = stop_wavelength.to(wavelength.unit)

    # Create a mask for the wavelength range
    mask = (wavelength >= start_wavelength) & (wavelength <= stop_wavelength)

    # Integrate the photon flux over the specified wavelength range for each time step
    integrated_flux = [np.trapz(flux[mask], wavelength[mask]) for flux in photon_flux]

    return u.Quantity(integrated_flux)

def determine_baseline(photon_flux, time_iso, baseline_time):
    """
    Determine the baseline photon flux at a specific time.

    Parameters:
    photon_flux (list of Quantity arrays): The photon flux arrays.
    time_iso (Time array): The time array in ISO format.
    baseline_time (Time): The baseline time to determine the flux.

    Returns:
    Quantity: The baseline photon flux.
    """
    if not isinstance(time_iso, Time):
        time_iso = Time(time_iso)
    if not isinstance(baseline_time, Time):
        baseline_time = Time(baseline_time)

    # Find the index of the closest time to the baseline_time
    time_diff = np.abs(time_iso - baseline_time)
    baseline_idx = np.argmin(time_diff)
    
    # Find the median of all the photon flux values from the series start up to the baseline_idx
    baseline_flux = np.median(photon_flux[:baseline_idx+1], axis=0)
    
    return baseline_flux

fig, axes = plt.subplots(3, 3, figsize=(18, 12))

for idx, file in enumerate(filenames):
    file_path = data_path + file
    hours, time_iso, photon_flux, wavelength = process_file(file_path)
    
    single_171 = integrate_photon_flux_over_wavelength(photon_flux, wavelength, 170.1 * u.AA, 172.1 * u.AA)
    single_177 = integrate_photon_flux_over_wavelength(photon_flux, wavelength, 176.2 * u.AA, 178.2 * u.AA)
    single_180 = integrate_photon_flux_over_wavelength(photon_flux, wavelength, 179.4 * u.AA, 181.4 * u.AA)
    single_195 = integrate_photon_flux_over_wavelength(photon_flux, wavelength, 194.1 * u.AA, 196.1 * u.AA)
    single_202 = integrate_photon_flux_over_wavelength(photon_flux, wavelength, 201.0 * u.AA, 203.0 * u.AA)
    single_211 = integrate_photon_flux_over_wavelength(photon_flux, wavelength, 210.3 * u.AA, 212.3 * u.AA)
    single_368 = integrate_photon_flux_over_wavelength(photon_flux, wavelength, 367.1 * u.AA, 369.1 * u.AA)

    combo_171_177_180_195_202 = single_171 + single_177 + single_180 + single_195 + single_202
    combo_171_177_180_195_211 = single_171 + single_177 + single_180 + single_195 + single_211
    combo_171_177_180_195_368 = single_171 + single_177 + single_180 + single_195 + single_368

    band_80_180 = integrate_photon_flux_over_wavelength(photon_flux, wavelength, 80 * u.AA, 180 * u.AA)
    band_150_250 = integrate_photon_flux_over_wavelength(photon_flux, wavelength, 150 * u.AA, 250 * u.AA)
    band_100_300 = integrate_photon_flux_over_wavelength(photon_flux, wavelength, 100 * u.AA, 300 * u.AA)

    baseline_time = '2011-08-04T04:00:00'
    baseline_levels = {
        'Single Line: 171 Å': determine_baseline(single_171, time_iso, baseline_time),
        'Single Line: 177 Å': determine_baseline(single_177, time_iso, baseline_time),
        'Single Line: 180 Å': determine_baseline(single_180, time_iso, baseline_time),
        'Combo: 171+177+180+195+202 Å': determine_baseline(combo_171_177_180_195_202, time_iso, baseline_time),
        'Combo: 171+177+180+195+211 Å': determine_baseline(combo_171_177_180_195_211, time_iso, baseline_time),
        'Combo: 171+177+180+195+368 Å': determine_baseline(combo_171_177_180_195_368, time_iso, baseline_time),
        'Band: 80-180 Å': determine_baseline(band_80_180, time_iso, baseline_time),
        'Band: 150-250 Å': determine_baseline(band_150_250, time_iso, baseline_time),
        'Band: 100-300 Å': determine_baseline(band_100_300, time_iso, baseline_time)
    }

    # Define the plot data for each column
    plot_data = [
        {
            'Single Line: 171 Å': single_171,
            'Single Line: 177 Å': single_177,
            'Single Line: 180 Å': single_180
        },
        {
            'Combo: 171+177+180+195+202 Å': combo_171_177_180_195_202,
            'Combo: 171+177+180+195+211 Å': combo_171_177_180_195_211,
            'Combo: 171+177+180+195+368 Å': combo_171_177_180_195_368
        },
        {
            'Band: 80-180 Å': band_80_180,
            'Band: 150-250 Å': band_150_250,
            'Band: 100-300 Å': band_100_300
        }
    ]

    # Loop through each column and plot the corresponding data
    for col, data_group in enumerate(plot_data):
        for row, (label, data) in enumerate(data_group.items()):
            ax = axes[row, col]
            ax.plot(hours, data, label='Photon Flux', color=colors[idx], linewidth=2)
            ax.axhline(baseline_levels[label].value, color=colors[idx], linestyle='--', linewidth=1)
            ax.set_xlabel('hours since start', fontsize=18)
            ax.set_ylabel('intensity [photons s$^{-1}$ cm$^{-2}$]', fontsize=18)
            ax.set_title(f'{label}', fontsize=20)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.minorticks_on()
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.tick_params(axis='both', which='minor', labelsize=10)

# Add legend
handles = [mpatches.Patch(color=colors[i], label=rf'log$_{{10}}$N(HI) [cm$^{{-2}}$]={isms[i]}') for i in range(len(isms))]
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=len(isms))

plt.tight_layout(rect=[0, 0, 1, 0.92])
filename_png = f"{save_path}png/stellar_dimming_light_curve_examples"
filename_pdf = f"{save_path}stellar_dimming_light_curve_examples"
plt.savefig(f"{filename_png}.png", bbox_inches='tight', dpi=300)
plt.savefig(f"{filename_pdf}.pdf", bbox_inches='tight')
plt.show()

pass

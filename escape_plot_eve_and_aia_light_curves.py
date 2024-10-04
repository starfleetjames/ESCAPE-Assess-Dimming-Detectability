import numpy as np
import astropy.units as u
from astropy.constants import h, c
from astropy.time import Time
from scipy.io import readsav
import matplotlib.pyplot as plt

save_path = '/Users/masonjp2/Dropbox/Apps/Overleaf/ESCAPE Dimming Detectability Paper/figures/'
eve_data_path = '/Users/masonjp2/Dropbox/Research/Data/ESCAPE/eve_for_escape/EVE Dimming Data for ESCAPE.sav'
tmp = readsav(eve_data_path)
eve = tmp['eve']

# Break out the relevant parts
irradiance = eve['irradiance'] # [W/m2/nm]
wavelength = eve['wavelength'][0] * 10. * u.angstrom
jd = eve['jd']

# Convert time
hours = (jd - jd[0]) * 24
time_iso = Time(jd, format='jd').iso
date_of_event = '2011-08-04T00:00:00Z' # just for reference

# Convert irradiance to photons/second/cm2/angstrom
flattened_irradiance = np.concatenate(irradiance)
flattened_irradiance[flattened_irradiance == -1] = np.nan
repeated_wavelengths = np.tile(wavelength, len(irradiance))
irradiance_quantity = flattened_irradiance * (u.W / u.m**2 / u.nm)
wavelength_m = repeated_wavelengths.to(u.m)
photon_energy = (h * c / wavelength_m).to(u.J)
photon_flux = (irradiance_quantity / photon_energy).to(1 / u.s / u.m**2 / u.nm)
photon_flux *= u.photon
photon_flux = photon_flux.to(u.photon / u.s / (u.cm**2) / u.AA)
photon_flux = np.split(photon_flux, len(irradiance))

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


fig, axes = plt.subplots(3, 3, figsize=(18, 12))

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
        ax.plot(hours, data, label='Photon Flux', color='black', linewidth=1.5)
        ax.axhline(baseline_levels[label].value, color='gray', linestyle='--', label='Baseline Level', linewidth=1.5)
        ax.set_xlabel('hours since start', fontsize=14)
        ax.set_ylabel('intensity [photons s$^{-1}$ cm$^{-2}$]', fontsize=14)
        ax.set_title(f'{label}', fontsize=16)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.minorticks_on()
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=10)

plt.tight_layout()
filename = f"{save_path}eve_dimming_light_curve_examples"
plt.savefig(f"{filename}.png", bbox_inches='tight', dpi=300)
plt.savefig(f"{filename}.pdf", bbox_inches='tight')
plt.show()

pass


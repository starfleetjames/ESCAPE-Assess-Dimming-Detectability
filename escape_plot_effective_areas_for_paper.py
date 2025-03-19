import numpy as np
import astropy.units as u
from astropy.constants import h, c
from astropy.time import Time
import pandas as pd
import matplotlib.pyplot as plt
save_path = '/Users/masonjp2/Dropbox/Apps/Overleaf/ESCAPE Dimming Detectability Paper/figures/'
data_path = '/Users/masonjp2/Dropbox/Research/Data/ESCAPE/escape_dimming_detectability_exploration/'

euve_aeff = pd.read_csv(data_path + 'euve_aeff.csv')
euve_deep_aeff = pd.read_csv(data_path + 'euve_deep_aeff.csv')
escape_gold_aeff = pd.read_csv(data_path + 'escape_gold_aeff.csv')

escape_gold_aeff['effective area [cm2]'] = escape_gold_aeff['effective area [cm2]'].apply(lambda x: np.nan if x < 0 else x)

fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(escape_gold_aeff['wavelength [Å]'], escape_gold_aeff['effective area [cm2]'], label='ESCAPE', color='black', linewidth=2)
ax.plot(euve_aeff['wavelength [Å]'], euve_aeff['effective area [cm2]'], label='EUVE DS/S spectrometers', color='tomato', linewidth=2, linestyle='--')
ax.plot(euve_deep_aeff['wavelength [Å]'], euve_deep_aeff['effective area [cm2]'], label='EUVE DS/S imager', color='dodgerblue', linewidth=2, linestyle='-.')

ax.set_xlabel('wavelength [Å]', fontsize=22)
ax.set_ylabel('effective area [cm²]', fontsize=22)
ax.set_yscale('log')
ax.set_xlim([100, 1000])
ax.set_ylim(bottom=1e-2)
ax.legend(fontsize=18, loc='upper right', bbox_to_anchor=(1, 1))
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.tick_params(axis='both', which='minor', labelsize=18)

# Add a shaded area for the O I + He I airglow
ax.axvspan(565, 600, color='grey', alpha=0.5, hatch='//', edgecolor='darkgrey')
mid_wavelength = (565 + 600) / 2
ax.text(mid_wavelength, ax.get_ylim()[1], 'O I + He I Earth atmospheric airglow', rotation=90, verticalalignment='top', horizontalalignment='center', fontsize=18, color='black')

filename_png = f"{save_path}png/effective_area_comparison"
filename_pdf = f"{save_path}effective_area_comparison"
plt.savefig(f"{filename_png}.png", bbox_inches='tight', dpi=300)
plt.savefig(f"{filename_pdf}.pdf", bbox_inches='tight')
plt.show()

pass


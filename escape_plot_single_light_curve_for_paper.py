from scipy.io import readsav
import matplotlib.pyplot as plt
import numpy as np

def setup_matplotlib_defaults():
    """Set default plotting parameters for publication quality figures"""
    plt.rc('font', size=20)          # controls default text sizes
    plt.rc('axes', titlesize=22)     # fontsize of the axes title
    plt.rc('axes', labelsize=22)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=20)    # fontsize of the tick labels
    plt.rc('legend', fontsize=20)    # legend fontsize
    plt.rc('lines', linewidth=3.0)   # default line width

def load_light_curve_data(file_path):
    """Load and extract light curve data from .sav file"""
    data = readsav(file_path)
    return {
        'time_hours': data['time_hours'],
        'light_curve': data['light_curve'],
        'noise': data['noise'],
        'best_detection': data['best_detection'],
        'depth_window_indices': data['depth_window_indices'],
        'preflare_window_indices': data['preflare_window_indices'],
        'preflare_baseline': data['preflare_baseline']
    }

def find_gap_locations(time_array, values_array, orbital_period_minutes, efficiency):
    """Find the locations of gaps in the time series based on orbital period"""
    orbital_period_hours = orbital_period_minutes / 60.0
    current_time = 0
    gap_times = []
    gap_values = []
    
    while current_time < np.max(time_array):
        dark_start = current_time + (orbital_period_hours * efficiency)
        dark_end = current_time + orbital_period_hours
        
        if dark_start < np.max(time_array):
            before_gap_idx = np.where(time_array <= dark_start)[0]
            after_gap_idx = np.where(time_array >= dark_end)[0]
            
            if len(before_gap_idx) > 0 and len(after_gap_idx) > 0:
                before_gap_idx = before_gap_idx[-1]
                after_gap_idx = after_gap_idx[0]
                
                gap_times.append((time_array[before_gap_idx], time_array[after_gap_idx]))
                gap_values.append((values_array[before_gap_idx], values_array[after_gap_idx]))
        
        current_time += orbital_period_hours
    
    return gap_times, gap_values

def plot_data_with_gaps(ax, time, values, color, label=None, noise=None, zorder=5):
    """Plot data with proper handling of gaps for partial efficiency"""
    if noise is not None:
        ax.errorbar(time, values, yerr=noise, fmt='-', 
                   ecolor='grey', capsize=5, elinewidth=1.0,
                   color=color, label=label)
    else:
        ax.plot(time, values, color=color, label=label, zorder=zorder)

def add_white_lines_over_gaps(ax, gap_times, gap_values):
    """Add white lines to create visual breaks in the data"""
    for (t1, t2), (v1, v2) in zip(gap_times, gap_values):
        ax.plot([t1, t2], [v1, v2], 
               '-', color='white', linewidth=3.5, zorder=6)

def add_grey_bars(ax, max_time, orbital_period_minutes, efficiency):
    """Add grey bars indicating periods of no observation"""
    ymin, ymax = ax.get_ylim()
    bar_height = (ymax - ymin) * 0.02
    bar_bottom = ymin
    
    current_time = 0
    orbital_period_hours = orbital_period_minutes / 60.0
    
    while current_time < max_time:
        dark_start = current_time + (orbital_period_hours * efficiency)
        dark_end = current_time + orbital_period_hours
        
        ax.add_patch(plt.Rectangle(
            (dark_start, bar_bottom),
            dark_end - dark_start,
            bar_height,
            facecolor='darkgrey',
            alpha=0.7,
            zorder=3
        ))
        
        current_time += orbital_period_hours
    
    return ymin, ymax

def format_plot(ax, wavelength_combo, ymin=None, ymax=None):
    """Apply final formatting to the plot"""
    ax.set_xlabel('hours since start', fontsize=22)
    ax.set_ylabel(f'{wavelength_combo}\nintensity [counts]', fontsize=22)
    ax.legend()
    ax.set_xlim(0, 25)
    
    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin, ymax)
    
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    plt.tight_layout()

def save_plot(fig, efficiency):
    """Save plot in both PDF and PNG formats"""
    save_path = '/Users/masonjp2/Dropbox/Apps/Overleaf/ESCAPE Dimming Detectability Paper/figures/'
    efficiency_str = f"_{int(efficiency*100)}percent" if efficiency < 1.0 else ""
    
    # Save PNG
    fig.savefig(f"{save_path}png/single_light_curve_with_noise{efficiency_str}.png", 
                bbox_inches='tight', dpi=300)
    
    # Save PDF
    fig.savefig(f"{save_path}single_light_curve_with_noise{efficiency_str}.pdf", 
                bbox_inches='tight')

def create_single_light_curve_plot(efficiency=1.0, orbital_period_minutes=90.0, save=True):
    """
    Create a single light curve plot with given observing efficiency
    
    Parameters:
    -----------
    efficiency : float
        Observing efficiency (0-1), default=1.0
    orbital_period_minutes : float
        Orbital period in minutes, default=90.0
    save : bool
        Whether to save the plot to disk, default=True
    """
    # Load data
    file_path = '/Users/masonjp2/Dropbox/Research/Data/ESCAPE/escape_dimming_detectability_exploration/ESCAPE Solid Gold Best Detection Light Curve.sav'
    data = load_light_curve_data(file_path)
    
    # Apply observing efficiency
    plot_time, (plot_light_curve, plot_noise) = apply_observing_efficiency(
        data['time_hours'], [data['light_curve'], data['noise']], 
        efficiency, orbital_period_minutes
    )
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot main light curve
    if efficiency < 1.0:
        gap_times, gap_values = find_gap_locations(plot_time, plot_light_curve, 
                                                 orbital_period_minutes, efficiency)
        plot_data_with_gaps(ax, plot_time, plot_light_curve, 'black', noise=plot_noise)
        add_white_lines_over_gaps(ax, gap_times, gap_values)
    else:
        plot_data_with_gaps(ax, plot_time, plot_light_curve, 'black', noise=plot_noise)
    
    # Handle preflare window
    plot_preflare_times, plot_preflare_values = apply_observing_efficiency(
        data['time_hours'][data['preflare_window_indices'][:-1]], 
        data['light_curve'][data['preflare_window_indices'][:-1]], 
        efficiency, orbital_period_minutes
    )
    
    if efficiency < 1.0:
        gap_times, gap_values = find_gap_locations(plot_preflare_times, plot_preflare_values, 
                                                 orbital_period_minutes, efficiency)
        plot_data_with_gaps(ax, plot_preflare_times, plot_preflare_values, 
                           'dodgerblue', label='used for baseline calculation', zorder=5)
        add_white_lines_over_gaps(ax, gap_times, gap_values)
    else:
        plot_data_with_gaps(ax, plot_preflare_times, plot_preflare_values, 
                           'dodgerblue', label='used for baseline calculation', zorder=5)
    
    # Add baseline
    ax.axhline(data['preflare_baseline'], color='dodgerblue', linestyle='--', 
               linewidth=3.5, label='baseline')
    
    # Handle depth window
    plot_depth_times, plot_depth_values = apply_observing_efficiency(
        data['time_hours'][data['depth_window_indices'][:-1]], 
        data['light_curve'][data['depth_window_indices'][:-1]], 
        efficiency, orbital_period_minutes
    )
    
    if efficiency < 1.0:
        gap_times, gap_values = find_gap_locations(plot_depth_times, plot_depth_values, 
                                                 orbital_period_minutes, efficiency)
        plot_data_with_gaps(ax, plot_depth_times, plot_depth_values, 
                           'tomato', label='used for depth calculation', zorder=5)
        add_white_lines_over_gaps(ax, gap_times, gap_values)
    else:
        plot_data_with_gaps(ax, plot_depth_times, plot_depth_values, 
                           'tomato', label='used for depth calculation', zorder=5)
    
    # Add grey bars and format plot
    ymin, ymax = None, None
    if efficiency < 1.0:
        ymin, ymax = add_grey_bars(ax, np.max(data['time_hours']), 
                                 orbital_period_minutes, efficiency)
    
    wavelength_combo = data['best_detection']['best_detection_wavelength_combo'][0].decode('utf-8')
    format_plot(ax, wavelength_combo, ymin, ymax)
    
    if save:
        save_plot(fig, efficiency)
    
    return fig, ax

def apply_observing_efficiency(time_hours, data_array, efficiency=1.0, orbital_period_minutes=90.0):
    """
    Apply observing efficiency by removing data points during 'dark' periods
    
    Parameters:
    -----------
    time_hours : array-like
        Time points in hours
    data_array : array-like or list of array-like
        Data arrays to mask (light curve, noise, etc.)
    efficiency : float
        Observing efficiency (0-1)
    orbital_period_minutes : float
        Orbital period in minutes
    
    Returns:
    --------
    tuple : (masked_time, masked_data_arrays)
        Arrays with data points removed during 'dark' periods
    """
    if efficiency >= 1.0:
        if isinstance(data_array, list):
            return time_hours, data_array
        return time_hours, data_array
    
    # Convert orbital period to hours
    orbital_period_hours = orbital_period_minutes / 60.0
    
    # Calculate dark time per orbit
    dark_time = orbital_period_hours * (1 - efficiency)
    
    # Create a mask for the observable periods
    mask = np.ones_like(time_hours, dtype=bool)
    
    # For each orbit
    current_orbit = 0
    while current_orbit * orbital_period_hours < np.max(time_hours):
        orbit_start = current_orbit * orbital_period_hours
        dark_start = orbit_start + (orbital_period_hours * efficiency)
        dark_end = orbit_start + orbital_period_hours
        
        # Mask out the dark period
        dark_period = (time_hours >= dark_start) & (time_hours < dark_end)
        mask[dark_period] = False
        
        current_orbit += 1
    
    # Apply mask to all data arrays
    if isinstance(data_array, list):
        masked_arrays = [arr[mask] for arr in data_array]
        return time_hours[mask], masked_arrays
    
    return time_hours[mask], data_array[mask]

if __name__ == "__main__":
    setup_matplotlib_defaults()
    create_single_light_curve_plot(efficiency=1.0)
    create_single_light_curve_plot(efficiency=0.78)
    plt.show()

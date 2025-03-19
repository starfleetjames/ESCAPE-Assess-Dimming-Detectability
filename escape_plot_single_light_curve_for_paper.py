from scipy.io import readsav
import matplotlib.pyplot as plt
import numpy as np

# Set font sizes for publication
plt.rc('font', size=20)          # controls default text sizes
plt.rc('axes', titlesize=22)     # fontsize of the axes title
plt.rc('axes', labelsize=22)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
plt.rc('ytick', labelsize=20)    # fontsize of the tick labels
plt.rc('legend', fontsize=20)    # legend fontsize

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

def create_single_light_curve_plot(efficiency=1.0, orbital_period_minutes=90.0):
    """
    Create a single light curve plot with given observing efficiency
    
    Parameters:
    -----------
    efficiency : float
        Observing efficiency (0-1), default=1.0
    orbital_period_minutes : float
        Orbital period in minutes, default=90.0
    """
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

    # Apply observing efficiency to the main light curve and noise
    plot_time, (plot_light_curve, plot_noise) = apply_observing_efficiency(
        time_hours, [light_curve, noise], efficiency, orbital_period_minutes
    )

    # Extract and decode the best detection wavelength combo
    best_detection_wavelength_combo = best_detection['best_detection_wavelength_combo'][0].decode('utf-8')

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    if efficiency < 1.0:
        # For efficiency < 100%, use thinner lines
        main_linewidth = 2.0
        dotted_linewidth = 0.8
        
        # Split the data into segments
        time_gaps = np.where(np.diff(plot_time) > orbital_period_minutes/60.0/2)[0]
        time_segments = np.split(plot_time, time_gaps + 1)
        lc_segments = np.split(plot_light_curve, time_gaps + 1)
        noise_segments = np.split(plot_noise, time_gaps + 1)
        
        # Plot first segment with label
        ax.errorbar(time_segments[0], lc_segments[0], yerr=noise_segments[0], fmt='-', 
                   ecolor='grey', capsize=5, elinewidth=1.0, linewidth=main_linewidth, 
                   color='black', label='')
        
        # Plot remaining segments without labels
        for t_seg, lc_seg, noise_seg in zip(time_segments[1:], lc_segments[1:], noise_segments[1:]):
            ax.errorbar(t_seg, lc_seg, yerr=noise_seg, fmt='-', 
                       ecolor='grey', capsize=5, elinewidth=1.0, linewidth=main_linewidth, 
                       color='black')
            
        # Connect segments with dotted lines
        for i in range(len(time_segments)-1):
            ax.plot([time_segments[i][-1], time_segments[i+1][0]], 
                   [lc_segments[i][-1], lc_segments[i+1][0]], 
                   ':', color='black', linewidth=dotted_linewidth)
    else:
        # Original plotting for 100% efficiency
        ax.errorbar(plot_time, plot_light_curve, yerr=plot_noise, fmt='-', 
                   ecolor='grey', capsize=5, elinewidth=1.0, linewidth=3.1, 
                   color='black', label='')

    # Apply efficiency to preflare window data
    preflare_times = time_hours[preflare_window_indices[:-1]]
    preflare_values = light_curve[preflare_window_indices[:-1]]
    plot_preflare_times, plot_preflare_values = apply_observing_efficiency(
        preflare_times, preflare_values, efficiency, orbital_period_minutes
    )

    if efficiency < 1.0:
        # Handle preflare window with gaps
        time_gaps = np.where(np.diff(plot_preflare_times) > orbital_period_minutes/60.0/2)[0]
        time_segments = np.split(plot_preflare_times, time_gaps + 1)
        value_segments = np.split(plot_preflare_values, time_gaps + 1)
        
        # Plot first segment with label
        ax.plot(time_segments[0], value_segments[0], 
                color='dodgerblue', linewidth=main_linewidth, 
                label='used for baseline calculation', zorder=5)
        
        # Plot remaining segments without labels
        for t_seg, v_seg in zip(time_segments[1:], value_segments[1:]):
            ax.plot(t_seg, v_seg, color='dodgerblue', linewidth=main_linewidth, zorder=5)
        
        # Connect segments with dotted lines
        for i in range(len(time_segments)-1):
            ax.plot([time_segments[i][-1], time_segments[i+1][0]], 
                   [value_segments[i][-1], value_segments[i+1][0]], 
                   ':', color='dodgerblue', linewidth=dotted_linewidth, zorder=5)
    else:
        ax.plot(plot_preflare_times, plot_preflare_values, 
                color='dodgerblue', linewidth=3.0, 
                label='used for baseline calculation', zorder=5)

    # Add baseline line
    ax.axhline(preflare_baseline, color='dodgerblue', linestyle='--', 
               linewidth=3.5, label='baseline')

    # Apply efficiency to depth window data and plot with gaps if needed
    depth_times = time_hours[depth_window_indices[:-1]]
    depth_values = light_curve[depth_window_indices[:-1]]
    plot_depth_times, plot_depth_values = apply_observing_efficiency(
        depth_times, depth_values, efficiency, orbital_period_minutes
    )

    if efficiency < 1.0:
        # Handle depth window with gaps
        time_gaps = np.where(np.diff(plot_depth_times) > orbital_period_minutes/60.0/2)[0]
        time_segments = np.split(plot_depth_times, time_gaps + 1)
        value_segments = np.split(plot_depth_values, time_gaps + 1)
        
        # Plot first segment with label
        ax.plot(time_segments[0], value_segments[0], 
                color='tomato', linewidth=main_linewidth, 
                label='used for depth calculation', zorder=5)
        
        # Plot remaining segments without labels
        for t_seg, v_seg in zip(time_segments[1:], value_segments[1:]):
            ax.plot(t_seg, v_seg, color='tomato', linewidth=main_linewidth, zorder=5)
        
        # Connect segments with dotted lines
        for i in range(len(time_segments)-1):
            ax.plot([time_segments[i][-1], time_segments[i+1][0]], 
                   [value_segments[i][-1], value_segments[i+1][0]], 
                   ':', color='tomato', linewidth=dotted_linewidth, zorder=5)
    else:
        ax.plot(plot_depth_times, plot_depth_values, 
                color='tomato', linewidth=3.1, 
                label='used for depth calculation', zorder=5)

    # Print the time_hours of depth_window_indices and calculate the delta
    print("Depth window times:", plot_depth_times)
    if len(plot_depth_times) > 1:
        delta = plot_depth_times[-1] - plot_depth_times[0]
        print("Delta between first and last depth window time:", delta)

    if efficiency < 1.0:
        # Add grey bars for missing data periods
        ymin, ymax = ax.get_ylim()
        bar_height = (ymax - ymin) * 0.02  # Height of the bar will be 2% of y-axis range
        bar_bottom = ymin  # Place at bottom of plot
        
        # Calculate the dark periods based on orbital period
        current_time = 0
        while current_time < np.max(time_hours):
            # Each orbit's dark period
            dark_start = current_time + (orbital_period_minutes/60.0 * efficiency)
            dark_end = current_time + (orbital_period_minutes/60.0)
            
            # Add grey rectangle for dark period
            ax.add_patch(plt.Rectangle(
                (dark_start, bar_bottom),  # (x, y)
                dark_end - dark_start,     # width
                bar_height,                # height
                facecolor='darkgrey',
                alpha=0.7,
                zorder=3
            ))
            
            current_time += orbital_period_minutes/60.0

    # Add labels
    ax.set_xlabel('hours since start', fontsize=22)
    ax.set_ylabel(f'{best_detection_wavelength_combo}\nintensity [counts]', fontsize=22)
    ax.legend()

    ax.set_xlim(0, 25)

    # If we added grey bars, adjust y-axis limit to show them
    if efficiency < 1.0:
        ax.set_ylim(ymin, ymax)

    # Tick formatting
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    plt.tight_layout()

    # Save with efficiency in filename if not 100%
    save_path = '/Users/masonjp2/Dropbox/Apps/Overleaf/ESCAPE Dimming Detectability Paper/figures/'
    efficiency_str = f"_{int(efficiency*100)}percent" if efficiency < 1.0 else ""
    
    plt.savefig(f"{save_path}png/single_light_curve_with_noise{efficiency_str}.png", 
                bbox_inches='tight', dpi=300)
    plt.savefig(f"{save_path}single_light_curve_with_noise{efficiency_str}.pdf", 
                bbox_inches='tight')
    plt.show()

# Example usage
if __name__ == "__main__":
    # Create plot with 100% efficiency (default)
    create_single_light_curve_plot()
    
    # Create plot with 80% efficiency
    create_single_light_curve_plot(efficiency=0.78)

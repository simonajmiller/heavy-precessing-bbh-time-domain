# import basics
import os, sys, tqdm, re, math
import numpy as np
from functools import reduce
import scipy.signal as sig

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import seaborn as sns
cp = sns.color_palette('muted')
cp2 = sns.color_palette('pastel')

'''
DEFINE CONSTANTS
'''

# make sure all full page figs are the same width 
DEFAULT_FIG_WIDTH = 15/1.3

# parameters we want to look at and their associate plotting limits
params = {
    'mtotal':'M', 
    'q':'q', 
    'chi_eff':'\chi_{\mathrm{eff}}',
    'chi_p':'\chi_p'
}
ymaxes = {
    'mtotal':0.04, 
    'q':4.5, 
    'chi_eff':3,
    'chi_p':2.5
}
xlims = {
    'mtotal':[200,320], 
    'q':[0.18, 1], 
    'chi_eff':[-0.6, 0.6],
    'chi_p':[0, 1]
}

# translate between cutoff cycles and cutoff times for the different mass runs
cutoff_cycles_all = ['-2.5', '-2.0', '-1.5', '-1.0', '-0.5', '0.0', '0.5', '1.0', '1.5', '2.0', '2.5', '3.0', '3.5']
cutoff_times_80 = ['-0.032', '-0.022', '-0.015', '-0.009', '-0.005', '0.0', '0.002', '0.005', '0.007', '0.009', '0.011']
cutoff_times_100 = ['-0.039', '-0.028', '-0.018', '-0.011', '-0.006', '0.0', '0.003', '0.007', '0.009', '0.012', '0.014', '0.018', '0.02']
cutoff_times_120 = ['-0.047', '-0.033', '-0.022', '-0.013', '-0.007', '-0.001', '0.003', '0.008', '0.011', '0.014', '0.017', '0.021', '0.025']
cutoff_times_140 = ['-0.055', '-0.038', '-0.026', '-0.015', '-0.008', '-0.001', '0.004', '0.009', '0.012', '0.017', '0.02', '0.025', '0.029']
cutoff_times_170  = ['-0.067', '-0.047', '-0.031', '-0.018', '-0.01', '-0.001', '0.005', '0.011', '0.015', '0.02', '0.025', '0.03', '0.034']
cutoff_times_220  = ['-0.086', '-0.06', '-0.04', '-0.024', '-0.012', '-0.001', '0.006', '0.014', '0.02', '0.026', '0.032', '0.039', '0.045']
cutoff_times_270  = ['-0.106', '-0.074', '-0.05', '-0.029', '-0.015', '-0.001', '0.008', '0.017', '0.024', '0.032', '0.039', '0.048', '0.055']
cutoff_times_320  = ['-0.125', '-0.088', '-0.059', '-0.034', '-0.018', '-0.001', '0.009', '0.02', '0.029', '0.038', '0.047', '0.057']
cutoff_times_400 = ['-0.157', '-0.11', '-0.074', '-0.043', '-0.023', '-0.002', '0.011', '0.025', '0.036', '0.048', '0.058', '0.071', '0.081']
cutoff_times_500 = ['-0.196', '-0.137', '-0.092', '-0.054', '-0.028', '-0.002', '0.014', '0.031', '0.045', '0.059', '0.073', '0.089', '0.102']
cutoff_times_600 = ['-0.235', '-0.165', '-0.11', '-0.065', '-0.034', '-0.003', '0.017', '0.038', '0.053', '0.071', '0.088', '0.106', '0.122']
cutoff_times_all  = {
    x:l for x, l in zip(
        [80, 100, 120, 140, 170, 220, 270, 320, 400, 500, 600],
        [cutoff_times_80, cutoff_times_100, cutoff_times_120, cutoff_times_140, cutoff_times_170, cutoff_times_220, cutoff_times_270, cutoff_times_320, 
         cutoff_times_400, cutoff_times_500, cutoff_times_600]
    )
}
cycles_to_times_dict = {
    f'total mass {mass}':{x1:x2 for x1,x2 in zip(cutoff_cycles_all,cutoff_times_all[mass])} for mass in cutoff_times_all.keys()
}



'''
PLOTTING FUNCTIONS
'''

def add_legend(fig, handle_lw=4, **legend_kws): 
    leg = fig.legend(**legend_kws)
    for i, h in enumerate(legend_kws['handles']):
        leg.get_lines()[i].set_linewidth(handle_lw)

def custom_axes(nrows, ncols_per_row):
    
    # function to get the lowest common multiple of a list of numbers
    def lcm_of_list(numbers):
        def calc_lcm(a, b):
            return abs(a * b) // math.gcd(a, b)
        return reduce(calc_lcm, numbers)

    # define over-all shape of the grid
    lcm = lcm_of_list(ncols_per_row) # lowest common multiple of the ncols_per_row
    shape = (nrows, lcm)
    
    axes = []
    
    # cycle through the rows
    for row in np.arange(nrows): 
        axes_row = []
        
        # get number of columns for this row 
        ncols = ncols_per_row[row]
        
        # how many subplots each column takes up in this row
        span = int(lcm/ncols)
        
        # cycle through the columns in this row
        for col in np.arange(ncols): 
            
            # this column's position in the grid
            position = (row, int(span*col))
            ax = plt.subplot2grid(shape, position, colspan=span)
            
            axes_row.append(ax)
        axes.append(axes_row) 
        
    return axes

def plot_posterior(ax, posterior, xlims, ymax, param_label, ylabel=None, **kws): 
    
    ax.hist(posterior, **kws)
    ax.set_xlabel(fr'${param_label}$')
    if ylabel is None:
        ax.set_ylabel(fr'$p({param_label})$')
    else: 
        ax.set_ylabel(ylabel)
    ax.grid(':', color='silver', alpha=0.5)
    ax.set_xlim(*xlims)
    ax.set_ylim(0, ymax)

def plot_posteriors_and_waveform(
    posteriors_dict, time_cuts, params_to_plot, true_params, ymaxes, plotting_kws, strain_data_dict, 
    ifo='L1', unit='s', prior_dict=None, JSD_dict=None, figsize=None, dxs=None, custom_xticks=None
):
        
    n_rows = len(time_cuts) + 1
    n_cols = len(params_to_plot.keys()) + 1
    
    # Make figure 
    if figsize is None: 
        figsize = (DEFAULT_FIG_WIDTH, 2.5*n_rows/1.3)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    for j,c in enumerate(['full', *time_cuts]):
                
        ## plot waveform 
        
        ax = axes[j][0]
        
        times = transform_times(strain_data_dict['time_dict'][ifo],  true_params[f'{ifo}_time'])
        times_M = transform_s_to_M(times, true_params['mtotal'])
        
        ax.plot(times_M, strain_data_dict['strain_wh'][ifo], color='silver', lw=0.75)
        ax.plot(times_M, strain_data_dict['data_dict_wh'][ifo], color='k')
                
        set_limits_and_labels_of_whitened_wf_plot(ax, unit=unit)
        
        # shade in the different parts (pre/post cutoff)
        if not isinstance(c, str):
            tc = get_tcut_from_Ncycles(strain_data_dict['data_dict'][ifo], times, c)
            tc_M = transform_s_to_M(tc, true_params['mtotal'])
            ax.axvline(x=tc_M, color='k', ls='--', alpha=0.9)
            ax.fill_betweenx(ax.get_ylim(), tc_M, ax.get_xlim()[1], color=cp[1], alpha=0.2, zorder=0)
            ax.fill_betweenx(ax.get_ylim(), ax.get_xlim()[0], tc_M, color=cp[0], alpha=0.2, zorder=0)
            
            # add label 
            tc_str = fr'$t_{{\rm cut}} = {np.round(tc_M,1)} \, {unit}$'
            ax.text(0.05, 0.94, tc_str, transform=ax.transAxes, verticalalignment='top', horizontalalignment='left', 
                   bbox=dict(facecolor='white', edgecolor='silver', boxstyle='round', alpha=0.7), fontsize=11)
        
        if j == n_rows - 1:
            ax.set_xlabel(fr'$t~[{unit}]$')
        else: 
            ax.set_xlabel('')
            ax.set_xticklabels([])
        
        ## plot posteriors 
        
        for i, p in enumerate(params_to_plot.keys()): 
            
            ax = axes[j][i+1]
            
            # fetch prior samples
            if prior_dict is None:
                prior_samps = posteriors_dict[list(posteriors_dict.keys())[0]]['prior'][p]
            else: 
                prior_samps = prior_dict[p]
            
            # get bounds and bins for histogram
            if p=='mtotal': 
                bounds = [210, 350]
            else:
                bounds = [min(prior_samps), max(prior_samps)]
            bins = np.linspace(*bounds, 30)

            ax.hist(prior_samps, **plotting_kws['prior'], bins=bins)
            
            for name in posteriors_dict.keys(): 
            
                if c=='full':
                    
                    # plot full histogram
                    full_samps = posteriors_dict[name]['full'][p]
                    ax.hist(full_samps, **plotting_kws[f'{name} full'], bins=bins)
                    
                    # add JSD text if desired
                    if JSD_dict is not None:
                        JSD_str = f'JSD = {format_in_scientific_notation(JSD_dict["full"][p])}'
                        
                        if p=='mtotal': 
                            ax.text(0.97, 0.95, JSD_str, transform=ax.transAxes, verticalalignment='top',
                                    horizontalalignment='right', fontsize=10.5)
                        else: 
                            ax.text(0.03, 0.95, JSD_str, transform=ax.transAxes, verticalalignment='top', 
                                    horizontalalignment='left', fontsize=10.5)
                        
                
                else:
                    for mode in ['pre', 'post']: 

                        run_key = f'{mode} {c}cycles' 
                        
                        # plot pre/post histograms
                        if run_key in posteriors_dict[name].keys():
                            samps = posteriors_dict[name][run_key][p]
                        else: 
                            samps = np.random.choice(prior_samps, size=5000)

                        ax.hist(samps, **plotting_kws[f'{name} {mode}'], bins=bins)
                        
                    # add JSD text if desired
                    if JSD_dict is not None:
                        
                        # pre
                        JSD_str_pre = f"JSD = {format_in_scientific_notation(JSD_dict[f'pre {c}cycles'][p])}"
                        color_pre = plotting_kws['maxL pre']['color']
                        
                        if p=='mtotal': 
                            ax.text(0.97, 0.95, JSD_str_pre, transform=ax.transAxes, verticalalignment='top',
                                    horizontalalignment='right', fontsize=10.5, color=color_pre)
                        else: 
                            ax.text(0.03, 0.95, JSD_str_pre, transform=ax.transAxes, verticalalignment='top', 
                                    horizontalalignment='left', fontsize=10.5, color=color_pre)
                            
                        # post
                        JSD_str_post = f"JSD = {format_in_scientific_notation(JSD_dict[f'post {c}cycles'][p])}"
                        color_post = plotting_kws['maxL post']['color']
                        
                        if p=='mtotal': 
                            ax.text(0.97, 0.83, JSD_str_post, transform=ax.transAxes, verticalalignment='top',
                                    horizontalalignment='right', fontsize=10.5, color=color_post)
                        else: 
                            ax.text(0.03, 0.83, JSD_str_post, transform=ax.transAxes, verticalalignment='top', 
                                    horizontalalignment='left', fontsize=10.5, color=color_post)

            # injected value
            ax.axvline(true_params[p], color=cp[3], label=r'max $\mathcal{L}$ injection: truth')     

            # format axes
            ax.set_xlim(*bounds)
            ax.set_ylim(0, ymaxes[p])
            
            if custom_xticks is not None: 
                ax.set_xticks(custom_xticks[p])
                    
            if j == n_rows - 1:
                ax.set_xlabel(fr'${params_to_plot[p]}$')
            else: 
                ax.set_xticklabels([])
            if i==0:
                ax.set_ylabel('Probability\ndensity')
                
            ax.grid(':', color='silver', alpha=0.5)

    # space the axes accordingly
    plt.subplots_adjust(wspace=0.4, hspace=0.1)
    
    for j,c in enumerate(['full', *time_cuts]): 
        dx = -0.03
        x0, y0, x1, y1 = axes[j][0].get_position().bounds
        axes[j][0].set_position([x0+dx, y0, x1, y1])
        
        if dxs is None: 
            dxs = [0.04, 0.02]
        for ii, dx in zip(range(1, n_cols), dxs): 
            x0, y0, x1, y1 = axes[j][ii].get_position().bounds
            axes[j][ii].set_position([x0+dx, y0, x1, y1])


    return fig, axes


def set_limits_and_labels_of_whitened_wf_plot(ax, unit='s', ifo='LLO'): 
    
    ax.set_ylim(-2.7, 2.8)
    if unit=='s':
        ax.set_xlim(-0.06, 0.075)
    elif unit=='M': 
        ax.set_xlim(-55,65)
    else: 
        print('Invalid `unit` given to `set_limits_and_labels_of_whitened_wf_plot`.')
        sys.exit()
    ax.set_ylabel(fr'$\hat{{h}}_{{\rm {ifo}}}~[\sigma]$')
    ax.set_xlabel(fr'$t~[{unit}]$')
    

    
'''
MISC FUNCTIONS
'''

def transform_times(times, ref_t): 
    return np.asarray(times) - ref_t

def transform_s_to_M(time_in_seconds, M): 
    mass_scaling = 4.925491025543576e-06 * M
    times_in_M = time_in_seconds / mass_scaling
    return times_in_M

def format_in_scientific_notation(num): 
    A, b = f"{num:.0e}".split("e")
    return fr"${A} \times 10^{{{int(b)}}}$"
    
def get_tcut_from_Ncycles(h_ifo, times, Ncycles):
    
    h_ifo = np.asarray(h_ifo)
    
    # Get indices of extrema 
    idxs, _ = sig.find_peaks(np.abs(h_ifo), height=0)
    
    # Get times of extrema 
    t_cycles_ifo = times[idxs]
    
    # Get the cycle we care about
    i0 = np.argmax(np.abs(h_ifo[idxs])) # index corresponding to tcut=0 (absolute peak time)
    n_i = 2 * Ncycles                    # one index = 1/2 cycle

    # If the desired cycle cut is at a peak/trough ...
    if isinstance(Ncycles, int) or (isinstance(Ncycles, float) and n_i.is_integer()):
        
        icut = i0 + int(n_i)           # index corresponding to the cycle we care about

        # Get time in H1
        tcut_ifo = t_cycles_ifo[icut]
    
    # Otherwise, linearly interpolate between nearest peak and trough  
    else: 
        # Our desired cut sits between these two times 
        tcut_ifo_min = t_cycles_ifo[i0 + int(np.floor(n_i))]
        tcut_ifo_max = t_cycles_ifo[i0 + int(np.ceil(n_i))]

        # How far between the extrema?
        frac_between = n_i - np.floor(n_i)

        # Interpolate
        tcut_ifo = tcut_ifo_min + frac_between*(tcut_ifo_max - tcut_ifo_min)
    
    return tcut_ifo

def get_unique_times(strings):
    ## get the unique cutoff times and their unit from a set of results

    times = set()
    for s in strings:
        # Split the string to get the time part, assuming format is consistent
        parts = s.split(' ')
        if len(parts) > 1:  # Ignore strings that do not have time information
            
            # extract the unit from the string
            time_with_unit = parts[1]       
            unit = re.sub(r'[0-9]', '', time_with_unit).replace('.','').replace('-','')
            
            # extract the time value from the string
            time_value = float(time_with_unit[:-len(unit)])
            times.add(time_value)
            
    # return the sorted times and their unit
    return sorted(times), unit

def get_key(x, parameter_name): 
    if parameter_name == 'phase': 
        return f'phase {x}'
    elif parameter_name == 'psi':
        if isinstance(x, str):
            return f'pol {x}'
        else:
            return f'pol {x:.2f}'
    elif parameter_name == 'iota': 
        return f'iota {x}'
    
def get_color(x, parameter_name, cmap):
    if parameter_name == 'psi' or parameter_name == 'pol':
        p = float(x) * np.pi 
    else: 
        p = float(x)
    return cmap(p/np.pi)

def fit_line_to_data(x, y, grid_x=None):
    """
    Fits a line to the given dataset using least squares regression and returns the best fit line over a grid.
    
    Parameters:
        x (array-like): Independent variable data points.
        y (array-like): Dependent variable data points.
        grid_x (array-like, optional): Grid of x values to evaluate the best fit line.
                                       If None, a grid is generated based on min and max of x.
    
    Returns:
        grid_x (numpy array): The x values of the grid.
        best_fit_y (numpy array): The corresponding y values of the best fit line.
    """
    # Convert to numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Compute least squares solution
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]  # Solve for slope (m) and intercept (c)
    
    # Define grid if not provided
    if grid_x is None:
        grid_x = np.linspace(x.min(), x.max(), 100)
    
    # Compute best fit line
    best_fit_y = m * grid_x + c
    
    return grid_x, best_fit_y
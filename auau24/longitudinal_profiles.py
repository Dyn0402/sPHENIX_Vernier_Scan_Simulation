#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 15 11:20 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/longitudinal_profiles

@author: Dylan Neff, dn277127
"""

import platform
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as cf
from scipy.signal import find_peaks

from plot_cad_measurements import gaus_pdf
from Measure import Measure


def main():
    if platform.system() == 'Windows':
        base_path = 'C:/Users/Dylan/Desktop/'
    else:
        base_path = '/local/home/dn277127/Bureau/'
    cad_measurements_path = f'{base_path}vernier_scan_AuAu24/CAD_Measurements/'

    # min_time, max_time = 30, 75
    min_time, max_time = 0, 106
    # min_val = 1.5
    min_val = 0.2
    fit_range = [35, 70]
    # max_pdf_val = 0.125 if vernier_scan_date == 'Aug12' else 0.1
    beam_colors = ['blue', 'yellow']
    plot_colors = ['blue', 'orange']
    # beam_colors = ['blue']
    # plot_colors = ['blue']
    # sufx = '_21:33'
    sufx = '_22'
    peak_list = define_peaks()

    write_out = True
    p0 = None

    fig, ax = plt.subplots(figsize=(12, 6))
    fig2, ax2 = plt.subplots(figsize=(12, 6))

    for beam_color, plot_color in zip(beam_colors, plot_colors):
        file_path = f'{cad_measurements_path}VernierScan_AuAu_longitudinal_{beam_color}{sufx}.dat'
        fit_out_path = f'{cad_measurements_path}VernierScan_AuAu_{beam_color}_longitudinal_fit{sufx}.dat'
        with open(file_path, 'r') as f:
            file_content = f.read()
        lines = file_content.split('\n')
        times, values = [[]], [[]]
        for line in lines[1:]:
            if line == '':
                continue
            columns = line.split('\t')
            time, value = float(columns[0]), float(columns[1])
            if len(times[-1]) > 0 and time < times[-1][-1]:
                times.append([])
                values.append([])
            times[-1].append(time)
            values[-1].append(value)

        # For times less than 30 ns and greater than 75 ns set to 0
        for bunch_i, (bunch_times, bunch_vals) in enumerate(zip(times, values)):
            bunch_vals = np.array(bunch_vals)
            bunch_vals[(np.array(bunch_times) < min_time) | (np.array(bunch_times) > max_time)] = 0
            values[bunch_i] = bunch_vals

        full_time = 0
        for bunch_i, (bunch_times, bunch_vals) in enumerate(zip(times, values)):
            bunch_vals = np.array(bunch_vals)
            max_bunch_val = np.max(bunch_vals)
            times_increasing = np.array(bunch_times)
            times_increasing += full_time
            full_time += bunch_times[-1]
            if max_bunch_val > min_val:
                ax.plot(bunch_times, bunch_vals / np.max(bunch_vals), color=plot_color)
            ax2.plot(times_increasing, bunch_vals, color=plot_color)

        # Fit all the bunches superimposed
        fit_times, fit_vals, weird_one_times, weird_one_vals, weird_ones = [], [], [], [], 0
        for bunch_i, (bunch_times, bunch_vals) in enumerate(zip(times, values)):
            if np.max(bunch_vals) < min_val:
                continue
            bin_width = bunch_times[1] - bunch_times[0]
            bunch_times, bunch_vals = np.array(bunch_times), np.array(bunch_vals)
            fit_mask = (bunch_times > fit_range[0]) & (bunch_times < fit_range[1])
            vals = bunch_vals[fit_mask] / np.sum(bunch_vals[fit_mask]) / bin_width
            fit_times.extend(list(bunch_times[fit_mask]))
            fit_vals.extend(list(vals))

            # if p0 is None:  # Get guess from first blue bunch
                # p0, bounds, peak_xs = make_initial_guess_from_data(np.array(bunch_times), np.array(vals))
                # p0, bounds, peak_xs = make_initial_guess_from_expectation(np.array(bunch_times), np.array(vals))
            p0, bounds = build_manual_initial_guess(peak_list[beam_color])

        print(f'{beam_color} {weird_ones} weird ones of {len(times)} bunches')

        popt, pcov = cf(multi_gaus_pdf, fit_times, fit_vals, p0=p0, bounds=bounds)
        perr = np.sqrt(np.diag(pcov))
        pmeas = [Measure(p, e) for p, e in zip(popt, perr)]

        fig_all, ax_all = plt.subplots(figsize=(8, 6))
        x_plot = np.linspace(fit_times[0], fit_times[-1], 1000)
        ax_all.plot(fit_times, fit_vals, color=plot_color, alpha=0.01, ls='none', marker='.', label='CAD Profiles')
        ax_all.plot(x_plot, multi_gaus_pdf(x_plot, *p0), color='green', ls='-', label='Guess')
        ax_all.plot(x_plot, p0[-3] / (1 + np.sum(p0[2::3])) * gaus_pdf(x_plot, *p0[-2:]), color='green', ls='--')
        ax_all.plot(x_plot, multi_gaus_pdf(x_plot, *popt), color='red', ls='-', label='Fit')
        ax_all.plot(x_plot, popt[-3] / (1 + np.sum(popt[2::3])) * gaus_pdf(x_plot, *popt[-2:]), color='red', ls='--')
        ax_all.set_xlabel('Time (ns)')
        ax_all.set_ylabel('Probability Density')
        ax_all.set_title(
            f'AuAu24 Vernier Scan {beam_color.capitalize()} Beam Longitudinal Bunch Density')
        ax_all.set_xlim(min_time, max_time)
        ax_all.legend(loc='upper right', fontsize=14)
        ax_all.grid(True)
        ax_all.set_xlim(30, 75)
        fig_all.tight_layout()

        if write_out:  # Write out fit parameters
            # fig_all.savefig(fit_plots_out_path)
            write_longitudinal_beam_profile_fit_parameters(fit_out_path, beam_color, pmeas)

    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Value')
    ax.set_title('Longitudinal Beam Measurements vs Time')
    ax.grid(True)

    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('Value')
    ax2.set_title('Longitudinal Beam Measurements vs Index')
    ax2.grid(True)

    fig.tight_layout()
    fig2.tight_layout()

    plt.show()

    print('donzo')


def define_peaks():
    spacing = 3.3
    sigma = 0.8
    center_blue = 52.55
    center_yellow = 52.36
    manual_peaks = {
        'blue' : [
            {'a': 1.0, 'mu': center_blue, 'sigma': sigma},
            {'a': 0.2, 'mu': center_blue + spacing + 0.1, 'sigma': sigma},
            {'a': 0.2, 'mu': center_blue - spacing - 0.3, 'sigma': sigma},
            {'a': 0.2, 'mu': center_blue + 2 * spacing - 0.3, 'sigma': sigma},
            {'a': 0.2, 'mu': center_blue - 2 * spacing + 0.25, 'sigma': sigma},
            {'a': 0.05, 'mu': center_blue + 3 * spacing - 1.2, 'sigma': sigma},
            {'a': 0.05, 'mu': center_blue - 3 * spacing + 1.0, 'sigma': sigma},
            {'a': 0.05, 'mu': center_blue + 4 * spacing - 1.8, 'sigma': sigma},
            {'a': 0.05, 'mu': center_blue - 4 * spacing + 1.5, 'sigma': sigma},
            # {'a': 0.00, 'mu': center_blue, 'sigma': sigma * 5},
        ],
        'yellow' : [
            {'a': 1.0, 'mu': center_yellow, 'sigma': sigma},
            {'a': 0.2, 'mu': center_yellow + spacing, 'sigma': sigma},
            {'a': 0.2, 'mu': center_yellow - spacing - 0.4, 'sigma': sigma},
            {'a': 0.2, 'mu': center_yellow + 2 * spacing, 'sigma': sigma},
            {'a': 0.2, 'mu': center_yellow - 2 * spacing + 0.5, 'sigma': sigma},
            {'a': 0.05, 'mu': center_yellow + 3 * spacing - 1, 'sigma': sigma},
            {'a': 0.05, 'mu': center_yellow - 3 * spacing + 1.5, 'sigma': sigma},
            {'a': 0.05, 'mu': center_yellow + 4 * spacing - 1.5, 'sigma': sigma},
            {'a': 0.05, 'mu': center_yellow - 4 * spacing + 1.8, 'sigma': sigma},
            {'a': 0.05, 'mu': center_yellow - 4 * spacing + 1.8, 'sigma': sigma},
            # {'a': 0.2, 'mu': center_yellow, 'sigma': sigma * 5},
        ]
    }

    return manual_peaks


def multi_gaus_pdf(x, *params):
    """
    Sum of N Gaussians, where the first Gaussian has fixed amplitude = 1
    and the rest have variable amplitudes.

    Parameters layout:
    [b1, c1, a2, b2, c2, a3, b3, c3, ..., aN, bN, cN]

    Returns:
        Sum of Gaussians normalized so that total amplitude = 1.
    """
    # First Gaussian (amplitude implicitly 1)
    b1, c1 = params[0], params[1]
    y = gaus_pdf(x, b1, c1)
    total_amp = 1.0

    # Remaining Gaussians
    i = 2
    while i + 2 < len(params):
        a = params[i]
        b = params[i + 1]
        c = params[i + 2]
        y += a * gaus_pdf(x, b, c)
        total_amp += a
        i += 3

    return y / total_amp


def make_initial_guess_from_data(x_data, y_data, n_peaks=8, sigma_guess=1.2):
    # Find all peaks
    peaks, properties = find_peaks(y_data, prominence=0.01)

    # Sort peaks by prominence (highest first)
    prominences = properties["prominences"]
    top_peak_indices = np.array(np.argsort(prominences)[-n_peaks:])
    top_peaks = peaks[top_peak_indices]
    peak_xs = x_data[top_peaks]
    peak_heights = y_data[top_peaks]

    # Sort peak_xs largest to smallest
    peak_heights, peak_xs = zip(*sorted(zip(peak_heights, peak_xs), reverse=True))

    # Generate initial guess parameters
    print(f'peak_xs[0]: {peak_xs[0]}, peak_heights[0]: {peak_heights[0]}')
    p0 = [peak_xs[0], sigma_guess]  # b1, c1
    bounds_lower = [peak_xs[0] - 4, 0]
    bounds_upper = [peak_xs[0] + 4, 10]

    for b, h in zip(peak_xs[1:], peak_heights[1:]):
        a = h / peak_heights[0]  # Amplitude
        c = np.abs(sigma_guess)
        p0 += [a, b, c]
        bounds_lower += [0, b - 4, 0]
        bounds_upper += [np.inf, b + 4, 10]

        for i, (p, l, u) in enumerate(zip(p0, bounds_lower, bounds_upper)):  # Check if any p0 outside bounds
            if p < l or p > u:
                print(f'p0[{i}] = {p} is outside bounds [{l}, {u}]')

    # Add final underlying wide Gaussian
    p0 += [0, peak_xs[0], 5]
    bounds_lower += [0, peak_xs[0] - 4, 0]
    bounds_upper += [np.inf, peak_xs[0] + 4, 20]

    return p0, (bounds_lower, bounds_upper), peak_xs


def make_initial_guess_from_expectation(x_data, y_data, n_peaks=8, sigma_guess=1.2):
    # Find all peaks
    peaks, properties = find_peaks(y_data, prominence=0.01)

    # Sort peaks by prominence (highest first)
    prominences = properties["prominences"]
    top_peak_indices = np.array(np.argsort(prominences)[-n_peaks:])
    top_peaks = peaks[top_peak_indices]
    peak_xs = x_data[top_peaks]
    peak_heights = y_data[top_peaks]

    # Sort peak_xs largest to smallest
    peak_heights, peak_xs = zip(*sorted(zip(peak_heights, peak_xs), reverse=True))

    # Generate initial guess parameters
    print(f'peak_xs[0]: {peak_xs[0]}, peak_heights[0]: {peak_heights[0]}')
    p0 = [peak_xs[0], sigma_guess]  # b1, c1
    bounds_lower = [peak_xs[0] - 4, 0]
    bounds_upper = [peak_xs[0] + 4, 10]

    for i in [-4, -3, -2, -1, 1, 2, 3, 4]:
        a = peak_heights[0] / 4  # Amplitude
        c = sigma_guess
        b = peak_xs[0] + i * 3.2
        p0 += [a, b, c]
        bounds_lower += [0, b - 4, 0]
        bounds_upper += [np.inf, b + 4, 10]

        for i, (p, l, u) in enumerate(zip(p0, bounds_lower, bounds_upper)):  # Check if any p0 outside bounds
            if p < l or p > u:
                print(f'p0[{i}] = {p} is outside bounds [{l}, {u}]')

    # Add final underlying wide Gaussian
    p0 += [peak_heights[0] / 20, peak_xs[0], 5]
    bounds_lower += [0, peak_xs[0] - 4, 0]
    bounds_upper += [np.inf, peak_xs[0] + 4, 20]

    return p0, (bounds_lower, bounds_upper), peak_xs


def build_manual_initial_guess(peak_list):
    """
    peak_list: list of dictionaries like:
        [{'a': 1.0, 'mu': 52.3, 'sigma': 1.0}, ...]
        First peak doesn't use 'a' (it's absorbed into normalization)
    """
    mu_range = 3
    p0 = [peak_list[0]['mu'], peak_list[0]['sigma']]
    bounds_lower = [peak_list[0]['mu'] - mu_range, 0]
    bounds_upper = [peak_list[0]['mu'] + mu_range, 5]

    for peak in peak_list[1:]:
        a = peak.get('a', 1.0)
        mu = peak['mu']
        sigma = peak['sigma']
        p0 += [a, mu, sigma]
        bounds_lower += [0, mu - mu_range, 0]
        bounds_upper += [np.inf, mu + mu_range, 5]

    # bounds_upper[-1] = 50  # Last sigma is wide

    return p0, (bounds_lower, bounds_upper)



def write_longitudinal_beam_profile_fit_parameters(fit_out_path, beam_color, fit_parameters):
    """
    Write longitudinal fit parameters to a file and include the dynamic LaTeX equation.
    """
    c = 299792458. * 1e6 / 1e9  # um/ns Speed of light

    # Infer number of Gaussians from parameter count
    n_params = len(fit_parameters)
    assert (n_params - 2) % 3 == 0, "Fit parameters do not match expected format"
    n_gaussians = 1 + (n_params - 2) // 3

    # Generate dynamic fit equation string
    fit_eq = generate_fit_equation_string(n_gaussians)

    # Create parameter name list
    param_names = ['mu1', 'sigma1']
    for i in range(2, n_gaussians + 1):
        param_names.extend([f'a{i}', f'mu{i}', f'sigma{i}'])

    with open(fit_out_path, 'w') as file:
        file.write(f'Fit Parameters for {beam_color.capitalize()} Beam Longitudinal Profile\n')
        file.write(f'Fit Equation: {fit_eq}\n')
        file.write(f'Fit Parameters:\n')
        mu1_val = fit_parameters[0].val
        for param, meas in zip(param_names, fit_parameters):
            val = meas.val
            if 'mu' in param:
                val -= mu1_val
            if 'mu' in param or 'sigma' in param:
                val *= c  # Convert from ns to um
            file.write(f'{param}: {val:.6g}\n')



def generate_fit_equation_string(n_gaussians):
    """
    Dynamically generate the LaTeX fit equation for n Gaussians.
    """
    terms = []
    for i in range(1, n_gaussians + 1):
        if i == 1:
            term = r'\frac{1}{\sigma_1 \sqrt{2 \pi}} \exp\left(-\frac{(t - \mu_1)^2}{2\sigma_1^2}\right)'
        else:
            term = (fr'\frac{{a_{i}}}{{\sigma_{i} \sqrt{{2 \pi}}}}'
                    fr' \exp\left(-\frac{{(t - \mu_{i})^2}}{{2\sigma_{i}^2}}\right)')
        terms.append(term)
    numerator = ' + '.join(terms)
    denominator = '1' + ''.join([f' + a_{i}' for i in range(2, n_gaussians + 1)])
    return fr'$p(t) = \frac{{{numerator}}}{{{denominator}}}$'



if __name__ == '__main__':
    main()

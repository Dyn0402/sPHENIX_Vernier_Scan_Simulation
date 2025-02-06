#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on February 04 3:16 PM 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/phenix_style_hourglass_analysis.py

@author: Dylan Neff, Dylan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit as cf

from BunchCollider import BunchCollider
from Measure import Measure
from vernier_z_vertex_fitting import read_cad_measurement_file, get_cw_rates, get_mbd_z_dists


def main():
    base_path = '/local/home/dn277127/Bureau/vernier_scan/'
    # base_path = '/home/dylan/Desktop/vernier_scan/'
    # base_path = 'C:/Users/Dylan/Desktop/vernier_scan/'
    simulate_vernier_scan(base_path)
    print('donzo')


def simulate_vernier_scan(base_path):
    run_scan = False  # If, false, read in data from file

    beta_star_actual = 85  # cm
    beta_star_off = 9e11  # cm Effectively turned off
    # beta_star = 9e11  # cm
    bunch_width = 135.  # microns Transverse Gaussian bunch width
    # bunch_width = 200.  # microns Transverse Gaussian bunch width

    # mbd_resolution = 2.0  # cm MBD resolution
    mbd_resolution = 0.0  # cm MBD resolution
    # gauss_eff_width = 500  # cm Gaussian efficiency width
    gauss_eff_width = None  # cm Gaussian efficiency width
    crossing_angles = False  # If True, use crossing angles from CAD data, if False set to exactly head on
    # bkg = 0.4e-16  # Background level
    bkg = 0.0  # Background level

    fit_func = vernier_scan_fit_bkg
    # fit_func = vernier_scan_fit_nobkg
    n_params = fit_func.__code__.co_argcount - 1  # Get number of parameters in fit function
    fit_pars_dict = {'p0': [1, 150, 0], 'names': ['Amp', 'Sigma', 'X0']}
    if n_params == 4:
        fit_pars_dict['p0'].append(0)
        fit_pars_dict['names'].append('B')

    vernier_scan_date = 'Aug12'
    scan_orientation = 'Horizontal'
    cad_measurement_path = f'{base_path}CAD_Measurements/VernierScan_{vernier_scan_date}_combined.dat'
    longitudinal_fit_path = f'{base_path}CAD_Measurements/VernierScan_{vernier_scan_date}_COLOR_longitudinal_fit.dat'

    beta_stars = {'On': beta_star_actual, 'Off': beta_star_off}

    if run_scan:
        collider_sim = BunchCollider()
        collider_sim.set_bunch_rs(np.array([0., 0., -6.e6]), np.array([0., 0., +6.e6]))
        collider_sim.set_bunch_sigmas(np.array([bunch_width, bunch_width]), np.array([bunch_width, bunch_width]))
        collider_sim.set_gaus_smearing_sigma(mbd_resolution)
        collider_sim.set_gaus_z_efficiency_width(gauss_eff_width)
        collider_sim.set_bkg(bkg)
        blue_fit_path = longitudinal_fit_path.replace('_COLOR_', '_blue_')
        yellow_fit_path = longitudinal_fit_path.replace('_COLOR_', '_yellow_')
        collider_sim.set_longitudinal_fit_parameters_from_file(blue_fit_path, yellow_fit_path)

        cad_data = read_cad_measurement_file(cad_measurement_path)
        cad_data_orientation = cad_data[cad_data['orientation'] == scan_orientation]

        data_dict = {'Offsets': []}
        data_dict.update({beta_star_name: [] for beta_star_name in beta_stars.keys()})
        for row_i, step_cad_data in cad_data_orientation.iterrows():
            print(f'Starting Step {row_i + 1} of {len(cad_data_orientation)}')
            offset = step_cad_data['offset_set_val'] * 1e3  # mm to um
            if scan_orientation == 'Horizontal':
                collider_sim.set_bunch_offsets(np.array([offset, 0.]), np.array([0., 0.]))
            elif scan_orientation == 'Vertical':
                collider_sim.set_bunch_offsets(np.array([0., offset]), np.array([0., 0.]))

            if crossing_angles:
                blue_angle_x, yellow_angle_x = -step_cad_data['bh8_avg'] / 1e3, -step_cad_data['yh8_avg'] / 1e3
            else:
                blue_angle_x, yellow_angle_x = 0., 0.
            collider_sim.set_bunch_crossing(blue_angle_x, 0., yellow_angle_x, 0.0)
            print(f'Offset: {offset}, Blue Angle X: {blue_angle_x * 1e3:.3f} mrad, Yellow Angle X: {yellow_angle_x * 1e3:.3f} mrad')

            data_dict['Offsets'].append(offset)
            for beta_star_name, beta_star in beta_stars.items():
                print(f'Starting {beta_star_name}')
                collider_sim.set_bunch_beta_stars(beta_star, beta_star)
                collider_sim.run_sim_parallel()
                lumi = collider_sim.get_naked_luminosity()
                data_dict[beta_star_name].append(lumi)


        # Write data_dict to file
        data_dict['Offsets'] = np.array(data_dict['Offsets'])
        for beta_star_name in beta_stars.keys():
            data_dict[beta_star_name] = np.array(data_dict[beta_star_name])
        data_df = pd.DataFrame(data_dict)
        data_df.to_csv(f'{base_path}simulated_phenix_scan/VernierScan_{vernier_scan_date}_{scan_orientation}.csv', index=False)

    if not run_scan:
        data_df = pd.read_csv(f'{base_path}simulated_phenix_scan/VernierScan_{vernier_scan_date}_{scan_orientation}.csv')

    fig_together, ax_together = plt.subplots(figsize=(8, 4))
    fig_2col, ax_2col = plt.subplots(1, 2, figsize=(10, 4), sharey='all')

    offsets = np.array(data_df['Offsets'])
    fit_results = {beta_star_name: {} for beta_star_name in beta_stars.keys()}
    for i, beta_star_name in enumerate(beta_stars.keys()):
        lumis = np.array(data_df[beta_star_name])  # Fit to vernier_scan fit
        p0 = fit_pars_dict['p0']
        p0[0] = max(lumis)
        popt, pcov = cf(fit_func, offsets, lumis, p0=p0)
        popt[1] = abs(popt[1])  # Ensure sigma is positive
        perr = np.sqrt(np.diag(pcov))
        # For perrs that are inf, set to 0
        perr[np.isinf(perr)] = 0
        print(popt)
        print(perr)
        pmeas = [Measure(popt[i], perr[i]) for i in range(len(popt))]
        for fit_par_i, fit_par_name in enumerate(fit_pars_dict['names']):
            fit_results[beta_star_name][fit_par_name] = pmeas[fit_par_i]

        x_fit = np.linspace(min(offsets), max(offsets), 1000)

        fig, ax = plt.subplots()
        ax.scatter(offsets, lumis, label=beta_star_name)
        ax.plot(x_fit, fit_func(x_fit, *popt), color='red', label='Fit')
        fit_str = f'Hourglass {beta_star_name} Fit\n'
        for fit_par_name, fit_par_meas in zip(fit_pars_dict['names'], pmeas):
            fit_str += f'{fit_par_name}: {fit_par_meas}\n'
        fit_str = fit_str.strip()
        ax.annotate(fit_str, (0.05, 0.95), xycoords='axes fraction', ha='left', va='top', fontsize=12,
                    bbox=dict(facecolor='wheat', alpha=0.5, boxstyle='round'))
        ax.set_ylim(bottom=0)
        ax.legend()
        ax.set_xlabel('Offset [μm]')
        ax.set_ylabel('Naked Luminosity')
        ax.set_title(f'Vernier Scan Hourglass {beta_star_name}')
        fig.tight_layout()

        line, = ax_together.plot(offsets, lumis, label=f'{beta_star_name} Data', marker='o', ls='none', zorder=10)
        color = line.get_color()
        ax_together.plot(x_fit, fit_func(x_fit, *popt), color=color, zorder=5, label=f'{beta_star_name} Fit')
        ax_together.annotate(fit_str, (0.02, 0.97 - i * 0.35), xycoords='axes fraction', ha='left', va='top',
                             fontsize=11, bbox=dict(facecolor=color, alpha=0.2, boxstyle='round'))

        ax_2col[i].plot(offsets, lumis, color=color, label=f'{beta_star_name} Data', marker='o', ls='none', zorder=10)
        ax_2col[i].plot(x_fit, fit_func(x_fit, *popt), color=color, label=f'{beta_star_name} Fit', zorder=5)
        ax_2col[i].set_xlabel('Offset [μm]')
        if i == 0:
            ax_2col[i].set_ylabel('Naked Luminosity')
        ax_2col[i].legend()

    ax_together.set_ylim(bottom=0)
    ax_together.set_xlabel('Offset [μm]')
    ax_together.set_ylabel('Naked Luminosity')
    ax_together.legend()
    fig_together.tight_layout()

    ax_2col[0].set_ylim(bottom=0)
    fig_2col.tight_layout()
    fig_2col.subplots_adjust(wspace=0.0)

    # Neglecting hourglass overestimates luminosity. Need to multiply by hourglass on / off, smaller than 1
    beta_star_amp_ratio = fit_results['On']['Amp'] / fit_results['Off']['Amp']  # On / Off

    # Neglecting hourglass overestimates σ, therefore underestimates lumi. Need to multiply by (on / off)**2, > 1
    beta_star_sigma_ratio = fit_results['On']['Sigma'] / fit_results['Off']['Sigma']  # On / Off

    lumi_correction = beta_star_amp_ratio * beta_star_sigma_ratio ** 2

    print(f'Ratio of beta star amps: {beta_star_amp_ratio}')
    print(f'Ratio of beta star sigmas: {beta_star_sigma_ratio}')
    print(f'Luminosity Correction: {lumi_correction}')
    print(f'Percent Correction: {(lumi_correction - 1) * 100}%')
    plt.show()


def vernier_scan_fit_bkg(x, a, sigma, x0, b):
    return a * np.exp(-(x - x0) ** 2 / (4 * sigma ** 2)) + b


def vernier_scan_fit_nobkg(x, a, sigma, x0):
    return a * np.exp(-(x - x0) ** 2 / (4 * sigma ** 2))


if __name__ == '__main__':
    main()

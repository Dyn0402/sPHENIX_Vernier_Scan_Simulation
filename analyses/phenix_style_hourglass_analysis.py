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
from scipy.optimize import curve_fit as cf

from BunchCollider import BunchCollider
from Measure import Measure
from vernier_z_vertex_fitting import read_cad_measurement_file, get_cw_rates, get_mbd_z_dists


def main():
    # base_path = '/local/home/dn277127/Bureau/vernier_scan/'
    # base_path = '/home/dylan/Desktop/vernier_scan/'
    base_path = 'C:/Users/Dylan/Desktop/vernier_scan/'
    simulate_vernier_scan(base_path)
    print('donzo')


def simulate_vernier_scan(base_path):
    x_offsets = np.linspace(-1., 1., 9) * 1e3  # mm to microns

    beta_star_actual = 85  # cm
    beta_star_off = 9e11  # cm Effectively turned off
    # beta_star = 9e11  # cm
    bunch_width = 135.  # microns Transverse Gaussian bunch width
    # bunch_width = 200.  # microns Transverse Gaussian bunch width

    mbd_resolution = 2.0  # cm MBD resolution
    gauss_eff_width = 500  # cm Gaussian efficiency width
    # bkg = 0.4e-16  # Background level
    bkg = 0.0  # Background level

    vernier_scan_date = 'Aug12'
    scan_orientation = 'Horizontal'
    cad_measurement_path = f'{base_path}CAD_Measurements/VernierScan_{vernier_scan_date}_combined.dat'
    longitudinal_fit_path = f'{base_path}CAD_Measurements/VernierScan_{vernier_scan_date}_COLOR_longitudinal_fit.dat'

    collider_sim = BunchCollider()
    collider_sim.set_bunch_rs(np.array([0., 0., -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_sigmas(np.array([bunch_width, bunch_width]), np.array([bunch_width, bunch_width]))
    collider_sim.set_gaus_smearing_sigma(mbd_resolution)
    collider_sim.set_gaus_z_efficiency_width(gauss_eff_width)
    collider_sim.set_bkg(bkg)
    blue_fit_path = longitudinal_fit_path.replace('_COLOR_', '_blue_')
    yellow_fit_path = longitudinal_fit_path.replace('_COLOR_', '_yellow_')
    collider_sim.set_longitudinal_fit_parameters_from_file(blue_fit_path, yellow_fit_path)

    beta_stars = {'On': beta_star_actual, 'Off': beta_star_off}

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

        blue_angle_x, yellow_angle_x = -step_cad_data['bh8_avg'] / 1e3, -step_cad_data['yh8_avg'] / 1e3
        collider_sim.set_bunch_crossing(blue_angle_x, 0., yellow_angle_x, 0.0)
        print(f'Offset: {offset}, Blue Angle X: {blue_angle_x * 1e3:.3f} mrad, Yellow Angle X: {yellow_angle_x * 1e3:.3f} mrad')

        data_dict['Offsets'].append(offset)
        for beta_star_name, beta_star in beta_stars.items():
            print(f'Starting {beta_star_name}')
            collider_sim.set_bunch_beta_stars(beta_star, beta_star)
            collider_sim.run_sim_parallel()
            lumi = collider_sim.get_naked_luminosity()
            data_dict[beta_star_name].append(lumi)


    offsets = data_dict['Offsets']
    fit_results = {beta_star_name: {} for beta_star_name in beta_stars.keys()}
    for beta_star_name in beta_stars.keys():
        # Fit to vernier_scan fit
        lumis = data_dict[beta_star_name]
        p0 = [max(lumis), 150, 0, 0]
        popt, pcov = cf(vernier_scan_fit, offsets, lumis, p0=p0)
        popt[1] = abs(popt[1])  # Ensure sigma is positive
        perr = np.sqrt(np.diag(pcov))
        pmeas = [Measure(popt[i], perr[i]) for i in range(len(popt))]
        fit_results[beta_star_name]['amp'] = pmeas[0]
        fit_results[beta_star_name]['sigma'] = pmeas[1]
        fit_results[beta_star_name]['x0'] = pmeas[2]
        fit_results[beta_star_name]['b'] = pmeas[3]

        x_fit = np.linspace(min(offsets), max(offsets), 1000)

        fig, ax = plt.subplots()
        ax.scatter(offsets, lumis, label=beta_star_name)
        ax.plot(x_fit, vernier_scan_fit(x_fit, *popt), color='red', label='Fit')
        fit_str = f'Amp: {str(pmeas[0])}\nSigma: {str(pmeas[1])} μm\nX0: {str(pmeas[2])} μm\nB: {str(pmeas[3])}'
        ax.annotate(fit_str, (0.05, 0.95), xycoords='axes fraction', ha='left', va='top', fontsize=12,
                    bbox=dict(facecolor='wheat', alpha=0.5, boxstyle='round,pad=1'))
        ax.legend()
        ax.set_xlabel('Offset [μm]')
        ax.set_ylabel('Naked Luminosity')
        ax.set_title(f'Vernier Scan Hourglass {beta_star_name}')
        fig.tight_layout()

    # Neglecting hourglass overestimates luminosity. Need to multiply by hourglass on / off, smaller than 1
    beta_star_amp_ratio = fit_results['On']['amp'] / fit_results['Off']['amp']  # On / Off

    # Neglecting hourglass overestimates σ, therefore underestimates lumi. Need to multiply by (on / off)**2, > 1
    beta_star_sigma_ratio = fit_results['On']['sigma'] / fit_results['Off']['amp']  # On / Off

    lumi_correction = beta_star_amp_ratio * beta_star_sigma_ratio ** 2

    print(f'Ratio of beta star amps: {beta_star_amp_ratio}')
    print(f'Ratio of beta star sigmas: {beta_star_sigma_ratio}')
    print(f'Luminosity Correction: {lumi_correction}')
    print(f'Percent Correction: {(lumi_correction - 1) * 100}%')
    plt.show()


def vernier_scan_fit(x, a, sigma, x0, b):
    return a * np.exp(-(x - x0) ** 2 / (4 * sigma ** 2)) + b


if __name__ == '__main__':
    main()

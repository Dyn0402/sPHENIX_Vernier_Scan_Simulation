#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 20 8:43 PM 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/head_on_crossing_angle_dep.py

@author: Dylan Neff, Dylan
"""

import numpy as np
import matplotlib.pyplot as plt

from BunchCollider import BunchCollider
from analyze_combined_lumi_data import read_max_rate, get_nblue_nyellow
from vernier_z_vertex_fitting import read_cad_measurement_file
from luminosity_calculation_rcf import get_bw_from_beta_star, get_bw_beta_star_fit_params


def main():
    vernier_scan_date = 'Aug12'
    # vernier_scan_date = 'Jul11'
    longitudinal_fit_path = f'../CAD_Measurements/VernierScan_{vernier_scan_date}_COLOR_longitudinal_fit.dat'

    max_rate_path = '../max_rate.txt'
    max_rate = read_max_rate(max_rate_path)
    print(max_rate)
    n_bunch = 111
    max_rate_per_bunch = max_rate / n_bunch

    cad_measurement_path = '../CAD_Measurements/VernierScan_Aug12_combined.dat'
    cad_data = read_cad_measurement_file(cad_measurement_path)

    f_beam = 78.4  # kHz
    n_blue, n_yellow = get_nblue_nyellow(cad_data, orientation='Horizontal', step=1, n_bunch=n_bunch)  # n_protons
    print(f'N Blue: {n_blue:.2e}, N Yellow: {n_yellow:.2e}')
    mb_to_um2 = 1e-19

    scan_date = 'Aug12'
    beta_star_bw_fit_path = f'../run_rcf_jobs/output/{scan_date}/bw_opt_vs_beta_star_fits.txt'
    bw_beta_star_fit_params = get_bw_beta_star_fit_params(beta_star_bw_fit_path)

    # Important parameters
    measured_beta_star = np.array([97., 82., 88., 95.])
    # beam_width = np.array([161.79, 157.08])

    bw_x = get_bw_from_beta_star(90, bw_beta_star_fit_params['x'])
    bw_y = get_bw_from_beta_star(90, bw_beta_star_fit_params['y'])
    beam_width = np.array([bw_x, bw_y])

    blue_x_offset, blue_y_offset = 0.0, 0.0  # um
    # blue_x_angle, blue_y_angle, yellow_x_angle, yellow_y_angle = -0.07e-3, 0.0, -0.115e-3, 0.0
    blue_x_angle, blue_y_angle, yellow_x_angle, yellow_y_angle = 0.0, +0.14e-3, 0.0, -0.07e-3  # Major bug!
    blue_len_scaling, yellow_len_scaling = 0.993863022403956, 0.991593955543314

    collider_sim = BunchCollider()
    collider_sim.set_bunch_rs(np.array([blue_x_offset, blue_y_offset, -6.e6]), np.array([0., 0., +6.e6]))
    # collider_sim.set_bunch_crossing(blue_x_angle, blue_y_angle, yellow_x_angle, yellow_y_angle)
    blue_fit_path = longitudinal_fit_path.replace('_COLOR_', '_blue_')
    yellow_fit_path = longitudinal_fit_path.replace('_COLOR_', '_yellow_')
    collider_sim.set_longitudinal_fit_parameters_from_file(blue_fit_path, yellow_fit_path)
    # collider_sim.set_longitudinal_fit_scaling(blue_len_scaling, yellow_len_scaling)

    collider_sim.set_bunch_beta_stars(*measured_beta_star)
    collider_sim.set_bunch_sigmas(beam_width, beam_width)

    # yellow_x_angles = np.linspace(-0.07e-3, -0.3e-3, 20)
    yellow_y_angles = np.linspace(-0.07e-3, +0.14e-3, 20)
    lumis, cross_sections, relative_angles = [], [], []
    # for yellow_x_angle in yellow_x_angles:
    for yellow_y_angle in yellow_y_angles:

        collider_sim.set_bunch_crossing(blue_x_angle, blue_y_angle, yellow_x_angle, yellow_y_angle)

        collider_sim.run_sim_parallel()
        naked_luminosity = collider_sim.get_naked_luminosity()

        luminosity = naked_luminosity * mb_to_um2 * f_beam * 1e3 * n_blue * n_yellow

        # relative_crossing_angle = blue_x_angle - yellow_x_angle
        relative_crossing_angle = blue_y_angle - yellow_y_angle

        print(f'Relative angle: {relative_crossing_angle}')
        print(f'Naked Luminosity: {naked_luminosity}')
        print(f'Luminosity: {luminosity}')

        cross_section = max_rate_per_bunch / luminosity
        print(f'Cross Section: {cross_section}')
        lumis.append(luminosity)
        cross_sections.append(cross_section)
        relative_angles.append(relative_crossing_angle)

    relative_angles = np.array(relative_angles) * 1e3

    # Plot lumis and cross sections (on separate y-axes) vs relative angles
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.errorbar(relative_angles, lumis, yerr=None, fmt='o-', label='Luminosity')
    ax2.errorbar(relative_angles, [cs.val for cs in cross_sections], yerr=[cs.err for cs in cross_sections], fmt='o-', label='Cross Section', color='red')
    ax1.set_xlabel('Relative Angle [mrad]')
    ax1.set_ylabel('Luminosity [$mb^{-1} s^{-1}$]')
    ax2.set_ylabel('Cross Section (mb)')
    ax1.legend(loc='upper center')
    ax2.legend(loc='lower center')
    fig.tight_layout()

    plt.show()


    print('donzo')


if __name__ == '__main__':
    main()

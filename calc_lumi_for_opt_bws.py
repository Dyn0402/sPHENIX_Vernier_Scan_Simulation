#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on February 20 09:47 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/calc_lumi_for_opt_bws

@author: Dylan Neff, dn277127
"""

import platform
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from BunchCollider import BunchCollider
from luminosity_calculation_rcf import get_bw_beta_star_fit_params, get_bw_from_beta_star


def main():
    scan_date = 'Aug12'
    beta_star_bw_fit_path = f'run_rcf_jobs/output/{scan_date}/bw_opt_vs_beta_star_fits.txt'
    if platform.system() == 'Linux':
        save_path = '/local/home/dn277127/Bureau/vernier_scan/presentation/analysis_note/Cross_Section/'
    else:  # Windows
        save_path = 'C:/Users/Dylan/OneDrive - UCLA IT Services/Research/Saclay/sPHENIX/Vernier_Scan/Analysis_Note/Cross_Section/'

    # Get fit parameters
    bw_beta_star_fit_params = get_bw_beta_star_fit_params(beta_star_bw_fit_path)
    print(bw_beta_star_fit_params)

    # Plot lines
    beta_stars = np.linspace(80, 110, 100)
    bw_xs = get_bw_from_beta_star(beta_stars, bw_beta_star_fit_params['x'])
    bw_ys = get_bw_from_beta_star(beta_stars, bw_beta_star_fit_params['y'])
    fig, ax = plt.subplots()
    ax.plot(beta_stars, bw_xs, label='X')
    ax.plot(beta_stars, bw_ys, label='Y')
    ax.set_title('Beam Width vs Beta Star')
    ax.set_xlabel('Beta Star [cm]')
    ax.set_ylabel('Beam Width [cm]')
    ax.legend()
    fig.tight_layout()

    # Lumi calc beta stars
    lumi_calc_beta_stars = [85.0, 90.0, 95.0, 100.0, 105.0, 110.0]

    longitudinal_fit_path = f'CAD_Measurements/VernierScan_{scan_date}_COLOR_longitudinal_fit.dat'

    # Important parameters
    measured_beta_star = np.array([97., 82., 88., 95.])

    blue_x_offset, blue_y_offset, offset_err = 0.0, 0.0, 2.0  # um
    blue_x_angle, blue_y_angle, yellow_x_angle, yellow_y_angle, angle_err = 0.0, +0.14e-3, 0.0, -0.07e-3, 0.05e-3

    # Gaussian approximation with bad beam width luminosity
    f_beam = 78.4  # kHz
    n_blue = 1.636e11  # n_protons
    n_yellow = 1.1e11  # n_protons
    machine_lumi = 317.1524  # mb^-1 s^-1  Corrected machine luminosity from Gaus approx

    # Convert machine lumi to naked lumi in um^-2 per bunch crossing
    mb_to_um2 = 1e-19
    naked_lumi_cw = machine_lumi / mb_to_um2 / (f_beam * 1e3) / n_blue / n_yellow
    print(f'Naked luminosity per bunch crossing: {naked_lumi_cw:.2e}')

    collider_sim = BunchCollider()
    collider_sim.set_bunch_rs(np.array([blue_x_offset, blue_y_offset, -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_crossing(blue_x_angle, blue_y_angle, yellow_x_angle, yellow_y_angle)
    blue_fit_path = longitudinal_fit_path.replace('_COLOR_', '_blue_')
    yellow_fit_path = longitudinal_fit_path.replace('_COLOR_', '_yellow_')
    collider_sim.set_longitudinal_fit_parameters_from_file(blue_fit_path, yellow_fit_path)

    lumis_dict = {}
    for beta_star in lumi_calc_beta_stars:
        print(f'Calculating luminosity for beta star: {beta_star:.2f} cm')
        beta_star_scale_factor_i = beta_star / 90  # Set 90 cm (average of measured) to default values, then scale from there
        beta_star_scaled_i = measured_beta_star * beta_star_scale_factor_i

        bw_x_i = get_bw_from_beta_star(beta_star, bw_beta_star_fit_params['x'])
        bw_y_i = get_bw_from_beta_star(beta_star, bw_beta_star_fit_params['y'])

        collider_sim.set_bunch_rs(np.array([blue_x_offset, blue_y_offset, -6.e6]), np.array([0., 0., +6.e6]))
        collider_sim.set_bunch_beta_stars(*beta_star_scaled_i)
        collider_sim.set_bunch_sigmas(np.array([bw_x_i, bw_y_i]), np.array([bw_x_i, bw_y_i]))
        collider_sim.set_bunch_crossing(blue_x_angle, blue_y_angle, yellow_x_angle, yellow_y_angle)

        collider_sim.run_sim_parallel()
        luminosity = collider_sim.get_naked_luminosity()
        lumis_dict.update({beta_star: luminosity})

    lumis = np.array(list(lumis_dict.values())) * 1e6  # Convert to 1/mm^2

    # Plot lumi vs beta star
    fig_lumis, ax_lumis = plt.subplots(nrows=2, figsize=(10, 5), sharex='all')
    ax_lumis[1].plot(lumi_calc_beta_stars, lumis, marker='o', markersize=10)
    ax_lumis[1].set_xlabel('Beta Star [cm]')
    ax_lumis[1].set_ylabel('Naked Luminosity [1/mm²]')
    ax_lumis[1].annotate('Zoomed on Numerical Integration', xy=(0.1, 0.2), fontsize=16, xycoords='axes fraction',
                         ha='left', va='center')

    # Plot again but with cw lumi as reference
    ax_lumis[0].plot(lumi_calc_beta_stars, lumis, marker='o', markersize=10, label='Full Numerical Integration')
    ax_lumis[0].axhline(y=naked_lumi_cw * 1e6, color='r', linestyle='--', label='Gaussian Approximation')
    ax_lumis[0].set_xlabel('Beta Star [cm]')
    ax_lumis[0].set_ylabel('Naked Luminosity [1/mm²]')
    ax_lumis[0].legend()
    fig_lumis.subplots_adjust(left=0.075, right=0.995, top=0.995, bottom=0.09, hspace=0.05)

    # Write lumis to file
    df = [
        {'beta_star': None, 'luminosity': naked_lumi_cw},
        {'beta_star': 90, 'luminosity': lumis_dict[90]},
        {'beta_star': 105, 'luminosity': lumis_dict[105]}
    ]
    df = pd.DataFrame(df)
    df.to_csv(f'lumi_vs_beta_star.csv', index=False)

    # Save plots
    fig_lumis.savefig(f'{save_path}lumi_vs_beta_star.png')
    fig_lumis.savefig(f'{save_path}lumi_vs_beta_star.pdf')

    plt.show()

    print('donzo')


if __name__ == '__main__':
    main()

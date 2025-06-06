#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 03 04:57 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/estimate_head_on_outside_of_200cm

@author: Dylan Neff, dn277127
"""

import platform
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd
import uproot

from BunchCollider import BunchCollider
from z_vertex_fitting_common import fit_amp_shift, fit_shift_only, get_profile_path


def main():
    if platform.system() == 'Windows':
        base_path = 'C:/Users/Dylan/Desktop/'
    else:
        base_path = '/local/home/dn277127/Bureau/'

    base_path_auau = f'{base_path}Vernier_Scans/auau_oct_16_24/'
    longitudinal_profiles_dir_path = f'{base_path_auau}profiles/'
    # z_vertex_zdc_data_path = f'{base_path_auau}vertex_data/54733_vertex_distributions.root'
    z_vertex_no_zdc_data_path = f'{base_path_auau}vertex_data/54733_vertex_distributions_no_zdc_coinc.root'
    combined_cad_step_data_csv_path = f'{base_path_auau}combined_cad_step_data.csv'

    cad_df = pd.read_csv(combined_cad_step_data_csv_path)

    fit_range = [-200, 200]
    step = 0

    collider_sim = BunchCollider()
    # collider_sim.set_grid_size(31, 31, 101, 31)
    collider_sim.set_grid_size(51, 51, 301, 51)
    collider_sim.set_z_bounds((-500. * 1e4, 500. * 1e4))
    bkg = 2.0e-17
    beam_width_x, beam_width_y = 130, 130  # in micrometers
    beta_star = 80
    gauss_eff_width = 500
    mbd_resolution = 2.0
    collider_sim.set_bkg(bkg)
    collider_sim.set_gaus_z_efficiency_width(gauss_eff_width)
    collider_sim.set_gaus_smearing_sigma(mbd_resolution)

    with uproot.open(z_vertex_no_zdc_data_path) as f:
        hist = f[f'step_{step}']
        centers = hist.axis().centers()
        counts = hist.counts()
        count_errs = hist.errors()
        count_errs[count_errs == 0] = 1

    cad_step_row = cad_df[cad_df['step'] == step].iloc[0]

    blue_widths = np.array([beam_width_x, beam_width_y])
    yellow_widths = np.array([beam_width_x, beam_width_y])

    collider_sim.set_bunch_beta_stars(beta_star, beta_star)
    collider_sim.set_bunch_sigmas(blue_widths, yellow_widths)

    blue_angle_x = -cad_step_row['blue angle h'] * 1e-3
    blue_angle_y = -cad_step_row['blue angle v'] * 1e-3
    yellow_angle_x = -cad_step_row['yellow angle h'] * 1e-3
    yellow_angle_y = -cad_step_row['yellow angle v'] * 1e-3

    blue_offset_x, blue_offset_y = cad_step_row['set offset h'] * 1e3, cad_step_row['set offset v'] * 1e3
    yellow_offset_x, yellow_offset_y = 0, 0

    collider_sim.set_bunch_crossing(blue_angle_x, blue_angle_y, yellow_angle_x, yellow_angle_y)

    collider_sim.set_bunch_offsets(
        [blue_offset_x, blue_offset_y],
        [yellow_offset_x, yellow_offset_y]
    )

    profile_path = get_profile_path(
        longitudinal_profiles_dir_path, cad_step_row['start'], cad_step_row['end'], False
    )
    collider_sim.set_longitudinal_profiles_from_file(
        profile_path.replace('COLOR_', 'blue_'),
        profile_path.replace('COLOR_', 'yellow_')
    )

    collider_sim.run_sim_parallel()

    fit_mask = (centers >= fit_range[0]) & (centers <= fit_range[1])

    fit_amp_shift(collider_sim, counts[fit_mask], centers[fit_mask], count_errs[fit_mask])

    zs, z_dist = collider_sim.get_z_density_dist()

    # Plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(centers, counts, yerr=count_errs, fmt='o', label='MBD Data', color='blue')
    plt.plot(zs, z_dist, label='Simulated Distribution', color='orange')
    plt.xlabel('Z Vertex Position (cm)')
    plt.ylabel('Counts')
    plt.title(f'Z Vertex Distribution for Step {step}')
    plt.legend()
    plt.grid()

    plt.tight_layout()

    zs_integrate = np.linspace(-500, 500, 10000)
    z_dist_interp = interp1d(zs, z_dist, bounds_error=False, fill_value=0)
    full_z_dist_integrated = np.sum(z_dist_interp(zs_integrate))
    plus_200_mask = np.abs(zs_integrate) >= 200
    plus_200_integrated = np.sum(z_dist_interp(zs_integrate[plus_200_mask]))
    print(f'Full Z Distribution Integrated: {full_z_dist_integrated}')
    print(f'Z Distribution Integrated from 200 cm: {plus_200_integrated}')
    print(f'Percentage of Z Distribution from 200 cm: {plus_200_integrated / full_z_dist_integrated * 100:.2f}%')

    plt.annotate(f'Fraction |z| >= 200 cm: {plus_200_integrated / full_z_dist_integrated * 100:.2f}%',
                 xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, va='top', ha='left',
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))


    plt.show()


    print('donzo')


if __name__ == '__main__':
    main()

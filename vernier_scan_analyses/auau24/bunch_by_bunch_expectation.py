#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on July 01 4:23 PM 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/bunch_by_bunch_expectation.py

@author: Dylan Neff, dylan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from BunchCollider import BunchCollider
from z_vertex_fitting_common import (fit_amp_shift, fit_shift_only, get_profile_path, compute_total_chi2,
                                     load_vertex_distributions, merge_cad_rates_df, set_sim)
from Measure import Measure
from common_logistics import set_base_path


def main():
    base_path = set_base_path()
    base_path += 'Vernier_Scans/auau_oct_16_24/'
    plot_bunch_by_bunch_sum(base_path)
    print('donzo')


def plot_bunch_by_bunch_sum(base_path):
    """
    Simulate bunch-by-bunch expectations for a vernier scan and plot the results.
    :param base_path: Base path to the vernier scan data.
    :return: None
    """

    longitudinal_profiles_dir_path = f'{base_path}profiles/'
    z_vertex_zdc_data_path = f'{base_path}vertex_data/54733_vertex_distributions.root'
    combined_cad_step_data_csv_path = f'{base_path}combined_cad_step_data.csv'

    cad_df = pd.read_csv(combined_cad_step_data_csv_path)

    fit_range = [-200, 200]
    scan_step = 3

    collider_sim = BunchCollider()
    collider_sim.set_grid_size(31, 31, 101, 31)
    beam_width_x, beam_width_y = 130.0, 130.0
    beta_star = 77.15
    bkg = 0.0e-17
    gauss_eff_width = 500
    mbd_resolution = 1.0
    rate_column = 'zdc_cor_rate'

    collider_sim.set_bkg(bkg)
    collider_sim.set_gaus_z_efficiency_width(gauss_eff_width)
    collider_sim.set_gaus_smearing_sigma(mbd_resolution)
    collider_sim.set_bunch_beta_stars(beta_star, beta_star)

    # Get nominal dcct ions and emittances
    step_0 = cad_df[cad_df['step'] == 0].iloc[0]
    em_blue_horiz_nom, em_blue_vert_nom = step_0['blue_horiz_emittance'], step_0['blue_vert_emittance']
    em_yel_horiz_nom, em_yel_vert_nom = step_0['yellow_horiz_emittance'], step_0['yellow_vert_emittance']

    cad_step_row = cad_df[cad_df['step'] == scan_step].iloc[0]

    # Preload data and histograms here
    vertex_data = load_vertex_distributions(z_vertex_zdc_data_path, [scan_step], cad_df, rate_column)

    fig, ax = plt.subplots(figsize=(10, 6))
    centers, counts, count_errs = vertex_data[scan_step]
    fit_mask = (centers > fit_range[0]) & (centers < fit_range[1])
    ax.errorbar(centers[fit_mask], counts[fit_mask], yerr=count_errs[fit_mask], color='black', fmt='o',
                label='Measured Counts')

    profile_colors = ['blue', 'red', 'green', 'purple', 'orange']
    for bunch_i in range(111):
        print(f'Processing Bunch {bunch_i + 1}/111')
        profile_paths = get_profile_path(
            longitudinal_profiles_dir_path, cad_step_row['start'], cad_step_row['end'], True, bunch_i
        )
        for profile_i, profile_path in enumerate(profile_paths):
            zs, z_dist, interp_sim = fit_sim(
                collider_sim, cad_step_row, beam_width_x, beam_width_y,
                {
                    'em_blue_nom': (em_blue_horiz_nom, em_blue_vert_nom),
                    'em_yel_nom': (em_yel_horiz_nom, em_yel_vert_nom),
                    'fit_range': fit_range
                }, vertex_data, scan_step, profile_path
            )

            ax.plot(centers[fit_mask], interp_sim, color=profile_colors[profile_i], alpha=0.1)
    ax.set_xlabel('Z Vertex Position (mm)')
    ax.set_ylabel('Counts')
    ax.set_title(f'Bunch-by-Bunch Expectation for Scan Step {scan_step}')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    plt.show()


def fit_sim(collider_sim, cad_step_row, beam_width_x, beam_width_y, sim_settings, vertex_data, scan_step, profile_path):
    # em_blue_horiz, em_blue_vert = cad_step_row['blue_horiz_emittance'], cad_step_row['blue_vert_emittance']
    # em_yel_horiz, em_yel_vert = cad_step_row['yellow_horiz_emittance'], cad_step_row['yellow_vert_emittance']
    #
    # em_blue_horiz_nom, em_blue_vert_nom = sim_settings['em_blue_nom']
    # em_yel_horiz_nom, em_yel_vert_nom = sim_settings['em_yel_nom']
    #
    # blue_widths = np.array([
    #     beam_width_x * np.sqrt(em_blue_horiz / em_blue_horiz_nom),
    #     beam_width_y * np.sqrt(em_blue_vert / em_blue_vert_nom)
    # ])
    # yellow_widths = np.array([
    #     beam_width_x * np.sqrt(em_yel_horiz / em_yel_horiz_nom),
    #     beam_width_y * np.sqrt(em_yel_vert / em_yel_vert_nom)
    # ])
    #
    # collider_sim.set_bunch_sigmas(blue_widths, yellow_widths)
    #
    # blue_angle_x = -cad_step_row['blue angle h'] * 1e-3
    # blue_angle_y = -cad_step_row['blue angle v'] * 1e-3
    # yellow_angle_x = -cad_step_row['yellow angle h'] * 1e-3
    # yellow_angle_y = -cad_step_row['yellow angle v'] * 1e-3
    #
    # blue_offset_x, blue_offset_y = cad_step_row['set offset h'] * 1e3, cad_step_row['set offset v'] * 1e3
    # yellow_offset_x, yellow_offset_y = 0, 0
    #
    # collider_sim.set_bunch_crossing(blue_angle_x, blue_angle_y, yellow_angle_x, yellow_angle_y)
    #
    # collider_sim.set_bunch_offsets(
    #     [blue_offset_x, blue_offset_y],
    #     [yellow_offset_x, yellow_offset_y]
    # )
    #
    # collider_sim.set_longitudinal_profiles_from_file(
    #     profile_path.replace('COLOR_', 'blue_'),
    #     profile_path.replace('COLOR_', 'yellow_')
    # )

    set_sim(collider_sim, cad_step_row, beam_width_x, beam_width_y,
            sim_settings['em_blue_nom'], sim_settings['em_yel_nom'], profile_path)

    collider_sim.run_sim_parallel()

    centers, counts, count_errs = vertex_data[scan_step]
    fit_mask = (centers > sim_settings['fit_range'][0]) & (centers < sim_settings['fit_range'][1])

    fit_amp_shift(collider_sim, counts[fit_mask], centers[fit_mask], count_errs[fit_mask])

    zs, z_dist = collider_sim.get_z_density_dist()
    interp_sim = np.interp(centers[fit_mask], zs, z_dist)

    return zs, z_dist, interp_sim


if __name__ == '__main__':
    main()

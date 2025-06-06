#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 10 12:33 PM 2024
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/luminosity_calculation.py

@author: Dylan Neff, Dylan
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime

from BunchCollider import BunchCollider
from vernier_z_vertex_fitting_clean import create_dir
from z_vertex_fitting_common import get_profile_path


def main():
    job_num = int(sys.argv[1])  # Get system arguments, should just be an integer for the job number
    base_path = '/sphenix/u/dneffsph/gpfs/Vernier_Scans/'
    vernier_scan = 'auau_oct_16_24'
    longitudinal_fit_path = f'{base_path}{vernier_scan}/profiles/'
    combined_cad_step_data_csv_path = f'{base_path}{vernier_scan}/combined_cad_step_data.csv'
    cad_df = pd.read_csv(combined_cad_step_data_csv_path)

    scan_step = 0  # Step in the scan to use for the luminosity estimation

    output_dir = f'output/{vernier_scan}/'
    create_dir(output_dir)
    output_path = f'{output_dir}Scan_Step_{scan_step}/'
    create_dir(output_path)

    estimate_final_luminosity(longitudinal_fit_path, cad_df, job_num, output_dir)
    print('donzo')


def estimate_final_luminosity(longitudinal_profiles_path, cad_df, job_num, output_dir, scan_step=0, n_samples=500, err_estimates='best'):
    """
    Use extracted beam parameters along with their estimated uncertainties to estimate the final luminosity.
    Repeatedly sample beam parameters according to their uncertainties and calculate the luminosity for each sample.
    Generate a distribution of luminosities and calculate the mean and standard deviation.
    :param longitudinal_profiles_path:
    :param cad_df:
    :param job_num:
    :param output_dir: Directory to save the output CSV file.
    :param scan_step: Step in the scan to use for the luminosity estimation.
    :param n_samples: Number of samples to take for the luminosity estimation.
    :param err_estimates: 'best' or 'conservative' error estimates.
    :return:
    """
    # Important parameters
    beta_star = 76.7  # Use same for all 4 components
    beam_width_x, beam_width_y = 129.2, 125.7
    bkg = 0.0e-17
    gauss_eff_width = 500
    mbd_resolution = 1.0

    if err_estimates == 'best':  # Best err estimates
        bw_x_err, bw_y_err, beta_star_err = 2.1, 2.9, 1.1  # um, um, cm
        offset_err, angle_err = 10, 0.01e-3  # um, rad
    else:  # Conservative err estimates
        bw_x_err, bw_y_err, beta_star_err = 5.0, 5.0, 2.0  # um, um, cm
        offset_err, angle_err = 50, 0.05e-3  # um, rad

    collider_sim = BunchCollider()

    collider_sim.set_bkg(bkg)
    collider_sim.set_gaus_z_efficiency_width(gauss_eff_width)
    collider_sim.set_gaus_smearing_sigma(mbd_resolution)

    cad_step_row = cad_df[cad_df['step'] == scan_step].iloc[0]

    em_blue_horiz, em_blue_vert = cad_step_row['blue_horiz_emittance'], cad_step_row['blue_vert_emittance']
    em_yel_horiz, em_yel_vert = cad_step_row['yellow_horiz_emittance'], cad_step_row['yellow_vert_emittance']

    # Get nominal emittances
    step_0 = cad_df[cad_df['step'] == 0].iloc[0]
    em_blue_horiz_nom, em_blue_vert_nom = step_0['blue_horiz_emittance'], step_0['blue_vert_emittance']
    em_yel_horiz_nom, em_yel_vert_nom = step_0['yellow_horiz_emittance'], step_0['yellow_vert_emittance']

    blue_widths = np.array([
        beam_width_x * np.sqrt(em_blue_horiz / em_blue_horiz_nom),
        beam_width_y * np.sqrt(em_blue_vert / em_blue_vert_nom)
    ])
    yellow_widths = np.array([
        beam_width_x * np.sqrt(em_yel_horiz / em_yel_horiz_nom),
        beam_width_y * np.sqrt(em_yel_vert / em_yel_vert_nom)
    ])

    blue_angle_x = -cad_step_row['blue angle h'] * 1e-3
    blue_angle_y = -cad_step_row['blue angle v'] * 1e-3
    yellow_angle_x = -cad_step_row['yellow angle h'] * 1e-3
    yellow_angle_y = -cad_step_row['yellow angle v'] * 1e-3

    blue_offset_x, blue_offset_y = cad_step_row['set offset h'] * 1e3, cad_step_row['set offset v'] * 1e3
    yellow_offset_x, yellow_offset_y = 0, 0

    profile_paths = get_profile_path(
        longitudinal_profiles_path, cad_step_row['start'], cad_step_row['end'], True
    )

    sample_results = []  # Sample beam parameters
    for i in range(n_samples):
        beta_star_i = np.random.normal(beta_star, beta_star_err)

        blue_bw_x_i = np.random.normal(blue_widths[0], bw_x_err)
        blue_bw_y_i = np.random.normal(blue_widths[1], bw_y_err)
        yellow_bw_x_i = np.random.normal(yellow_widths[0], bw_x_err)
        yellow_bw_y_i = np.random.normal(yellow_widths[1], bw_y_err)

        blue_x_offset_i = np.random.normal(blue_offset_x, offset_err)
        blue_y_offset_i = np.random.normal(blue_offset_y, offset_err)
        blue_x_angle_i = np.random.normal(blue_angle_x, angle_err)
        blue_y_angle_i = np.random.normal(blue_angle_y, angle_err)
        yellow_x_angle_i = np.random.normal(yellow_angle_x, angle_err)
        yellow_y_angle_i = np.random.normal(yellow_angle_y, angle_err)

        profile_number_i = np.random.randint(len(profile_paths))
        profile_path_i = profile_paths[profile_number_i]

        collider_sim.set_bunch_beta_stars(beta_star_i, beta_star_i)
        collider_sim.set_bunch_sigmas(np.array([blue_bw_x_i, blue_bw_y_i]), np.array([yellow_bw_x_i, yellow_bw_y_i]))
        collider_sim.set_bunch_crossing(blue_x_angle_i, blue_y_angle_i, yellow_x_angle_i, yellow_y_angle_i)
        collider_sim.set_bunch_offsets([blue_x_offset_i, blue_y_offset_i], [yellow_offset_x, yellow_offset_y])
        collider_sim.set_longitudinal_profiles_from_file(
            profile_path_i.replace('COLOR_', 'blue_'),
            profile_path_i.replace('COLOR_', 'yellow_')
        )

        collider_sim.run_sim_parallel()

        luminosity = collider_sim.get_naked_luminosity()
        print(f'{datetime.now()} Sample {i + 1}/{n_samples} luminosity: {luminosity:.2e}')
        sample_dict = {
            'luminosity': luminosity,
            'blue_bw_x': blue_bw_x_i,
            'blue_bw_y': blue_bw_y_i,
            'yellow_bw_x': yellow_bw_x_i,
            'yellow_bw_y': yellow_bw_y_i,
            'beta_star': beta_star_i,
            'blue_x_offset': blue_x_offset_i,
            'blue_y_offset': blue_y_offset_i,
            'blue_x_angle': blue_x_angle_i,
            'blue_y_angle': blue_y_angle_i,
            'yellow_x_angle': yellow_x_angle_i,
            'yellow_y_angle': yellow_y_angle_i,
            'profile_number': profile_number_i
        }
        sample_results.append(sample_dict)

    # Write results to pandas csv
    sample_df = pd.DataFrame(sample_results)
    output_dir = f'{output_dir}{err_estimates}_err/'
    create_dir(output_dir)
    sample_df.to_csv(f'{output_dir}luminosity_samples_{job_num}.csv', index=False)


if __name__ == '__main__':
    main()

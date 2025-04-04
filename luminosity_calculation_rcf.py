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


def main():
    # Get system arguments, should just be an integer for the job number
    job_num = int(sys.argv[1])
    vernier_scan_date = 'Aug12'
    # vernier_scan_date = 'Jul11'
    longitudinal_fit_path = f'../CAD_Measurements/VernierScan_{vernier_scan_date}_COLOR_longitudinal_fit.dat'
    bw_beta_star_linear_fit_path = 'bw_opt_vs_beta_star_fits.txt'
    bw_beta_star_fit_params = get_bw_beta_star_fit_params(bw_beta_star_linear_fit_path)
    estimate_final_luminosity(longitudinal_fit_path, bw_beta_star_fit_params, job_num)
    print('donzo')


def get_bw_beta_star_fit_params(fit_path):
    """
    Read in the beam width vs beta star fit parameters from a file.
    """

    fit_dict = {}

    with open(fit_path, 'r') as file:
        for line in file:
            if "Beam Width X Linear Fit:" in line or "Beam Width Y Linear Fit:" in line:
                parts = line.split(', ')
                key = parts[0].split(':')[0]
                key = 'x' if ' X ' in key else 'y'

                a_part = parts[0].split('=')[1].strip()
                a, a_error = a_part.split(' +- ')

                b_part = parts[1].split('=')[1].strip()
                b, b_error = b_part.split(' +- ')

                fit_dict[key] = {
                    "a": float(a),
                    "a_error": float(a_error),
                    "b": float(b),
                    "b_error": float(b_error)
                }

    return fit_dict


def get_bw_from_beta_star(beta_star, fit_params_dict):
    """
    Calculate the beam width from the beta star using the linear fit parameters.
    """
    bw = beta_star * fit_params_dict['a'] + fit_params_dict['b']
    return bw


def estimate_final_luminosity(longitudinal_fit_path, bw_beta_star_fit_params, job_num):
    """
    Use extracted beam parameters along with their estimated uncertainties to estimate the final luminosity.
    Repeatedly sample beam parameters according to their uncertainties and calculate the luminosity for each sample.
    Generate a distribution of luminosities and calculate the mean and standard deviation.
    :param longitudinal_fit_path:
    :param bw_beta_star_fit_params:
    :param job_num:
    :return:
    """
    n_samples = 500

    err_estimates = 'conservative'  # 'best'

    # Important parameters
    beta_star_low, beta_star_high = 80.0, 105.0 # cm
    measured_beta_star = np.array([97., 82., 88., 95.])

    blue_x_offset, blue_y_offset = 0.0, 0.0  # um
    # blue_x_angle, blue_y_angle, yellow_x_angle, yellow_y_angle = 0.0, +0.14e-3, 0.0, -0.07e-3  # Major bug!
    blue_x_angle, blue_y_angle, yellow_x_angle, yellow_y_angle = -0.07e-3, 0.0, -0.115e-3, 0.0
    blue_len_scaling, yellow_len_scaling = 0.993863022403956, 0.991593955543314

    if err_estimates == 'best':  # Best err estimates
        bw_err, beta_star_err, offset_err, angle_err, len_scale_err = 0.5, 5.0, 2.0, 0.05e-3, 0.001  # um, cm, um, rad, %
        after_burner_lumi_err = 0.0
    else:  # Conservative err estimates
        bw_err, beta_star_err, offset_err, angle_err, len_scale_err = 1.0, 10.0, 5.0, 0.1e-3, 0.005  # um, cm, um, rad, %
        after_burner_lumi_err = 0.04 * 2.1e-6  # 4% error on 2.1e-6 um^-2 s^-1, to account for tuning?
        # after_burner_lumi_err = 2.25e-6 - 2.1e-6  # CW minus most probable
        # after_burner_lumi_err += 2.21e-6 - 2.1e-6  # betastar 105 minus most probable

    collider_sim = BunchCollider()
    collider_sim.set_bunch_rs(np.array([blue_x_offset, blue_y_offset, -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_crossing(blue_x_angle, blue_y_angle, yellow_x_angle, yellow_y_angle)
    blue_fit_path = longitudinal_fit_path.replace('_COLOR_', '_blue_')
    yellow_fit_path = longitudinal_fit_path.replace('_COLOR_', '_yellow_')
    collider_sim.set_longitudinal_fit_parameters_from_file(blue_fit_path, yellow_fit_path)
    collider_sim.set_longitudinal_fit_scaling(blue_len_scaling, yellow_len_scaling)

    # Sample beam parameters
    sample_results = []
    for i in range(n_samples):
        beta_star = np.random.uniform(beta_star_low, beta_star_high)
        beta_star_scale_factor_i = beta_star / 90  # Set 90 cm (average of measured) to default values, then scale from there
        beta_star_scaled_i = measured_beta_star * beta_star_scale_factor_i
        for index in range(len(beta_star_scaled_i)):  # Sample each component separately
            beta_star_scaled_i[index] = np.random.normal(beta_star_scaled_i[index], beta_star_err)

        bw_x_i = get_bw_from_beta_star(beta_star, bw_beta_star_fit_params['x'])
        bw_y_i = get_bw_from_beta_star(beta_star, bw_beta_star_fit_params['y'])

        bw_x_i = np.random.normal(bw_x_i, bw_err)
        bw_y_i = np.random.normal(bw_y_i, bw_err)
        blue_x_offset_i = np.random.normal(blue_x_offset, offset_err)
        blue_y_offset_i = np.random.normal(blue_y_offset, offset_err)
        blue_x_angle_i = np.random.normal(blue_x_angle, angle_err)
        blue_y_angle_i = np.random.normal(blue_y_angle, angle_err)
        yellow_x_angle_i = np.random.normal(yellow_x_angle, angle_err)
        yellow_y_angle_i = np.random.normal(yellow_y_angle, angle_err)
        blue_len_scale_i = np.random.normal(blue_len_scaling, len_scale_err)
        yellow_len_scale_i = np.random.normal(yellow_len_scaling, len_scale_err)

        collider_sim.set_bunch_rs(np.array([blue_x_offset_i, blue_y_offset_i, -6.e6]), np.array([0., 0., +6.e6]))
        collider_sim.set_bunch_beta_stars(*beta_star_scaled_i)
        collider_sim.set_bunch_sigmas(np.array([bw_x_i, bw_y_i]), np.array([bw_x_i, bw_y_i]))
        collider_sim.set_bunch_crossing(blue_x_angle_i, blue_y_angle_i, yellow_x_angle_i, yellow_y_angle_i)
        collider_sim.set_longitudinal_fit_scaling(blue_len_scale_i, yellow_len_scale_i)

        collider_sim.run_sim_parallel()
        luminosity_pre_afterburner = collider_sim.get_naked_luminosity()
        luminosity = np.random.normal(luminosity_pre_afterburner, after_burner_lumi_err)
        print(f'{datetime.now()} Sample {i + 1}/{n_samples} luminosity: {luminosity:.2e}')
        sample_dict = {
            'luminosity': luminosity,
            'luminosity_pre_afterburner': luminosity_pre_afterburner,
            'bw_x': bw_x_i,
            'bw_y': bw_y_i,
            'beta_star': beta_star,
            'blue_x_offset': blue_x_offset_i,
            'blue_y_offset': blue_y_offset_i,
            'blue_x_angle': blue_x_angle_i,
            'blue_y_angle': blue_y_angle_i,
            'yellow_x_angle': yellow_x_angle_i,
            'yellow_y_angle': yellow_y_angle_i,
            'blue_len_scaling': blue_len_scale_i,
            'yellow_len_scaling': yellow_len_scale_i
        }
        sample_results.append(sample_dict)

    # Write results to pandas csv
    sample_df = pd.DataFrame(sample_results)
    output_dir = f'output/{err_estimates}_err/'
    create_dir(output_dir)
    sample_df.to_csv(f'{output_dir}luminosity_samples_{job_num}.csv', index=False)


if __name__ == '__main__':
    main()

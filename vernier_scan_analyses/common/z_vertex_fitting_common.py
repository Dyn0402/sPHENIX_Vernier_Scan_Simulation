#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 01 03:38 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/z_vertex_fitting_common

@author: Dylan Neff, dn277127
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from datetime import datetime, time
import uproot


def load_vertex_distributions(z_vertex_data_path, steps, cad_df, rate_column='zdc_raw_rate'):
    """
    Load vertex data from a ROOT file for the specified steps.
    :param z_vertex_data_path: Path to the ROOT file containing vertex distributions.
    :param steps: List of steps to load data for.
    :param cad_df: DataFrame containing CAD step data.
    :param rate_column: Column name in cad_df for the rate to scale the histograms.
    :return: Dictionary with step as key and (centers, counts, count_errs) as values.
    """
    step_0 = cad_df[cad_df['step'] == 0].iloc[0]
    dcct_blue_nom, dcct_yellow_nom = step_0['blue_dcct_ions'], step_0['yellow_dcct_ions']

    vertex_data = {}
    with uproot.open(z_vertex_data_path) as f:
        for step in steps:
            hist = f[f'step_{step}']
            centers = hist.axis().centers()
            counts = hist.counts()
            count_errs = hist.errors()
            count_errs[count_errs == 0] = 1  # Avoid division by zero

            cad_step_row = cad_df[cad_df['step'] == step].iloc[0]
            if rate_column:
                raw_rate = cad_step_row[rate_column]
                hist_scaling_factor = raw_rate / np.sum(counts)
            else:
                hist_scaling_factor = 1

            dcct_scale = (dcct_blue_nom * dcct_yellow_nom) / (
                    cad_step_row['blue_dcct_ions'] * cad_step_row['yellow_dcct_ions'])
            counts *= hist_scaling_factor * dcct_scale
            count_errs *= hist_scaling_factor * dcct_scale

            vertex_data[step] = (centers, counts, count_errs)
    return vertex_data


def merge_cad_rates_df(cad_df, rates_df):
    """
    Merge CAD DataFrame with rates DataFrame on 'step' column. Calculate the corrected raw MBD rate.
    :param cad_df: DataFrame containing CAD step data.
    :param rates_df: DataFrame containing rates data.
    :return: Merged DataFrame.
    """
    rates_df['corrected_raw_mbd_rate'] = rates_df['mbd_raw_rate_mean'] * (1 - rates_df['mbd_cut_correction_fraction'] / 100)
    rates_df['corrected_raw_mbd_rate_err'] = np.sqrt(rates_df['mbd_raw_rate_std']**2 + (rates_df['mbd_raw_rate_mean'] * (rates_df['simulated_cut_fraction'] / 100)**2))
    merged_df = pd.merge(cad_df, rates_df, on='step', how='inner')
    return merged_df


def compute_total_chi2(params, collider_sim, cad_df, centers_list, counts_list, count_errs_list, sim_settings,
                       metrics=('chi2',)):
    beam_width_x, beam_width_y, beta_star_x, beta_star_y, yellow_angle_dx, blue_offset_dx, yellow_angle_dy, blue_offset_dy = params
    # collider_sim.set_bunch_beta_stars(beta_star_x, beta_star_y)
    collider_sim.set_bunch_beta_stars(beta_star_x, beta_star_x)

    total_chi2, total_log_likelihood, total_scaled_resid = 0.0, 0.0, 0.0
    for i, scan_step in enumerate(sim_settings['steps']):
        cad_step_row = cad_df[cad_df['step'] == scan_step].iloc[0]

        em_blue_horiz, em_blue_vert = cad_step_row['blue_horiz_emittance'], cad_step_row['blue_vert_emittance']
        em_yel_horiz, em_yel_vert = cad_step_row['yellow_horiz_emittance'], cad_step_row['yellow_vert_emittance']

        em_blue_horiz_nom, em_blue_vert_nom = sim_settings['em_blue_nom']
        em_yel_horiz_nom, em_yel_vert_nom = sim_settings['em_yel_nom']

        blue_widths = np.array([
            beam_width_x * np.sqrt(em_blue_horiz / em_blue_horiz_nom),
            beam_width_y * np.sqrt(em_blue_vert / em_blue_vert_nom)
        ])
        yellow_widths = np.array([
            beam_width_x * np.sqrt(em_yel_horiz / em_yel_horiz_nom),
            beam_width_y * np.sqrt(em_yel_vert / em_yel_vert_nom)
        ])

        collider_sim.set_bunch_sigmas(blue_widths, yellow_widths)

        blue_angle_x = -cad_step_row['blue angle h'] * 1e-3
        blue_angle_y = -cad_step_row['blue angle v'] * 1e-3
        yellow_angle_x = -cad_step_row['yellow angle h'] * 1e-3
        yellow_angle_y = -cad_step_row['yellow angle v'] * 1e-3

        blue_offset_x, blue_offset_y = cad_step_row['set offset h'] * 1e3, cad_step_row['set offset v'] * 1e3
        yellow_offset_x, yellow_offset_y = 0, 0

        if i != 0:
            yellow_angle_x += yellow_angle_dx * 1e-3
            yellow_angle_y += yellow_angle_dy * 1e-3
            blue_offset_x += blue_offset_dx
            blue_offset_y += blue_offset_dy

        collider_sim.set_bunch_crossing(blue_angle_x, blue_angle_y, yellow_angle_x, yellow_angle_y)

        collider_sim.set_bunch_offsets(
            [blue_offset_x, blue_offset_y],
            [yellow_offset_x, yellow_offset_y]
        )

        profile_path = get_profile_path(
            sim_settings['profiles_path'], cad_step_row['start'], cad_step_row['end'], False
        )
        collider_sim.set_longitudinal_profiles_from_file(
            profile_path.replace('COLOR_', 'blue_'),
            profile_path.replace('COLOR_', 'yellow_')
        )

        collider_sim.run_sim_parallel()

        data_index = i if isinstance(centers_list, list) else scan_step
        centers = centers_list[data_index]
        counts = counts_list[data_index]
        count_errs = count_errs_list[data_index]
        fit_mask = (centers > sim_settings['fit_range'][0]) & (centers < sim_settings['fit_range'][1])

        if data_index == 0:  # Fix the amplitude in the first head-on step
        # if True:  # Fix the amplitude in the first head-on step
            fit_amp_shift(collider_sim, counts[fit_mask], centers[fit_mask], count_errs[fit_mask])
        else:
            # fit_amp_shift(collider_sim, counts[fit_mask], centers[fit_mask], count_errs[fit_mask])
            fit_shift_only(collider_sim, counts[fit_mask], centers[fit_mask], count_errs[fit_mask])

        zs, z_dist = collider_sim.get_z_density_dist()
        interp_sim = np.interp(centers[fit_mask], zs, z_dist)
        if 'chi2' in metrics:
            chi2 = np.sum(((counts[fit_mask] - interp_sim) / count_errs[fit_mask]) ** 2) / len(counts[fit_mask])
            total_chi2 += chi2
        if 'log_like' in metrics:
            log_likelihood = -np.sum(counts[fit_mask] * np.log(interp_sim) - interp_sim)
            total_log_likelihood += log_likelihood
        if 'scaled_resid' in metrics:
            scaled_resid = np.sqrt(np.sum(((counts[fit_mask] - interp_sim) / np.mean(counts[fit_mask])) ** 2))
            total_scaled_resid += scaled_resid

        if sim_settings.get('plot', False):
            fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
            ax[0].errorbar(centers, counts, yerr=count_errs, fmt='o', label='Data', color='black', markersize=4)
            ax[0].plot(centers, interp_sim, label='Simulation', color='blue')
            ax[0].set_ylabel('Counts')
            ax[0].legend()
            ax[0].set_title(f'Step {scan_step}, Chi2={chi2:.2f}')

            residuals = (counts - interp_sim) / count_errs
            ax[1].plot(centers[fit_mask], residuals[fit_mask], 'o', color='red')
            ax[1].axhline(0, color='gray', linestyle='--')
            ax[1].set_ylabel('Residuals')
            ax[1].set_xlabel('z position')

            plt.tight_layout()

    print(f'Total Chi2: {total_chi2}, likelihood: {total_log_likelihood}, residuals: {total_scaled_resid}\n'
          f'Current parameters: {params}')

    if sim_settings.get('plot', False):
        plt.show()

    if metrics == ('chi2', 'log_like', 'scaled_resid'):
        return total_chi2, total_log_likelihood, total_scaled_resid

    if metrics == ('chi2', 'log_like'):
        return total_chi2, total_log_likelihood

    if metrics == ('scaled_resid',):
        return total_scaled_resid

    if metrics == ('log_like',):
        return total_log_likelihood

    if metrics == ('chi2',):
        return total_chi2

    return None


def fit_amp_shift(collider_sim, z_dist_data, zs_data, z_dist_errs=None):
    collider_sim.set_amplitude(1)
    zs, z_dist = collider_sim.get_z_density_dist()
    scale = max(z_dist_data) / max(z_dist)

    best_result = None
    best_fun = np.inf
    best_initial_shift = None
    initial_shifts = np.linspace(-5e4, 5e4, 5)

    for init_shift in initial_shifts:
        res = minimize(
            amp_shift_residual,
            np.array([1.0, 0.0]),  # initial scale factor and shift delta
            method='Nelder-Mead',
            # method='L-BFGS-B',  # Doesn't call outside of bounds but needs to be smooth (should be fine)
            args=(collider_sim, scale, init_shift, z_dist_data, zs_data, z_dist_errs),
            # bounds=((0.0, 2.0), (-34e4, 34e4))
        )

        if res.fun < best_fun:
            best_fun = res.fun
            best_result = res
            best_initial_shift = init_shift

    # Update scale and shift using best result
    scale = best_result.x[0] * scale if best_result is not None else scale
    shift = best_result.x[1] + best_initial_shift if best_result is not None else 0

    collider_sim.set_amplitude(scale)
    collider_sim.set_z_shift(shift)


def amp_shift_residual(x, collider_sim, scale_0, shift_0, z_dist_data, zs_data, z_dist_errs=None):
    collider_sim.set_amplitude(x[0] * scale_0)
    collider_sim.set_z_shift(x[1] + shift_0)
    sim_zs, sim_z_dist = collider_sim.get_z_density_dist()
    sim_interp = interp1d(sim_zs, sim_z_dist)
    if min(sim_zs) > min(zs_data) or max(sim_zs) < max(zs_data):
        # print(f'Warning: Simulation z range {min(sim_zs)} to {max(sim_zs)} '
        #       f'does not cover data z range {min(zs_data)} to {max(zs_data)}. ')
        return 1e20
    # return np.sqrt(np.sum((z_dist_data - sim_interp(zs_data)) ** 2)) / np.mean(z_dist_data)
    if z_dist_errs is not None:  # Calculate chi2 per degree of freedom
        resid = np.sum((z_dist_data - sim_interp(zs_data)) ** 2 / z_dist_errs ** 2) / (len(z_dist_data) - 1)
    else:
        resid = np.sqrt(np.sum((z_dist_data - sim_interp(zs_data)) ** 2)) / np.mean(z_dist_data)
    # print(f'Amplitude: {x[0] * scale_0:.3f}, Shift: {x[1] + shift_0:.3f} um, dshift: {x[1]} um, Residual: {resid:.3f}')
    return resid


def fit_shift_only(collider_sim, z_dist_data, zs_data, z_dist_errs=None):
    best_result = None
    best_fun = np.inf
    best_initial_shift = None
    initial_shifts = np.linspace(-5e4, 5e4, 5)

    for init_shift in initial_shifts:
        res = minimize(
            shift_only_residual,
            np.array([0.0]),  # initial shift delta
            method='Nelder-Mead',
            # method='L-BFGS-B',  # Doesn't call outside of bounds but needs to be smooth (should be fine)
            args=(collider_sim, init_shift, z_dist_data, zs_data, z_dist_errs),
            # bounds=((-34e4, 34e4),)
        )

        if res.fun < best_fun:
            best_fun = res.fun
            best_result = res
            best_initial_shift = init_shift

    # Update only the shift using best result
    shift = best_result.x[0] + best_initial_shift if best_result is not None else 0

    collider_sim.set_z_shift(shift)


def shift_only_residual(x, collider_sim, shift_0, z_dist_data, zs_data, z_dist_errs=None):
    collider_sim.set_z_shift(x[0] + shift_0)
    sim_zs, sim_z_dist = collider_sim.get_z_density_dist()
    sim_interp = interp1d(sim_zs, sim_z_dist)
    if min(sim_zs) > min(zs_data) or max(sim_zs) < max(zs_data):
        # print(f'Warning: Simulation z range {min(sim_zs)} to {max(sim_zs)} '
        #       f'does not cover data z range {min(zs_data)} to {max(zs_data)}. Shift: {x[0] + shift_0:.3f} um')
        return 1e20
    if z_dist_errs is not None:
        resid = np.sum((z_dist_data - sim_interp(zs_data)) ** 2 / z_dist_errs ** 2) / (len(z_dist_data) - 1)
    else:
        resid = np.sqrt(np.sum((z_dist_data - sim_interp(zs_data)) ** 2)) / np.mean(z_dist_data)

    return resid


def get_profile_path(profile_dir_path, start_time, end_time, return_all=False):
    """
    Returns a list of full paths to all blue/yellow profile files in the given directory
    between start_datetime and end_datetime (inclusive),
    where both a blue_ and yellow_ version exist for the same time.
    """
    if isinstance(start_time, str):
        start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    if isinstance(end_time, str):
        end_time = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')

    # Get all files in the directory
    all_files = [f for f in os.listdir(profile_dir_path) if f.endswith('.dat') and f.startswith('avg_')]

    # Filter files that match the expected pattern
    blue_files = [f for f in all_files if f.startswith("avg_blue_profile_24_")]
    blue_times = [datetime.combine(start_time.date(), extract_time_from_filename(f)) for f in blue_files]
    blue_times, blue_files = zip(*sorted(zip(blue_times, blue_files)))  # Sort blue files and times together by time
    good_files = []
    for file_time, file_name in zip(blue_times, blue_files):
        if file_time < start_time or file_time > end_time:
            continue
        if file_name.replace('blue_', 'yellow_') not in all_files:
            continue
        good_files.append(os.path.join(profile_dir_path, file_name.replace('blue_', 'COLOR_')))

    if return_all:
        return good_files
    else:  # Get middle file in list
        return good_files[len(good_files) // 2]


def extract_time_from_filename(filename):
    parts = filename.split("_")
    try:
        hour = int(parts[-3])
        minute = int(parts[-2])
        second = int(parts[-1].split(".")[0])
        return time(hour=hour, minute=minute, second=second)
    except ValueError:
        return None
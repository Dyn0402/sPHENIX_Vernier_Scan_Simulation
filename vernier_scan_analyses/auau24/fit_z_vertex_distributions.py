#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 04 01:40 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/fit_z_vertex_distributions

@author: Dylan Neff, dn277127
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import interp1d
import uproot
from datetime import datetime, time

from BunchCollider import BunchCollider
from z_vertex_fitting_common import (fit_amp_shift, fit_shift_only, get_profile_path, compute_total_chi2,
                                     load_vertex_distributions, merge_cad_rates_df)
from Measure import Measure
from common_logistics import set_base_path


def main():
    base_path = set_base_path()
    base_path = f'{base_path}Vernier_Scans/auau_oct_16_24/'

    # fit_beta_star_to_head_on_steps(base_path)
    # fit_beta_stars_bws_to_all_steps(base_path)
    # fit_beam_widths(base_path)
    plot_beta_star_head_on_fit_results(base_path)
    # plot_beam_width_fit_results(base_path)
    plt.show()
    print('donzo')


def fit_beta_stars_bws_to_all_steps(base_path):
    """
    Fit beta star and beam widths in both x and y directions to all steps.
    :param base_path: Base path to the vernier scan data.
    :return: None
    """

    longitudinal_profiles_dir_path = f'{base_path}profiles/'
    z_vertex_zdc_data_path = f'{base_path}vertex_data/54733_vertex_distributions.root'
    # z_vertex_no_zdc_data_path = f'{base_path}vertex_data/54733_vertex_distributions_no_zdc_coinc.root'
    combined_cad_step_data_csv_path = f'{base_path}combined_cad_step_data.csv'

    cad_df = pd.read_csv(combined_cad_step_data_csv_path)

    fit_range = [-200, 200]
    # steps = [0]
    steps = [0, 6, 12, 18]
    # steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # steps = np.arange(0, 24)

    collider_sim = BunchCollider()
    # collider_sim.set_grid_size(31, 31, 101, 31)
    collider_sim.set_grid_size(31, 31, 101, 31)
    bkg = 0.0e-17
    gauss_eff_width = 500
    mbd_resolution = 1.0
    collider_sim.set_bkg(bkg)
    collider_sim.set_gaus_z_efficiency_width(gauss_eff_width)
    collider_sim.set_gaus_smearing_sigma(mbd_resolution)

    # Get nominal dcct ions and emittances
    step_0 = cad_df[cad_df['step'] == 0].iloc[0]
    dcct_blue_nom, dcct_yellow_nom = step_0['blue_dcct_ions'], step_0['yellow_dcct_ions']
    em_blue_horiz_nom, em_blue_vert_nom = step_0['blue_horiz_emittance'], step_0['blue_vert_emittance']
    em_yel_horiz_nom, em_yel_vert_nom = step_0['yellow_horiz_emittance'], step_0['yellow_vert_emittance']

    # Preload data and histograms here
    centers_list, counts_list, count_errs_list = [], [], []
    for step in steps:
        with uproot.open(z_vertex_zdc_data_path) as f:
            # with uproot.open(z_vertex_no_zdc_data_path) as f:
            hist = f[f'step_{step}']
            centers = hist.axis().centers()
            counts = hist.counts()
            count_errs = hist.errors()
            count_errs[count_errs == 0] = 1

        cad_step_row = cad_df[cad_df['step'] == step].iloc[0]
        zdc_raw_rate = cad_step_row['zdc_raw_rate']
        hist_scaling_factor = zdc_raw_rate / np.sum(counts)
        dcct_scale = (dcct_blue_nom * dcct_yellow_nom) / (
                cad_step_row['blue_dcct_ions'] * cad_step_row['yellow_dcct_ions'])

        counts *= hist_scaling_factor * dcct_scale
        count_errs *= hist_scaling_factor * dcct_scale

        centers_list.append(centers)
        counts_list.append(counts)
        count_errs_list.append(count_errs)

    sim_settings = {
        'steps': steps,
        'fit_range': fit_range,
        'profiles_path': longitudinal_profiles_dir_path,
        'em_blue_nom': (em_blue_horiz_nom, em_blue_vert_nom),
        'em_yel_nom': (em_yel_horiz_nom, em_yel_vert_nom),
        'plot': False  # Set to True if you want to plot the results
    }

    # beam_width_x, beam_width_y, beta_star, yellow_angle_x (mrad), blue_offset_x (microns)
    # initial_guess = np.array([120.0, 110.0, 72.0, 0.0, 0.0])
    # initial_guess = np.array([125, 125, 72, 1e-2, -14.91])
    # initial_guess = np.array([124.1, 125, 72, 1.2e-3, -81.64])
    initial_guess = '[ 1.286e+02  1.436e+02  6.605e+01  3.833e-04 -6.498e-01 -4.741e-02 -9.745e+00]'
    initial_guess = np.array([128.6, 143.6, 66.05, 0.0003833, -0.6498, -0.04741, -9.745])
    # bounds = [(120, 170), (120, 170), (55, 90), (-0.2, +0.2), (-200, 200)]
    # bounds = [(110, 145), (110, 145), (65, 120), (65, 120), (-0.1, 0.1), (-100.0, 100.0), (-0.05, 0.05), (-50.0, 50.0)]
    bounds = [(130, 130), (130, 130), (60, 85), (72, 72), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]

    # result = minimize(
    #     compute_total_chi2,
    #     initial_guess,
    #     args=(collider_sim, cad_df, centers_list, counts_list, count_errs_list, sim_settings),
    #     # method='Nelder-Mead',  # or 'L-BFGS-B' with bounds
    #     method='L-BFGS-B',
    #     bounds=bounds,
    #     options={'maxiter': 10000}
    # )

    result = differential_evolution(
        compute_total_chi2,
        bounds=bounds,
        args=(collider_sim, cad_df, centers_list, counts_list, count_errs_list, sim_settings, ('scaled_resid',)),
        strategy='best1bin',
        maxiter=10000,
        popsize=15,
        tol=0.01,
        mutation=(0.5, 1),
        recombination=0.7,
        polish=True,
        # updating='deferred',
        updating='immediate',
        disp=True
    )

    print('Optimization result:', result)


def fit_beta_star_to_head_on_steps(base_path):
    """
    Use the head-on steps to fit the beta star x and y values.
    Head-on is insensitive to beam width and less sensitive to other beam parameters.
    Beta star affects the width of the z-vertex distribution, though more a combination of the x and y beta star
    components. Slight symmetry breaking in the larger horizontal crossing angle.
    But generally get a line in the beta star x vs beta star y space.
    :param base_path: Base path to the vernier scan data.
    :return: None
    """

    longitudinal_profiles_dir_path = f'{base_path}profiles/'
    z_vertex_zdc_data_path = f'{base_path}vertex_data/54733_vertex_distributions.root'
    combined_cad_step_data_csv_path = f'{base_path}combined_cad_step_data.csv'
    # out_name = 'beta_star_fit_results.csv'
    # out_name = 'beta_star_fit_results_zdc_cor_rate_bw130.csv'
    # out_name = 'beta_star_fit_results_zdc_raw_rate_bw130.csv'
    out_name_base = 'beta_star_fit_results_'
    # out_name = 'beta_star_fit_results_mbd_z200_rate_bw110.csv'
    # rate_column = 'zdc_raw_rate'  # 'zdc_raw_rate', 'zdc_cor_rate', 'mbd_z200_rate', or 'mbd_bkg_cor_rate'
    # rate_column = 'mbd_z200_rate'  # 'zdc_raw_rate', 'zdc_cor_rate', 'mbd_z200_rate', or 'mbd_bkg_cor_rate'

    rate_cols = ['zdc_sasha_cor_rate', 'mbd_sasha_z200_rate', 'mbd_sasha_bkg_cor_rate']
    bws = [130, 110]

    cad_df = pd.read_csv(combined_cad_step_data_csv_path)

    fit_range = [-200, 200]
    steps = [0, 6, 12, 18, 24]

    collider_sim = BunchCollider()
    collider_sim.set_grid_size(31, 31, 101, 31)
    bkg = 0.0e-17
    gauss_eff_width = 500
    mbd_resolution = 1.0
    collider_sim.set_bkg(bkg)
    collider_sim.set_gaus_z_efficiency_width(gauss_eff_width)
    collider_sim.set_gaus_smearing_sigma(mbd_resolution)

    # Get nominal dcct ions and emittances
    step_0 = cad_df[cad_df['step'] == 0].iloc[0]
    em_blue_horiz_nom, em_blue_vert_nom = step_0['blue_horiz_emittance'], step_0['blue_vert_emittance']
    em_yel_horiz_nom, em_yel_vert_nom = step_0['yellow_horiz_emittance'], step_0['yellow_vert_emittance']

    for rate_column in rate_cols:
        # Preload data and histograms here
        vertex_data = load_vertex_distributions(z_vertex_zdc_data_path, steps, cad_df, rate_column)

        for bw in bws:
            # beam_width_x, beam_width_y = 130.0, 130.0
            beam_width_x, beam_width_y = bw, bw
            out_name = f'{out_name_base}{rate_column}_bw{bw}.csv'

            sim_settings = {
                'steps': steps,
                'fit_range': fit_range,
                'profiles_path': longitudinal_profiles_dir_path,
                'em_blue_nom': (em_blue_horiz_nom, em_blue_vert_nom),
                'em_yel_nom': (em_yel_horiz_nom, em_yel_vert_nom),
                'plot': False  # Set to True if you want to plot the results
            }

            beta_stars = np.linspace(60, 95, 150)

            results = []
            for beta_star in beta_stars:  # For given beta_star, amplitude is fixed at step=0 and stays there.
                for step in steps:
                    sim_settings['steps'] = [step]
                    centers, counts, count_errs = vertex_data[step]
                    initial_guess = np.array([beam_width_x, beam_width_y, beta_star, beta_star, 0.0, 0.0, 0.0, 0.0])

                    chi2, log_like, resid = compute_total_chi2(
                        initial_guess,
                        collider_sim,
                        cad_df,
                        {step: centers},
                        {step: counts},
                        {step: count_errs},
                        sim_settings,
                        metrics=('chi2', 'log_like', 'scaled_resid')
                    )
                    results.append({'step': step, 'beta_star': beta_star, 'chi2': chi2, 'log_like': log_like, 'resids': resid})

            # Write results to a CSV file
            results_df = pd.DataFrame(results)
            results_df.to_csv(f'{base_path}{out_name}', index=False)

    plt.show()


def fit_beam_widths(base_path):
    """
    Fit beam widths in x to all horizontal scan steps and in y to all vertical scan steps.
    :param base_path: Base path to the vernier scan data.
    :return: None
    """

    longitudinal_profiles_dir_path = f'{base_path}profiles/'
    z_vertex_zdc_data_path = f'{base_path}vertex_data/54733_vertex_distributions.root'
    combined_cad_step_data_csv_path = f'{base_path}combined_cad_step_data.csv'
    # rate_column = 'zdc_cor_rate'  # 'zdc_raw_rate', 'zdc_cor_rate', 'mbd_z200_rate', or 'mbd_bkg_cor_rate'
    rate_column = 'mbd_z200_rate'  # 'zdc_raw_rate', 'zdc_cor_rate', 'mbd_z200_rate', or 'mbd_bkg_cor_rate'

    cad_df = pd.read_csv(combined_cad_step_data_csv_path)

    fit_range = [-200, 200]
    # steps = np.arange(0, 25)
    steps = [i for i in np.arange(0, 24) if (i % 6 or i == 0)]  # Only first head-on to fix amplitude

    collider_sim = BunchCollider()
    collider_sim.set_grid_size(31, 31, 101, 31)
    beta_star = 77.15  # cm
    bkg = 0.0e-17
    gauss_eff_width = 500
    mbd_resolution = 1.0
    collider_sim.set_bkg(bkg)
    collider_sim.set_gaus_z_efficiency_width(gauss_eff_width)
    collider_sim.set_gaus_smearing_sigma(mbd_resolution)
    beam_widths = np.linspace(110, 160, 100)  # in micrometers

    # Get nominal dcct ions and emittances
    step_0 = cad_df[cad_df['step'] == 0].iloc[0]
    em_blue_horiz_nom, em_blue_vert_nom = step_0['blue_horiz_emittance'], step_0['blue_vert_emittance']
    em_yel_horiz_nom, em_yel_vert_nom = step_0['yellow_horiz_emittance'], step_0['yellow_vert_emittance']

    # Preload data and histograms here
    vertex_data = load_vertex_distributions(z_vertex_zdc_data_path, steps, cad_df, rate_column)

    sim_settings = {
        'steps': steps,
        'fit_range': fit_range,
        'profiles_path': longitudinal_profiles_dir_path,
        'em_blue_nom': (em_blue_horiz_nom, em_blue_vert_nom),
        'em_yel_nom': (em_yel_horiz_nom, em_yel_vert_nom),
        'plot': False  # Set to True if you want to plot the results
    }

    results = []
    for beam_width in beam_widths:
        # Run the first head-on step for each beam width to fix the amplitude
        for step in steps:
            sim_settings['steps'] = [step]
            centers, counts, count_errs = vertex_data[step]
            initial_guess = np.array([beam_width, beam_width, beta_star, beta_star, 0.0, 0.0, 0.0, 0.0])

            chi2, log_like, resid = compute_total_chi2(
                initial_guess,
                collider_sim,
                cad_df,
                {step: centers},
                {step: counts},
                {step: count_errs},
                sim_settings,
                metrics=('chi2', 'log_like', 'scaled_resid')
            )
            if step != 0:  # Skip the first head-on step, just to fix the amplitude
                orientation = 'horizontal' if step < 12 else 'vertical'
                results.append({'step': step, 'orientation': orientation, 'beam_width': beam_width,
                                'chi2': chi2, 'log_like': log_like, 'resids': resid})

    # Write results to a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{base_path}beam_width_fit_results.csv', index=False)

    plt.show()


def plot_beta_star_head_on_fit_results(base_path):
    """
    Plot the results of the beta star head-on fit.
    """
    # results_df = pd.read_csv(f'{base_path}beta_star_fit_results_130bw.csv')
    # results_df = pd.read_csv(f'{base_path}beta_star_fit_results.csv')

    # results_df = pd.read_csv(f'{base_path}beta_star_fit_results_zdc_sasha_cor_rate_bw110.csv')
    results_df = pd.read_csv(f'{base_path}beta_star_fit_results_zdc_sasha_cor_rate_bw130.csv')  # Nominal
    # results_df = pd.read_csv(f'{base_path}beta_star_fit_results_zdc_raw_rate_bw110.csv')
    # results_df = pd.read_csv(f'{base_path}beta_star_fit_results_zdc_raw_rate_bw130.csv')
    # results_df = pd.read_csv(f'{base_path}beta_star_fit_results_mbd_sasha_z200_rate_bw110.csv')
    # results_df = pd.read_csv(f'{base_path}beta_star_fit_results_mbd_sasha_z200_rate_bw130.csv')
    # results_df = pd.read_csv(f'{base_path}beta_star_fit_results_mbd_sasha_bkg_cor_rate_bw110.csv')
    # results_df = pd.read_csv(f'{base_path}beta_star_fit_results_mbd_sasha_bkg_cor_rate_bw130.csv')

    # Plot chi2 vs beta star for each step
    fig, axs = plt.subplots(figsize=(10, 6), nrows=3, sharex='all')
    min_bses = []
    for step in results_df['step'].unique():
        step_data = results_df[results_df['step'] == step]
        log_like_avg = step_data['log_like'].mean()
        line0, = axs[0].plot(step_data['beta_star'], step_data['chi2'], label=f'Step {step}')
        line1, = axs[1].plot(step_data['beta_star'], step_data['log_like'] - log_like_avg, label=f'Step {step}')
        line2, = axs[2].plot(step_data['beta_star'], step_data['resids'], label=f'Step {step}')
        min_chi2_beta_star = step_data['beta_star'][step_data['chi2'].idxmin()]
        min_log_like_beta_star = step_data['beta_star'][step_data['log_like'].idxmin()]
        min_resid_beta_star = step_data['beta_star'][step_data['resids'].idxmin()]
        axs[0].axvline(min_chi2_beta_star, color=line0.get_color(), linestyle='--', linewidth=0.5)
        axs[1].axvline(min_log_like_beta_star, color=line1.get_color(), linestyle='--', linewidth=0.5)
        axs[2].axvline(min_resid_beta_star, color=line2.get_color(), linestyle='--', linewidth=0.5)
        min_bses.append(min_chi2_beta_star)
        min_bses.append(min_log_like_beta_star)
        min_bses.append(min_resid_beta_star)

    # Group by beta_star and compute the mean chi2
    step_avg_df = results_df.groupby('beta_star')[['chi2', 'log_like', 'resids']].mean().reset_index()
    axs[0].plot(step_avg_df['beta_star'], step_avg_df['chi2'], color='black', linestyle='--', label='Mean')
    axs[1].plot(step_avg_df['beta_star'], step_avg_df['log_like'] - step_avg_df['log_like'].mean(), color='black',
                linestyle='--')
    axs[2].plot(step_avg_df['beta_star'], step_avg_df['resids'], color='black', linestyle='--')

    min_chi2_avg_beta_star = step_avg_df['beta_star'][step_avg_df['chi2'].idxmin()]
    min_log_like_avg_beta_star = step_avg_df['beta_star'][step_avg_df['log_like'].idxmin()]
    min_resid_avg_beta_star = step_avg_df['beta_star'][step_avg_df['resids'].idxmin()]

    opt_beta_star = Measure(min_resid_avg_beta_star, np.std(min_bses))

    axs[0].axvline(min_chi2_avg_beta_star, color='black', linestyle=':', lw=3, linewidth=0.5)
    axs[1].axvline(min_log_like_avg_beta_star, color='black', linestyle=':', lw=3, linewidth=0.5)
    axs[2].axvline(min_resid_avg_beta_star, color='black', linestyle=':', lw=3, linewidth=0.5)

    axs[2].axvspan(opt_beta_star.val - opt_beta_star.err, opt_beta_star.val + opt_beta_star.err, color='black',
                   alpha=0.2)

    axs[2].annotate(f'Minimum Scaled Residual at {opt_beta_star} cm',
                    xy=(min_resid_avg_beta_star, step_avg_df['resids'].min()),
                    xytext=(min_resid_avg_beta_star + 4, 1.6),
                    arrowprops=dict(arrowstyle='->', color='black'), fontsize=10, color='black')

    axs[2].set_xlabel('Beta Star (cm)')
    axs[0].set_ylabel('Chi2')
    axs[1].set_ylabel('Log Likelihood')
    axs[2].set_ylabel('Scaled Residual')
    # axs.set_title('Chi2 vs Beta Star for Head-On Steps')
    axs[0].legend()
    for i in range(3):
        axs[i].grid()
    fig.tight_layout()
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.075, top=0.995, hspace=0.0)


def plot_beam_width_fit_results(base_path):
    """
    Plot the results of the beam width fit.
    """
    # all_results_df = pd.read_csv(f'{base_path}beam_width_fit_results.csv')
    all_results_df = pd.read_csv(f'{base_path}beam_width_fit_results_zdc_cor_rate.csv')
    skip_steps = [1, 7, 13, 19]  # Skip steps at 100 microns, less sensitive to beam width
    bw_range = (110, 150)  # Range of beam widths in micrometers

    orientations = ['horizontal', 'vertical']
    # orientations = ['horizontal']
    fig_avgs, ax_avgs = plt.subplots(figsize=(10, 7), nrows=2, sharex='all')
    ax_avgs = dict(zip(orientations, ax_avgs))
    for orientation in orientations:
        results_df = all_results_df[all_results_df['orientation'] == orientation]
        results_df = results_df[(results_df['beam_width'] >= bw_range[0]) & (results_df['beam_width'] <= bw_range[1])]

        # Plot chi2 vs beta star for each step
        fig, axs = plt.subplots(figsize=(10, 7), nrows=3, sharex='all')
        min_bses = []
        for step in results_df['step'].unique():
            if step in skip_steps:
                continue
            step_data = results_df[results_df['step'] == step]
            log_like_avg = step_data['log_like'].mean()
            line0, = axs[0].plot(step_data['beam_width'], step_data['chi2'], label=f'Step {step}')
            line1, = axs[1].plot(step_data['beam_width'], step_data['log_like'] - log_like_avg, label=f'Step {step}')
            line2, = axs[2].plot(step_data['beam_width'], step_data['resids'], label=f'Step {step}')
            min_chi2_beam_width = step_data['beam_width'][step_data['chi2'].idxmin()]
            min_log_like_beam_width = step_data['beam_width'][step_data['log_like'].idxmin()]
            min_resid_beam_width = step_data['beam_width'][step_data['resids'].idxmin()]
            axs[0].axvline(min_chi2_beam_width, color=line0.get_color(), linestyle='--', linewidth=0.5)
            axs[1].axvline(min_log_like_beam_width, color=line1.get_color(), linestyle='--', linewidth=0.5)
            axs[2].axvline(min_resid_beam_width, color=line2.get_color(), linestyle='--', linewidth=0.5)
            min_bses.append(min_chi2_beam_width)
            min_bses.append(min_log_like_beam_width)
            min_bses.append(min_resid_beam_width)

        # Group by beam_width and compute the mean chi2
        step_avg_df = results_df.groupby('beam_width')[['chi2', 'log_like', 'resids']].mean().reset_index()
        axs[0].plot(step_avg_df['beam_width'], step_avg_df['chi2'], color='black', linestyle='--', label='Mean')
        axs[1].plot(step_avg_df['beam_width'], step_avg_df['log_like'] - step_avg_df['log_like'].mean(), color='black',
                    linestyle='--')
        axs[2].plot(step_avg_df['beam_width'], step_avg_df['resids'], color='black', linestyle='--')

        min_chi2_avg_beam_width = step_avg_df['beam_width'][step_avg_df['chi2'].idxmin()]
        min_log_like_avg_beam_width = step_avg_df['beam_width'][step_avg_df['log_like'].idxmin()]
        min_resid_avg_beam_width = step_avg_df['beam_width'][step_avg_df['resids'].idxmin()]

        opt_beam_width = Measure(min_resid_avg_beam_width, np.std(min_bses))

        axs[0].axvline(min_chi2_avg_beam_width, color='black', linestyle=':', lw=3, linewidth=0.5)
        axs[1].axvline(min_log_like_avg_beam_width, color='black', linestyle=':', lw=3, linewidth=0.5)
        axs[2].axvline(min_resid_avg_beam_width, color='black', linestyle=':', lw=3, linewidth=0.5)

        axs[2].axvspan(opt_beam_width.val - opt_beam_width.err, opt_beam_width.val + opt_beam_width.err, color='black',
                       alpha=0.2)

        max_val = results_df['resids'].max()
        axs[2].annotate(f'Minimum Scaled Residual at {opt_beam_width} cm',
                        xy=(min_resid_avg_beam_width, 0), xytext=(min_resid_avg_beam_width + 4, max_val * 0.9),
                        arrowprops=dict(arrowstyle='->', color='black'), fontsize=10, color='black')

        axs[2].set_xlabel('Beam Width (um)')
        axs[0].set_ylabel('Chi2')
        axs[1].set_ylabel('Log Likelihood')
        axs[2].set_ylabel('Scaled Residual')
        # axs.set_title('Chi2 vs Beta Star for Head-On Steps')
        axs[0].legend()
        for i in range(3):
            axs[i].grid()
        fig.tight_layout()
        fig.suptitle(f'Beam Width Fit Results - {orientation.capitalize()} Orientation', fontsize=16,
                     bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3'))
        fig.subplots_adjust(left=0.07, right=0.995, bottom=0.075, top=0.995, hspace=0.0)

        # Normalize all metrics to range of 0 to 1
        chi2_norm = (step_avg_df['chi2'] - step_avg_df['chi2'].min()) / (step_avg_df['chi2'].max() - step_avg_df['chi2'].min())
        log_like_norm = step_avg_df['log_like'] - step_avg_df['log_like'].mean()
        log_like_norm = (log_like_norm - log_like_norm.min()) / (step_avg_df['log_like'].max() - step_avg_df['log_like'].min())
        resids_norm = (step_avg_df['resids'] - step_avg_df['resids'].min()) / (step_avg_df['resids'].max() - step_avg_df['resids'].min())

        # Plot average results
        ax = ax_avgs[orientation]
        line_chi, = ax.plot(step_avg_df['beam_width'], chi2_norm, label=f'Chi2')
        line_log, = ax.plot(step_avg_df['beam_width'], log_like_norm, label=f'Log Likelihood')
        line_res, = ax.plot(step_avg_df['beam_width'], resids_norm, label=f'Scaled Residuals')
        ax.axvline(min_chi2_avg_beam_width, color=line_chi.get_color(), linestyle=':', lw=3, linewidth=0.5)
        ax.axvline(min_log_like_avg_beam_width, color=line_log.get_color(), linestyle=':', lw=3, linewidth=0.5)
        ax.axvline(min_resid_avg_beam_width, color=line_res.get_color(), linestyle=':', lw=3, linewidth=0.5)
        ax.axvspan(opt_beam_width.val - opt_beam_width.err, opt_beam_width.val + opt_beam_width.err, color='black',
                   alpha=0.2)
        ax.annotate(f'Minimum Scaled Residual at {opt_beam_width} um',
                    xy=(min_resid_avg_beam_width, 0), xytext=(min_resid_avg_beam_width + 4, 0.8),
                    arrowprops=dict(arrowstyle='->', color='black'), fontsize=10, color='black')
        ax.annotate(f'{orientation.capitalize()} Scan', xy=(0.08, 0.9), xycoords='axes fraction',
                    fontsize=18, ha='left', va='top')
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax.axhline(1, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylabel('Normalized Metrics')
        ax.legend(loc='lower right')

    ax_avgs['vertical'].set_xlabel('Beam Width (um)')
    fig_avgs.tight_layout()
    fig_avgs.subplots_adjust(hspace=0.0)


if __name__ == '__main__':
    main()

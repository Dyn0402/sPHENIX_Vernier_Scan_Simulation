#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on July 01 6:12 PM 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/rate_vs_offset_analysis.py

@author: Dylan Neff, dylan
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit as cf
from time import time
from datetime import datetime, timedelta
from itertools import product

from BunchCollider import BunchCollider
from z_vertex_fitting_common import (fit_amp_shift, fit_shift_only, get_profile_path, compute_total_chi2,
                                     load_vertex_distributions, merge_cad_rates_df, set_sim)
from Measure import Measure
from common_logistics import set_base_path

def main():
    base_path = set_base_path()
    # base_path += 'Vernier_Scans/auau_oct_16_24/'
    # base_path += 'Vernier_Scans/auau_july_17_25/'
    base_path += 'Vernier_Scans/pp_aug_12_24/'
    # plot_head_on_accuracy_for_corrections(base_path)
    # fit_beam_widths(base_path)
    # fit_beam_widths_bunch_by_bunch(base_path)
    # plot_fit_beam_widths(base_path)
    # compare_gl1_and_gl1p_rates(base_path)
    # plot_lumi_vs_step_zdc_cor(base_path)
    # lumi_test(base_path)
    bunch_by_bunch_cross_section(base_path)
    print('donzo')


def lumi_test(base_path):
    longitudinal_profiles_dir_path = f'{base_path}profiles/'
    combined_cad_step_data_csv_path = f'{base_path}combined_cad_step_data.csv'
    cad_df = pd.read_csv(combined_cad_step_data_csv_path)

    scan_step = 0
    cad_step_row = cad_df[cad_df['step'] == scan_step].iloc[0]

    collider_sim = BunchCollider()
    collider_sim.set_grid_size(30, 30, 101, 31)
    beam_width_x, beam_width_y = 130.0, 130.0
    beam_length = 1e6
    beta_star = 77.15
    bkg = 0.0e-17
    gauss_eff_width = 500
    mbd_resolution = 1.0

    collider_sim.set_bkg(bkg)
    collider_sim.set_gaus_z_efficiency_width(gauss_eff_width)
    collider_sim.set_gaus_smearing_sigma(mbd_resolution)
    collider_sim.set_bunch_beta_stars(beta_star, beta_star)
    collider_sim.set_bunch_sigmas(np.array([beam_width_x, beam_width_y, beam_length]),
                                  np.array([beam_width_x, beam_width_y, beam_length]))

    profile_path = get_profile_path(
        longitudinal_profiles_dir_path, cad_step_row['start'], cad_step_row['end'], False, None
    )
    collider_sim.set_longitudinal_profiles_from_file(
        profile_path.replace('COLOR_', 'blue_'),
        profile_path.replace('COLOR_', 'yellow_')
    )

    collider_sim.run_sim_parallel()
    naked_lumi_old = collider_sim.get_naked_luminosity_old()
    naked_lumi = collider_sim.get_naked_luminosity()
    zs, z_dist = collider_sim.get_z_density_dist()
    print(f'Naked Luminosity: {naked_lumi}, naked_lumi_old: {naked_lumi_old}, z_dist sum: {np.sum(z_dist)}, max_z: {np.max(z_dist)}')

    collider_sim.x_lim_sigma = 15
    collider_sim.y_lim_sigma = 15
    collider_sim.set_bunch_beta_stars(beta_star, beta_star)
    collider_sim.run_sim_parallel()
    naked_lumi_old = collider_sim.get_naked_luminosity_old()
    naked_lumi = collider_sim.get_naked_luminosity()
    zs, z_dist = collider_sim.get_z_density_dist()
    print(f'Naked Luminosity: {naked_lumi}, naked_lumi_old: {naked_lumi_old}, z_dist sum: {np.sum(z_dist)}, max_z: {np.max(z_dist)}')

    collider_sim.set_grid_size(45, 45, 101, 31)
    collider_sim.set_bunch_beta_stars(beta_star, beta_star)
    collider_sim.run_sim_parallel()
    naked_lumi_old = collider_sim.get_naked_luminosity_old()
    naked_lumi = collider_sim.get_naked_luminosity()
    zs, z_dist = collider_sim.get_z_density_dist()
    print(f'Naked Luminosity: {naked_lumi}, naked_lumi_old: {naked_lumi_old}, z_dist sum: {np.sum(z_dist)}, max_z: {np.max(z_dist)}')


def plot_head_on_accuracy_for_corrections(base_path):
    """
    Simulate bunch-by-bunch expectations for a vernier scan and plot the results.
    :param base_path: Base path to the vernier scan data.
    :return: None
    """
    longitudinal_profiles_dir_path = f'{base_path}profiles/'
    combined_cad_step_data_csv_path = f'{base_path}combined_cad_step_data.csv'

    f_beam = 78.4  # kHz
    mb_to_um2 = 1e-19
    # lumi_z_cut = 200
    lumi_z_cut = None
    observed = False  # True for MBD
    # observed = True  # True for MBD
    ions = 'dcct'
    # ions = 'wcm'

    if base_path.split('/')[-2] == 'auau_oct_16_24':
        beta_star = 80.3  # in cm
        norm_step = 6
    elif base_path.split('/')[-2] == 'auau_july_17_25':
        beta_star = 82.1  # in cm
        norm_step = 8
    elif base_path.split('/')[-2] == 'pp_aug_12_24':
        beta_star = 80.3 # in cm
        norm_step = 0
    else:
        raise ValueError(f'Unknown run number for base path: {base_path}')

    cad_df = pd.read_csv(combined_cad_step_data_csv_path)

    scan_steps = np.arange(0, cad_df['step'].max() + 1, 1)
    head_on_steps = [row['step'] for _, row in cad_df.iterrows() if row['set offset h'] == 0 and row['set offset v'] == 0]

    collider_sim = BunchCollider()
    collider_sim.set_grid_size(31, 31, 101, 31)
    beam_width_x, beam_width_y = 130.0, 130.0
    bkg = 0.0e-17
    gauss_eff_width = 500
    mbd_resolution = 1.0

    collider_sim.set_bkg(bkg)
    collider_sim.set_gaus_z_efficiency_width(gauss_eff_width)
    collider_sim.set_gaus_smearing_sigma(mbd_resolution)
    collider_sim.set_bunch_beta_stars(beta_star, beta_star)

    # Get nominal dcct ions and emittances
    step_0 = cad_df[cad_df['step'] == 0].iloc[0]
    em_blue_horiz_nom, em_blue_vert_nom = step_0['blue_horiz_emittance'], step_0['blue_vert_emittance']
    em_yel_horiz_nom, em_yel_vert_nom = step_0['yellow_horiz_emittance'], step_0['yellow_vert_emittance']
    em_blue_nom = (em_blue_horiz_nom, em_blue_vert_nom)
    em_yel_nom = (em_yel_horiz_nom, em_yel_vert_nom)

    rates_data = [
        {'col_name': 'zdc_cor_rate', 'name': 'ZDC Uncorrected', 'data': [], 'errs': [], 'marker': 's', 'color':'black', 'ls': '-'},
        {'col_name': 'zdc_acc_multi_cor_rate', 'name': 'ZDC Angelika Corrected', 'data': [], 'errs': [], 'marker': 's', 'color':'orange', 'ls': '-'},
        {'col_name': 'zdc_sasha_cor_rate', 'name': 'ZDC Sasha Corrected', 'data': [], 'errs': [], 'marker': 's', 'color':'green', 'ls': '-'},
        {'col_name': 'mbd_cor_rate', 'name': 'MBD Uncorrected', 'data': [], 'errs': [], 'marker': 'o', 'color':'black', 'ls': '-'},
    ]
    if 'pp_' in base_path:
        rates_data.extend([
            {'col_name': 'mbd_acc_multi_cor_rate', 'name': 'MBD Angelika Corrected', 'data': [], 'errs': [], 'marker': 'o', 'color': 'orange', 'ls': '-'},
            {'col_name': 'mbd_sasha_cor_rate', 'name': 'MBD Sasha Corrected', 'data': [], 'errs': [], 'marker': 'o', 'color': 'green', 'ls': '-'},
        ])
    if 'auau_' in base_path:
        rates_data.extend([
            {'col_name': 'mbd_z200_rate', 'name': 'MBD |z|<200 Angelika Corrected', 'data': [], 'errs': [], 'marker': 'o', 'color':'orange', 'ls': '-'},
            {'col_name': 'mbd_bkg_cor_rate', 'name': 'MBD Angelika Bkg Corrected', 'data': [], 'errs': [], 'marker': 'o', 'color':'orange', 'ls': '--'},
            {'col_name': 'mbd_sasha_z200_rate', 'name': 'MBD |z|<200 Sasha Corrected', 'data': [], 'errs': [], 'marker': 'o', 'color':'green', 'ls': '-'},
            {'col_name': 'mbd_sasha_bkg_cor_rate', 'name': 'MBD Sasha Bkg Corrected', 'data': [], 'errs': [], 'marker': 'o', 'color':'green', 'ls': '--'},
            {'col_name': 'mbd_zdc_coinc_cor_rate', 'name': 'MBD ZDC Coinc Uncorrected', 'data': [], 'errs': [], 'marker': '^', 'color': 'black', 'ls': '--'},
            {'col_name': 'mbd_zdc_coinc_acc_multi_cor_rate', 'name': 'MBD ZDC Coinc Angelika Corrected', 'data': [], 'errs': [], 'marker': '^', 'color': 'orange', 'ls': '--'},
            {'col_name': 'mbd_zdc_coinc_sasha_cor_rate', 'name': 'MBD ZDC Coinc Sasha Corrected', 'data': [], 'errs': [], 'marker': '^', 'color': 'green', 'ls': '--'},
        ])

    lumis, scan_steps_plt = [], []
    for scan_step in scan_steps:
        print(f'Scan Step: {scan_step}')
        cad_step_row = cad_df[cad_df['step'] == scan_step].iloc[0]
        profile_paths = get_profile_path(
            longitudinal_profiles_dir_path, cad_step_row['start'], cad_step_row['end'], True
        )
        for profile_path in profile_paths:
            set_sim(collider_sim, cad_step_row, beam_width_x, beam_width_y, em_blue_nom, em_yel_nom, profile_path)
            collider_sim.run_sim_parallel()
            if lumi_z_cut is None:
                naked_lumi = collider_sim.get_naked_luminosity()
            else:
                zs, z_dist = collider_sim.get_z_density_dist()
                zs_cut, z_dist_cut = zs[np.abs(zs) < lumi_z_cut], z_dist[np.abs(zs) < lumi_z_cut]
                naked_lumi = collider_sim.get_naked_luminosity(observed=observed)
                cut_fraction = np.trapezoid(z_dist_cut, zs_cut) / np.trapezoid(z_dist, zs)
                naked_lumi *= cut_fraction
            n_blue, n_yellow = cad_step_row[f'blue_{ions}_ions'], cad_step_row[f'yellow_{ions}_ions']
            lumi = naked_lumi * mb_to_um2 * f_beam * 1e3 * n_blue * n_yellow
            lumis.append(lumi)
            scan_steps_plt.append(scan_step)
            for rate_data in rates_data:
                rate_data['data'].append(cad_step_row[rate_data['col_name']])
                dur = cad_step_row['rate_calc_duration']
                rate_data['errs'].append(np.sqrt(cad_step_row[rate_data['col_name']] * dur) / dur)

    scan_steps_plt = np.array(scan_steps_plt)

    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.plot(scan_steps_plt, lumis, marker='o', linestyle='-', color='b', label='Luminosity')
    for rate_data in rates_data:
        ax2.plot(scan_steps_plt, rate_data['data'], marker='o', linestyle='-', label=rate_data['name'])
    ax.set_xlabel('Scan Step')
    ax.set_ylabel(r'Luminosity [$mb^{-1} s^{-1}$]')
    ax2.set_ylabel('ZDC Rate [Hz]')
    ax.set_title('Luminosity vs Scan Step')
    ax.legend(loc='upper center')
    ax2.legend(loc='upper right')
    ax.set_ylim(bottom=0, top=1.2 * np.max(lumis))
    ax2.set_ylim(bottom=0)
    plt.tight_layout()

    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.plot(scan_steps_plt, lumis, marker='o', linestyle='-', color='b', label='Luminosity')
    max_step_0 = np.max([rate_data['data'][0] for rate_data in rates_data])
    for rate_data in rates_data:
        scale = max_step_0 / rate_data['data'][0]
        ax2.plot(scan_steps_plt, np.array(rate_data['data']) * scale, marker='o', linestyle='-', label=rate_data['name'])
    ax.set_xlabel('Scan Step')
    ax.set_ylabel(r'Luminosity [$mb^{-1} s^{-1}$]')
    ax2.set_ylabel('ZDC Rate [Hz]')
    ax.set_title('Luminosity vs Scan Step')
    ax.legend(loc='upper center')
    ax2.legend(loc='upper right')
    ax.set_ylim(bottom=0, top=1.2 * np.max(lumis))
    ax2.set_ylim(bottom=0, top=1.2 * max_step_0)
    plt.tight_layout()

    step_mask = np.isin(scan_steps_plt, np.array(head_on_steps))

    fig, ax = plt.subplots()
    ax.axhline(0, color='k', ls='-', zorder=0)
    for rate_data in rates_data:
        # Normalize rates to the first step
        norm_rates = np.array(rate_data['data']) / rate_data['data'][norm_step] * lumis[norm_step]
        percent_rate = (norm_rates - lumis) / norm_rates * 100
        ax.plot(scan_steps_plt[step_mask], percent_rate[step_mask], marker='o', linestyle='-', label=rate_data["name"])
    ax.set_xlabel('Scan Step')
    ax.set_ylabel('Percent Difference (Data - Sim) / Data [%]')
    ax.set_title('Luminosity vs Scan Step')
    ax.legend()
    plt.tight_layout()

    # Calculate the average luminosity for each scan step and use standard deviation for error bars
    lumis = np.array(lumis)
    lumis_mean = np.array([np.mean(lumis[scan_steps_plt == step]) for step in scan_steps])
    lumis_std = np.array([np.std(lumis[scan_steps_plt == step]) for step in scan_steps])
    lumis_measures = np.array([Measure(lumis_mean[i], lumis_std[i]) for i in range(len(lumis_mean))])
    for rate_data in rates_data:
        rate_data['data'] = np.array(rate_data['data'])  # All values for each step should be the same, just get mean
        rate_data['errs'] = np.array(rate_data['errs'])
        rate_data['data_step'] = np.array([np.mean(rate_data['data'][scan_steps_plt == step]) for step in scan_steps])
        rate_data['err_step'] = np.array([np.mean(rate_data['errs'][scan_steps_plt == step]) for step in scan_steps])
        rate_data['data_measures'] = np.array([Measure(rate_data['data_step'][i], rate_data['err_step'][i]) for i in range(len(rate_data['data_step']))])

    step_mask = np.isin(scan_steps, head_on_steps)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(0, color='k', ls='-', zorder=0)
    for rate_data in rates_data:
        norm_rates = np.array(rate_data['data_measures']) / rate_data['data_measures'][norm_step] * lumis_measures[norm_step]
        percent_rate = (norm_rates - lumis_measures) / norm_rates * 100
        percent_rate_vals = np.array([m.val for m in percent_rate])
        percent_rate_errs = np.array([m.err for m in percent_rate])
        ax.errorbar(scan_steps[step_mask], percent_rate_vals[step_mask], yerr=percent_rate_errs[step_mask],
                    color=rate_data['color'], marker=rate_data['marker'], linestyle=rate_data['ls'],
                    label=rate_data['name'])
    ax.set_xlabel('Scan Step')
    ax.set_ylabel('Percent Difference (Data - Sim) / Data [%]')
    ax.set_title('Simulation Luminosity Prediction Accuracy vs Scan Step (Head On Steps Only)')
    ax.annotate(
        'Normalization Step',
        xy=(scan_steps[norm_step], percent_rate_vals[norm_step]),
        xytext=(0.2, 0.8),
        textcoords='axes fraction',
        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=4),
        ha='center',
        fontsize=10,
        color='black'
    )
    ax.legend()
    plt.tight_layout()

    plt.show()


def fit_beam_widths(base_path):
    """
    Simulate bunch-by-bunch expectations for a vernier scan and plot the results.
    :param base_path: Base path to the vernier scan data.
    :return: None
    """
    longitudinal_profiles_dir_path = f'{base_path}profiles/'
    combined_cad_step_data_csv_path = f'{base_path}combined_cad_step_data.csv'
    out_fig_path = f'{base_path}Figures/Beam_Param_Inferences/Beam_Width_Rate_Only_Fits/'
    os.makedirs(out_fig_path, exist_ok=True)
    out_csv_path = f'{base_path}beam_widths_rate_only_fit_results.csv'

    f_beam = 78.4  # kHz
    mb_to_um2 = 1e-19
    lumi_z_cut = 200
    # ions = 'dcct'
    ions = 'wcm'

    cad_df = pd.read_csv(combined_cad_step_data_csv_path)

    orientations = ['Horizontal', 'Vertical']
    # orientations = ['Horizontal']

    if base_path.split('/')[-2] == 'auau_oct_16_24':
        beta_stars = [80.3, 79, 81.5]  # in cm
        beam_widths_hz = {'Horizontal': np.linspace(120, 150, 20), 'Vertical': np.linspace(110, 140, 20)}
        beam_width_x_nom, beam_width_y_nom = 135.0, 129.0
    elif base_path.split('/')[-2] == 'auau_july_17_25':
        beta_stars = [82.1, 81.6, 82.6]  # in cm
        beam_widths_hz = {'Horizontal': np.linspace(120, 150, 20), 'Vertical': np.linspace(110, 140, 20)}
        beam_width_x_nom, beam_width_y_nom = 124.0, 114.0
    elif base_path.split('/')[-2] == 'pp_aug_12_24':
        beta_stars = [111.6, 109.6, 113.6]  # in cm
        beam_widths_hz = {'Horizontal': np.linspace(120, 200, 20), 'Vertical': np.linspace(120, 200, 20)}
        beam_width_x_nom, beam_width_y_nom = 160.0, 160.0
    else:
        raise ValueError(f'Unknown run number for base path: {base_path}')

    scan_steps = cad_df['step']
    norm_steps = [row['step'] for _, row in cad_df.iterrows() if row['set offset h'] == 0 and row['set offset v'] == 0]

    min_offset, max_offset = 150, 750  # um

    for beta_star, orientation in product(beta_stars, orientations):
        collider_sim = BunchCollider()
        collider_sim.set_grid_size(31, 31, 101, 31)
        # collider_sim.set_grid_size(21, 21, 51, 21)
        beam_widths = beam_widths_hz[orientation]  # Use the beam widths for the current orientation
        # beta_star = 80.3
        bkg = 0.0e-17
        gauss_eff_width = 500
        mbd_resolution = 1.0

        norm_bw = beam_widths[len(beam_widths) // 2]  # Use the middle beam width for normalization

        collider_sim.set_bkg(bkg)
        collider_sim.set_gaus_z_efficiency_width(gauss_eff_width)
        collider_sim.set_gaus_smearing_sigma(mbd_resolution)
        collider_sim.set_bunch_beta_stars(beta_star, beta_star)

        # Get nominal dcct ions and emittances
        step_0 = cad_df[cad_df['step'] == 0].iloc[0]
        em_blue_horiz_nom, em_blue_vert_nom = step_0['blue_horiz_emittance'], step_0['blue_vert_emittance']
        em_yel_horiz_nom, em_yel_vert_nom = step_0['yellow_horiz_emittance'], step_0['yellow_vert_emittance']
        em_blue_nom = (em_blue_horiz_nom, em_blue_vert_nom)
        em_yel_nom = (em_yel_horiz_nom, em_yel_vert_nom)

        rates_data = [
            {'col_name': 'zdc_cor_rate', 'name': 'ZDC Uncorrected', 'lumi_type': 'lumi', 'marker': 's', 'color':'black', 'ls': '-'},
            {'col_name': 'zdc_acc_multi_cor_rate', 'name': 'ZDC Angelika Corrected', 'lumi_type': 'lumi', 'marker': 's', 'color':'orange', 'ls': '-'},
            {'col_name': 'zdc_sasha_cor_rate', 'name': 'ZDC Sasha Corrected', 'lumi_type': 'lumi', 'marker': 's', 'color':'green', 'ls': '-'},
            {'col_name': 'mbd_cor_rate', 'name': 'MBD Uncorrected', 'lumi_type': 'lumi_observed', 'marker': 'o', 'color':'black', 'ls': '-'},
            {'col_name': 'mbd_z200_rate', 'name': 'MBD |z|<200 Angelika Corrected', 'lumi_type': 'lumi_z_cut', 'marker': 'o', 'color':'orange', 'ls': '-'},
            {'col_name': 'mbd_bkg_cor_rate', 'name': 'MBD Angelika Bkg Corrected', 'lumi_type': 'lumi_observed', 'marker': 'o', 'color':'orange', 'ls': '--'},
            {'col_name': 'mbd_sasha_z200_rate', 'name': 'MBD |z|<200 Sasha Corrected', 'lumi_type': 'lumi_z_cut', 'marker': 'o', 'color':'green', 'ls': '-'},
            {'col_name': 'mbd_sasha_bkg_cor_rate', 'name': 'MBD Sasha Bkg Corrected', 'lumi_type': 'lumi_observed', 'marker': 'o', 'color':'green', 'ls': '--'},
            {'col_name': 'mbd_zdc_coinc_sasha_cor_rate', 'name': 'MBD ZDC Coinc Sasha Corrected', 'lumi_type': 'lumi_observed', 'marker': 'o', 'color': 'red', 'ls': '--'},
        ]

        for rate_data in rates_data:
            rate_data.update({'data': [], 'errs': [], 'norm_scale': None,
                              'lumis': {bwx: [] for bwx in beam_widths}, 'lumi_stds': {bwx: [] for bwx in beam_widths},
                              'norm_step_lumis': {}})

        scan_steps_plt = []
        for scan_step in scan_steps:
            print(f'Scan Step: {scan_step}')
            cad_step_row = cad_df[cad_df['step'] == scan_step].iloc[0]
            if cad_step_row['orientation'] != orientation:
                continue
            offset_col = 'set offset h' if cad_step_row['orientation'] == 'Horizontal' else 'set offset v'
            if ((max_offset < abs(cad_step_row[offset_col] * 1e3) or abs(cad_step_row[offset_col] * 1e3) < min_offset)
                    and scan_step not in norm_steps):
                print(f'Skipping scan step {scan_step} with offset |{cad_step_row[offset_col] * 1e3}| < {min_offset}')
                continue
            if scan_step not in norm_steps:
                scan_steps_plt.append(scan_step)

            profile_paths = get_profile_path(
                longitudinal_profiles_dir_path, cad_step_row['start'], cad_step_row['end'], True
            )
            for beam_width in beam_widths:
                profile_lumis = {'lumi': [], 'lumi_observed': [], 'lumi_z_cut': []}
                for profile_path in profile_paths:
                    if orientation == 'Horizontal':
                        beam_width_x, beam_width_y = beam_width, beam_width_y_nom
                    elif orientation == 'Vertical':
                        beam_width_x, beam_width_y = beam_width_x_nom, beam_width
                    else:
                        raise NotImplementedError
                    set_sim(collider_sim, cad_step_row, beam_width_x, beam_width_y, em_blue_nom, em_yel_nom, profile_path)
                    collider_sim.run_sim_parallel()
                    naked_lumi = collider_sim.get_naked_luminosity()
                    zs, z_dist = collider_sim.get_z_density_dist()
                    zs_cut, z_dist_cut = zs[np.abs(zs) < lumi_z_cut], z_dist[np.abs(zs) < lumi_z_cut]
                    naked_lumi_obs = collider_sim.get_naked_luminosity(observed=True)
                    cut_fraction = np.trapezoid(z_dist_cut, zs_cut) / np.trapezoid(z_dist, zs)
                    naked_lumi_z_cut = naked_lumi_obs * cut_fraction
                    n_blue, n_yellow = cad_step_row[f'blue_{ions}_ions'], cad_step_row[f'yellow_{ions}_ions']
                    profile_lumis['lumi'].append(naked_lumi * mb_to_um2 * f_beam * 1e3 * n_blue * n_yellow)
                    profile_lumis['lumi_observed'].append(naked_lumi_obs * mb_to_um2 * f_beam * 1e3 * n_blue * n_yellow)
                    profile_lumis['lumi_z_cut'].append(naked_lumi_z_cut * mb_to_um2 * f_beam * 1e3 * n_blue * n_yellow)
                for rate_data in rates_data:
                    if scan_step in norm_steps:
                        rate_data['norm_step_lumis'][beam_width] = np.nanmean(profile_lumis[rate_data['lumi_type']])
                    else:
                        rate_data['lumis'][beam_width].append(np.nanmean(profile_lumis[rate_data['lumi_type']]))
                        rate_data['lumi_stds'][beam_width].append(np.nanstd(profile_lumis[rate_data['lumi_type']]))
            for rate_data in rates_data:
                if scan_step in norm_steps:
                    rate_data['norm_scale'] = rate_data['norm_step_lumis'][norm_bw] / cad_step_row[rate_data['col_name']]
                else:
                    rate_data['data'].append(cad_step_row[rate_data['col_name']])
                    dur = cad_step_row['rate_calc_duration']
                    rate_data['errs'].append(np.sqrt(cad_step_row[rate_data['col_name']] * dur) / dur)

        fig, ax = plt.subplots(figsize=(10, 6))
        for rate_data in rates_data:
            rate_data['data'] = np.array(rate_data['data'])  # All should be the same for each step, just get mean
            rate_data['errs'] = np.array(rate_data['errs'])

            rate_data['scaled_rate'] = rate_data['data'] * rate_data['norm_scale']
            rate_data['scaled_errs'] = rate_data['errs'] * rate_data['norm_scale']

            ax.errorbar(scan_steps_plt, rate_data['scaled_rate'], yerr=rate_data['scaled_errs'],
                        marker=rate_data['marker'], linestyle=rate_data['ls'],
                        color=rate_data['color'], label=rate_data['name'])

        chi2s = {rd['name']: {} for rd in rates_data}
        for beam_width in beam_widths:
            for rate_data in rates_data:
                lumis_mean = np.array(rate_data['lumis'][beam_width])
                lumi_std = np.array(rate_data['lumi_stds'][beam_width])
                scale = rate_data['norm_step_lumis'][norm_bw] / rate_data['norm_step_lumis'][beam_width]
                chi2s[rate_data['name']][beam_width] = (lumis_mean * scale - rate_data['scaled_rate'])**2 / (rate_data['scaled_errs']**2 + lumi_std**2)

                label = rf'Lumi $\sigma=$ {beam_width:.1f} um' if rate_data['name'] == rates_data[0]['name'] else None
                ax.errorbar(scan_steps_plt, lumis_mean * scale, yerr=lumi_std * scale, alpha=0.7,
                            marker='.', linestyle='-', label=label)

        ax.axhline(0, color='k', ls='-', zorder=0)
        ax.set_xlabel('Scan Step')
        ax.set_ylabel(r'Luminosity [$mb^{-1} s^{-1}$] / Rate [Hz]')
        ax.set_title(f'{orientation} Beam Width Scan')
        ax.legend()
        fig.tight_layout()
        fig.savefig(f'{out_fig_path}bw_{orientation}_rates_vs_step_betastar_{beta_star:.1f}.png')
        fig.savefig(f'{out_fig_path}bw_{orientation}_rates_vs_step_betastar_{beta_star:.1f}.pdf')

        df_out = []
        for rate_data in rates_data:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.axhline(0, color='k', ls='-', zorder=0)
            chi2_array = np.array([chi2s[rate_data['name']][beam_width] for beam_width in beam_widths])
            min_chi2_bws = []
            for step_i, step in enumerate(scan_steps_plt):
                step_chis = chi2_array[:, step_i]
                l = ax.plot(beam_widths, step_chis, linestyle='-', label=f'Step {step}')
                min_chi2_bw, min_chi2_value = get_minimum_chi2(step_chis, beam_widths)
                min_chi2_bws.append(min_chi2_bw)
                ax.axvline(min_chi2_bw, color=l[0].get_color(), linestyle='--')

            # Plot average chi2 across all steps
            avg_chi2 = np.mean(chi2_array, axis=1)
            min_chi2_bw, min_chi2_value = get_minimum_chi2(avg_chi2, beam_widths)
            std_min_chi2 = np.std(min_chi2_bws)
            ax.axvspan(min_chi2_bw - std_min_chi2, min_chi2_bw + std_min_chi2, alpha=0.2, color='black')
            ax.plot(beam_widths, avg_chi2, linestyle='-', color='black', label='Average', linewidth=2)
            ax.axvline(min_chi2_bw, color='black', linestyle='--', linewidth=2)
            bw_min = Measure(min_chi2_bw, std_min_chi2)
            ax.annotate(f'Beam Width: {bw_min} μm', xy=(min_chi2_bw, min_chi2_value), xycoords='data',
                        xytext=(0.1, 0.93), textcoords='axes fraction',
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=4),
                        ha='left', va='top', fontsize=10, color='black')

            df_out.append({'rate_name': rate_data['name'], 'orientation': orientation, 'beta_star': beta_star,
                           'transverse_beam_width': beam_width_x_nom if orientation == 'Horizontal' else beam_width_y_nom,
                           'lumi_type': rate_data['lumi_type'], 'lumi_z_cut': lumi_z_cut, 'ions': ions,
                           'beam_width': bw_min.val, 'beam_width_err': bw_min.err})

            ax.set_xlabel(f'{orientation} Beam Width [μm]')
            ax.set_ylabel(r'$\chi^2$')
            ax.set_title(f'Chi2 vs {orientation} Beam Width for Each Step : {rate_data["name"]}')
            ax.legend(loc='upper right')
            fig.tight_layout()
            fig.savefig(f'{out_fig_path}bw_{orientation}_rate_only_chi2_vs_bw_{rate_data["col_name"]}_betastar_{beta_star:.1f}.png')
            fig.savefig(f'{out_fig_path}bw_{orientation}_rate_only_chi2_vs_bw_{rate_data["col_name"]}_betastar_{beta_star:.1f}.pdf')

        df_out = pd.DataFrame(df_out)
        if os.path.exists(out_csv_path):
            df_existing = pd.read_csv(out_csv_path)

            # Remove existing rows that match any of the new rate_name + orientation combinations
            mask = pd.Series([True] * len(df_existing))

            for _, row in df_out.iterrows():
                mask &= ~((df_existing['rate_name'] == row['rate_name']) &
                          (df_existing['orientation'] == row['orientation']) &
                          (df_existing['beta_star'] == row['beta_star']) &
                          (df_existing['transverse_beam_width'] == row['transverse_beam_width']) &
                          (df_existing['lumi_type'] == row['lumi_type']) &
                          (df_existing['lumi_z_cut'] == row['lumi_z_cut']) &
                          (df_existing['ions'] == row['ions']))

            df_existing = df_existing[mask]

            # Append the new rows
            df_out = pd.concat([df_existing, df_out], ignore_index=True)

        # Save to CSV
        df_out.to_csv(out_csv_path, index=False)

    plt.show()


def fit_beam_widths_bunch_by_bunch(base_path):
    """
    Simulate bunch-by-bunch expectations for a vernier scan and plot the results.
    :param base_path: Base path to the vernier scan data.
    :return: None
    """
    longitudinal_profiles_dir_path = f'{base_path}profiles/'
    combined_cad_step_data_csv_path = f'{base_path}combined_cad_step_data.csv'
    gl1p_rate_data_csv_path = f'{base_path}gl1p_bunch_by_bunch_step_rates.csv'
    out_fig_path = f'{base_path}Figures/Beam_Param_Inferences/Beam_Width_Rate_Only_Fits/'
    os.makedirs(out_fig_path, exist_ok=True)
    out_csv_path = f'{base_path}beam_widths_bunch_by_bunch_rate_only_fit_results.csv'
    out_summary_csv_path = f'{base_path}beam_widths_bunch_by_bunch_rate_only_fit_results_summary.csv'

    bunches = np.arange(0, 111, 1)
    # bunches = np.arange(0, 2, 1)

    f_beam = 78.4  # kHz
    mb_to_um2 = 1e-19
    lumi_z_cut = 200
    ions = 'dcct'
    # ions = 'wcm'
    average_profiles = False  # If False, just take the middle

    cad_df = pd.read_csv(combined_cad_step_data_csv_path)
    gl1p_df = pd.read_csv(gl1p_rate_data_csv_path)
    cad_df = pd.merge(cad_df, gl1p_df, how='left', on='step')

    orientations = ['Horizontal', 'Vertical']
    # orientations = ['Horizontal']

    if base_path.split('/')[-2] == 'auau_oct_16_24':
        beta_star = 80.3  # in cm
        beam_widths_hz = {'Horizontal': np.linspace(120, 150, 20), 'Vertical': np.linspace(110, 140, 20)}
        beam_width_x_nom, beam_width_y_nom = 135.0, 129.0
    elif base_path.split('/')[-2] == 'auau_july_17_25':
        beta_star = 82.1  # in cm
        beam_widths_hz = {'Horizontal': np.linspace(120, 150, 20), 'Vertical': np.linspace(110, 140, 20)}
        beam_width_x_nom, beam_width_y_nom = 124.0, 114.0
    elif base_path.split('/')[-2] == 'pp_aug_12_24':
        beta_star = 111.6  # in cm
        beam_widths_hz = {'Horizontal': np.linspace(120, 200, 20), 'Vertical': np.linspace(120, 200, 20)}
        beam_width_x_nom, beam_width_y_nom = 160.0, 160.0
    else:
        raise ValueError(f'Unknown run number for base path: {base_path}')

    scan_steps = cad_df['step']
    norm_steps = [row['step'] for _, row in cad_df.iterrows() if row['set offset h'] == 0 and row['set offset v'] == 0]

    min_offset, max_offset = 150, 750  # um

    plot_bunches = False

    for orientation in orientations:
        collider_sim = BunchCollider()
        collider_sim.set_grid_size(31, 31, 101, 31)
        beam_widths = beam_widths_hz[orientation]
        bkg = 0.0e-17
        gauss_eff_width = 500
        mbd_resolution = 1.0

        norm_bw = beam_widths[len(beam_widths) // 2]  # Use the middle beam width for normalization

        collider_sim.set_bkg(bkg)
        collider_sim.set_gaus_z_efficiency_width(gauss_eff_width)
        collider_sim.set_gaus_smearing_sigma(mbd_resolution)
        collider_sim.set_bunch_beta_stars(beta_star, beta_star)

        # Get nominal dcct ions and emittances
        step_0 = cad_df[cad_df['step'] == 0].iloc[0]
        em_blue_horiz_nom, em_blue_vert_nom = step_0['blue_horiz_emittance'], step_0['blue_vert_emittance']
        em_yel_horiz_nom, em_yel_vert_nom = step_0['yellow_horiz_emittance'], step_0['yellow_vert_emittance']
        em_blue_nom = (em_blue_horiz_nom, em_blue_vert_nom)
        em_yel_nom = (em_yel_horiz_nom, em_yel_vert_nom)

        start_time = time()
        bw_mins = {}
        df_out = []
        for bunch_i, bunch in enumerate(bunches):
            time_elapsed = (time() - start_time) / 60  # Convert to minutes
            rate = time_elapsed / bunch_i if bunch_i > 0 else 0
            est_time_remaining = rate * (len(bunches) - bunch_i) if bunch_i != 0 else 0
            finish_time = datetime.now() + timedelta(minutes=est_time_remaining)
            print(f'Bunch: {bunch}, time elapsed: {time_elapsed:.1f} min, time remaining: {est_time_remaining:.1f} min, '
                  f'est. finish: {finish_time.strftime("%H:%M")}')
            rates_data = [
                {'col_name': 'zdc_cor_rate', 'name': 'ZDC Uncorrected', 'lumi_type': 'lumi', 'marker': 's', 'color':'black', 'ls': '-'},
                {'col_name': 'zdc_acc_multi_cor_rate', 'name': 'ZDC Angelika Corrected', 'lumi_type': 'lumi', 'marker': 's', 'color':'orange', 'ls': '-'},
                {'col_name': 'zdc_sasha_cor_rate', 'name': 'ZDC Sasha Corrected', 'lumi_type': 'lumi', 'marker': 's', 'color':'green', 'ls': '-'},
                {'col_name': 'mbd_cor_rate', 'name': 'MBD Uncorrected', 'lumi_type': 'lumi_observed', 'marker': 'o', 'color':'black', 'ls': '-'},
                {'col_name': 'mbd_z200_rate', 'name': 'MBD |z|<200 Angelika Corrected', 'lumi_type': 'lumi_z_cut', 'marker': 'o', 'color':'orange', 'ls': '-'},
                {'col_name': 'mbd_bkg_cor_rate', 'name': 'MBD Angelika Bkg Corrected', 'lumi_type': 'lumi_observed', 'marker': 'o', 'color':'orange', 'ls': '--'},
                {'col_name': 'mbd_sasha_z200_rate', 'name': 'MBD |z|<200 Sasha Corrected', 'lumi_type': 'lumi_z_cut', 'marker': 'o', 'color':'green', 'ls': '-'},
                {'col_name': 'mbd_sasha_bkg_cor_rate', 'name': 'MBD Sasha Bkg Corrected', 'lumi_type': 'lumi_observed', 'marker': 'o', 'color':'green', 'ls': '--'},
                {'col_name': 'mbd_zdc_coinc_sasha_cor_rate', 'name': 'MBD ZDC Coinc Sasha Corrected', 'lumi_type': 'lumi_observed', 'marker': 'o', 'color': 'red', 'ls': '--'},
            ]
            for rate_data in rates_data:
                rate_data.update({'data': [], 'errs': [], 'norm_scale': None, 'lumis': {bwx: [] for bwx in beam_widths},
                                  'lumi_stds': {bwx: [] for bwx in beam_widths}, 'norm_step_lumis': {}})

            scan_steps_plt = []
            for scan_step in scan_steps:
                cad_step_row = cad_df[cad_df['step'] == scan_step].iloc[0]
                if cad_step_row['orientation'] != orientation:
                    continue
                offset_col = 'set offset h' if cad_step_row['orientation'] == 'Horizontal' else 'set offset v'
                if ((max_offset < abs(cad_step_row[offset_col] * 1e3) or abs(cad_step_row[offset_col] * 1e3) < min_offset)
                        and scan_step not in norm_steps):
                    print(
                        f'Skipping scan step {scan_step} with offset |{cad_step_row[offset_col] * 1e3}| < {min_offset}')
                    continue
                if scan_step not in norm_steps:
                    scan_steps_plt.append(scan_step)

                profile_paths = get_profile_path(longitudinal_profiles_dir_path, cad_step_row['start'], cad_step_row['end'],
                                                 True, bunch_num=bunch)
                if not average_profiles:
                    profile_paths = [profile_paths[len(profile_paths) // 2]]
                for beam_width in beam_widths:
                    profile_lumis = {'lumi': [], 'lumi_observed': [], 'lumi_z_cut': []}
                    for profile_path in profile_paths:
                        if orientation == 'Horizontal':
                            beam_width_x, beam_width_y = beam_width, beam_width_y_nom
                        elif orientation == 'Vertical':
                            beam_width_x, beam_width_y = beam_width_x_nom, beam_width
                        else:
                            raise NotImplementedError
                        set_sim(collider_sim, cad_step_row, beam_width_x, beam_width_y, em_blue_nom, em_yel_nom, profile_path)
                        collider_sim.run_sim_parallel()
                        naked_lumi = collider_sim.get_naked_luminosity()
                        zs, z_dist = collider_sim.get_z_density_dist()
                        zs_cut, z_dist_cut = zs[np.abs(zs) < lumi_z_cut], z_dist[np.abs(zs) < lumi_z_cut]
                        naked_lumi_obs = collider_sim.get_naked_luminosity(observed=True)
                        cut_fraction = np.trapezoid(z_dist_cut, zs_cut) / np.trapezoid(z_dist, zs)
                        naked_lumi_z_cut = naked_lumi_obs * cut_fraction
                        n_blue, n_yellow = cad_step_row[f'blue_{ions}_ions'], cad_step_row[f'yellow_{ions}_ions']
                        profile_lumis['lumi'].append(naked_lumi * mb_to_um2 * f_beam * 1e3 * n_blue * n_yellow)
                        profile_lumis['lumi_observed'].append(
                            naked_lumi_obs * mb_to_um2 * f_beam * 1e3 * n_blue * n_yellow)
                        profile_lumis['lumi_z_cut'].append(
                            naked_lumi_z_cut * mb_to_um2 * f_beam * 1e3 * n_blue * n_yellow)
                    for rate_data in rates_data:
                        if scan_step in norm_steps:
                            rate_data['norm_step_lumis'][beam_width] = np.nanmean(profile_lumis[rate_data['lumi_type']])
                        else:
                            rate_data['lumis'][beam_width].append(np.nanmean(profile_lumis[rate_data['lumi_type']]))
                            rate_data['lumi_stds'][beam_width].append(np.nanstd(profile_lumis[rate_data['lumi_type']]))
                for rate_data in rates_data:
                    col_name = rate_data['col_name'].replace('_bunch_x_', f'_bunch_{bunch}_')

                    if cad_step_row[col_name] < 0:
                        print(f'Step {scan_step} has negative rate {cad_step_row[col_name]} for {col_name}, setting to 0.')
                        if scan_step in norm_steps:
                            rate_data['norm_scale'] = rate_data['norm_step_lumis'][norm_bw] / 1
                        else:
                            rate_data['data'].append(0.0)
                            rate_data['errs'].append(1.0)

                    if scan_step in norm_steps:
                        rate_data['norm_scale'] = rate_data['norm_step_lumis'][norm_bw] / cad_step_row[col_name]
                    else:
                        rate_data['data'].append(cad_step_row[col_name])
                        dur = cad_step_row['rate_calc_duration']
                        rate_data['errs'].append(np.sqrt(cad_step_row[col_name] * dur) / dur)

            for rate_data in rates_data:
                rate_data['data'] = np.array(rate_data['data'])  # All should be the same for each step, just get mean
                rate_data['errs'] = np.array(rate_data['errs'])

                rate_data['scaled_rate'] = rate_data['data'] * rate_data['norm_scale']
                rate_data['scaled_errs'] = rate_data['errs'] * rate_data['norm_scale']

            chi2s = {rd['name']: {} for rd in rates_data}
            for beam_width in beam_widths:
                for rate_data in rates_data:
                    lumis_mean = np.array(rate_data['lumis'][beam_width])
                    lumi_std = np.array(rate_data['lumi_stds'][beam_width])
                    scale = rate_data['norm_step_lumis'][norm_bw] / rate_data['norm_step_lumis'][beam_width]
                    chi2s[rate_data['name']][beam_width] = (lumis_mean * scale - rate_data['scaled_rate']) ** 2 / (
                                rate_data['scaled_errs'] ** 2 + lumi_std ** 2)

            for rate_data in rates_data:
                chi2_array = np.array([chi2s[rate_data['name']][beam_width] for beam_width in beam_widths])
                min_chi2_bws = []
                for step_i, step in enumerate(scan_steps_plt):
                    step_chis = chi2_array[:, step_i]
                    min_chi2_bw, min_chi2_value = get_minimum_chi2(step_chis, beam_widths)
                    min_chi2_bws.append(min_chi2_bw)

                # Plot average chi2 across all steps
                avg_chi2 = np.mean(chi2_array, axis=1)
                min_chi2_bw, min_chi2_value = get_minimum_chi2(avg_chi2, beam_widths)
                std_min_chi2 = np.std(min_chi2_bws)
                bw_min = Measure(min_chi2_bw, std_min_chi2)
                if rate_data['name'] not in bw_mins:
                    bw_mins[rate_data['name']] = [bw_min]
                else:
                    bw_mins[rate_data['name']].append(bw_min)

                df_out.append({
                    'bunch': bunch, 'orientation': orientation, 'beta_star': beta_star,
                    'transverse_beam_width': beam_width_x_nom if orientation == 'Horizontal' else beam_width_y_nom,
                    'lumi_type': rate_data['lumi_type'], 'lumi_z_cut': lumi_z_cut,
                    'rate_name': rate_data['name'], 'ions': ions, 'average_profiles': average_profiles,
                    'beam_width': bw_min.val, 'beam_width_err': bw_min.err
                })

            if plot_bunches:
                fig, ax = plt.subplots(figsize=(10, 6))
                for rate_data in rates_data:
                    rate_data['data'] = np.array(rate_data['data'])  # All should be the same for each step, just get mean
                    rate_data['errs'] = np.array(rate_data['errs'])

                    rate_data['scaled_rate'] = rate_data['data'] * rate_data['norm_scale']
                    rate_data['scaled_errs'] = rate_data['errs'] * rate_data['norm_scale']

                    ax.errorbar(scan_steps_plt, rate_data['scaled_rate'], yerr=rate_data['scaled_errs'],
                                marker=rate_data['marker'], linestyle=rate_data['ls'],
                                color=rate_data['color'], label=rate_data['name'], zorder=10)

                chi2s = {rd['name']: {} for rd in rates_data}
                for beam_width in beam_widths:
                    # lumis_mean = np.array(lumis[beam_width_x])
                    # lumi_std = np.array(lumi_stds[beam_width_x])
                    # scale = norm_step_lumis[norm_bw] / norm_step_lumis[beam_width_x]
                    #
                    # for rate_data in rates_data:
                    #     chi2s[rate_data['name']][beam_width_x] = (lumis_mean * scale - rate_data['scaled_rate'])**2 / (rate_data['scaled_errs']**2 + lumi_std**2)
                    for rate_data in rates_data:
                        lumis_mean = np.array(rate_data['lumis'][beam_width])
                        lumi_std = np.array(rate_data['lumi_stds'][beam_width])
                        scale = rate_data['norm_step_lumis'][norm_bw] / rate_data['norm_step_lumis'][beam_width]
                        chi2s[rate_data['name']][beam_width] = (lumis_mean * scale - rate_data['scaled_rate']) ** 2 / (
                                    rate_data['scaled_errs'] ** 2 + lumi_std ** 2)

                        label = rf'Lumi $\sigma=$ {beam_width:.1f} um' if rate_data['name'] == rates_data[0][
                            'name'] else None
                        ax.errorbar(scan_steps_plt, lumis_mean * scale, yerr=lumi_std * scale, alpha=0.7,
                                    marker='.', linestyle='-', label=label)

                ax.axhline(0, color='k', ls='-', zorder=0)
                ax.set_xlabel('Scan Step')
                ax.set_ylabel(r'Luminosity [$mb^{-1} s^{-1}$] (Rate Scaled to Lumi)')
                ax.set_title(f'Beam Width X Scan: Bunch {bunch}')
                ax.legend()
                fig.tight_layout()

                for rate_data in rates_data:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.axhline(0, color='k', ls='-', zorder=0)
                    chi2_array = np.array([chi2s[rate_data['name']][beam_width] for beam_width in beam_widths])
                    min_chi2_bws = []
                    for step_i, step in enumerate(scan_steps_plt):
                        step_chis = chi2_array[:, step_i]
                        l = ax.plot(beam_widths, step_chis, linestyle='-', label=f'Step {step}')
                        min_chi2_bw, min_chi2_value = get_minimum_chi2(step_chis, beam_widths)
                        min_chi2_bws.append(min_chi2_bw)
                        ax.axvline(min_chi2_bw, color=l[0].get_color(), linestyle='--')

                    # Plot average chi2 across all steps
                    avg_chi2 = np.mean(chi2_array, axis=1)
                    min_chi2_bw, min_chi2_value = get_minimum_chi2(avg_chi2, beam_widths)
                    std_min_chi2 = np.std(min_chi2_bws)
                    ax.axvspan(min_chi2_bw - std_min_chi2, min_chi2_bw + std_min_chi2, alpha=0.2, color='black')
                    ax.plot(beam_widths, avg_chi2, linestyle='-', color='black', label='Average', linewidth=2)
                    ax.axvline(min_chi2_bw, color='black', linestyle='--', linewidth=2)
                    bw_min = Measure(min_chi2_bw, std_min_chi2)
                    ax.annotate(f'Beam Width: {bw_min} μm', xy=(min_chi2_bw, min_chi2_value), xycoords='data',
                                xytext=(0.1, 0.93), textcoords='axes fraction',
                                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=4),
                                ha='left', va='top', fontsize=10, color='black')

                    ax.set_xlabel(f'{orientation} Beam Width [μm]')
                    ax.set_ylabel(r'$\chi^2$')
                    ax.set_title(f'Chi2 vs {orientation} Beam Width, Bunch {bunch}, {rate_data["name"]}')
                    ax.legend(loc='upper right')
                    fig.tight_layout()

        df_summary_out = []
        for rate_data_name, bw_mins_i in bw_mins.items():
            fig, ax = plt.subplots(figsize=(10, 6))
            bw_min_vals, bw_min_errs = [m.val for m in bw_mins_i], [m.err for m in bw_mins_i]
            ax.errorbar(bunches, bw_min_vals, yerr=bw_min_errs, marker='.', linestyle='none', color='black')
            ax.set_xlabel('Bunch Number')
            ax.set_ylabel(f'{orientation} Beam Width [μm]')
            ax.set_title(f'{orientation} Beam Width Min vs Bunch Number for {rate_data_name} ({ions})')
            mean_bw_min = np.mean(bw_mins_i)
            std_bw_min = np.std(bw_mins_i)
            ax.axhspan(mean_bw_min.val - std_bw_min.val, mean_bw_min.val + std_bw_min.val, alpha=0.2, color='k')
            ax.axhspan(mean_bw_min.val - mean_bw_min.err, mean_bw_min.val + mean_bw_min.err, alpha=0.4, color='k')
            ax.axhline(mean_bw_min.val, color='k', linestyle='-', label='Mean')
            ax.annotate(f'Bunch Averaged {orientation} Beam Width: {mean_bw_min}', xy=(0.05, 0.95), xycoords='axes fraction',
                        textcoords='axes fraction', fontsize=10, ha='left', va='top', color='k')

            fig.tight_layout()
            fig.savefig(f'{out_fig_path}{orientation}_bunch_by_bunch_bw_min_{rate_data_name}_{ions}.png')
            fig.savefig(f'{out_fig_path}{orientation}_bunch_by_bunch_bw_min_{rate_data_name}_{ions}.pdf')

            # Make a histogram of the beam width mins
            fig, ax = plt.subplots(figsize=(10, 6))
            counts, bins = np.histogram(bw_min_vals, bins=20)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            ax.bar(bin_centers, counts, width=np.diff(bins), align='center', alpha=0.7, color='blue', edgecolor='black')
            try:
                popt, pcov = cf(gaus, bin_centers, counts, p0=[np.max(counts), np.mean(bw_min_vals), np.std(bw_min_vals)])
                perr = np.sqrt(np.diag(pcov))
                xs_plot = np.linspace(bins[0], bins[-1], 250)
                ax.plot(xs_plot, gaus(xs_plot, *popt), color='red')
                pmeas = [Measure(v, e) for v, e in zip(popt, perr)]
                ax.annotate(f'Gaussian Fit:\nAmplitude: {pmeas[0]}\nMean: {pmeas[1]} μm\nStd Dev: {pmeas[2]} μm',
                            xy=(0.03, 0.97), xycoords='axes fraction', textcoords='axes fraction',
                            fontsize=10, ha='left', va='top', color='black')
            except RuntimeError:
                print(f'Could not fit Gaussian to {orientation} beam width mins for {rate_data_name} ({ions})')
            ax.set_xlabel('Beam Width X [μm]')
            ax.set_ylabel('Count')
            ax.set_title(f'Bunch by Bunch {orientation} Beam Width for {rate_data_name} ({ions})')
            fig.tight_layout()

            fig.savefig(f'{out_fig_path}{orientation}_bunch_by_bunch_bw_hist_{rate_data_name}_{ions}.png')
            fig.savefig(f'{out_fig_path}{orientation}_bunch_by_bunch_bw_hist_{rate_data_name}_{ions}.pdf')

            df_summary_out.append({
                'orientation': orientation, 'beta_star': beta_star, 'lumi_type': rate_data['lumi_type'],
                'lumi_z_cut': lumi_z_cut, 'rate_name': rate_data_name,
                'transverse_beam_width': beam_width_x_nom if orientation == 'Horizontal' else beam_width_y_nom,
                'ions': ions, 'average_profiles': average_profiles,
                'beam_width': mean_bw_min.val, 'beam_width_err': mean_bw_min.err,
                'beam_width_std': std_bw_min.val, 'beam_width_std_err': std_bw_min.err,
            }
            )

        df_out = pd.DataFrame(df_out)
        if os.path.exists(out_csv_path):
            df_existing = pd.read_csv(out_csv_path)

            # Remove existing rows that match any of the new rate_name + orientation combinations
            mask = pd.Series([True] * len(df_existing))

            for _, row in df_out.iterrows():
                mask &= ~((df_existing['rate_name'] == row['rate_name']) &
                          (df_existing['orientation'] == row['orientation']) &
                          (df_existing['beta_star'] == row['beta_star']) &
                          (df_existing['lumi_type'] == row['lumi_type']) &
                          (df_existing['lumi_z_cut'] == row['lumi_z_cut']) &
                            (df_existing['ions'] == row['ions']) &
                          (df_existing['average_profiles'] == row['average_profiles']))
            df_existing = df_existing[mask]

            # Append the new rows
            df_out = pd.concat([df_existing, df_out], ignore_index=True)

        # Save to CSV
        df_out.to_csv(out_csv_path, index=False)

        df_summary_out = pd.DataFrame(df_summary_out)
        if os.path.exists(out_summary_csv_path):
            df_existing_summary = pd.read_csv(out_summary_csv_path)

            # Remove existing rows that match any of the new rate_name + orientation combinations
            mask = pd.Series([True] * len(df_existing_summary))

            for _, row in df_summary_out.iterrows():
                mask &= ~((df_existing_summary['rate_name'] == row['rate_name']) &
                          (df_existing_summary['orientation'] == row['orientation']) &
                          (df_existing_summary['beta_star'] == row['beta_star']) &
                          (df_existing_summary['lumi_type'] == row['lumi_type']) &
                          (df_existing_summary['lumi_z_cut'] == row['lumi_z_cut']) &
                            (df_existing_summary['ions'] == row['ions']) &
                          (df_existing_summary['average_profiles'] == row['average_profiles']))
            df_existing_summary = df_existing_summary[mask]

            # Append the new rows
            df_summary_out = pd.concat([df_existing_summary, df_summary_out], ignore_index=True)

        # Save to CSV
        df_summary_out.to_csv(out_summary_csv_path, index=False)

    plt.show()


def plot_fit_beam_widths(base_path):
    """
    Plot the fitted beam widths for the various rates.
    :param base_path: Base path to the vernier scan data.
    """
    out_csv_path = f'{base_path}beam_widths_rate_only_fit_results.csv'
    out_fig_path = f'{base_path}Figures/Beam_Param_Inferences/'
    df = pd.read_csv(out_csv_path)
    orientations = ['Horizontal', 'Vertical']
    fig_both, ax_both = plt.subplots(figsize=(10, 6))

    for orientation in orientations:
        df_orientation = df[df['orientation'] == orientation]
        df_orientation = df_orientation.sort_values(by='rate_name')  # Sort by rate_name
        ax_both.errorbar(df_orientation['beam_width'], df_orientation['rate_name'],
                         xerr=df_orientation['beam_width_err'],
                         marker='s', linestyle='none', label=f'{orientation} Beam Widths')

    ax_both.set_xlabel('Beam Width [μm]')
    ax_both.set_title('Fitted Beam Widths for Different Relative Rate Estimates')
    ax_both.legend()
    fig_both.tight_layout()
    fig_both.savefig(f'{out_fig_path}fitted_rate_only_beam_widths.png')
    fig_both.savefig(f'{out_fig_path}fitted_rate_only_beam_widths.pdf')

    plt.show()


def gaus(x, a, b, c):
    """
    Gaussian function.
    :param x: Input value.
    :param a: Amplitude.
    :param b: Mean.
    :param c: Standard deviation.
    :return: Gaussian value at x.
    """
    return a * np.exp(-((x - b) ** 2) / (2 * c ** 2))


def compare_gl1_and_gl1p_rates(base_path):
    """
    Simulate bunch-by-bunch expectations for a vernier scan and plot the results.
    :param base_path: Base path to the vernier scan data.
    :return: None
    """
    longitudinal_profiles_dir_path = f'{base_path}profiles/'
    combined_cad_step_data_csv_path = f'{base_path}combined_cad_step_data.csv'
    gl1p_rate_data_csv_path = f'{base_path}gl1p_bunch_by_bunch_step_rates.csv'

    f_beam = 78.4  # kHz
    mb_to_um2 = 1e-19
    # lumi_z_cut = 200
    lumi_z_cut = None
    observed = False  # True for MBD

    cad_df = pd.read_csv(combined_cad_step_data_csv_path)
    gl1p_df = pd.read_csv(gl1p_rate_data_csv_path)
    cad_df = pd.merge(cad_df, gl1p_df, how='left', on='step')

    # scan_steps = np.arange(0, 25, 1)
    scan_steps = np.arange(0, 12, 1)
    norm_step = 0  # Step to normalize the rates to
    # scan_steps = np.arange(0, 4, 1)

    min_offset = 0  # um
    max_offset = 1650  # um

    plot_bunches = False

    collider_sim = BunchCollider()
    collider_sim.set_grid_size(31, 31, 101, 31)
    beam_width_x, beam_width_y = 130.0, 130.0
    beam_width_xs = np.linspace(120, 170, 9)
    beta_star = 76.4
    bkg = 0.0e-17
    gauss_eff_width = 500
    mbd_resolution = 1.0
    # gauss_eff_width = None
    # mbd_resolution = None

    norm_bw = beam_width_xs[len(beam_width_xs) // 2]  # Use the middle beam width for normalization

    collider_sim.set_bkg(bkg)
    collider_sim.set_gaus_z_efficiency_width(gauss_eff_width)
    collider_sim.set_gaus_smearing_sigma(mbd_resolution)
    collider_sim.set_bunch_beta_stars(beta_star, beta_star)

    # Get nominal dcct ions and emittances
    step_0 = cad_df[cad_df['step'] == 0].iloc[0]
    em_blue_horiz_nom, em_blue_vert_nom = step_0['blue_horiz_emittance'], step_0['blue_vert_emittance']
    em_yel_horiz_nom, em_yel_vert_nom = step_0['yellow_horiz_emittance'], step_0['yellow_vert_emittance']
    em_blue_nom = (em_blue_horiz_nom, em_blue_vert_nom)
    em_yel_nom = (em_yel_horiz_nom, em_yel_vert_nom)

    rates_data = [
        # {'col_name': 'zdc_cor_rate', 'name': 'ZDC Uncorrected', 'data': [], 'errs': [], 'norm_scale': None, 'marker': 's', 'color':'black', 'ls': '-'},
        # {'col_name': 'zdc_acc_multi_cor_rate', 'name': 'ZDC Angelika Corrected', 'data': [], 'errs': [], 'norm_scale': None, 'marker': 's', 'color':'orange', 'ls': '-'},
        # {'col_name': 'zdc_sasha_cor_rate', 'name': 'ZDC Sasha Corrected', 'data': [], 'errs': [], 'norm_scale': None,
        #  'marker': 's', 'color': 'green', 'ls': '-'},
        {'col_name': 'mbd_cor_rate', 'name': 'MBD Uncorrected', 'data': [], 'errs': [], 'norm_scale': None, 'marker': 'o', 'color':'black', 'ls': '-'},
        # {'col_name': 'mbd_z200_rate', 'name': 'MBD |z|<200 Angelika Corrected', 'data': [], 'errs': [], 'norm_scale': None, 'marker': 'o', 'color':'orange', 'ls': '-'},
        # {'col_name': 'mbd_bkg_cor_rate', 'name': 'MBD Angelika Bkg Corrected', 'data': [], 'errs': [], 'norm_scale': None, 'marker': 'o', 'color':'orange', 'ls': '--'},
        {'col_name': 'mbd_sasha_z200_rate', 'name': 'MBD |z|<200 Sasha Corrected', 'data': [], 'errs': [],
         'norm_scale': None, 'marker': 'o', 'color': 'green', 'ls': '-'},
        # {'col_name': 'mbd_sasha_bkg_cor_rate', 'name': 'MBD Sasha Bkg Corrected', 'data': [], 'errs': [],
        #  'norm_scale': None, 'marker': 'o', 'color': 'green', 'ls': '--'},
    ]

    lumis, lumi_stds, scan_steps_plt = {bwx: [] for bwx in beam_width_xs}, {bwx: [] for bwx in beam_width_xs}, []
    norm_step_lumis = {}
    norm_step_lumis[norm_bw] = 100
    for scan_step in scan_steps:
        cad_step_row = cad_df[cad_df['step'] == scan_step].iloc[0]
        offset_col = 'set offset h' if cad_step_row['orientation'] == 'Horizontal' else 'set offset v'
        if ((max_offset < abs(cad_step_row[offset_col] * 1e3) or abs(cad_step_row[offset_col] * 1e3) < min_offset)
                and scan_step != norm_step):
            continue
        if scan_step != norm_step:
            scan_steps_plt.append(scan_step)
        print(f' Scan Step: {scan_step}')

        # profile_paths = get_profile_path(longitudinal_profiles_dir_path, cad_step_row['start'], cad_step_row['end'],
        #                                  True)
        # for beam_width_x in beam_width_xs:
        #     profile_lumis = []
        #     for profile_path in profile_paths:
        #         set_sim(collider_sim, cad_step_row, beam_width_x, beam_width_y, em_blue_nom, em_yel_nom, profile_path)
        #         collider_sim.run_sim_parallel()
        #         if lumi_z_cut is None:
        #             naked_lumi = collider_sim.get_naked_luminosity()
        #         else:
        #             zs, z_dist = collider_sim.get_z_density_dist()
        #             zs_cut, z_dist_cut = zs[np.abs(zs) < lumi_z_cut], z_dist[np.abs(zs) < lumi_z_cut]
        #             naked_lumi = collider_sim.get_naked_luminosity(observed=observed)
        #             cut_fraction = np.trapezoid(z_dist_cut, zs_cut) / np.trapezoid(z_dist, zs)
        #             naked_lumi *= cut_fraction
        #         n_blue, n_yellow = cad_step_row['blue_wcm_ions'], cad_step_row['yellow_wcm_ions']
        #         lumi = naked_lumi * mb_to_um2 * f_beam * 1e3 * n_blue * n_yellow
        #         profile_lumis.append(lumi)
        #     lumi_mean, lumi_std = np.nanmean(profile_lumis), np.nanstd(profile_lumis)
        #     if scan_step == norm_step:
        #         norm_step_lumis[beam_width_x] = lumi_mean
        #     else:
        #         lumis[beam_width_x].append(lumi_mean)
        #         lumi_stds[beam_width_x].append(lumi_std)
        for rate_data in rates_data:
            col_name = rate_data['col_name']
            if cad_step_row[col_name] < 0:
                print(f'Step {scan_step} has negative rate {cad_step_row[col_name]} for {col_name}, setting to 0.')
                if scan_step == norm_step:
                    rate_data['norm_scale'] = norm_step_lumis[norm_bw] / 1
                else:
                    rate_data['data'].append(0.0)
                    rate_data['errs'].append(1.0)

            if scan_step == norm_step:
                rate_data['norm_scale'] = norm_step_lumis[norm_bw] / cad_step_row[col_name]
            else:
                rate_data['data'].append(cad_step_row[col_name])
                dur = cad_step_row['rate_calc_duration']
                rate_data['errs'].append(np.sqrt(cad_step_row[col_name] * dur) / dur)

    fig, ax = plt.subplots(figsize=(10, 6))
    for rate_data in rates_data:
        rate_data['data'] = np.array(rate_data['data'])  # All should be the same for each step, just get mean
        rate_data['errs'] = np.array(rate_data['errs'])
        rate_data['scaled_rate'] = rate_data['data'] * rate_data['norm_scale']
        rate_data['scaled_errs'] = rate_data['errs'] * rate_data['norm_scale']
        ax.errorbar(scan_steps_plt, rate_data['scaled_rate'], yerr=rate_data['scaled_errs'],
                    marker=rate_data['marker'], linestyle=rate_data['ls'],
                    color=rate_data['color'], label=rate_data['name'])

    bunches = np.arange(0, 111, 1)
    for bunch in bunches:
        print(f'Bunch: {bunch}')
        rates_data = [
            # {'col_name': 'zdc_cor_bunch_x_rate', 'name': 'ZDC Uncorrected', 'data': [], 'errs': [], 'norm_scale': None, 'marker': 's', 'color':'black', 'ls': '-'},
            # {'col_name': 'zdc_acc_multi_cor_bunch_x_rate', 'name': 'ZDC Angelika Corrected', 'data': [], 'errs': [], 'norm_scale': None, 'marker': 's', 'color':'orange', 'ls': '-'},
            # {'col_name': 'zdc_sasha_cor_bunch_x_rate', 'name': 'ZDC Sasha Corrected', 'data': [], 'errs': [], 'norm_scale': None, 'marker': 's', 'color':'green', 'ls': '-'},
            # {'col_name': 'mbd_cor_bunch_x_rate', 'name': 'MBD Uncorrected', 'data': [], 'errs': [], 'norm_scale': None, 'marker': 'o', 'color':'black', 'ls': '-'},
            # {'col_name': 'mbd_z200_bunch_x_rate', 'name': 'MBD |z|<200 Angelika Corrected', 'data': [], 'errs': [], 'norm_scale': None, 'marker': 'o', 'color':'orange', 'ls': '-'},
            # {'col_name': 'mbd_z200_uncor_bunch_x_rate', 'name': 'MBD |z|<200 Uncorrected', 'data': [], 'errs': [], 'norm_scale': None, 'marker': 'o', 'color':'orange', 'ls': '-'},
            # {'col_name': 'mbd_bkg_cor_bunch_x_rate', 'name': 'MBD Angelika Bkg Corrected', 'data': [], 'errs': [], 'norm_scale': None, 'marker': 'o', 'color':'orange', 'ls': '--'},
            {'col_name': 'mbd_sasha_z200_bunch_x_rate', 'name': 'MBD |z|<200 Sasha Corrected', 'data': [], 'errs': [], 'norm_scale': None, 'marker': 'o', 'color':'green', 'ls': '-'},
            # {'col_name': 'mbd_sasha_bkg_cor_bunch_x_rate', 'name': 'MBD Sasha Bkg Corrected', 'data': [], 'errs': [], 'norm_scale': None, 'marker': 'o', 'color':'green', 'ls': '--'},
        ]

        lumis, lumi_stds, scan_steps_plt = {bwx: [] for bwx in beam_width_xs}, {bwx: [] for bwx in beam_width_xs}, []
        # norm_step_lumis = {}
        for scan_step in scan_steps:
            cad_step_row = cad_df[cad_df['step'] == scan_step].iloc[0]
            offset_col = 'set offset h' if cad_step_row['orientation'] == 'Horizontal' else 'set offset v'
            if ((max_offset < abs(cad_step_row[offset_col] * 1e3) or abs(cad_step_row[offset_col] * 1e3) < min_offset)
                    and scan_step != norm_step):
                continue
            if scan_step != norm_step:
                scan_steps_plt.append(scan_step)
            print(f' Scan Step: {scan_step}')

            # profile_paths = get_profile_path(longitudinal_profiles_dir_path, cad_step_row['start'], cad_step_row['end'],
            #                                  True, bunch_num=bunch)
            # for beam_width_x in beam_width_xs:
                # profile_lumis = []
                # for profile_path in profile_paths:
                #     set_sim(collider_sim, cad_step_row, beam_width_x, beam_width_y, em_blue_nom, em_yel_nom, profile_path)
                #     collider_sim.run_sim_parallel()
                #     if lumi_z_cut is None:
                #         naked_lumi = collider_sim.get_naked_luminosity()
                #     else:
                #         zs, z_dist = collider_sim.get_z_density_dist()
                #         zs_cut, z_dist_cut = zs[np.abs(zs) < lumi_z_cut], z_dist[np.abs(zs) < lumi_z_cut]
                #         naked_lumi = collider_sim.get_naked_luminosity(observed=observed)
                #         cut_fraction = np.trapezoid(z_dist_cut, zs_cut) / np.trapezoid(z_dist, zs)
                #         naked_lumi *= cut_fraction
                #     n_blue, n_yellow = cad_step_row['blue_wcm_ions'], cad_step_row['yellow_wcm_ions']
                #     lumi = naked_lumi * mb_to_um2 * f_beam * 1e3 * n_blue * n_yellow
                #     profile_lumis.append(lumi)
                # lumi_mean, lumi_std = np.nanmean(profile_lumis), np.nanstd(profile_lumis)
                # if scan_step == norm_step:
                #     norm_step_lumis[beam_width_x] = lumi_mean
                # else:
                #     lumis[beam_width_x].append(lumi_mean)
                #     lumi_stds[beam_width_x].append(lumi_std)
            for rate_data in rates_data:
                col_name = rate_data['col_name'].replace('_bunch_x_', f'_bunch_{bunch}_')
                # if cad_step_row[col_name] < 0:
                #     print(f'Step {scan_step} has negative rate {cad_step_row[col_name]} for {col_name}, setting to 0.')
                #     if scan_step == norm_step:
                #         rate_data['norm_scale'] = norm_step_lumis[norm_bw] / 1
                #     else:
                #         rate_data['data'].append(0.0)
                #         rate_data['errs'].append(1.0)

                if scan_step == norm_step:
                    rate_data['norm_scale'] = norm_step_lumis[norm_bw] / cad_step_row[col_name]
                else:
                    rate_data['data'].append(cad_step_row[col_name])
                    dur = cad_step_row['rate_calc_duration']
                    rate_data['errs'].append(np.sqrt(cad_step_row[col_name] * dur) / dur)

        for rate_data in rates_data:
            rate_data['data'] = np.array(rate_data['data'])  # All should be the same for each step, just get mean
            rate_data['errs'] = np.array(rate_data['errs'])
            rate_data['scaled_rate'] = rate_data['data'] * rate_data['norm_scale']
            rate_data['scaled_errs'] = rate_data['errs'] * rate_data['norm_scale']
            ax.errorbar(scan_steps_plt, rate_data['scaled_rate'], yerr=rate_data['scaled_errs'], label=f'Bunch {bunch}')
    ax.axhline(0, color='k', ls='-', zorder=0)
    ax.grid(True, zorder=0)
    # ax.legend()
    fig.tight_layout()

    plt.show()


def bunch_by_bunch_cross_section(base_path):
    """
    Use bunch by bunch beam widths and profiles to calculate individual bunch lumis.
    Then use bunch by bunch rates to calculate cross sections.
    """
    longitudinal_profiles_dir_path = f'{base_path}profiles/'
    combined_cad_step_data_csv_path = f'{base_path}combined_cad_step_data.csv'
    gl1p_rate_data_csv_path = f'{base_path}gl1p_bunch_by_bunch_step_rates.csv'
    out_fig_path = f'{base_path}Figures/Beam_Param_Inferences/Beam_Width_Rate_Only_Fits/'
    bunch_width_csv_path = f'{base_path}beam_widths_bunch_by_bunch_rate_only_fit_results.csv'

    bunches = np.arange(0, 111, 1)

    f_beam = 78.4  # kHz
    mb_to_um2 = 1e-19
    lumi_z_cut = 200
    ions = 'dcct'
    # ions = 'wcm'
    average_profiles = False  # If False, just take the middle

    # rate_name_for_beam_width = 'MBD |z|<200 Sasha Corrected'
    # rate_names_for_absolute_rate = ['mbd_sasha_bkg_cor_bunch_x_rate', 'zdc_sasha_cor_bunch_x_rate']
    analysis_rates = [
        {'name': 'MBD', 'bw_rate_name': 'MBD |z|<200 Sasha Corrected', 'abs_rate_name': 'mbd_sasha_bkg_cor_bunch_x_rate', 'lumi_type': 'MBD'},
        {'name': 'ZDC', 'bw_rate_name': 'ZDC Sasha Corrected', 'abs_rate_name': 'zdc_sasha_cor_bunch_x_rate', 'lumi_type': 'ZDC'},
    ]

    cad_df = pd.read_csv(combined_cad_step_data_csv_path)
    gl1p_df = pd.read_csv(gl1p_rate_data_csv_path)
    cad_df = pd.merge(cad_df, gl1p_df, how='left', on='step')

    bunch_width_df = pd.read_csv(bunch_width_csv_path)

    if base_path.split('/')[-2] == 'auau_oct_16_24':
        beta_star = 80.3  # in cm
    elif base_path.split('/')[-2] == 'auau_july_17_25':
        beta_star = 82.1  # in cm
    elif base_path.split('/')[-2] == 'pp_aug_12_24':
        beta_star = 111.6  # in cm
    else:
        raise ValueError(f'Unknown run number for base path: {base_path}')

    print(bunch_width_df)
    print(bunch_width_df.columns)

    bunch_width_df = bunch_width_df[bunch_width_df['ions'] == ions]
    bunch_width_df = bunch_width_df[bunch_width_df['beta_star'] == beta_star]
    bunch_width_df = bunch_width_df[bunch_width_df['average_profiles'] == average_profiles]
    # bunch_width_df = bunch_width_df[bunch_width_df['rate_name'] == rate_name_for_beam_width]

    head_on_steps = [row['step'] for _, row in cad_df.iterrows() if row['set offset h'] == 0 and row['set offset v'] == 0]

    print(f'Head-on steps: {head_on_steps}')
    print(cad_df.columns)

    df_out = []
    df_summary_out = []

    collider_sim = BunchCollider()
    collider_sim.set_grid_size(31, 31, 101, 31)
    bkg = 0.0e-17
    gauss_eff_width = 500
    mbd_resolution = 1.0

    collider_sim.set_bkg(bkg)
    collider_sim.set_gaus_z_efficiency_width(gauss_eff_width)
    collider_sim.set_gaus_smearing_sigma(mbd_resolution)
    collider_sim.set_bunch_beta_stars(beta_star, beta_star)

    # Get nominal dcct ions and emittances
    step_0 = cad_df[cad_df['step'] == 0].iloc[0]
    em_blue_horiz_nom, em_blue_vert_nom = step_0['blue_horiz_emittance'], step_0['blue_vert_emittance']
    em_yel_horiz_nom, em_yel_vert_nom = step_0['yellow_horiz_emittance'], step_0['yellow_vert_emittance']
    em_blue_nom = (em_blue_horiz_nom, em_blue_vert_nom)
    em_yel_nom = (em_yel_horiz_nom, em_yel_vert_nom)

    for step in head_on_steps:
        cad_df_step = cad_df[cad_df['step'] == step].iloc[0]
        print(f'Processing step {step}')
        # rates = {abs_rate_name: [] for abs_rate_name in rate_names_for_absolute_rate}
        # rates = {analysis_rate['name']: [] for analysis_rate in analysis_rates}
        # beam_widths = {'horizontal': [], 'vertical': []}
        n_protons = {'blue': [], 'yellow': []}
        # lumis = {'naked': [], 'lumi': [], 'observed': [], 'z_cut': []}
        analysis_data = {analysis_rate['name']: {
            'rate': [], 'beam_widths': {'horizontal': [], 'vertical': []}, 'lumis': {'naked': [], 'lumi': [], 'observed': [], 'z_cut': []}
        } for analysis_rate in analysis_rates}
        for bunch_i in bunches:
            profile_path = get_profile_path(longitudinal_profiles_dir_path, cad_df_step['start'], cad_df_step['end'],
                                            False, bunch_num=bunch_i)
            bunch_n_proton_fracs = {}
            for color in ['blue', 'yellow']:
                file_name = os.path.basename(profile_path).replace(f'bunch_{bunch_i}_', '').replace('COLOR', color)
                bunch_rel_n_protons = np.loadtxt(f'{longitudinal_profiles_dir_path}bunch_norm_factors_{file_name}',
                                                 dtype=float,
                                                 delimiter=',')
                bunch_n_proton_frac = bunch_rel_n_protons[bunch_i] / np.sum(bunch_rel_n_protons)
                bunch_n_proton_fracs[color] = bunch_n_proton_frac

            n_blue, n_yellow = cad_df_step['blue_wcm_ions'], cad_df_step['yellow_wcm_ions']
            n_blue *= bunch_n_proton_fracs['blue']
            n_yellow *= bunch_n_proton_fracs['yellow']
            n_protons['blue'].append(n_blue)
            n_protons['yellow'].append(n_yellow)

            # for absolute_rate_name in rate_names_for_absolute_rate:
            for analysis_rate in analysis_rates:
                absolute_rate_name = analysis_rate['abs_rate_name']
                print(f'Processing bunch {bunch_i} for absolute rate {absolute_rate_name}')
                rate_data = cad_df_step[absolute_rate_name.replace('bunch_x', f'bunch_{bunch_i}')]
                # rates[absolute_rate_name].append(rate_data)
                analysis_data[analysis_rate['name']]['rate'].append(rate_data)

                bunch_widths = bunch_width_df[(bunch_width_df['bunch'] == bunch_i) & (bunch_width_df['rate_name'] == analysis_rate['bw_rate_name'])]
                bunch_width_horizontal = bunch_widths[bunch_widths['orientation'] == 'Horizontal'].iloc[0]['beam_width']
                bunch_width_vertical = bunch_widths[bunch_widths['orientation'] == 'Vertical'].iloc[0]['beam_width']
                # beam_widths['horizontal'].append(bunch_width_horizontal)
                # beam_widths['vertical'].append(bunch_width_vertical)
                analysis_data[analysis_rate['name']]['beam_widths']['horizontal'].append(bunch_width_horizontal)
                analysis_data[analysis_rate['name']]['beam_widths']['vertical'].append(bunch_width_vertical)

                set_sim(collider_sim, cad_df_step, bunch_width_horizontal, bunch_width_vertical, em_blue_nom, em_yel_nom, profile_path)
                collider_sim.run_sim_parallel()
                naked_lumi = collider_sim.get_naked_luminosity()

                zs, z_dist = collider_sim.get_z_density_dist()
                zs_cut, z_dist_cut = zs[np.abs(zs) < lumi_z_cut], z_dist[np.abs(zs) < lumi_z_cut]
                naked_lumi_obs = collider_sim.get_naked_luminosity(observed=True)
                cut_fraction = np.trapezoid(z_dist_cut, zs_cut) / np.trapezoid(z_dist, zs)
                naked_lumi_z_cut = naked_lumi_obs * cut_fraction

                # lumis['naked'].append(naked_lumi)
                # lumis['lumi'].append(naked_lumi * mb_to_um2 * f_beam * 1e3 * n_blue * n_yellow)
                # lumis['observed'].append(naked_lumi_obs * mb_to_um2 * f_beam * 1e3 * n_blue * n_yellow)
                # lumis['z_cut'].append(naked_lumi_z_cut * mb_to_um2 * f_beam * 1e3 * n_blue * n_yellow)
                lumi_factor = mb_to_um2 * f_beam * 1e3 * n_blue * n_yellow
                analysis_data[analysis_rate['name']]['lumis']['naked'].append(naked_lumi)
                analysis_data[analysis_rate['name']]['lumis']['lumi'].append(naked_lumi * lumi_factor)
                analysis_data[analysis_rate['name']]['lumis']['observed'].append(naked_lumi_obs * lumi_factor)
                analysis_data[analysis_rate['name']]['lumis']['z_cut'].append(naked_lumi_z_cut * lumi_factor)

        # for absolute_rate_name, rate_data in rates.items():
        for absolute_rate_name, analysis_data_rate in analysis_data.items():
            rate_data = analysis_data_rate['rate']
            beam_widths = analysis_data_rate['beam_widths']
            lumis = analysis_data_rate['lumis']

            cross_sections = np.array(rate_data) / np.array(lumis['lumi'])

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(bunches, rate_data, marker='o', label=absolute_rate_name)
            ax_bw = ax.twinx()
            ax_bw.plot(bunches, beam_widths['horizontal'], marker='s', linestyle='--', color='orange', label='Horizontal Beam Width')
            ax_bw.plot(bunches, beam_widths['vertical'], marker='^', linestyle='--', color='green', label='Vertical Beam Width')
            ax_bw.set_ylabel('Beam Width [μm]')
            ax.set_xlabel('Bunch Number')
            ax.set_ylabel('Rate')
            ax.set_title(f'Absolute Rates for Step {step}')
            ax.legend()
            ax_bw.legend()
            fig.tight_layout()

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(bunches, n_protons['blue'], marker='o', color='blue', label='Blue Protons')
            ax.plot(bunches, n_protons['yellow'], marker='s', color='orange', label='Yellow Protons')
            ax.set_xlabel('Bunch Number')
            ax.set_ylabel('Number of Protons')
            ax.legend()
            fig.tight_layout()

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(bunches, lumis['lumi'], marker='o', label='Lumi')
            ax.plot(bunches, lumis['observed'], marker='s', linestyle='--', color='orange', label='Observed Lumi')
            ax.plot(bunches, lumis['z_cut'], marker='^', linestyle='--', color='green', label='Lumi |z|<200')
            ax.set_xlabel('Bunch Number')
            ax.set_ylabel('Luminosity [cm⁻² s⁻¹]')
            ax.legend()
            fig.tight_layout()

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(bunches, cross_sections, marker='o', label=f'Cross Sections for {absolute_rate_name}')
            ax.set_xlabel('Bunch Number')
            ax.set_ylabel('Cross Section [cm²]')
            ax.set_title(f'Cross Sections for {absolute_rate_name} at Step {step}')
            ax.legend()
            fig.tight_layout()
    plt.show()


    # for step in cad_df['step'].unique():
    #     print(f'Processing step {step}')
    #     cad_step_row = cad_df[cad_df['step'] == step].iloc[0]
    #     orientation = cad_step_row['orientation']
        # if orientation == 'Horizontal':
        #     beam_width_x_nom = cad_step_row['beam_width_x']
        #     beam_width_y_nom = cad_step_row['beam_width_y']
        # else:
        #     beam_width_x_nom = cad_step_row['beam_width_y']
        #     beam_width_y_nom = cad_step_row['beam_width_x']
        #
        # # Get nominal emittances
        # em_blue_horiz_nom, em_blue_vert_nom = cad_step_row['blue_horiz_emittance'], cad_step_row['blue_vert_emittance']
        # em_yel_horiz_nom, em_yel_vert_nom = cad_step_row['yellow_horiz_emittance'], cad_step_row['yellow_vert_emittance']
        # em_blue_nom = (em_blue_horiz_nom, em_blue_vert_nom)
        # em_yel_nom = (em_yel_horiz_nom, em_yel_vert_nom)
        #
        # # Set up collider sim
        # collider_sim = BunchCollider()
        # collider_sim.set_grid_size(31, 31, 101, 31)
        # collider_sim.set_bkg(0.0e-17)
        # collider_sim.set_gaus_z_efficiency_width(500)
        # collider_sim.set_gaus_smearing_sigma(1.0)
        # collider_sim.set_bunch_beta_stars(beta_star, beta_star)
        #
        # # Get the beam widths for this step
        # beam_widths = np.linspace(120, 170, 9)


def get_minimum_chi2(chi2s, xs, n_pts=5):
    """
    Find the minimum chi2. Find minimum point, select range around it, and fit a parabola to find the minimum.
    :param chi2s: Numpy 1D array of chi2 values.
    :param xs: Numpy 1D array of x values corresponding to chi2s.
    :param n_pts: Number of points to use for the parabola fit around the minimum.
    :return: Location and value of minimum.
    """
    min_index = np.argmin(chi2s)

    # Select a range around the minimum point
    half_range = n_pts // 2
    start_index = max(0, min_index - half_range)
    end_index = min(len(chi2s), min_index + half_range + 1)
    range_indices = np.arange(start_index, end_index)

    # Fit a parabola to the selected range
    x_data = xs[range_indices]
    y_data = chi2s[range_indices]

    popt, _ = cf(parabola, x_data, y_data)

    # Find the vertex of the parabola
    vertex_x = -popt[1] / (2 * popt[0])
    vertex_y = parabola(vertex_x, *popt)

    return vertex_x, vertex_y


def parabola(x, a, b, c):
    return a * x**2 + b * x + c


def find_best_scale_to_lumis(data, lumis, errs):
    weights = 1 / errs**2
    numerator = np.sum(weights * data * lumis)
    denominator = np.sum(weights * data**2)
    return numerator / denominator


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on July 01 6:12 PM 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/rate_vs_offset_analysis.py

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
    plot_lumi_vs_step(base_path)
    # plot_lumi_vs_step_zdc_cor(base_path)
    # lumi_test(base_path)
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


def plot_lumi_vs_step(base_path):
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

    cad_df = pd.read_csv(combined_cad_step_data_csv_path)

    scan_steps = np.arange(0, 25, 1)

    collider_sim = BunchCollider()
    collider_sim.set_grid_size(31, 31, 101, 31)
    beam_width_x, beam_width_y = 130.0, 130.0
    beta_star = 76.7
    bkg = 0.0e-17
    gauss_eff_width = 500
    mbd_resolution = 1.0
    # gauss_eff_width = None
    # mbd_resolution = None

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
        {'col_name': 'zdc_cor_rate', 'name': 'ZDC Uncorrected', 'data': [], 'errs': []},
        {'col_name': 'zdc_acc_multi_cor_rate', 'name': 'ZDC Angelika Corrected', 'data': [], 'errs': []},
        {'col_name': 'zdc_sasha_cor_rate', 'name': 'ZDC Sasha Corrected', 'data': [], 'errs': []},
        {'col_name': 'mbd_z200_rate', 'name': 'MBD Angelika Corrected Z200', 'data': [], 'errs': []},
        {'col_name': 'mbd_bkg_cor_rate', 'name': 'MBD Angelika Bkg Corrected', 'data': [], 'errs': []},
        {'col_name': 'mbd_sasha_z200_rate', 'name': 'MBD Sasha Corrected Z200', 'data': [], 'errs': []},
        {'col_name': 'mbd_sasha_bkg_cor_rate', 'name': 'MBD Sasha Bkg Corrected', 'data': [], 'errs': []},
        {'col_name': 'mbd_cor_rate', 'name': 'MBD Uncorrected', 'data': [], 'errs': []},
    ]

    lumis, scan_steps_plt = [], []
    for scan_step in scan_steps:
        print(f'Scan Step: {scan_step}')
        cad_step_row = cad_df[cad_df['step'] == scan_step].iloc[0]
        # profile_path = get_profile_path(
        #     longitudinal_profiles_dir_path, cad_step_row['start'], cad_step_row['end'], False
        # )
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
            # n_blue, n_yellow = cad_step_row['blue_dcct_ions'], cad_step_row['yellow_dcct_ions']
            n_blue, n_yellow = cad_step_row['blue_wcm_ions'], cad_step_row['yellow_wcm_ions']
            lumi = naked_lumi * mb_to_um2 * f_beam * 1e3 * n_blue * n_yellow
            lumis.append(lumi)
            scan_steps_plt.append(scan_step)
            for rate_data in rates_data:
                rate_data['data'].append(cad_step_row[rate_data['col_name']])
                dur = cad_step_row['duration'] - 2  # Should be pretty close (~1%) to actual duration used to get rate.
                rate_data['errs'].append(np.sqrt(cad_step_row[rate_data['col_name']] * dur) / dur)

        for bunch_i in range(111):
            pass

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
    # ax2.set_ylim(bottom=0, top=1.2 * np.max(zdc_rates))
    ax2.set_ylim(bottom=0)
    plt.tight_layout()

    steps = np.array([0, 6, 12, 18, 24])
    step_mask = np.isin(scan_steps_plt, steps)
    norm_step = 0

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

    bkg_frac = 0.17
    lumis_plus_bkg = np.array(lumis) + lumis[norm_step] * bkg_frac

    fig, ax = plt.subplots()
    ax.axhline(0, color='k', ls='-', zorder=0)
    for rate_data in rates_data:
        # Normalize rates to the first step
        norm_rates = np.array(rate_data['data']) / rate_data['data'][norm_step] * lumis_plus_bkg[norm_step]
        percent_rate = (norm_rates - lumis_plus_bkg) / norm_rates * 100
        ax.plot(scan_steps_plt[step_mask], percent_rate[step_mask], marker='o', linestyle='-', label=rate_data["name"])
    ax.set_xlabel('Scan Step')
    ax.set_ylabel('Percent Difference (Data - Sim) / Data [%]')
    ax.set_title('Luminosity Plus Background vs Scan Step')
    ax.legend()
    plt.tight_layout()

    # Calculate the average luminosity for each scan step and use standard deviation for error bars
    lumis = np.array(lumis)
    lumis_mean = np.array([np.mean(lumis[scan_steps_plt == step]) for step in scan_steps])
    lumis_std = np.array([np.std(lumis[scan_steps_plt == step]) for step in scan_steps])
    for rate_data in rates_data:
        rate_data['data'] = np.array(rate_data['data'])  # All values for each step should be the same, just get mea
        rate_data['data_step'] = np.array([np.mean(rate_data['data'][scan_steps_plt == step]) for step in scan_steps])
        rate_data['err_step'] = np.array([np.mean(rate_data['errs'][scan_steps_plt == step]) for step in scan_steps])
        rate_data['data_measures'] = np.array([Measure(rate_data['data_step'][i], rate_data['err_step'][i]) for i in range(len(rate_data['data_step']))])

    step_mask = np.isin(scan_steps, steps)
    norm_step = 0

    fig, ax = plt.subplots()
    ax.axhline(0, color='k', ls='-', zorder=0)
    for rate_data in rates_data:
        norm_rates = np.array(rate_data['data_step']) / rate_data['data_step'][norm_step] * lumis_mean[norm_step]
        lumi_errs = lumis_std[step_mask] / norm_rates * 100
        percent_rate = (norm_rates - lumis_mean) / norm_rates * 100
        ax.errorbar(scan_steps[step_mask], percent_rate[step_mask],
                    yerr=lumis_std[step_mask] / norm_rates[step_mask] * 100, fmt='o-', label=rate_data["name"])
    ax.set_xlabel('Scan Step')
    ax.set_ylabel('Percent Difference (Data - Sim) / Data [%]')
    ax.set_title('Luminosity vs Scan Step with Error Bars')
    ax.legend()
    plt.tight_layout()

    fig, ax = plt.subplots()
    ax.axhline(0, color='k', ls='-', zorder=0)
    for rate_data in rates_data:
        norm_rates = np.array(rate_data['data_step']) / rate_data['data_step'][norm_step] * lumis_mean[norm_step]
        ax.errorbar(scan_steps[step_mask], norm_rates[step_mask],
                    yerr=lumis_std[step_mask] / norm_rates[step_mask] * 100, fmt='o-', label=rate_data["name"])
    ax.set_xlabel('Scan Step')
    ax.set_ylabel('Luminosity ')
    ax.set_title('Luminosity vs Scan Step with Error Bars')
    ax.legend()
    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    main()

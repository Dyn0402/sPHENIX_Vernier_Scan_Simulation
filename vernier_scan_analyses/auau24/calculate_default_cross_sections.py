#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on July 17 12:37 PM 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/calculate_default_cross_sections.py

@author: Dylan Neff, dylan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from BunchCollider import BunchCollider
from z_vertex_fitting_common import get_profile_path, set_sim
from common_logistics import set_base_path
from Measure import Measure


def main():
    base_path = set_base_path()
    base_path += 'Vernier_Scans/auau_oct_16_24/'
    calculate_default_cross_sections(base_path)
    # compare_to_gaus_lumi(base_path)
    # compare_to_gaus_lumi_simpler(base_path)
    print('donzo')


def calculate_default_cross_sections(base_path):
    """ Calculate the default cross sections for MBD and ZDC detectors in the sPHENIX experiment.
    Use head on steps to calculate the cross sections.
    :param base_path: The base path to the simulation directory.
    """
    longitudinal_profiles_dir_path = f'{base_path}profiles/'
    combined_cad_step_data_csv_path = f'{base_path}combined_cad_step_data.csv'
    beam_width_csv_path = f'{base_path}beam_widths_rate_only_fit_results.csv'

    f_beam = 78.4 * 1e3  # Hz  Frequency of single bunch
    mb_to_um2 = 1e-19
    n_bunches = 111  # Number of bunches in the collider
    # lumi_z_cut = 200
    lumi_z_cut = None
    observed = False  # Should always be False for cross section calculations

    cad_df = pd.read_csv(combined_cad_step_data_csv_path)
    beam_width_df = pd.read_csv(beam_width_csv_path)

    rate_name_for_beam_width = 'MBD |z|<200 Sasha Corrected'
    rate_names_for_absolute_rate = ['mbd_sasha_bkg_cor_rate', 'zdc_sasha_cor_rate']

    collider_sim = BunchCollider()
    collider_sim.set_grid_size(31, 31, 101, 31)
    beta_star = 80.3
    bkg = 0.0e-17
    gauss_eff_width = 500
    mbd_resolution = 1.0

    beam_width_df_rate = beam_width_df[beam_width_df['rate_name'] == rate_name_for_beam_width]
    beam_width_x = beam_width_df_rate[beam_width_df['orientation'] == 'Horizontal'].iloc[0]['beam_width']
    beam_width_y = beam_width_df_rate[beam_width_df['orientation'] == 'Vertical'].iloc[0]['beam_width']

    scan_steps = cad_df['step']

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

    steps_plt, lumis, lumi_errs = [], [], []
    rates = {rate_name: [] for rate_name in rate_names_for_absolute_rate}
    rate_errs = {rate_name: [] for rate_name in rate_names_for_absolute_rate}
    for step in scan_steps:
        cad_step_row = cad_df[cad_df['step'] == step].iloc[0]
        if cad_step_row['set offset h'] != 0 or cad_step_row['set offset v'] != 0:
            continue  # Only calculate for head-on steps
        profile_paths = get_profile_path(
            longitudinal_profiles_dir_path, cad_step_row['start'], cad_step_row['end'], True
        )
        profile_lumis = []
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
            # n_blue, n_yellow = cad_step_row['blue_wcm_ions'], cad_step_row['yellow_wcm_ions']  # Total for all bunches
            n_blue, n_yellow = cad_step_row['blue_dcct_ions'], cad_step_row['yellow_dcct_ions']  # Total for all bunches
            lumi = naked_lumi * mb_to_um2 * f_beam * n_blue * n_yellow / n_bunches * 1e3  # in b⁻¹s⁻¹
            profile_lumis.append(lumi)
        lumi_mean, lumi_std = np.nanmean(profile_lumis), np.nanstd(profile_lumis)
        steps_plt.append(step)
        lumis.append(lumi_mean)
        lumi_errs.append(lumi_std)
        for rate_col in rate_names_for_absolute_rate:
            rate_val = cad_step_row[rate_col]
            dur = cad_step_row['rate_calc_duration']
            rate_err = np.sqrt(cad_step_row[rate_col] * dur) / dur
            rates[rate_col].append(rate_val)
            rate_errs[rate_col].append(rate_err)

    # Plot rates and lumis vs step
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(steps_plt, lumis, yerr=lumi_errs, fmt='o', label='Lumi')
    ax.set_xlabel('Step')
    for rate_col in rate_names_for_absolute_rate:
        scale = lumis[0] / rates[rate_col][0]  # Scale rates to match lumi
        ax.errorbar(steps_plt, np.array(rates[rate_col]) * scale, yerr=np.array(rate_errs[rate_col]) * scale, fmt='o',
                    label=rate_col)
    ax.set_ylabel('Lumi ($b^-1s^-1$) & Rate (Hz) (scaled)')
    ax.set_title('Lumi and Rates vs Step')
    ax.legend()
    fig.tight_layout()

    # Calculate cross sections
    lumi_meases = np.array([Measure(val, err) for val, err in zip(lumis, lumi_errs)])
    cross_sections = {}
    for rate_col in rate_names_for_absolute_rate:
        rate_meases = np.array([Measure(val, err) for val, err in zip(rates[rate_col], rate_errs[rate_col])])
        cross_sections[rate_col] = rate_meases / lumi_meases  # Should be in units of mb

    # Plot cross sections vs step
    for rate_col in rate_names_for_absolute_rate:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(steps_plt, [cs.val for cs in cross_sections[rate_col]],
                    yerr=[cs.err for cs in cross_sections[rate_col]], fmt='o')
        ax.set_xlabel('Step')
        ax.set_ylabel('Cross Section (b)')
        ax.set_title(f'Cross Section for {rate_col}')
        fig.tight_layout()

    plt.show()


def compare_to_gaus_lumi(base_path):
    """ Compare the luminosity with realistic beam params with Gaussian approximation luminosity.
    :param base_path: The base path to the simulation directory.
    """
    longitudinal_profiles_dir_path = f'{base_path}profiles/'
    combined_cad_step_data_csv_path = f'{base_path}combined_cad_step_data.csv'
    beam_width_csv_path = f'{base_path}beam_widths_rate_only_fit_results.csv'

    f_beam = 78.4 * 1e3  # Hz  Frequency of single bunch
    mb_to_um2 = 1e-19
    n_bunches = 111  # Number of bunches in the collider
    # lumi_z_cut = 200
    lumi_z_cut = None
    observed = False  # Should always be False for cross section calculations

    cad_df = pd.read_csv(combined_cad_step_data_csv_path)
    beam_width_df = pd.read_csv(beam_width_csv_path)

    rate_name_for_beam_width = 'MBD |z|<200 Sasha Corrected'

    collider_sim = BunchCollider()
    collider_sim.set_grid_size(31, 31, 101, 31)
    beta_star = 76.4
    bkg = 0.0e-17
    gauss_eff_width = 500
    mbd_resolution = 1.0

    beam_width_df_rate = beam_width_df[beam_width_df['rate_name'] == rate_name_for_beam_width]
    beam_width_x = beam_width_df_rate[beam_width_df['orientation'] == 'Horizontal'].iloc[0]['beam_width']
    beam_width_y = beam_width_df_rate[beam_width_df['orientation'] == 'Vertical'].iloc[0]['beam_width']

    scan_steps = cad_df['step']

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

    steps_plt, lumis, lumi_errs, lumi_gaus = [], [], [], []
    for step in scan_steps:
        cad_step_row = cad_df[cad_df['step'] == step].iloc[0]
        if cad_step_row['set offset h'] != 0 or cad_step_row['set offset v'] != 0:
            continue  # Only calculate for head-on steps
        profile_paths = get_profile_path(
            longitudinal_profiles_dir_path, cad_step_row['start'], cad_step_row['end'], True
        )
        profile_lumis = []
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
            n_blue, n_yellow = cad_step_row['blue_wcm_ions'], cad_step_row['yellow_wcm_ions']  # Total for all bunches
            lumi = naked_lumi * mb_to_um2 * f_beam * n_blue * n_yellow / n_bunches * 1e3  # in b⁻¹s⁻¹
            profile_lumis.append(lumi)
        lumi_mean, lumi_std = np.nanmean(profile_lumis), np.nanstd(profile_lumis)
        steps_plt.append(step)
        lumis.append(lumi_mean)
        lumi_errs.append(lumi_std)

        collider_sim.set_bunch_crossing(0, 0, 0, 0)
        collider_sim.set_longitudinal_profiles_from_file(None, None)
        collider_sim.set_bunch_lengths(1e9, 1e9)  # Set bunch lengths to 1 m for Gaussian approximation
        collider_sim.run_sim_parallel()
        if lumi_z_cut is None:
            naked_lumi = collider_sim.get_naked_luminosity()
        else:
            zs, z_dist = collider_sim.get_z_density_dist()
            zs_cut, z_dist_cut = zs[np.abs(zs) < lumi_z_cut], z_dist[np.abs(zs) < lumi_z_cut]
            naked_lumi = collider_sim.get_naked_luminosity(observed=observed)
            cut_fraction = np.trapezoid(z_dist_cut, zs_cut) / np.trapezoid(z_dist, zs)
            naked_lumi *= cut_fraction
        n_blue, n_yellow = cad_step_row['blue_wcm_ions'], cad_step_row['yellow_wcm_ions']  # Total for all bunches
        lumi = naked_lumi * mb_to_um2 * f_beam * n_blue * n_yellow / n_bunches * 1e3  # in b⁻¹s⁻¹
        lumi_gaus.append(lumi)

    # Plot rates and lumis vs step
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(steps_plt, lumis, yerr=lumi_errs, fmt='o', label='Lumi')
    ax.scatter(steps_plt, lumi_gaus, label='Gaussian Lumi', color='orange')
    ax.set_xlabel('Step')
    ax.set_ylabel('Lumi ($b^-1s^-1$) & Rate (Hz) (scaled)')
    ax.set_title('Lumi and Rates vs Step')
    ax.legend()
    fig.tight_layout()

    plt.show()


def compare_to_gaus_lumi_simpler(base_path):
    """ Compare the luminosity with realistic beam params with Gaussian approximation luminosity.
    :param base_path: The base path to the simulation directory.
    """
    longitudinal_profiles_dir_path = f'{base_path}profiles/'
    combined_cad_step_data_csv_path = f'{base_path}combined_cad_step_data.csv'
    beam_width_csv_path = f'{base_path}beam_widths_rate_only_fit_results.csv'

    f_beam = 78.4 * 1e3  # Hz  Frequency of single bunch
    mb_to_um2 = 1e-19
    n_bunches = 111  # Number of bunches in the collider
    # lumi_z_cut = 200
    lumi_z_cut = None
    observed = False  # Should always be False for cross section calculations

    cad_df = pd.read_csv(combined_cad_step_data_csv_path)
    beam_width_df = pd.read_csv(beam_width_csv_path)

    rate_name_for_beam_width = 'MBD |z|<200 Sasha Corrected'
    rate_names_for_absolute_rate = ['mbd_sasha_bkg_cor_rate', 'zdc_sasha_cor_rate']

    collider_sim = BunchCollider()
    collider_sim.set_grid_size(31, 31, 101, 31)
    beta_star = 76.4
    bkg = 0.0e-17
    gauss_eff_width = 500
    mbd_resolution = 1.0

    beam_width_df_rate = beam_width_df[beam_width_df['rate_name'] == rate_name_for_beam_width]
    beam_width_x = beam_width_df_rate[beam_width_df['orientation'] == 'Horizontal'].iloc[0]['beam_width']
    beam_width_y = beam_width_df_rate[beam_width_df['orientation'] == 'Vertical'].iloc[0]['beam_width']

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

    cad_step_row = cad_df[cad_df['step'] == 0].iloc[0]
    profile_path = get_profile_path(
        longitudinal_profiles_dir_path, cad_step_row['start'], cad_step_row['end'], False
    )
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
    print(f'Naked lumi: {naked_lumi}')
    n_blue, n_yellow = cad_step_row['blue_wcm_ions'], cad_step_row['yellow_wcm_ions']  # Total for all bunches
    lumi = naked_lumi * mb_to_um2 * f_beam * n_blue * n_yellow / n_bunches * 1e3  # in b⁻¹s⁻¹
    collider_sim.check_profile_normalizations()

    bunch1_zs = collider_sim.bunch1.longitudinal_profile_zs
    bunch2_zs = collider_sim.bunch2.longitudinal_profile_zs
    bunch1_densities = collider_sim.bunch1.longitudinal_profile_densities
    bunch2_densities = collider_sim.bunch2.longitudinal_profile_densities
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(bunch1_zs, bunch1_densities, color='blue', label='Blue Beam Profile')
    ax.plot(bunch2_zs, bunch2_densities, color='orange', label='Yellow Beam Profile')
    ax.plot(bunch1_zs, gaus(bunch1_zs, 0, 1e6), color='red', linestyle='--',
            label='Gaussian Approximation', zorder=0)
    ax.set_xlabel('Z (um)')
    ax.set_ylabel('Density (1/um)')
    ax.set_title('Longitudinal Profiles of Bunches')
    ax.legend()
    fig.tight_layout()

    z_cut = 0.5e7
    z_mask1 = abs(bunch1_zs) > z_cut
    z_mask2 = abs(bunch2_zs) > z_cut

    # Integrate the profiles over the z range
    integral_bunch1 = np.trapezoid(bunch1_densities[z_mask1], bunch1_zs[z_mask1])
    integral_bunch2 = np.trapezoid(bunch2_densities[z_mask2], bunch2_zs[z_mask2])
    print(f'Integral of Blue Beam Profile (|z| > {z_cut} um): {integral_bunch1}')
    print(f'Integral of Yellow Beam Profile (|z| > {z_cut} um): {integral_bunch2}')

    print(f'Lumi: {lumi}')

    collider_sim.set_bunch_crossing(0, 0, 0, 0)
    collider_sim.run_sim_parallel()
    if lumi_z_cut is None:
        naked_lumi = collider_sim.get_naked_luminosity()
    else:
        zs, z_dist = collider_sim.get_z_density_dist()
        zs_cut, z_dist_cut = zs[np.abs(zs) < lumi_z_cut], z_dist[np.abs(zs) < lumi_z_cut]
        naked_lumi = collider_sim.get_naked_luminosity(observed=observed)
        cut_fraction = np.trapezoid(z_dist_cut, zs_cut) / np.trapezoid(z_dist, zs)
        naked_lumi *= cut_fraction
    print(f'Naked lumi no crossing angle: {naked_lumi}')
    n_blue, n_yellow = cad_step_row['blue_wcm_ions'], cad_step_row['yellow_wcm_ions']  # Total for all bunches
    lumi = naked_lumi * mb_to_um2 * f_beam * n_blue * n_yellow / n_bunches * 1e3  # in b⁻¹s⁻¹

    print(f'Lumi no crossing angle: {lumi}')

    collider_sim.set_bunch_crossing(0, 0, 0, 0)
    collider_sim.bunch1.longitudinal_profile_densities = gaus(collider_sim.bunch1.longitudinal_profile_zs, 0, 1e6)
    collider_sim.bunch2.longitudinal_profile_densities = gaus(collider_sim.bunch2.longitudinal_profile_zs, 0, 1e6)
    collider_sim.run_sim_parallel()
    if lumi_z_cut is None:
        naked_lumi = collider_sim.get_naked_luminosity()
    else:
        zs, z_dist = collider_sim.get_z_density_dist()
        zs_cut, z_dist_cut = zs[np.abs(zs) < lumi_z_cut], z_dist[np.abs(zs) < lumi_z_cut]
        naked_lumi = collider_sim.get_naked_luminosity(observed=observed)
        cut_fraction = np.trapezoid(z_dist_cut, zs_cut) / np.trapezoid(z_dist, zs)
        naked_lumi *= cut_fraction
    print(f'Naked lumi no crossing angle gaus profile: {naked_lumi}')
    n_blue, n_yellow = cad_step_row['blue_wcm_ions'], cad_step_row['yellow_wcm_ions']  # Total for all bunches
    lumi = naked_lumi * mb_to_um2 * f_beam * n_blue * n_yellow / n_bunches * 1e3  # in b⁻¹s⁻¹

    print(f'Lumi no crossing angle gaus profile: {lumi}')


    collider_sim.set_longitudinal_profiles_from_file(None, None)
    collider_sim.set_bunch_lengths(1e6, 1e6)  # Set bunch lengths to 1 m for Gaussian approximation
    print(f'Beam length = {collider_sim.bunch1.get_beam_length()} m, {collider_sim.bunch2.get_beam_length()} m')
    collider_sim.run_sim_parallel()
    if lumi_z_cut is None:
        naked_lumi = collider_sim.get_naked_luminosity()
    else:
        zs, z_dist = collider_sim.get_z_density_dist()
        zs_cut, z_dist_cut = zs[np.abs(zs) < lumi_z_cut], z_dist[np.abs(zs) < lumi_z_cut]
        naked_lumi = collider_sim.get_naked_luminosity(observed=observed)
        cut_fraction = np.trapezoid(z_dist_cut, zs_cut) / np.trapezoid(z_dist, zs)
        naked_lumi *= cut_fraction
    print(f'Naked lumi gaus: {naked_lumi}')
    n_blue, n_yellow = cad_step_row['blue_wcm_ions'], cad_step_row['yellow_wcm_ions']  # Total for all bunches
    lumi = naked_lumi * mb_to_um2 * f_beam * n_blue * n_yellow / n_bunches * 1e3  # in b⁻¹s⁻¹

    print(f'Lumi gaus: {lumi}')


    plt.show()


def gaus(x, mu, sigma):
    """ Gaussian function.
    :param x: The x values.
    :param mu: The mean of the Gaussian.
    :param sigma: The standard deviation of the Gaussian.
    :return: The Gaussian function evaluated at x.
    """
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

if __name__ == '__main__':
    main()

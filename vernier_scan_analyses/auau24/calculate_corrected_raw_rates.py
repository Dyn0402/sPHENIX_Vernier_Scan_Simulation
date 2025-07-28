#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 07 04:50 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/calculate_raw_rates

@author: Dylan Neff, dn277127
"""

import platform
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd

from BunchCollider import BunchCollider
from z_vertex_fitting_common import (load_vertex_distributions, get_profile_path, fit_amp_shift,
                                     load_gl1p_vertex_distributions, get_bunches_from_gl1p_rates_df)
from common_logistics import set_base_path


def main():
    base_path = set_base_path()

    # base_path = f'{base_path}Vernier_Scans/auau_oct_16_24/'
    base_path = f'{base_path}Vernier_Scans/auau_july_17_25/'
    run_number = 69561

    longitudinal_profiles_dir_path = f'{base_path}profiles/'
    z_vertex_zdc_data_path = f'{base_path}vertex_data/{run_number}_vertex_distributions.root'
    z_vertex_no_zdc_data_path = f'{base_path}vertex_data/{run_number}_vertex_distributions_no_zdc_coinc.root'
    z_vertex_no_zdc_gl1_data_path = f'{base_path}vertex_data/{run_number}_vertex_distributions_no_zdc_coinc_bunch_by_bunch.root'
    combined_cad_step_data_csv_path = f'{base_path}combined_cad_step_data.csv'
    gl1p_step_rates_csv_path = f'{base_path}gl1p_bunch_by_bunch_step_rates.csv'
    plot_out_path = f'{base_path}Figures/zvertex_cut_plots/'

    get_mbd_cut_rates(longitudinal_profiles_dir_path, z_vertex_no_zdc_data_path, z_vertex_zdc_data_path, combined_cad_step_data_csv_path, plot_out_path)
    add_bkg_cor_mbd_rate_to_cad_df(combined_cad_step_data_csv_path)
    if gl1p_step_rates_csv_path:
        get_mbd_gl1p_cut_rates(z_vertex_no_zdc_gl1_data_path, gl1p_step_rates_csv_path, combined_cad_step_data_csv_path)
        add_bkg_cor_mbd_rate_to_gl1p_df(gl1p_step_rates_csv_path)
    # compare_mbd_corrected_to_zdc(base_path)

    print('donzo')


def add_bkg_cor_mbd_rate_to_cad_df(combined_cad_step_data_csv_path):
    """
    Add the background corrected MBD rate to the combined CAD step data CSV.
    """
    cad_df = pd.read_csv(combined_cad_step_data_csv_path)

    # Calculate the background corrected MBD rate
    cad_df['mbd_bkg_cor_rate'] = cad_df['mbd_acc_multi_cor_rate'] * (1 - cad_df['mbd_cut_correction_fraction'] / 100)
    cad_df['mbd_z200_rate'] = cad_df['mbd_acc_multi_cor_rate'] * (1 - (cad_df['mbd_cut_correction_fraction'] + cad_df['simulated_cut_fraction']) / 100)
    cad_df['mbd_sasha_bkg_cor_rate'] = cad_df['mbd_sasha_cor_rate'] * (1 - cad_df['mbd_cut_correction_fraction'] / 100)
    cad_df['mbd_sasha_z200_rate'] = cad_df['mbd_sasha_cor_rate'] * (1 - (cad_df['mbd_cut_correction_fraction'] + cad_df['simulated_cut_fraction']) / 100)

    # Save the updated DataFrame back to the CSV file
    cad_df.to_csv(combined_cad_step_data_csv_path, index=False)
    print(f'Saved updated cad_df with mbd_bkg_cor_rate to {combined_cad_step_data_csv_path}')


def add_bkg_cor_mbd_rate_to_gl1p_df(gl1p_rates_csv_path):
    """
    Add the background corrected MBD rates to the gl1p_rates_df.
    """
    gl1p_rates_df = pd.read_csv(gl1p_rates_csv_path)
    bunches = get_bunches_from_gl1p_rates_df(gl1p_rates_df)

    new_columns = {}
    for bunch in bunches:
        cut_fraction = gl1p_rates_df[f'mbd_cut_correction_fraction_bunch_{bunch}'] / 100
        sim_cut_fraction = gl1p_rates_df['simulated_cut_fraction'] / 100

        mbd_uncor = gl1p_rates_df[f'mbd_cor_bunch_{bunch}_rate']
        mbd_acc_multi = gl1p_rates_df[f'mbd_acc_multi_cor_bunch_{bunch}_rate']
        mbd_sasha_cor = gl1p_rates_df[f'mbd_sasha_cor_bunch_{bunch}_rate']

        new_columns[f'mbd_bkg_cor_bunch_{bunch}_rate'] = mbd_acc_multi * (1 - cut_fraction)
        new_columns[f'mbd_z200_bunch_{bunch}_rate'] = mbd_acc_multi * (1 - (cut_fraction + sim_cut_fraction))
        new_columns[f'mbd_z200_uncor_bunch_{bunch}_rate'] = mbd_uncor * (1 - (cut_fraction + sim_cut_fraction))
        row_num = 6
        print(f'Bunch {bunch}: cut_fraction = {cut_fraction.iloc[6]}, sim_cut_fraction = {sim_cut_fraction.iloc[6]}, mbd_uncor = {mbd_uncor.iloc[6]}, mbd_acc_multi = {mbd_acc_multi.iloc[6]}, mbd_sasha_cor = {mbd_sasha_cor.iloc[6]}')
        new_columns[f'mbd_sasha_bkg_cor_bunch_{bunch}_rate'] = mbd_sasha_cor * (1 - cut_fraction)
        new_columns[f'mbd_sasha_z200_bunch_{bunch}_rate'] = mbd_sasha_cor * (1 - (cut_fraction + sim_cut_fraction))

    # Drop any existing columns that are going to be overwritten
    gl1p_rates_df = gl1p_rates_df.drop(columns=new_columns.keys(), errors='ignore')

    # Concatenate new columns in one operation → no fragmentation
    gl1p_rates_df = pd.concat([gl1p_rates_df, pd.DataFrame(new_columns)], axis=1)

    gl1p_rates_df.to_csv(gl1p_rates_csv_path, index=False)
    print(f'Saved updated cad_df with mbd_bkg_cor_rate to {gl1p_rates_csv_path}')


def get_mbd_cut_rates(longitudinal_profiles_dir_path, z_vertex_data_path, z_vertex_zdc_data_path, combined_cad_step_data_csv_path, plot_out_path=None):
    """
    Calculate the raw rates for the MBD cuts.
    """
    cad_df = pd.read_csv(combined_cad_step_data_csv_path)
    fit_range = [-200, 200]
    z_cut = 200  # cm, cut for the MBD
    # steps = [5]
    # steps = [0, 6, 12, 18]
    steps = np.arange(0, 25)
    # plot_out_path = None

    # Get the vz fraction outside of the cut for data from no ZDC coincidence and from simulation with ZDC coincidence
    vertex_data = load_vertex_distributions(z_vertex_data_path, steps, cad_df, rate_column=None)
    vertex_data_with_zdc = load_vertex_distributions(z_vertex_zdc_data_path, steps, cad_df, rate_column=None)

    collider_sim = BunchCollider()
    # collider_sim.set_grid_size(71, 71, 1001, 81)
    collider_sim.set_grid_size(51, 51, 101, 51)
    z_sim_range = np.array([-805., 805.])
    collider_sim.set_z_bounds(z_sim_range * 1e4)
    beta_star = 76.7  # cm
    beam_width_x, beam_width_y = 129.2, 125.7
    bkg = 0.0e-17
    gauss_eff_width = 500
    mbd_resolution = 1.0
    collider_sim.set_bkg(bkg)
    collider_sim.set_gaus_z_efficiency_width(gauss_eff_width)
    collider_sim.set_gaus_smearing_sigma(mbd_resolution)

    for step in steps:
        print(f'Processing step {step}...')
        centers, counts, count_errs = vertex_data[step]
        centers_zdc, counts_zdc, count_errs_zdc = vertex_data_with_zdc[step]

        cad_step_row = cad_df[cad_df['step'] == step].iloc[0]

        em_blue_horiz, em_blue_vert = cad_step_row['blue_horiz_emittance'], cad_step_row['blue_vert_emittance']
        em_yel_horiz, em_yel_vert = cad_step_row['yellow_horiz_emittance'], cad_step_row['yellow_vert_emittance']

        # Get nominal emittances
        step_0 = cad_df[cad_df['step'] == 0].iloc[0]
        em_blue_horiz_nom, em_blue_vert_nom = step_0['blue_horiz_emittance'], step_0['blue_vert_emittance']
        em_yel_horiz_nom, em_yel_vert_nom = step_0['yellow_horiz_emittance'], step_0['yellow_vert_emittance']

        # blue_widths = np.array([
        #     beam_width_x * np.sqrt(em_blue_horiz / em_blue_horiz_nom),
        #     beam_width_y * np.sqrt(em_blue_vert / em_blue_vert_nom)
        # ])
        # yellow_widths = np.array([
        #     beam_width_x * np.sqrt(em_yel_horiz / em_yel_horiz_nom),
        #     beam_width_y * np.sqrt(em_yel_vert / em_yel_vert_nom)
        # ])

        blue_widths = np.array([
            beam_width_x * em_blue_horiz / em_blue_horiz_nom,
            beam_width_y * em_blue_vert / em_blue_vert_nom
        ])
        yellow_widths = np.array([
            beam_width_x * em_yel_horiz / em_yel_horiz_nom,
            beam_width_y * em_yel_vert / em_yel_vert_nom
        ])

        blue_angle_x = -cad_step_row['blue angle h'] * 1e-3
        blue_angle_y = -cad_step_row['blue angle v'] * 1e-3
        yellow_angle_x = -cad_step_row['yellow angle h'] * 1e-3
        yellow_angle_y = -cad_step_row['yellow angle v'] * 1e-3

        blue_offset_x, blue_offset_y = cad_step_row['set offset h'] * 1e3, cad_step_row['set offset v'] * 1e3
        yellow_offset_x, yellow_offset_y = 0, 0

        collider_sim.set_bunch_beta_stars(beta_star, beta_star)
        collider_sim.set_bunch_sigmas(blue_widths, yellow_widths)
        collider_sim.set_bunch_crossing(blue_angle_x, blue_angle_y, yellow_angle_x, yellow_angle_y)
        collider_sim.set_bunch_offsets([blue_offset_x, blue_offset_y], [yellow_offset_x, yellow_offset_y])

        # profile_path = get_profile_path(
        #     longitudinal_profiles_dir_path, cad_step_row['start'], cad_step_row['end'], False
        # )
        #
        # collider_sim.set_longitudinal_profiles_from_file(
        #     profile_path.replace('COLOR_', 'blue_'),
        #     profile_path.replace('COLOR_', 'yellow_')
        # )
        #
        # collider_sim.run_sim_parallel()

        profile_paths = get_profile_path(
            longitudinal_profiles_dir_path, cad_step_row['start'], cad_step_row['end'], True
        )

        z_dists = []
        for profile_i, profile_path in enumerate(profile_paths):
            print(f' Profile {profile_i + 1}/{len(profile_paths)}')
            collider_sim.set_longitudinal_profiles_from_file(
                profile_path.replace('COLOR_', 'blue_'),
                profile_path.replace('COLOR_', 'yellow_')
            )

            collider_sim.run_sim_parallel()
            z_dists.append(collider_sim.z_dist)

        # Use the average z_dist across all profiles
        z_dist = np.mean(z_dists, axis=0)
        collider_sim.z_dist = z_dist  # Update the collider_sim with the averaged z_dist

        fit_mask = (centers > fit_range[0]) & (centers < fit_range[1])

        # fit_amp_shift(collider_sim, counts[fit_mask], centers[fit_mask], count_errs[fit_mask])
        fit_amp_shift(collider_sim, counts_zdc[fit_mask], centers_zdc[fit_mask], count_errs_zdc[fit_mask])

        zs, z_dist = collider_sim.get_z_density_dist()
        center_steps = centers[1] - centers[0]  # Uniform step size
        sim_zs = np.arange(z_sim_range[0], z_sim_range[1] + center_steps, center_steps)
        interp_sim = interp1d(zs, z_dist, bounds_error=False, fill_value=0)(sim_zs)

        # Get fraction of counts where |z| > z_cut
        z_cut_mask = np.abs(centers) > z_cut
        total_counts = counts.sum()
        cut_counts = counts[z_cut_mask].sum()
        cut_fraction = cut_counts / total_counts if total_counts > 0 else 0

        # Do the same for the simulation
        zim_z_cut_mask = np.abs(sim_zs) > z_cut
        sim_cut_counts = interp_sim[zim_z_cut_mask].sum()
        sim_cut_fraction = sim_cut_counts / interp_sim.sum() if interp_sim.sum() > 0 else 0

        # Get bounds for plotting
        sim_cdf = np.cumsum(interp_sim)
        sim_cdf /= sim_cdf[-1]  # Normalize to 1
        lower_idx = np.searchsorted(sim_cdf, 0.002)
        upper_idx = np.searchsorted(sim_cdf, 0.998)
        plot_z_min = min(sim_zs[lower_idx], -305)
        plot_z_max = max(sim_zs[upper_idx], 305)

        cut_fraction, sim_cut_fraction = cut_fraction * 100, sim_cut_fraction * 100
        print(f'Step {step}: MBD cut fraction = {cut_fraction:.2f}%, Simulated cut fraction = {sim_cut_fraction:.2f}%')

        correction_fraction = cut_fraction - sim_cut_fraction

        # Save correction fraction and sim_cut_fraction to rates_df
        # rates_df.loc[rates_df['step'] == step, 'mbd_cut_correction_fraction'] = correction_fraction
        # rates_df.loc[rates_df['step'] == step, 'simulated_cut_fraction'] = sim_cut_fraction
        cad_df.loc[cad_df['step'] == step, 'mbd_cut_correction_fraction'] = correction_fraction
        cad_df.loc[cad_df['step'] == step, 'simulated_cut_fraction'] = sim_cut_fraction

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.errorbar(centers, counts, yerr=count_errs, fmt='o', label='MBD Trigger', color='black', markersize=3)
        ax.errorbar(centers_zdc, counts_zdc, yerr=count_errs_zdc, fmt='o', label='MBD+ZDC', color='gray', markersize=2,
                    alpha=0.5)
        ax.plot(sim_zs, interp_sim, label='Simulation', color='blue')
        ax.axvline(-z_cut, color='red', ls='--', label=f'|z| = {z_cut} cm')
        ax.axvline(z_cut, color='red', ls='--')
        ax.axhline(0, color='black', ls='-', lw=0.5, zorder=-1)
        ax.set_ylabel('Counts')
        ax.set_xlabel('MBD z Vertex Position (cm)')
        ax.set_ylim(bottom=0, top=np.max(counts[fit_mask]) * 1.2)
        ax.set_xlim(left=plot_z_min, right=plot_z_max)
        ax.legend()
        ax.annotate(f'Step {step}\nMBD vz outside of cut: {cut_fraction:.2f}%\n'
                    f'Simulated vz outside of cut: {sim_cut_fraction:.2f}%\n'
                    f'MBD fraction - Sim fraction: {correction_fraction:.2f}%',
                xy=(0.02, 0.98), xycoords='axes fraction', fontsize=12, va='top', ha='left',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

        fig.tight_layout()
        if plot_out_path:
            fig.savefig(f'{plot_out_path}mbd_cut_step_{step}.png')
            fig.savefig(f'{plot_out_path}mbd_cut_step_{step}.pdf')
    if not plot_out_path:
        plt.show()

    # Save rates_df to rates_path
    # rates_df.to_csv(rates_path, index=False)
    # print(f'Saved updated rates to {rates_path}')
    # Update the cad_df with the new rates
    cad_df.to_csv(combined_cad_step_data_csv_path, index=False)
    print(f'Saved updated cad_df to {combined_cad_step_data_csv_path}')


def get_mbd_gl1p_cut_rates(z_vertex_data_path, gl1p_step_rate_data_csv_path, combined_cad_step_data_csv_path):
    """
    Calculate the fraction of MBD z vertex positions outside of the cut for each step. For GL1P bunch-by-bunch rates,
    not enough stats to do fit, just get the fraction of counts outside of the cut.
    """
    gl1p_rates_df = pd.read_csv(gl1p_step_rate_data_csv_path)
    cad_df = pd.read_csv(combined_cad_step_data_csv_path)
    z_cut = 200  # cm, cut for the MBD
    steps = np.arange(0, 25)

    bunches = get_bunches_from_gl1p_rates_df(gl1p_rates_df)

    # Add simulated_cut_fraction column from cad_df to gl1p_rates_df
    # gl1p_rates_df['simulated_cut_fraction'] = cad_df['simulated_cut_fraction']
    gl1p_rates_df = gl1p_rates_df.merge(cad_df[['step', 'simulated_cut_fraction']], on='step', how='left')
    print(f'columns in gl1p_rates_df: {gl1p_rates_df.columns}')

    # Get the vz fraction outside of the cut for data from no ZDC coincidence and from simulation with ZDC coincidence
    vertex_data = load_gl1p_vertex_distributions(z_vertex_data_path, steps, gl1p_rates_df)

    # Build all new columns in dict of lists
    # new_columns = {f'mbd_cut_correction_fraction_bunch_{bunch}': [] for bunch in bunches}

    for step in steps:
        sim_cut_fraction = gl1p_rates_df[gl1p_rates_df['step'] == step].iloc[0]['simulated_cut_fraction']

        for bunch in bunches:
            centers, counts, count_errs = vertex_data[step][bunch]

            # Get fraction of counts where |z| > z_cut
            z_cut_mask = np.abs(centers) > z_cut
            total_counts = counts.sum()
            cut_counts = counts[z_cut_mask].sum()
            cut_fraction = cut_counts / total_counts if total_counts > 0 else 0

            cut_fraction = cut_fraction * 100
            print(f'Step {step}, bunch {bunch}: MBD cut fraction = {cut_fraction:.2f}%, Simulated cut fraction = {sim_cut_fraction:.2f}%')

            correction_fraction = cut_fraction - sim_cut_fraction

            # new_columns[f'mbd_cut_correction_fraction_bunch_{bunch}'].append(correction_fraction)
            gl1p_rates_df.loc[gl1p_rates_df['step'] == step, f'mbd_cut_correction_fraction_bunch_{bunch}'] = correction_fraction

    # Update the cad_df with the new rates
    # gl1p_rates_df = pd.concat([gl1p_rates_df, pd.DataFrame(new_columns)], axis=1)
    gl1p_rates_df.to_csv(gl1p_step_rate_data_csv_path, index=False)
    print(f'Saved updated cad_df to {gl1p_step_rate_data_csv_path}')


def compare_mbd_corrected_to_zdc(base_path):
    z_vertex_zdc_data_path = f'{base_path}vertex_data_old/54733_vertex_distributions.root'
    combined_cad_step_data_csv_path = f'{base_path}combined_cad_step_data.csv'
    rates_path = f'{base_path}step_raw_rates.csv'
    out_name = 'beta_star_fit_results_mbd_cor_fit_all_amps.csv'
    rate_column = 'corrected_raw_mbd_rate'  # 'zdc_raw_rate' or 'corrected_raw_mbd_rate'
    # rate_column = 'zdc_raw_rate'  # 'zdc_raw_rate' or 'corrected_raw_mbd_rate'

    cad_df = pd.read_csv(combined_cad_step_data_csv_path)
    # rates_df = pd.read_csv(rates_path)
    # cad_df = merge_cad_rates_df(cad_df, rates_df)

    print(cad_df.head())
    print(cad_df.columns)

    fig, ax = plt.subplots()
    ax.plot(cad_df['step'], cad_df['zdc_raw_rate'], 's-', color='orange', alpha=0.8, label='ZDC Raw Rate')
    ax.plot(cad_df['step'], cad_df['zdc_cor_rate'], 's-', color='blue', alpha=0.8, label='ZDC Raw Rate from Live')
    ax.plot(cad_df['step'], cad_df['mbd_raw_rate'], 'o-', color='salmon', alpha=0.8, label='MBD Raw Rate')
    ax.plot(cad_df['step'], cad_df['mbd_cor_rate'], 'o-', color='purple', alpha=0.8, label='MBD Raw Rate from Live')
    bkg_cor_mbd = cad_df['mbd_cor_rate'] * (1 - cad_df['mbd_cut_correction_fraction'] / 100)
    z200_cor_mbd = cad_df['mbd_cor_rate'] * (1 - (cad_df['mbd_cut_correction_fraction'] + cad_df['simulated_cut_fraction']) / 100)
    ax.plot(cad_df['step'], z200_cor_mbd, 'o-', color='green', alpha=0.8, label='MBD Rate |z|<200cm')
    ax.plot(cad_df['step'], bkg_cor_mbd, 'o-', color='red', alpha=0.8, label='MBD Rate Corrected for Background')
    ax.axhline(0, color='black', ls='-', lw=0.5, zorder=-1)
    ax.set_ylim(top=63000)
    ax.legend()
    ax.set_xlabel('Step')
    ax.set_ylabel('Rate (Hz)')
    fig.tight_layout()
    fig.subplots_adjust(left=0.12, right=0.995, bottom=0.1, top=0.995)

    # Scale bkg_cor_mbd to match the ZDC rates
    bkg_cor_mbd_scaled = bkg_cor_mbd * (cad_df['zdc_cor_rate'].iloc[1] / bkg_cor_mbd.iloc[1])
    z200_cor_mbd_scaled = z200_cor_mbd * (cad_df['zdc_cor_rate'].iloc[1] / z200_cor_mbd.iloc[1])
    mbd_scaled = cad_df['mbd_cor_rate'] * (cad_df['zdc_cor_rate'].iloc[1] / cad_df['mbd_cor_rate'].iloc[1])
    fig, ax = plt.subplots()
    ax.plot(cad_df['step'], mbd_scaled, 'o-', color='gray', alpha=0.5, label='MBD Raw Rate from Live (Scaled)')
    ax.plot(cad_df['step'], cad_df['zdc_cor_rate'], 's-', color='blue', alpha=0.8, label='ZDC Raw Rate from Live')
    ax.plot(cad_df['step'], z200_cor_mbd_scaled, 'o-', color='green', alpha=0.8, label='MBD Rate |z|<200cm (Scaled)')
    ax.plot(cad_df['step'], bkg_cor_mbd_scaled, 'o-', color='red', alpha=0.8, label='MBD Rate Corrected for Background (Scaled)')
    ax.axhline(0, color='black', ls='-', lw=0.5, zorder=-1)
    ax.legend()
    ax.set_ylim(top=59000)
    ax.set_xlabel('Step')
    ax.set_ylabel('Rate (Hz)')
    fig.tight_layout()
    fig.subplots_adjust(left=0.12, right=0.995, bottom=0.1, top=0.995)

    plt.show()


if __name__ == '__main__':
    main()

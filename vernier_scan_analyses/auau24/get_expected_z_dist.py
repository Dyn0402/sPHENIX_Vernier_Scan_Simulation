#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 15 18:37 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/get_expected_z_dist

@author: Dylan Neff, dn277127
"""

import platform
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import interp1d
import uproot
from datetime import datetime, time

from BunchCollider import BunchCollider
from z_vertex_fitting_common import fit_amp_shift, fit_shift_only, get_profile_path, compute_total_chi2


def main():
    # comp_pp_auau()
    # auau_data_comp()
    # auau_compare_profiles()
    # auau_pull_cad_step_data()
    # auau_plot_all_steps()
    # auau_tests()
    # auau_minimization()
    # auau_manual_simple_minimization()
    auau_manual_simple_beta_star_minimization()
    # auau_single_dist_bw_min()
    # auau_residuals_vs_beam_width()
    # auau_single_dist_multi_par_opt()
    print('donzo')


def auau_data_comp():
    if platform.system() == 'Windows':
        base_path = 'C:/Users/Dylan/Desktop/'
    else:
        base_path = '/local/home/dn277127/Bureau/'
    base_path_auau = f'{base_path}vernier_scan_AuAu24/'
    longitudinal_fit_path = f'{base_path_auau}CAD_Measurements/VernierScan_AuAu_COLOR_longitudinal_fit_22.dat'
    longitudinal_profile_path = f'{base_path_auau}CAD_Measurements/profiles_test/avg_COLOR_profile_24_14_12.dat'
    z_vertex_data_path = f'{base_path_auau}vertex_data/plot2.root'
    z_vertex_zdc_data_path = f'{base_path_auau}vertex_data/zdc_mbd_plot2.root'

    scan_step_dict = define_scan_step_dict()
    scan_step = 10

    with uproot.open(z_vertex_data_path) as f:
        hist = f[f'hist{scan_step}']
        centers_no_zdc = hist.axis().centers()
        counts_no_zdc = hist.counts()
        count_errs_no_zdc = hist.errors()
        count_errs_no_zdc[count_errs_no_zdc == 0] = 1  # Where errors are zero, set them to 1

    with uproot.open(z_vertex_zdc_data_path) as f:
        hist = f[f'hist{scan_step}']
        centers = hist.axis().centers()
        counts = hist.counts()
        count_errs = hist.errors()
        count_errs[count_errs == 0] = 1  # Where errors are zero, set them to 1

    fit_range = [-230, 230]

    # beam_width_scales = [0.95]
    beam_width_scales = [0.75]

    beta_star = 72  # cm
    blue_offset_x, blue_offset_y = scan_step_dict[scan_step]['x'] * 1e3, scan_step_dict[scan_step]['y'] * 1e3  # um
    yellow_offset_x, yellow_offset_y = 0, 0  # um
    # bkg = 20.0e-17  # Background rate
    bkg = 2.0e-17  # Background rate
    blue_angle_x = -0.125e-3  # rad
    blue_angle_y = -0.0e-3  # rad
    yellow_angle_x = -0.07e-3  # rad
    yellow_angle_y = -0.04e-3  # rad

    fig, ax = plt.subplots()

    for beam_width_scale in beam_width_scales:
        beam_width_x = 175 / np.sqrt(2) * beam_width_scale  # um
        beam_width_y = 287 / np.sqrt(2) * beam_width_scale  # um

        collider_sim = BunchCollider()
        collider_sim.set_bunch_beta_stars(beta_star, beta_star)
        collider_sim.set_bunch_sigmas(np.array([beam_width_x, beam_width_y]), np.array([beam_width_x, beam_width_y]))
        collider_sim.set_bunch_offsets([blue_offset_x, blue_offset_y], [yellow_offset_x, yellow_offset_y])
        collider_sim.set_bkg(bkg)
        collider_sim.set_bunch_crossing(blue_angle_x, blue_angle_y, yellow_angle_x, yellow_angle_y)
        blue_fit_path = longitudinal_fit_path.replace('_COLOR_', '_blue_')
        yellow_fit_path = longitudinal_fit_path.replace('_COLOR_', '_yellow_')
        blue_profile_path = longitudinal_profile_path.replace('_COLOR_', '_blue_')
        yellow_profile_path = longitudinal_profile_path.replace('_COLOR_', '_yellow_')
        # collider_sim.set_longitudinal_fit_parameters_from_file(blue_fit_path, yellow_fit_path)
        collider_sim.set_longitudinal_profiles_from_file(blue_profile_path, yellow_profile_path)

        collider_sim.run_sim_parallel()

        fit_mask = (centers > fit_range[0]) & (centers < fit_range[1])
        fit_amp_shift(collider_sim, counts[fit_mask], centers[fit_mask], count_errs[fit_mask])

        print(f'Z-shift: {collider_sim.z_shift} um')
        print(f'Amplitude: {collider_sim.amplitude} um')

        # collider_sim.set_amplitude(6.937150280233128e+25)

        zs, z_dist = collider_sim.get_z_density_dist()

        # Normalize the simulation distribution such that the maximum value is equal to the maximum value of the data
        # z_dist = z_dist / np.max(z_dist) * np.max(counts)

        # Manually shift the simulation distribution to match the data
        # zs += 4.5

        ax.plot(zs, z_dist, label=f'bw_x={beam_width_x:.1f} um', linewidth=2)
    ax.step(centers_no_zdc, counts_no_zdc, where='mid', label='Data No ZDC', linewidth=1, alpha=0.3, color='red')
    ax.step(centers, counts, where='mid', label='Data', linewidth=2)
    ax.set_xlabel('z (cm)')
    ax.set_ylabel('Density')
    ax.set_title(f'{scan_step_dict[scan_step]["scan_orientation"]} Scan Step {scan_step}')
    ax.set_ylim(bottom=0)
    ax.grid()
    ax.legend()
    ax.set_ylim(top=np.max(counts[(centers < 180) & (centers > -180)]) * 1.2)
    fig.tight_layout()
    plt.show()


def auau_compare_profiles():
    if platform.system() == 'Windows':
        base_path = 'C:/Users/Dylan/Desktop/'
    else:
        base_path = '/local/home/dn277127/Bureau/'
    base_path_auau = f'{base_path}vernier_scan_AuAu24/'
    longitudinal_profile_path = f'{base_path_auau}CAD_Measurements/profiles/avg_COLOR_profile_24_TIMESTRING.dat'
    z_vertex_data_path = f'{base_path_auau}vertex_data/plot2.root'
    z_vertex_zdc_data_path = f'{base_path_auau}vertex_data/zdc_mbd_plot2.root'

    scan_step_dict = define_scan_step_dict()
    scan_step = 10

    with uproot.open(z_vertex_data_path) as f:
        hist = f[f'hist{scan_step}']
        centers_no_zdc = hist.axis().centers()
        counts_no_zdc = hist.counts()
        count_errs_no_zdc = hist.errors()
        count_errs_no_zdc[count_errs_no_zdc == 0] = 1  # Where errors are zero, set them to 1

    with uproot.open(z_vertex_zdc_data_path) as f:
        hist = f[f'hist{scan_step}']
        centers = hist.axis().centers()
        counts = hist.counts()
        count_errs = hist.errors()
        count_errs[count_errs == 0] = 1  # Where errors are zero, set them to 1

    fit_range = [-230, 230]

    beam_width_scale = 0.95
    # time_strings = ['21_50_00', '21_55_00', '22_00_00', '22_05_00', '22_10_00', '22_15_00', '22_20_00', '22_25_00']
    time_strings = ['21_50_00', '22_10_00', '22_25_00']

    beta_star = 72  # cm
    blue_offset_x, blue_offset_y = scan_step_dict[scan_step]['x'] * 1e3, scan_step_dict[scan_step]['y'] * 1e3  # um
    yellow_offset_x, yellow_offset_y = 0, 0  # um
    # bkg = 20.0e-17  # Background rate
    bkg = 2.0e-17  # Background rate
    blue_angle_x = -0.125e-3  # rad
    blue_angle_y = -0.0e-3  # rad
    yellow_angle_x = -0.07e-3  # rad
    yellow_angle_y = -0.04e-3  # rad

    fig, ax = plt.subplots()

    for time_string in time_strings:
        beam_width_x = 175 / np.sqrt(2) * beam_width_scale  # um
        beam_width_y = 287 / np.sqrt(2) * beam_width_scale  # um

        collider_sim = BunchCollider()
        collider_sim.set_bunch_beta_stars(beta_star, beta_star)
        collider_sim.set_bunch_sigmas(np.array([beam_width_x, beam_width_y]), np.array([beam_width_x, beam_width_y]))
        collider_sim.set_bunch_offsets([blue_offset_x, blue_offset_y], [yellow_offset_x, yellow_offset_y])
        collider_sim.set_bkg(bkg)
        collider_sim.set_bunch_crossing(blue_angle_x, blue_angle_y, yellow_angle_x, yellow_angle_y)
        blue_profile_path = longitudinal_profile_path.replace('_COLOR_', '_blue_').replace('TIMESTRING', time_string)
        yellow_profile_path = longitudinal_profile_path.replace('_COLOR_', '_yellow_').replace('TIMESTRING', time_string)
        collider_sim.set_longitudinal_profiles_from_file(blue_profile_path, yellow_profile_path)

        collider_sim.run_sim_parallel()

        fit_mask = (centers > fit_range[0]) & (centers < fit_range[1])
        fit_amp_shift(collider_sim, counts[fit_mask], centers[fit_mask], count_errs[fit_mask])

        print(f'Z-shift: {collider_sim.z_shift} um')
        print(f'Amplitude: {collider_sim.amplitude} um')

        # collider_sim.set_amplitude(6.937150280233128e+25)

        zs, z_dist = collider_sim.get_z_density_dist()

        # Normalize the simulation distribution such that the maximum value is equal to the maximum value of the data
        # z_dist = z_dist / np.max(z_dist) * np.max(counts)

        # Manually shift the simulation distribution to match the data
        # zs += 4.5

        ax.plot(zs, z_dist, label=time_string, linewidth=2)
    ax.step(centers_no_zdc, counts_no_zdc, where='mid', label='Data No ZDC', linewidth=1, alpha=0.3, color='red')
    ax.step(centers, counts, where='mid', label='Data', linewidth=2)
    ax.set_xlabel('z (cm)')
    ax.set_ylabel('Density')
    ax.set_title(f'{scan_step_dict[scan_step]["scan_orientation"]} Scan Step {scan_step}')
    ax.set_ylim(bottom=0)
    ax.grid()
    ax.legend()
    ax.set_ylim(top=np.max(counts[(centers < 180) & (centers > -180)]) * 1.2)
    fig.tight_layout()
    plt.show()


def auau_pull_cad_step_data():
    if platform.system() == 'Windows':
        base_path = 'C:/Users/Dylan/Desktop/'
    else:
        base_path = '/local/home/dn277127/Bureau/'
    base_path_auau = f'{base_path}Vernier_Scans/auau_oct_16_24/'
    longitudinal_profiles_dir_path = f'{base_path_auau}profiles/'
    z_vertex_data_path = f'{base_path_auau}vertex_data/plot2.root'
    z_vertex_zdc_data_path = f'{base_path_auau}vertex_data/zdc_mbd_plot2.root'
    combined_cad_step_data_csv_path = f'{base_path_auau}combined_cad_step_data.csv'

    cad_df = pd.read_csv(combined_cad_step_data_csv_path)

    scan_step_dict = define_scan_step_dict()
    scan_step = 5
    cad_step_row = cad_df[cad_df['step'] == scan_step].iloc[0]

    with uproot.open(z_vertex_data_path) as f:
        hist = f[f'hist{scan_step}']
        centers_no_zdc = hist.axis().centers()
        counts_no_zdc = hist.counts()
        count_errs_no_zdc = hist.errors()
        count_errs_no_zdc[count_errs_no_zdc == 0] = 1  # Where errors are zero, set them to 1

    with uproot.open(z_vertex_zdc_data_path) as f:
        hist = f[f'hist{scan_step}']
        centers = hist.axis().centers()
        counts = hist.counts()
        count_errs = hist.errors()
        count_errs[count_errs == 0] = 1  # Where errors are zero, set them to 1

    fit_range = [-230, 230]
    long_paths = get_profile_path(longitudinal_profiles_dir_path, cad_step_row['start'], cad_step_row['end'], True)

    beam_width_scale = 0.95

    beta_star = 72  # cm
    blue_offset_x, blue_offset_y = cad_step_row['set offset h'] * 1e3, cad_step_row['set offset v'] * 1e3  # um
    yellow_offset_x, yellow_offset_y = 0, 0  # um
    bkg = 2.0e-17  # Background rate
    gauss_eff_width = 500  # cm Gaussian efficiency width
    mbd_resolution = 2.0  # cm MBD resolution
    blue_angle_x = -cad_step_row['blue angle h'] * 1e-3  # rad
    blue_angle_y = -cad_step_row['blue angle v'] * 1e-3  # rad
    yellow_angle_x = -cad_step_row['yellow angle h'] * 1e-3  # rad
    yellow_angle_y = -cad_step_row['yellow angle v'] * 1e-3  # rad

    fig, ax = plt.subplots()
    ax.step(centers_no_zdc, counts_no_zdc, where='mid', label='Data No ZDC', linewidth=1, alpha=0.3, color='red')
    ax.step(centers, counts, where='mid', label='Data', linewidth=2)

    print(f'Longitudinal profile paths: {long_paths}')
    for longitudinal_profile_path in long_paths:
        print(f'Longitudinal profile path: {longitudinal_profile_path}')
        beam_width_x = 175 / np.sqrt(2) * beam_width_scale  # um
        beam_width_y = 287 / np.sqrt(2) * beam_width_scale  # um

        collider_sim = BunchCollider()
        collider_sim.set_bunch_beta_stars(beta_star, beta_star)
        collider_sim.set_bunch_sigmas(np.array([beam_width_x, beam_width_y]), np.array([beam_width_x, beam_width_y]))
        collider_sim.set_bunch_offsets([blue_offset_x, blue_offset_y], [yellow_offset_x, yellow_offset_y])
        collider_sim.set_bkg(bkg)
        collider_sim.set_gaus_z_efficiency_width(gauss_eff_width)
        collider_sim.set_gaus_smearing_sigma(mbd_resolution)
        collider_sim.set_bunch_crossing(blue_angle_x, blue_angle_y, yellow_angle_x, yellow_angle_y)
        blue_profile_path = longitudinal_profile_path.replace('COLOR_', 'blue_')
        yellow_profile_path = longitudinal_profile_path.replace('COLOR_', 'yellow_')
        collider_sim.set_longitudinal_profiles_from_file(blue_profile_path, yellow_profile_path)

        collider_sim.run_sim_parallel()

        fit_mask = (centers > fit_range[0]) & (centers < fit_range[1])
        fit_amp_shift(collider_sim, counts[fit_mask], centers[fit_mask], count_errs[fit_mask])

        zs, z_dist = collider_sim.get_z_density_dist()
        print(f'Z-shift: {collider_sim.z_shift} um')

        ax.plot(zs, z_dist, label=f'Numerical Integration', linewidth=2)
    ax.set_xlabel('z (cm)')
    ax.set_ylabel('Density')
    ax.set_title(f'{scan_step_dict[scan_step]["scan_orientation"]} Scan Step {scan_step}')
    ax.set_ylim(bottom=0)
    ax.grid()
    ax.legend()
    ax.set_ylim(top=np.max(counts[(centers < 180) & (centers > -180)]) * 1.2)
    fig.tight_layout()
    plt.show()


def auau_plot_all_steps():
    if platform.system() == 'Windows':
        base_path = 'C:/Users/Dylan/Desktop/'
    else:
        base_path = '/local/home/dn277127/Bureau/'

    base_path_auau = f'{base_path}Vernier_Scans/auau_oct_16_24/'
    longitudinal_profiles_dir_path = f'{base_path_auau}profiles/'
    z_vertex_data_path = f'{base_path_auau}vertex_data/54733_vertex_distributions_no_zdc_coinc.root'
    z_vertex_zdc_data_path = f'{base_path_auau}vertex_data/54733_vertex_distributions.root'
    combined_cad_step_data_csv_path = f'{base_path_auau}combined_cad_step_data.csv'

    cad_df = pd.read_csv(combined_cad_step_data_csv_path)

    fit_range = [-230, 230]

    for scan_type, steps in zip(["horizontal", "vertical"], [range(12), range(12, 24)]):
        fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(13.3, 7.5), sharex=True)
        axes = axes.T.flatten()  # Fill columns first

        collider_sim = BunchCollider()
        first_fit = True

        # Get nominal dcct ions and emittances
        step_0 = cad_df[cad_df['step'] == 0].iloc[0]
        dcct_blue_nom, dcct_yellow_nom = step_0['blue_dcct_ions'], step_0['yellow_dcct_ions']
        em_blue_horiz_nom, em_blue_vert_nom = step_0['blue_horiz_emittance'], step_0['blue_vert_emittance']
        em_yel_horiz_nom, em_yel_vert_nom = step_0['yellow_horiz_emittance'], step_0['yellow_vert_emittance']

        for i, scan_step in enumerate(steps):
            print(f'{scan_type.capitalize()} Step {scan_step}')
            ax = axes[i]
            cad_step_row = cad_df[cad_df['step'] == scan_step].iloc[0]

            with uproot.open(z_vertex_data_path) as f:
                hist = f[f'step_{scan_step}']
                centers_no_zdc = hist.axis().centers()
                counts_no_zdc = hist.counts()
                count_errs_no_zdc = hist.errors()
                count_errs_no_zdc[count_errs_no_zdc == 0] = 1

            with uproot.open(z_vertex_zdc_data_path) as f:
                hist = f[f'step_{scan_step}']
                centers = hist.axis().centers()
                counts = hist.counts()
                count_errs = hist.errors()
                count_errs[count_errs == 0] = 1

            # Normalize counts to ZDC rate
            zdc_raw_rate = cad_step_row['zdc_raw_rate']
            zdc_hist_counts = np.sum(counts)
            hist_scaling_factor = zdc_raw_rate / zdc_hist_counts

            counts *= hist_scaling_factor
            counts_no_zdc *= hist_scaling_factor

            # Scale for dcct
            step_dcct_blue, step_dcct_yellow = cad_step_row['blue_dcct_ions'], cad_step_row['yellow_dcct_ions']
            dcct_scale = (dcct_blue_nom * dcct_yellow_nom) / (step_dcct_blue * step_dcct_yellow)

            counts *= dcct_scale
            counts_no_zdc *= dcct_scale

            beam_width_scale = 0.9
            beta_star = 72  # cm
            bkg = 2.0e-17
            gauss_eff_width = 500
            mbd_resolution = 2.0
            blue_angle_x = -cad_step_row['blue angle h'] * 1e-3
            blue_angle_y = -cad_step_row['blue angle v'] * 1e-3
            yellow_angle_x = -cad_step_row['yellow angle h'] * 1e-3
            yellow_angle_y = -cad_step_row['yellow angle v'] * 1e-3

            collider_sim.set_bunch_beta_stars(beta_star, beta_star)

            em_blue_horiz, em_blue_vert = cad_step_row['blue_horiz_emittance'], cad_step_row['blue_vert_emittance']
            em_yel_horiz, em_yel_vert = cad_step_row['yellow_horiz_emittance'], cad_step_row['yellow_vert_emittance']

            beam_width_x = 205 / np.sqrt(2) * beam_width_scale
            beam_width_y = 205 / np.sqrt(2) * beam_width_scale

            blue_widths = np.array([beam_width_x * np.sqrt(em_blue_horiz / em_blue_horiz_nom),
                                    beam_width_y * np.sqrt(em_blue_vert / em_blue_vert_nom)])
            yellow_widths = np.array([beam_width_x * np.sqrt(em_yel_horiz / em_yel_horiz_nom),
                                      beam_width_y * np.sqrt(em_yel_vert / em_yel_vert_nom)])

            collider_sim.set_bunch_sigmas(blue_widths, yellow_widths)
            collider_sim.set_bkg(bkg)
            collider_sim.set_gaus_z_efficiency_width(gauss_eff_width)
            collider_sim.set_gaus_smearing_sigma(mbd_resolution)

            collider_sim.set_bunch_crossing(blue_angle_x, blue_angle_y, yellow_angle_x, yellow_angle_y)

            long_paths = get_profile_path(longitudinal_profiles_dir_path, cad_step_row['start'], cad_step_row['end'],
                                          True)

            ax.step(centers_no_zdc, counts_no_zdc, where='mid', linewidth=1, alpha=0.3, color='red', label='Data No ZDC')
            ax.step(centers, counts, where='mid', linewidth=2, label='Data With ZDC')

            for longitudinal_profile_path in long_paths:
                blue_profile_path = longitudinal_profile_path.replace('COLOR_', 'blue_')
                yellow_profile_path = longitudinal_profile_path.replace('COLOR_', 'yellow_')
                collider_sim.set_longitudinal_profiles_from_file(blue_profile_path, yellow_profile_path)

                for offset_type in ['set', 'measured']:
                    if offset_type == 'set':
                        blue_offset_x, blue_offset_y = cad_step_row['set offset h'] * 1e3, cad_step_row['set offset v'] * 1e3
                    elif offset_type == 'measured':
                        blue_offset_x, blue_offset_y = cad_step_row['h_offset_shifted'], cad_step_row['v_offset_shifted']
                    ls = '-' if offset_type == 'set' else '--'
                    yellow_offset_x, yellow_offset_y = 0, 0

                    collider_sim.set_bunch_offsets([blue_offset_x, blue_offset_y], [yellow_offset_x, yellow_offset_y])

                    collider_sim.run_sim_parallel()

                    fit_mask = (centers > fit_range[0]) & (centers < fit_range[1])
                    if first_fit:  # Fit amplitude on first fit only and fix
                        fit_amp_shift(collider_sim, counts[fit_mask], centers[fit_mask], count_errs[fit_mask])
                        first_fit = False
                    else:
                        fit_shift_only(collider_sim, counts[fit_mask], centers[fit_mask], count_errs[fit_mask])

                    zs, z_dist = collider_sim.get_z_density_dist()
                    ax.plot(zs, z_dist, linewidth=2, ls=ls)

            if scan_type == 'horizontal':
                ax.annotate(f'Offset: {blue_offset_x:.1f} μm', xy=(0.05, 0.9), xycoords='axes fraction',
                            ha='left', va='top', fontsize=12)
            else:
                ax.annotate(f'Offset: {blue_offset_y:.1f} μm', xy=(0.05, 0.9), xycoords='axes fraction',
                            ha='left', va='top', fontsize=12)

            ax.set_ylim(bottom=0)
            # ax.set_xlim(fit_range)
            ax.grid()

            if i == 0:  # top row
                ax.legend()
            if i == 5 or i == 11:  # bottom row
                ax.set_xlabel('z (cm)')
            if i < 6:  # left column
                ax.set_ylabel('Rate (Hz)')

            # Adjust Y-axis upper limit
            max_y = np.max(counts[(centers < 180) & (centers > -180)]) * 1.2
            ax.set_ylim(top=max_y)

        fig.suptitle(f'{scan_type.capitalize()} Scan Steps', fontsize=16)
        fig.subplots_adjust(left=0.06, right=0.995, top=0.95, bottom=0.06, wspace=0.1, hspace=0.02)

        save_base = f'{base_path_auau}{scan_type}_scan_all_steps'
        fig.savefig(f'{save_base}.png')
        fig.savefig(f'{save_base}.pdf')
        # plt.show()
        plt.close(fig)


def auau_tests():
    if platform.system() == 'Windows':
        base_path = 'C:/Users/Dylan/Desktop/'
    else:
        base_path = '/local/home/dn277127/Bureau/'

    base_path_auau = f'{base_path}Vernier_Scans/auau_oct_16_24/'
    longitudinal_profiles_dir_path = f'{base_path_auau}profiles/'
    z_vertex_data_path = f'{base_path_auau}vertex_data/54733_vertex_distributions_no_zdc_coinc.root'
    z_vertex_zdc_data_path = f'{base_path_auau}vertex_data/54733_vertex_distributions.root'
    combined_cad_step_data_csv_path = f'{base_path_auau}combined_cad_step_data.csv'

    cad_df = pd.read_csv(combined_cad_step_data_csv_path)

    fit_range = [-230, 230]
    scan_type = 'horizontal'
    steps = [0, 3]

    collider_sim = BunchCollider()
    # collider_sim.set_grid_size(25, 25, 51, 21)
    first_fit = True

    # Get nominal dcct ions and emittances
    step_0 = cad_df[cad_df['step'] == 0].iloc[0]
    dcct_blue_nom, dcct_yellow_nom = step_0['blue_dcct_ions'], step_0['yellow_dcct_ions']
    em_blue_horiz_nom, em_blue_vert_nom = step_0['blue_horiz_emittance'], step_0['blue_vert_emittance']
    em_yel_horiz_nom, em_yel_vert_nom = step_0['yellow_horiz_emittance'], step_0['yellow_vert_emittance']

    for i, scan_step in enumerate(steps):
        print(f'{scan_type.capitalize()} Step {scan_step}')
        fig, ax = plt.subplots()
        cad_step_row = cad_df[cad_df['step'] == scan_step].iloc[0]

        with uproot.open(z_vertex_data_path) as f:
            hist = f[f'step_{scan_step}']
            centers_no_zdc = hist.axis().centers()
            counts_no_zdc = hist.counts()
            count_errs_no_zdc = hist.errors()
            count_errs_no_zdc[count_errs_no_zdc == 0] = 1

        with uproot.open(z_vertex_zdc_data_path) as f:
            hist = f[f'step_{scan_step}']
            centers = hist.axis().centers()
            counts = hist.counts()
            count_errs = hist.errors()
            count_errs[count_errs == 0] = 1

        # Normalize counts to ZDC rate
        zdc_raw_rate = cad_step_row['zdc_raw_rate']
        zdc_hist_counts = np.sum(counts)
        hist_scaling_factor = zdc_raw_rate / zdc_hist_counts

        counts *= hist_scaling_factor
        counts_no_zdc *= hist_scaling_factor

        # Scale for dcct
        step_dcct_blue, step_dcct_yellow = cad_step_row['blue_dcct_ions'], cad_step_row['yellow_dcct_ions']
        dcct_scale = (dcct_blue_nom * dcct_yellow_nom) / (step_dcct_blue * step_dcct_yellow)

        counts *= dcct_scale
        counts_no_zdc *= dcct_scale

        beam_width_scale = 0.95
        beta_star = 72  # cm
        # beta_star = 59.5  # cm
        bkg = 2.0e-17
        gauss_eff_width = 500
        mbd_resolution = 2.0
        blue_angle_x = -cad_step_row['blue angle h'] * 1e-3
        blue_angle_y = -cad_step_row['blue angle v'] * 1e-3
        yellow_angle_x = -cad_step_row['yellow angle h'] * 1e-3
        yellow_angle_y = -cad_step_row['yellow angle v'] * 1e-3

        collider_sim.set_bunch_beta_stars(beta_star, beta_star)

        em_blue_horiz, em_blue_vert = cad_step_row['blue_horiz_emittance'], cad_step_row['blue_vert_emittance']
        em_yel_horiz, em_yel_vert = cad_step_row['yellow_horiz_emittance'], cad_step_row['yellow_vert_emittance']

        # beam_width_x = 205 / np.sqrt(2) * beam_width_scale
        # beam_width_y = 205 / np.sqrt(2) * beam_width_scale
        beam_width_x = 123.9
        beam_width_y = 119.7

        blue_widths = np.array([beam_width_x * np.sqrt(em_blue_horiz / em_blue_horiz_nom),
                                beam_width_y * np.sqrt(em_blue_vert / em_blue_vert_nom)])
        yellow_widths = np.array([beam_width_x * np.sqrt(em_yel_horiz / em_yel_horiz_nom),
                                  beam_width_y * np.sqrt(em_yel_vert / em_yel_vert_nom)])

        collider_sim.set_bunch_sigmas(blue_widths, yellow_widths)
        collider_sim.set_bkg(bkg)
        collider_sim.set_gaus_z_efficiency_width(gauss_eff_width)
        collider_sim.set_gaus_smearing_sigma(mbd_resolution)

        longitudinal_profile_path = get_profile_path(longitudinal_profiles_dir_path, cad_step_row['start'], cad_step_row['end'],
                                      False)

        ax.step(centers_no_zdc, counts_no_zdc, where='mid', linewidth=1, alpha=0.3, color='red', label='Data No ZDC')
        ax.step(centers, counts, where='mid', linewidth=2, label='Data With ZDC')

        blue_profile_path = longitudinal_profile_path.replace('COLOR_', 'blue_')
        yellow_profile_path = longitudinal_profile_path.replace('COLOR_', 'yellow_')
        collider_sim.set_longitudinal_profiles_from_file(blue_profile_path, yellow_profile_path)

        for blue_offset_dx in [-70, 0, 70]:
            for yellow_angle_dx in [-0.07, 0, 0.07]:
                collider_sim.set_bunch_crossing(blue_angle_x, blue_angle_y, yellow_angle_x + yellow_angle_dx * 1e-3, yellow_angle_y)

                # for offset_type in ['set', 'measured']:
                for offset_type in ['set']:
                    if offset_type == 'set':
                        blue_offset_x, blue_offset_y = cad_step_row['set offset h'] * 1e3, cad_step_row['set offset v'] * 1e3
                    elif offset_type == 'measured':
                        blue_offset_x, blue_offset_y = cad_step_row['h_offset_shifted'], cad_step_row['v_offset_shifted']
                    ls = '-' if offset_type == 'set' else '--'
                    yellow_offset_x, yellow_offset_y = 0, 0

                    collider_sim.set_bunch_offsets([blue_offset_x + blue_offset_dx, blue_offset_y], [yellow_offset_x, yellow_offset_y])

                    collider_sim.run_sim_parallel()

                    fit_mask = (centers > fit_range[0]) & (centers < fit_range[1])
                    if first_fit:  # Fit amplitude on first fit only and fix
                        fit_amp_shift(collider_sim, counts[fit_mask], centers[fit_mask], count_errs[fit_mask])
                        first_fit = False
                    else:
                        fit_shift_only(collider_sim, counts[fit_mask], centers[fit_mask], count_errs[fit_mask])

                    zs, z_dist = collider_sim.get_z_density_dist()
                    ax.plot(zs, z_dist, linewidth=2, ls=ls)

        if scan_type == 'horizontal':
            ax.annotate(f'Offset: {blue_offset_x:.1f} μm', xy=(0.05, 0.9), xycoords='axes fraction',
                        ha='left', va='top', fontsize=12)
        else:
            ax.annotate(f'Offset: {blue_offset_y:.1f} μm', xy=(0.05, 0.9), xycoords='axes fraction',
                        ha='left', va='top', fontsize=12)

        ax.set_ylim(bottom=0)
        ax.grid()

        ax.legend()
        ax.set_xlabel('z (cm)')
        ax.set_ylabel('Rate (Hz)')

        # Adjust Y-axis upper limit
        max_y = np.max(counts[(centers < 180) & (centers > -180)]) * 1.2
        ax.set_ylim(top=max_y)

    plt.show()


def auau_minimization():
    if platform.system() == 'Windows':
        base_path = 'C:/Users/Dylan/Desktop/'
    else:
        base_path = '/local/home/dn277127/Bureau/'

    base_path_auau = f'{base_path}Vernier_Scans/auau_oct_16_24/'
    longitudinal_profiles_dir_path = f'{base_path_auau}profiles/'
    z_vertex_zdc_data_path = f'{base_path_auau}vertex_data/54733_vertex_distributions.root'
    z_vertex_no_zdc_data_path = f'{base_path_auau}vertex_data/54733_vertex_distributions_no_zdc_coinc.root'
    combined_cad_step_data_csv_path = f'{base_path_auau}combined_cad_step_data.csv'

    cad_df = pd.read_csv(combined_cad_step_data_csv_path)

    fit_range = [-200, 200]
    steps = [0]
    # steps = [0]
    # steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # steps = np.arange(0, 24)

    collider_sim = BunchCollider()
    # collider_sim.set_grid_size(31, 31, 101, 31)
    collider_sim.set_grid_size(31, 31, 101, 31)
    bkg = 2.0e-17
    gauss_eff_width = 500
    mbd_resolution = 2.0
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
    bounds = [(100, 145), (130, 130), (65, 120), (65, 120), (-0.1, 0.1), (-100.0, 100.0), (-0.05, 0.05), (-50.0, 50.0)]

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
        args=(collider_sim, cad_df, centers_list, counts_list, count_errs_list, sim_settings),
        strategy='best1bin',
        maxiter=10000,
        popsize=15,
        tol=0.01,
        mutation=(0.5, 1),
        recombination=0.7,
        polish=True,
        updating='deferred',
        disp=True
    )

    print('Optimization result:', result)


def auau_manual_simple_minimization():
    if platform.system() == 'Windows':
        base_path = 'C:/Users/Dylan/Desktop/'
    else:
        base_path = '/local/home/dn277127/Bureau/'

    base_path_auau = f'{base_path}Vernier_Scans/auau_oct_16_24/'
    longitudinal_profiles_dir_path = f'{base_path_auau}profiles/'
    z_vertex_zdc_data_path = f'{base_path_auau}vertex_data/54733_vertex_distributions.root'
    combined_cad_step_data_csv_path = f'{base_path_auau}combined_cad_step_data.csv'

    cad_df = pd.read_csv(combined_cad_step_data_csv_path)

    fit_range = [-230, 230]
    steps = np.arange(0, 24)

    collider_sim = BunchCollider()
    collider_sim.set_grid_size(31, 31, 101, 31)
    bkg = 2.0e-17
    gauss_eff_width = 500
    mbd_resolution = 2.0
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

    bws_x, bws_y = np.linspace(110, 140, 5), np.linspace(110, 140, 5)

    bwxs, bwys, total_chi2s = [], [], []
    for beam_width_x in bws_x:
        for beam_width_y in bws_y:
            # beam_width_x, beam_width_y, beta_star, yellow_angle_x (mrad), blue_offset_x (microns)
            initial_guess = np.array([beam_width_x, beam_width_y, 72.0, 0.0, 0.0, 0.0, 0.0])

            chi2 = compute_total_chi2(
                initial_guess,
                collider_sim,
                cad_df,
                centers_list,
                counts_list,
                count_errs_list,
                sim_settings
            )
            bwxs.append(beam_width_x)
            bwys.append(beam_width_y)
            total_chi2s.append(chi2)

    bwxs = np.array(bwxs)
    bwys = np.array(bwys)
    total_chi2s = np.array(total_chi2s)

    # Reshape the data to 2D grids
    bws_x_unique = np.unique(bwxs)
    bws_y_unique = np.unique(bwys)
    chi2_grid = total_chi2s.reshape(len(bws_x_unique), len(bws_y_unique))

    # Find the minimum chi2 and corresponding parameters
    min_index = np.argmin(total_chi2s)
    min_bw_x = bwxs[min_index]
    min_bw_y = bwys[min_index]
    min_chi2 = total_chi2s[min_index]

    print(f"Minimum chi2: {min_chi2:.2f} at beam_width_x = {min_bw_x:.2f}, beam_width_y = {min_bw_y:.2f}")

    # Plotting
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(bws_x_unique, bws_y_unique, chi2_grid.T, levels=20, cmap="viridis")
    plt.colorbar(cp, label='Total chi²')
    plt.plot(min_bw_x, min_bw_y, 'r*', markersize=10, label='Minimum chi²')
    plt.xlabel('beam_width_x [µm]')
    plt.ylabel('beam_width_y [µm]')
    plt.title('Chi² Landscape')
    plt.legend()
    plt.tight_layout()
    plt.show()


def auau_manual_simple_beta_star_minimization():
    if platform.system() == 'Windows':
        base_path = 'C:/Users/Dylan/Desktop/'
    else:
        base_path = '/local/home/dn277127/Bureau/'

    base_path_auau = f'{base_path}Vernier_Scans/auau_oct_16_24/'
    longitudinal_profiles_dir_path = f'{base_path_auau}profiles/'
    z_vertex_zdc_data_path = f'{base_path_auau}vertex_data/54733_vertex_distributions.root'
    combined_cad_step_data_csv_path = f'{base_path_auau}combined_cad_step_data.csv'

    cad_df = pd.read_csv(combined_cad_step_data_csv_path)

    fit_range = [-230, 230]
    steps = np.arange(0, 24)

    collider_sim = BunchCollider()
    collider_sim.set_grid_size(31, 31, 101, 31)
    bkg = 2.0e-17
    gauss_eff_width = 500
    mbd_resolution = 2.0
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

    beam_width_x, beam_width_y = 130.0, 130.0
    beta_star_xs, beta_star_ys = np.linspace(60, 110, 20), np.linspace(60, 110, 20)

    bsxs, bsys, total_chi2s, total_log_likes = [], [], [], []
    for beta_star_x in beta_star_xs:
        for beta_star_y in beta_star_ys:
            # beam_width_x, beam_width_y, beta_star, yellow_angle_x (mrad), blue_offset_x (microns)
            initial_guess = np.array([beam_width_x, beam_width_y, beta_star_x, beta_star_y, 0.0, 0.0, 0.0, 0.0])

            chi2, log_like = compute_total_chi2(
                initial_guess,
                collider_sim,
                cad_df,
                centers_list,
                counts_list,
                count_errs_list,
                sim_settings,
                metrics=('chi2', 'log_like')
            )
            bsxs.append(beta_star_x)
            bsys.append(beta_star_y)
            total_chi2s.append(chi2)
            total_log_likes.append(log_like)

    bsxs = np.array(bsxs)
    bsys = np.array(bsys)
    total_chi2s = np.array(total_chi2s)
    total_log_likes = np.array(total_log_likes)

    # Reshape the data to 2D grids
    bs_x_unique = np.unique(bsxs)
    bs_y_unique = np.unique(bsys)
    chi2_grid = total_chi2s.reshape(len(bs_x_unique), len(bs_y_unique))
    log_like_grid = total_log_likes.reshape(len(bs_x_unique), len(bs_y_unique))

    # Find the minimum chi2 and corresponding parameters
    min_index = np.argmin(total_chi2s)
    min_bs_x = bsxs[min_index]
    min_bs_y = bsys[min_index]
    min_chi2 = total_chi2s[min_index]

    # Print the maximum log likelihood
    max_index = np.argmax(total_log_likes)
    max_bs_x = bsxs[max_index]
    max_bs_y = bsys[max_index]
    max_log_like = total_log_likes[max_index]

    print(f"Minimum chi2: {min_chi2:.2f} at beta_star_x = {min_bs_x:.2f}, beta_star_y = {min_bs_y:.2f}")
    print(f"Maximum log likelihood: {max_log_like:.2f} at beta_star_x = {max_bs_x:.2f}, beta_star_y = {max_bs_y:.2f}")

    # Plotting
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(bs_x_unique, bs_y_unique, chi2_grid.T, levels=20, cmap="viridis")
    plt.scatter(bs_x_unique, bs_y_unique, c=total_chi2s, cmap='viridis', marker='o', s=10, alpha=0.5)
    plt.colorbar(cp, label='Total chi²')
    plt.plot(min_bs_x, min_bs_y, 'r*', markersize=10, label='Minimum chi²')
    plt.xlabel('beam_width_x [µm]')
    plt.ylabel('beam_width_y [µm]')
    plt.title('Chi² Landscape')
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(8, 6))
    cp = plt.contourf(bs_x_unique, bs_y_unique, log_like_grid.T, levels=20, cmap="plasma")
    plt.scatter(bs_x_unique, bs_y_unique, c=total_log_likes, cmap='plasma', marker='o', s=10, alpha=0.5)
    plt.colorbar(cp, label='Total Log Likelihood')
    plt.plot(max_bs_x, max_bs_y, 'r*', markersize=10, label='Maximum Log Likelihood')
    plt.xlabel('beta_star_x [cm]')
    plt.ylabel('beta_star_y [cm]')
    plt.legend()
    plt.tight_layout()

    plt.show()


def auau_single_dist_bw_min():
    if platform.system() == 'Windows':
        base_path = 'C:/Users/Dylan/Desktop/'
    else:
        base_path = '/local/home/dn277127/Bureau/'
    base_path_auau = f'{base_path}vernier_scan_AuAu24/'
    longitudinal_fit_path = f'{base_path_auau}CAD_Measurements/VernierScan_AuAu_COLOR_longitudinal_fit_22.dat'
    z_vertex_data_path = f'{base_path_auau}vertex_data/plot2.root'
    z_vertex_zdc_data_path = f'{base_path_auau}vertex_data/zdc_mbd_plot2.root'

    scan_step_dict = define_scan_step_dict()
    scan_step = 3

    with uproot.open(z_vertex_data_path) as f:
        hist = f[f'hist{scan_step}']
        centers_no_zdc = hist.axis().centers()
        counts_no_zdc = hist.counts()
        count_errs_no_zdc = hist.errors()

    with uproot.open(z_vertex_zdc_data_path) as f:
        hist = f[f'hist{scan_step}']
        centers = hist.axis().centers()
        counts = hist.counts()
        count_errs = hist.errors()
        count_errs[count_errs == 0] = 1  # Where errors are zero, set them to 1

    fit_range = [-230, 230]
    # fit_range = [-50, 50]

    # beam_width_scales = [0.7, 0.9, 1.1]
    beam_width_scales = np.linspace(0.8, 1.1, 30)

    beta_star = 72  # cm
    blue_offset_x, blue_offset_y = scan_step_dict[scan_step]['x'] * 1e3, scan_step_dict[scan_step]['y'] * 1e3  # um
    yellow_offset_x, yellow_offset_y = 0, 0  # um
    # bkg = 20.0e-17  # Background rate
    bkg = 2.0e-17  # Background rate
    blue_angle_x = -0.125e-3  # rad
    blue_angle_y = -0.0e-3  # rad
    yellow_angle_x = -0.07e-3  # rad
    yellow_angle_y = -0.04e-3  # rad

    fig, ax = plt.subplots()

    resids = []
    for beam_width_scale in beam_width_scales:
        print(f'Beam width scale: {beam_width_scale:.2f}')
        beam_width_x = 175 / np.sqrt(2) * beam_width_scale  # um
        # beam_width_x = 175 / np.sqrt(2) * 0.935  # um
        beam_width_y = 287 / np.sqrt(2) * beam_width_scale  # um
        # beam_width_y = 287 / np.sqrt(2) * 0.935  # um

        collider_sim = BunchCollider()
        collider_sim.set_bunch_beta_stars(beta_star, beta_star)
        collider_sim.set_bunch_sigmas(np.array([beam_width_x, beam_width_y]), np.array([beam_width_x, beam_width_y]))
        collider_sim.set_bunch_offsets([blue_offset_x, blue_offset_y], [yellow_offset_x, yellow_offset_y])
        collider_sim.set_bkg(bkg)
        collider_sim.set_bunch_crossing(blue_angle_x, blue_angle_y, yellow_angle_x, yellow_angle_y)
        blue_fit_path = longitudinal_fit_path.replace('_COLOR_', '_blue_')
        yellow_fit_path = longitudinal_fit_path.replace('_COLOR_', '_yellow_')
        collider_sim.set_longitudinal_fit_parameters_from_file(blue_fit_path, yellow_fit_path)

        collider_sim.run_sim_parallel()

        fit_mask = (centers > fit_range[0]) & (centers < fit_range[1])
        fit_amp_shift(collider_sim, counts[fit_mask], centers[fit_mask], count_errs[fit_mask])

        zs, z_dist = collider_sim.get_z_density_dist()
        sim_interp = interp1d(zs, z_dist)
        # resid = np.sqrt(np.sum((counts[fit_mask] - sim_interp(centers[fit_mask])) ** 2)) / np.mean(counts[fit_mask])
        # Calculate chi2 per degree of freedom
        resid = np.sum((counts[fit_mask] - sim_interp(centers[fit_mask])) ** 2 / count_errs[fit_mask] ** 2) / (len(fit_mask) - 1)
        resids.append(resid)

    # Get the minimum residual and the corresponding beam width scale
    min_resid = np.min(resids)
    min_resid_index = np.argmin(resids)
    min_beam_width_scale = beam_width_scales[min_resid_index]

    beam_width_x = 175 / np.sqrt(2) * min_beam_width_scale  # um
    # beam_width_x = 175 / np.sqrt(2) * 0.935  # um
    beam_width_y = 287 / np.sqrt(2) * min_beam_width_scale  # um
    # beam_width_y = 287 / np.sqrt(2) * 0.935  # um

    collider_sim = BunchCollider()
    collider_sim.set_bunch_beta_stars(beta_star, beta_star)
    collider_sim.set_bunch_sigmas(np.array([beam_width_x, beam_width_y]), np.array([beam_width_x, beam_width_y]))
    collider_sim.set_bunch_offsets([blue_offset_x, blue_offset_y], [yellow_offset_x, yellow_offset_y])
    collider_sim.set_bkg(bkg)
    collider_sim.set_bunch_crossing(blue_angle_x, blue_angle_y, yellow_angle_x, yellow_angle_y)
    blue_fit_path = longitudinal_fit_path.replace('_COLOR_', '_blue_')
    yellow_fit_path = longitudinal_fit_path.replace('_COLOR_', '_yellow_')
    collider_sim.set_longitudinal_fit_parameters_from_file(blue_fit_path, yellow_fit_path)

    collider_sim.run_sim_parallel()

    fit_mask = (centers > fit_range[0]) & (centers < fit_range[1])
    fit_amp_shift(collider_sim, counts[fit_mask], centers[fit_mask])

    zs, z_dist = collider_sim.get_z_density_dist()

    ax.step(centers, counts, where='mid', label='Data', linewidth=2)
    ax.plot(zs, z_dist, label='Simulation', linewidth=2)
    ax.set_xlabel('z (cm)')
    ax.set_ylabel('Density')
    ax.set_title(f'{scan_step_dict[scan_step]["scan_orientation"]} Scan Step {scan_step}')
    ax.set_ylim(bottom=0)
    ax.grid()
    ax.legend()
    ax.set_ylim(top=np.max(counts[(centers < 180) & (centers > -180)]) * 1.2)
    fig.tight_layout()

    fig, ax = plt.subplots()
    ax.plot(beam_width_scales, resids, marker='o', linestyle='-', color='blue')

    plt.show()


def auau_residuals_vs_beam_width():
    if platform.system() == 'Windows':
        base_path = 'C:/Users/Dylan/Desktop/'
    else:
        base_path = '/local/home/dn277127/Bureau/'
    base_path_auau = f'{base_path}vernier_scan_AuAu24/'
    longitudinal_fit_path = f'{base_path_auau}CAD_Measurements/VernierScan_AuAu_COLOR_longitudinal_fit_22.dat'
    z_vertex_zdc_data_path = f'{base_path_auau}vertex_data/zdc_mbd_plot2.root'

    scan_step_dict = define_scan_step_dict()

    beam_width_scales = np.linspace(0.75, 1.05, 40)
    fit_range = [-230, 230]

    beta_star = 72  # cm
    bkg = 2.0e-17  # Background rate
    blue_angle_x = -0.125e-3
    blue_angle_y = 0.0e-3
    yellow_angle_x = -0.07e-3
    yellow_angle_y = -0.04e-3

    residuals_dict = {}

    for scan_step in range(12):
        print(f"Processing scan step {scan_step}")
        residuals = []

        with uproot.open(z_vertex_zdc_data_path) as f:
            hist = f[f'hist{scan_step}']
            centers = hist.axis().centers()
            counts = hist.counts()
            count_errs = hist.errors()
            count_errs[count_errs == 0] = 1  # Where errors are zero, set them to 1

        blue_offset_x, blue_offset_y = scan_step_dict[scan_step]['x'] * 1e3, scan_step_dict[scan_step]['y'] * 1e3
        yellow_offset_x, yellow_offset_y = 0, 0

        for beam_width_scale in beam_width_scales:
            beam_width_x = 175 / np.sqrt(2) * beam_width_scale
            beam_width_y = 287 / np.sqrt(2) * beam_width_scale

            collider_sim = BunchCollider()
            collider_sim.set_bunch_beta_stars(beta_star, beta_star)
            collider_sim.set_bunch_sigmas(np.array([beam_width_x, beam_width_y]), np.array([beam_width_x, beam_width_y]))
            collider_sim.set_bunch_offsets([blue_offset_x, blue_offset_y], [yellow_offset_x, yellow_offset_y])
            collider_sim.set_bkg(bkg)
            collider_sim.set_bunch_crossing(blue_angle_x, blue_angle_y, yellow_angle_x, yellow_angle_y)

            blue_fit_path = longitudinal_fit_path.replace('_COLOR_', '_blue_')
            yellow_fit_path = longitudinal_fit_path.replace('_COLOR_', '_yellow_')
            collider_sim.set_longitudinal_fit_parameters_from_file(blue_fit_path, yellow_fit_path)

            collider_sim.run_sim_parallel()

            fit_mask = (centers > fit_range[0]) & (centers < fit_range[1])
            fit_amp_shift(collider_sim, counts[fit_mask], centers[fit_mask], count_errs[fit_mask])

            zs, z_dist = collider_sim.get_z_density_dist()
            sim_interp = interp1d(zs, z_dist, bounds_error=False, fill_value=0)

            resid = np.sum((counts[fit_mask] - sim_interp(centers[fit_mask])) ** 2 / count_errs[fit_mask] ** 2)
            resid /= (np.sum(fit_mask) - 1)
            residuals.append(resid)

        residuals_dict[f'step_{scan_step}'] = residuals

    residuals_df = pd.DataFrame(residuals_dict, index=beam_width_scales)
    residuals_df.index.name = 'beam_width_scale'
    residuals_df.to_csv(f'{base_path_auau}beam_width_residuals.csv')

    # Plot residuals per scan step
    fig, ax = plt.subplots()
    for step in residuals_df.columns:
        ax.plot(np.array(residuals_df.index), np.array(residuals_df[step]), label=step)
    ax.set_xlabel('Beam Width Scale')
    ax.set_ylabel('Chi² / DOF')
    ax.set_title('Residuals per Scan Step vs Beam Width Scale')
    ax.legend(ncol=2, fontsize='small')
    ax.grid(True)
    fig.tight_layout()

    # Plot average residual
    avg_residuals = residuals_df.mean(axis=1)
    fig, ax = plt.subplots()
    ax.plot(np.array(residuals_df.index), np.array(avg_residuals), marker='o', color='black')
    ax.set_xlabel('Beam Width Scale')
    ax.set_ylabel('Average Chi² / DOF')
    ax.set_title('Average Residual vs Beam Width Scale')
    ax.grid(True)
    fig.tight_layout()

    # Plot optimal simulation overlayed with data for each scan step
    best_scale = avg_residuals.idxmin()
    print(f"Best beam width scale: {best_scale:.4f}")

    fig, axs = plt.subplots(4, 3, figsize=(15, 10), sharex=True, sharey=True)
    axs = axs.flatten()

    for scan_step in range(12):
        ax = axs[scan_step]
        with uproot.open(z_vertex_zdc_data_path) as f:
            hist = f[f'hist{scan_step}']
            centers = hist.axis().centers()
            counts = hist.counts()
            count_errs = hist.errors()
            count_errs[count_errs == 0] = 1  # Where errors are zero, set them to 1

        blue_offset_x, blue_offset_y = scan_step_dict[scan_step]['x'] * 1e3, scan_step_dict[scan_step]['y'] * 1e3
        yellow_offset_x, yellow_offset_y = 0, 0

        beam_width_x = 175 / np.sqrt(2) * best_scale
        beam_width_y = 287 / np.sqrt(2) * best_scale

        collider_sim = BunchCollider()
        collider_sim.set_bunch_beta_stars(beta_star, beta_star)
        collider_sim.set_bunch_sigmas(np.array([beam_width_x, beam_width_y]), np.array([beam_width_x, beam_width_y]))
        collider_sim.set_bunch_offsets([blue_offset_x, blue_offset_y], [yellow_offset_x, yellow_offset_y])
        collider_sim.set_bkg(bkg)
        collider_sim.set_bunch_crossing(blue_angle_x, blue_angle_y, yellow_angle_x, yellow_angle_y)

        blue_fit_path = longitudinal_fit_path.replace('_COLOR_', '_blue_')
        yellow_fit_path = longitudinal_fit_path.replace('_COLOR_', '_yellow_')
        collider_sim.set_longitudinal_fit_parameters_from_file(blue_fit_path, yellow_fit_path)

        collider_sim.run_sim_parallel()
        fit_mask = (centers > fit_range[0]) & (centers < fit_range[1])
        fit_amp_shift(collider_sim, counts[fit_mask], centers[fit_mask], count_errs[fit_mask])

        zs, z_dist = collider_sim.get_z_density_dist()

        ax.errorbar(centers, counts, yerr=count_errs, fmt='o', label='Data', markersize=2)
        ax.plot(zs, z_dist, '-', label='Sim', color='red', lw=1)
        ax.set_title(f'Scan Step {scan_step}')
        ax.grid(True)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2)
    fig.suptitle('Data vs Simulation (Best Fit Scale)', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


    
    
def auau_single_dist_multi_par_opt():
    if platform.system() == 'Windows':
        base_path = 'C:/Users/Dylan/Desktop/'
    else:
        base_path = '/local/home/dn277127/Bureau/'
    base_path_auau = f'{base_path}vernier_scan_AuAu24/'
    longitudinal_fit_path = f'{base_path_auau}CAD_Measurements/VernierScan_AuAu_COLOR_longitudinal_fit_22.dat'
    z_vertex_data_path = f'{base_path_auau}vertex_data/plot2.root'

    scan_step_dict = define_scan_step_dict()
    scan_step = 3

    with uproot.open(z_vertex_data_path) as f:
        hist1 = f[f'hist{scan_step}']
        centers = hist1.axis().centers()
        counts = hist1.counts()
        count_errs = hist1.errors()

    fit_range = [-180, 180]
    # fit_range = [-50, 50]

    beta_star = 72  # cm
    blue_offset_x, blue_offset_y = scan_step_dict[scan_step]['x'] * 1e3, scan_step_dict[scan_step]['y'] * 1e3  # um
    yellow_offset_x, yellow_offset_y = 0, 0  # um
    bkg = 2.0e-17  # Background rate
    blue_angle_x = -0.125e-3  # rad
    blue_angle_y = -0.0e-3  # rad
    yellow_angle_x = -0.07e-3  # rad
    yellow_angle_y = -0.04e-3  # rad
    beam_width_scale = 0.935

    fig, ax = plt.subplots()

    beam_width_x = 175 / np.sqrt(2) * beam_width_scale  # um
    beam_width_y = 287 / np.sqrt(2) * beam_width_scale  # um

    initial_params = {
        'beta_star_x': 72,
        'beta_star_y': 72,
        'beam_width_x': 175 / np.sqrt(2) * 0.935,
        'beam_width_y': 287 / np.sqrt(2) * 0.935,
        'blue_offset_x': scan_step_dict[scan_step]['x'] * 1e3,
        'blue_offset_y': scan_step_dict[scan_step]['y'] * 1e3,
        'yellow_offset_x': 0.0,
        'yellow_offset_y': 0.0,
        'bkg': 2.0e-17,
        'blue_angle_x': -0.125e-3,
        'blue_angle_y': -0.0e-3,
        'yellow_angle_x': -0.07e-3,
        'yellow_angle_y': -0.04e-3,
    }

    param_bounds = {
        'beta_star_x': (40, 150),
        # 'beta_star_y': (50, 100),
        'beta_star_y': 'fixed',
        'beam_width_x': (100, 200),
        # 'beam_width_y': (100, 300),
        'beam_width_y': 'fixed',
        'blue_offset_x': (initial_params['blue_offset_x'], initial_params['blue_offset_x']),  # fixed
        'blue_offset_y': (initial_params['blue_offset_y'], initial_params['blue_offset_y']),  # fixed
        'yellow_offset_x': (0.0, 0.0),  # fixed
        'yellow_offset_y': (0.0, 0.0),  # fixed
        # 'bkg': (0e-17, 20e-17),
        'bkg': 'fixed',
        'blue_angle_x': (-1e-3, 1e-3),
        # 'blue_angle_y': (-1e-3, 1e-3),
        # 'yellow_angle_x': (-1e-3, 1e-3),
        # 'yellow_angle_y': (-1e-3, 1e-3),
        # 'blue_angle_x': 'fixed',
        'blue_angle_y': 'fixed',
        'yellow_angle_x': 'fixed',
        'yellow_angle_y': 'fixed',
    }

    free_param_names = []
    free_param_vals = []
    bounds = []

    for name, val in initial_params.items():
        b = param_bounds.get(name, (None, None))
        if b=='fixed' or b[0] == b[1]:  # fixed
            continue
        free_param_names.append(name)
        free_param_vals.append(val)
        bounds.append(b)

    fixed_params = {
        name: val for name, val in initial_params.items()
        if name not in free_param_names
    }

    result = minimize(
        residual_wrapper_free_params,
        free_param_vals,
        args=(free_param_names, fixed_params, longitudinal_fit_path, fit_range, centers, counts),
        bounds=bounds,
        method='L-BFGS-B',  # or 'Powell' if your function isn't smooth
        options={'maxiter': 100}
    )

    # Final parameters
    fitted_params = fixed_params.copy()
    fitted_params.update(dict(zip(free_param_names, result.x)))

    print("Final fitted parameters:")
    for key, val in fitted_params.items():
        print(f"{key} = {val:.6g}")

    print("\nResidual:", result.fun)
    print(result)

    resid, collider_sim = residual_wrapper_free_params(result.x, free_param_names, fixed_params, longitudinal_fit_path,
                                                       fit_range, centers, counts, return_collider_sim=True)

    zs, z_dist = collider_sim.get_z_density_dist()

    ax.step(centers, counts, where='mid', label='Data', linewidth=2)
    ax.plot(zs, z_dist, label='Simulation', linewidth=2)
    ax.set_xlabel('z (cm)')
    ax.set_ylabel('Density')
    ax.set_title(f'{scan_step_dict[scan_step]["scan_orientation"]} Scan Step {scan_step}')
    ax.set_ylim(bottom=0)
    ax.grid()
    ax.legend()
    ax.set_ylim(top=np.max(counts[(centers < 180) & (centers > -180)]) * 1.2)
    fig.tight_layout()

    plt.show()


def comp_pp_auau():
    if platform.system() == 'Windows':
        base_path = 'C:/Users/Dylan/Desktop/'
    else:
        base_path = '/local/home/dn277127/Bureau/'
    base_path_auau = f'{base_path}vernier_scan_AuAu24/'
    longitudinal_fit_path_auau = f'{base_path_auau}CAD_Measurements/VernierScan_AuAu_COLOR_longitudinal_fit_22.dat'
    base_path_pp = f'{base_path}vernier_scan/'
    longitudinal_fit_path_pp = f'{base_path_pp}CAD_Measurements/VernierScan_Aug12_COLOR_longitudinal_fit.dat'

    beam_width_x = 175 / np.sqrt(2) * 0.9  # um
    beam_width_y = 287 / np.sqrt(2) * 0.9  # um
    beta_star = 72  # cm
    bkg = 0.0e-17  # Background rate
    blue_angle_x = 0.0e-3  # rad
    blue_angle_y = -0.05e-3  # rad
    yellow_angle_x = 0.0e-3  # rad
    yellow_angle_y = +0.0e-3  # rad
    z_init = 6.0e6  # cm

    print(f'Beam width x: {beam_width_x} um, Beam width y: {beam_width_y} um')

    fig, ax = plt.subplots()

    for longitudinal_fit_path in [longitudinal_fit_path_auau, longitudinal_fit_path_pp]:
        collider_sim = BunchCollider()
        collider_sim.set_bunch_rs(np.array([0., 0., -z_init]), np.array([0., 0., z_init]))
        collider_sim.set_bunch_beta_stars(beta_star, beta_star)
        collider_sim.set_bunch_sigmas(np.array([beam_width_x, beam_width_y]), np.array([beam_width_x, beam_width_y]))
        collider_sim.set_bkg(bkg)
        collider_sim.set_bunch_crossing(blue_angle_x, blue_angle_y, yellow_angle_x, yellow_angle_y)
        blue_fit_path = longitudinal_fit_path.replace('_COLOR_', '_blue_')
        yellow_fit_path = longitudinal_fit_path.replace('_COLOR_', '_yellow_')
        collider_sim.set_longitudinal_fit_parameters_from_file(blue_fit_path, yellow_fit_path)

        collider_sim.run_sim_parallel()

        zs, z_dist = collider_sim.get_z_density_dist()

        label = 'AuAu' if longitudinal_fit_path == longitudinal_fit_path_auau else 'pp'
        ax.plot(zs, z_dist, label=label, linewidth=2)
    ax.set_xlabel('z (cm)')
    ax.set_ylabel('Density')
    ax.set_title('Z-Vertex Distribution')
    ax.set_ylim(bottom=0)
    ax.grid()
    ax.legend()
    fig.tight_layout()
    plt.show()


def define_scan_step_dict():
    scan_steps = {
        0: {"x": 0.0, "y": 0.0, "scan_orientation": "Horizontal"},
        1: {"x": 0.1, "y": 0.0, "scan_orientation": "Horizontal"},
        2: {"x": 0.2, "y": 0.0, "scan_orientation": "Horizontal"},
        3: {"x": 0.35, "y": 0.0, "scan_orientation": "Horizontal"},
        4: {"x": 0.5, "y": 0.0, "scan_orientation": "Horizontal"},
        5: {"x": 0.8, "y": 0.0, "scan_orientation": "Horizontal"},
        6: {"x": 0.0, "y": 0.0, "scan_orientation": "Horizontal"},
        7: {"x": -0.1, "y": 0.0, "scan_orientation": "Horizontal"},
        8: {"x": -0.2, "y": 0.0, "scan_orientation": "Horizontal"},
        9: {"x": -0.35, "y": 0.0, "scan_orientation": "Horizontal"},
        10: {"x": -0.5, "y": 0.0, "scan_orientation": "Horizontal"},
        11: {"x": -0.8, "y": 0.0, "scan_orientation": "Horizontal"},
        12: {"x": 0.0, "y": 0.0, "scan_orientation": "Vertical"},
        13: {"x": 0.0, "y": 0.1, "scan_orientation": "Vertical"},
        14: {"x": 0.0, "y": 0.25, "scan_orientation": "Vertical"},
        15: {"x": 0.0, "y": 0.45, "scan_orientation": "Vertical"},
        16: {"x": 0.0, "y": 0.7, "scan_orientation": "Vertical"},
        17: {"x": 0.0, "y": 1.0, "scan_orientation": "Vertical"},
        18: {"x": 0.0, "y": 0.0, "scan_orientation": "Vertical"},
        19: {"x": 0.0, "y": -0.1, "scan_orientation": "Vertical"},
        20: {"x": 0.0, "y": -0.25, "scan_orientation": "Vertical"},
        21: {"x": 0.0, "y": -0.45, "scan_orientation": "Vertical"},
        22: {"x": 0.0, "y": -0.7, "scan_orientation": "Vertical"},
        23: {"x": 0.0, "y": -1.0, "scan_orientation": "Vertical"},
    }

    return scan_steps


def run_sim(beta_star_x, beta_star_y, beam_width_x, beam_width_y, blue_offset_x, blue_offset_y,
            yellow_offset_x, yellow_offset_y, bkg, blue_angle_x, blue_angle_y,
            yellow_angle_x, yellow_angle_y, longitudinal_fit_path, fit_range,
            centers, counts, return_collider_sim=False):
    collider_sim = BunchCollider()
    collider_sim.set_bunch_beta_stars(beta_star_x, beta_star_y)
    collider_sim.set_bunch_sigmas(np.array([beam_width_x, beam_width_y]), np.array([beam_width_x, beam_width_y]))
    collider_sim.set_bunch_offsets([blue_offset_x, blue_offset_y], [yellow_offset_x, yellow_offset_y])
    collider_sim.set_bkg(bkg)
    collider_sim.set_bunch_crossing(blue_angle_x, blue_angle_y, yellow_angle_x, yellow_angle_y)
    blue_fit_path = longitudinal_fit_path.replace('_COLOR_', '_blue_')
    yellow_fit_path = longitudinal_fit_path.replace('_COLOR_', '_yellow_')
    collider_sim.set_longitudinal_fit_parameters_from_file(blue_fit_path, yellow_fit_path)

    collider_sim.run_sim_parallel()

    fit_mask = (centers > fit_range[0]) & (centers < fit_range[1])
    fit_amp_shift(collider_sim, counts[fit_mask], centers[fit_mask])

    zs, z_dist = collider_sim.get_z_density_dist()
    sim_interp = interp1d(zs, z_dist)
    resid = np.sqrt(np.sum((counts[fit_mask] - sim_interp(centers[fit_mask])) ** 2)) / np.mean(counts[fit_mask])

    print(f'Res: {resid:.3f}: b*x: {beta_star_x:.1f} cm, b*y: {beta_star_y:.1f} cm, '
          f'bwx: {beam_width_x:.1f} um, bwy: {beam_width_y:.1f} um, bkg: {bkg:.1e}, '
          # f'Blue offset x: {blue_offset_x:.1f} um, Blue offset y: {blue_offset_y:.1f} um, '
          # f'Yellow offset x: {yellow_offset_x:.1f} um, Yellow offset y: {yellow_offset_y:.1f} um, '
          f'bxx: {blue_angle_x * 1e3:.1e} mrad, bxy: {blue_angle_y * 1e3:.1e} mrad, '
          f'yxx: {yellow_angle_x * 1e3:.1e} mrad, yxy: {yellow_angle_y * 1e3:.1e} mrad')
    if return_collider_sim:
        return resid, collider_sim
    return resid


def residual_wrapper(params, longitudinal_fit_path, fit_range, centers, counts):
    return run_sim(
        beta_star_x=params[0],
        beta_star_y=params[1],
        beam_width_x=params[2],
        beam_width_y=params[3],
        blue_offset_x=params[4],
        blue_offset_y=params[5],
        yellow_offset_x=params[6],
        yellow_offset_y=params[7],
        bkg=params[8],
        blue_angle_x=params[9],
        blue_angle_y=params[10],
        yellow_angle_x=params[11],
        yellow_angle_y=params[12],
        longitudinal_fit_path=longitudinal_fit_path,
        fit_range=fit_range,
        centers=centers,
        counts=counts
    )


def residual_wrapper_free_params(free_params, param_names, fixed_params, longitudinal_fit_path, fit_range, centers, counts, return_collider_sim=False):
    full_params = fixed_params.copy()
    full_params.update(dict(zip(param_names, free_params)))
    return run_sim(
        beta_star_x=full_params['beta_star_x'],
        beta_star_y=full_params['beta_star_y'],
        beam_width_x=full_params['beam_width_x'],
        beam_width_y=full_params['beam_width_y'],
        blue_offset_x=full_params['blue_offset_x'],
        blue_offset_y=full_params['blue_offset_y'],
        yellow_offset_x=full_params['yellow_offset_x'],
        yellow_offset_y=full_params['yellow_offset_y'],
        bkg=full_params['bkg'],
        blue_angle_x=full_params['blue_angle_x'],
        blue_angle_y=full_params['blue_angle_y'],
        yellow_angle_x=full_params['yellow_angle_x'],
        yellow_angle_y=full_params['yellow_angle_y'],
        longitudinal_fit_path=longitudinal_fit_path,
        fit_range=fit_range,
        centers=centers,
        counts=counts,
        return_collider_sim=return_collider_sim
    )


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on July 08 9:55 PM 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/lumi_decay_vs_time.py

@author: Dylan Neff, dylan
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from datetime import timedelta

from common_logistics import set_base_path
from analyze_ions import read_ions_file
from analyze_emittance import read_emittance_file, parametrize_emittances_vs_time
from bpm_analysis import read_bpm_file
from analyze_sphnx_root_file import get_root_data_time, get_bco_offset
from rate_corrections import solve_sasha_equation
from z_vertex_fitting_common import get_profile_path
from BunchCollider import BunchCollider


def main():
    base_path = set_base_path()
    scan_path = f'{base_path}Vernier_Scans/auau_oct_16_24/'
    root_file_name = 'calofit_54733.root'
    # scan_path = f'{base_path}Vernier_Scans/auau_july_17_25/'
    # root_file_name = '69561.root'
    # scan_path = f'{base_path}Vernier_Scans/pp_aug_12_24/'
    # root_file_name = 'calofitting_51195.root'
    plot_lumi_decay(scan_path, root_file_name)
    print('donzo')


def plot_lumi_decay(base_path, root_file_name='calofit_69561.root'):
    """
    Plot the luminosity decay over time from the ions data.
    """
    longitudinal_profiles_dir_path = f'{base_path}profiles/'
    combined_cad_step_data_csv_path = f'{base_path}combined_cad_step_data.csv'
    ions_path = f'{base_path}COLOR_ions.dat'
    emittance_file_path = f'{base_path}emittance.dat'
    bpm_file_path = f'{base_path}bpms.dat'

    ions_data = {'blue': None, 'yellow': None}
    for color in ions_data.keys():
        ions_color_path = ions_path.replace('COLOR_', f'{color}_')
        ion_data = read_ions_file(ions_color_path)
        ions_data[color] = ion_data

    emittance_df = read_emittance_file(emittance_file_path)

    f_beam = 78.4 * 1e3  # kHz
    n_bunch = 111
    mb_to_um2 = 1e-19
    ions_avg_window = timedelta(seconds=10)

    cad_df = pd.read_csv(combined_cad_step_data_csv_path)

    bco_offset = get_bco_offset(base_path, cad_df, root_file_name)
    print(f'BCO offset: {bco_offset}')
    root_branches = ['BCO', 'GL1_clock_count', 'GL1_live_count', 'mbd_live_count', 'mbd_S_live_count',
                     'mbd_N_live_count', 'zdc_live_count', 'zdc_S_live_count', 'zdc_N_live_count']
    rate_data, rate_times = get_root_data_time(base_path, root_file_name, branches=root_branches, bco_offset=bco_offset)

    profile_paths, profile_times = get_profile_path(
        longitudinal_profiles_dir_path, cad_df['start'].min(), cad_df['end'].max(), True, return_times=True
    )
    # profile_paths, profile_times = get_profile_path(
    #     longitudinal_profiles_dir_path, cad_df['start'].min(), cad_df['end'].iloc[2], True, return_times=True
    # )

    bpm_data, blue_xing_h, yellow_xing_h, blue_xing_v, yellow_xing_v, rel_xing_h, rel_xing_v = (
        read_bpm_file(bpm_file_path, cad_df['start'].min(), cad_df['end'].max()))

    if '/pp_' in base_path:
        emittance_poly_order = 0
    else:
        emittance_poly_order = 2

    emittance_df = parametrize_emittances_vs_time(emittance_df, pd.array(profile_times), poly_order=emittance_poly_order)
    # Replace column names
    emittance_df = emittance_df.rename(columns={'BlueHoriz_fit': 'blue_horiz_emittance', 'BlueVert_fit': 'blue_vert_emittance',
                                                    'YellowHoriz_fit': 'yellow_horiz_emittance', 'YellowVert_fit': 'yellow_vert_emittance'})

    if base_path.split('/')[-2] == 'auau_oct_16_24':
        run_name = 'AuAu 2024'
        beta_star = 80.3  # in cm
        beam_width_x, beam_width_y = 130.0, 130.0
    elif base_path.split('/')[-2] == 'auau_july_17_25':
        run_name = 'AuAu 25'
        beta_star = 82.1  # in cm
        beam_width_x, beam_width_y = 130.0, 130.0
    elif base_path.split('/')[-2] == 'pp_aug_12_24':
        run_name = 'pp 24'
        beta_star = 111.6 # in cm
        beam_width_x, beam_width_y = 130.0, 130.0
    else:
        raise ValueError(f'Unknown run number for base path: {base_path}')


    collider_sim = BunchCollider()
    # collider_sim.set_grid_size(31, 31, 101, 31)
    collider_sim.set_grid_size(21, 21, 51, 21)
    # beta_star = 76.7
    bkg = 0.0e-17
    # gauss_eff_width = 500
    # mbd_resolution = 1.0
    gauss_eff_width = None
    mbd_resolution = None

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

    # WCM to DCCT scaling factor
    blue_time_mask = (
            (ions_data['blue']['Time'] >= profile_times[0] - ions_avg_window / 2) &
            (ions_data['blue']['Time'] <= profile_times[0] + ions_avg_window / 2)
    )
    yellow_time_mask = (
            (ions_data['yellow']['Time'] >= profile_times[0] - ions_avg_window / 2) &
            (ions_data['yellow']['Time'] <= profile_times[0] + ions_avg_window / 2)
    )
    n_blue_dcct0 = np.mean(ions_data['blue']['blue_dcct_ions'][blue_time_mask])
    n_yellow_dcct0 = np.mean(ions_data['yellow']['yellow_dcct_ions'][yellow_time_mask])
    n_blue_wcm0 = np.mean(ions_data['blue']['blue_wcm_ions'][blue_time_mask])
    n_yellow_wcm0 = np.mean(ions_data['yellow']['yellow_wcm_ions'][yellow_time_mask])

    xing_angle_mask0 = ((bpm_data['Time'] >= profile_times[0] - ions_avg_window / 2) &
                        (bpm_data['Time'] <= profile_times[0] + ions_avg_window / 2))

    run_start = pd.to_datetime(cad_df.iloc[0]['start'])

    lumi_runs = [
        {'angle': False, 'emittance': False, 'profile': False, 'n_protons': False, 'lumis': [], 'name': 'Baseline', 'color': 'black', 'ls': '-', 'lw': 1},
        {'angle': True, 'emittance': False, 'profile': False, 'n_protons': False, 'lumis': [], 'name': 'Crossing Angles Variation', 'color': 'gray', 'ls': '-', 'lw': 1},
        {'angle': False, 'emittance': True, 'profile': False, 'n_protons': False, 'lumis': [], 'name': 'Emittance Growth', 'color': 'orange', 'ls': '-', 'lw': 2},
        {'angle': False, 'emittance': False, 'profile': True, 'n_protons': False, 'lumis': [], 'name': 'Longitudinal Profiles', 'color': 'purple', 'ls': '-', 'lw': 2},
        {'angle': False, 'emittance': False, 'profile': False, 'n_protons': 'dcct', 'lumis': [], 'name': 'DCCT Proton Burn Off', 'color': 'green', 'ls': '--', 'lw': 2},
        {'angle': False, 'emittance': False, 'profile': False, 'n_protons': 'wcm', 'lumis': [], 'name': 'WCM Proton Burn Off', 'color': 'green', 'ls': '-', 'lw': 2},
        {'angle': True, 'emittance': True, 'profile': True, 'n_protons': 'dcct', 'lumis': [], 'name': 'All Effects (DCCT)', 'color': 'cyan', 'ls': '--', 'lw': 2},
        {'angle': True, 'emittance': True, 'profile': True, 'n_protons': 'wcm', 'lumis': [], 'name': 'All Effects (WCM)', 'color': 'cyan', 'ls': '-', 'lw': 2},
    ]

    mbd_rates, zdc_rates, mbd_uncor_rates, zdc_uncor_rates = [], [], [], []
    lumi_baseline, baseline_i, rate_data_exists = None, None, True
    for profile_i, (profile_path, time) in enumerate(zip(profile_paths, profile_times)):
        print(f'Time: {time}')
        time_low, time_high = time - ions_avg_window / 2, time + ions_avg_window / 2
        time_low_sec, time_high_sec = (time_low - run_start).total_seconds(), (time_high - run_start).total_seconds()
        rate_period_data = rate_data[(rate_times >= time_low_sec) & (rate_times <= time_high_sec)]
        rate_period_times = rate_times[(rate_times >= time_low_sec) & (rate_times <= time_high_sec)]
        if rate_period_data.empty:
            print(f'No rate data for time {time}')
            rate_data_exists = False
            mbd_rates.append(np.nan)
            zdc_rates.append(np.nan)
            mbd_uncor_rates.append(np.nan)
            zdc_uncor_rates.append(np.nan)
        else:
            rate_data_exists = True
            step_fine_dur = rate_period_times.iloc[-1] - rate_period_times.iloc[0]
            clock_raw_rate = (rate_period_data['GL1_clock_count'].iloc[-1] - rate_period_data['GL1_clock_count'].iloc[0]) / step_fine_dur
            clock_live_rate = (rate_period_data['GL1_live_count'].iloc[-1] - rate_period_data['GL1_live_count'].iloc[0]) / step_fine_dur
            clock_scale = clock_raw_rate / clock_live_rate

            zdc_rate = (rate_period_data['zdc_live_count'].iloc[-1] - rate_period_data['zdc_live_count'].iloc[0]) / step_fine_dur * clock_scale
            zdc_n_rate = (rate_period_data['zdc_N_live_count'].iloc[-1] - rate_period_data['zdc_N_live_count'].iloc[0]) / step_fine_dur * clock_scale
            zdc_s_rate = (rate_period_data['zdc_S_live_count'].iloc[-1] - rate_period_data['zdc_S_live_count'].iloc[0]) / step_fine_dur * clock_scale
            mbd_rate = (rate_period_data['mbd_live_count'].iloc[-1] - rate_period_data['mbd_live_count'].iloc[0]) / step_fine_dur * clock_scale
            mbd_n_rate = (rate_period_data['mbd_N_live_count'].iloc[-1] - rate_period_data['mbd_N_live_count'].iloc[0]) / step_fine_dur * clock_scale
            mbd_s_rate = (rate_period_data['mbd_S_live_count'].iloc[-1] - rate_period_data['mbd_S_live_count'].iloc[0]) / step_fine_dur * clock_scale

            zdc_cor_rate = solve_sasha_equation(zdc_n_rate, zdc_s_rate, zdc_rate, n_bunch * f_beam, plot=False)
            mbd_cor_rate = solve_sasha_equation(mbd_n_rate, mbd_s_rate, mbd_rate, n_bunch * f_beam, plot=False)
            mbd_rates.append(mbd_cor_rate)
            zdc_rates.append(zdc_cor_rate)
            mbd_uncor_rates.append(mbd_rate)
            zdc_uncor_rates.append(zdc_rate)

        for lumi_run in lumi_runs:
            if lumi_run['emittance']:
                emittance_time = emittance_df[emittance_df['Time'] == time].iloc[0]
            else:
                emittance_time = emittance_df[emittance_df['Time'] == profile_times[0]].iloc[0]

            em_blue_horiz_nom, em_blue_vert_nom = em_blue_nom
            em_yel_horiz_nom, em_yel_vert_nom = em_yel_nom

            em_blue_horiz, em_blue_vert = emittance_time['blue_horiz_emittance'], emittance_time['blue_vert_emittance']
            em_yel_horiz, em_yel_vert = emittance_time['yellow_horiz_emittance'], emittance_time['yellow_vert_emittance']

            # blue_widths = np.array([
            #     beam_width_x * (em_blue_horiz / em_blue_horiz_nom),
            #     beam_width_y * (em_blue_vert / em_blue_vert_nom)
            # ])
            # yellow_widths = np.array([
            #     beam_width_x * (em_yel_horiz / em_yel_horiz_nom),
            #     beam_width_y * (em_yel_vert / em_yel_vert_nom)
            # ])
            blue_widths = np.array([
                beam_width_x * np.sqrt(em_blue_horiz / em_blue_horiz_nom),
                beam_width_y * np.sqrt(em_blue_vert / em_blue_vert_nom)
            ])
            yellow_widths = np.array([
                beam_width_x * np.sqrt(em_yel_horiz / em_yel_horiz_nom),
                beam_width_y * np.sqrt(em_yel_vert / em_yel_vert_nom)
            ])

            collider_sim.set_bunch_sigmas(blue_widths, yellow_widths)

            if lumi_run['angle']:
                xing_angle_mask = ((bpm_data['Time'] >= time_low) & (bpm_data['Time'] <= time_high))
            else:
                xing_angle_mask = xing_angle_mask0

            blue_angle_x = -np.mean(blue_xing_h[xing_angle_mask]) * 1e-3
            blue_angle_y = -np.mean(blue_xing_v[xing_angle_mask]) * 1e-3
            yellow_angle_x = -np.mean(yellow_xing_h[xing_angle_mask]) * 1e-3
            yellow_angle_y = -np.mean(yellow_xing_v[xing_angle_mask]) * 1e-3

            collider_sim.set_bunch_crossing(blue_angle_x, blue_angle_y, yellow_angle_x, yellow_angle_y)

            if lumi_run['profile']:
                collider_sim.set_longitudinal_profiles_from_file(
                    profile_path.replace('COLOR_', 'blue_'),
                    profile_path.replace('COLOR_', 'yellow_')
                )
            else:
                collider_sim.set_longitudinal_profiles_from_file(
                    profile_paths[0].replace('COLOR_', 'blue_'),
                    profile_paths[0].replace('COLOR_', 'yellow_'),
                )

            collider_sim.run_sim_parallel()
            naked_lumi = collider_sim.get_naked_luminosity()

            if lumi_run['n_protons']:
                n_blue_key = f'blue_{lumi_run["n_protons"]}_ions'
                n_yellow_key = f'yellow_{lumi_run["n_protons"]}_ions'
                blue_time_mask = (
                    (ions_data['blue']['Time'] >= time_low) & (ions_data['blue']['Time'] <= time_high)
                )
                yellow_time_mask = (
                    (ions_data['yellow']['Time'] >= time - ions_avg_window / 2) &
                    (ions_data['yellow']['Time'] <= time + ions_avg_window / 2)
                )
                n_blue = np.mean(ions_data['blue'][n_blue_key][blue_time_mask])
                n_yellow = np.mean(ions_data['yellow'][n_yellow_key][yellow_time_mask])
                if lumi_run['n_protons'] == 'wcm':  # Scale to DCCT
                    n_blue *= n_blue_dcct0 / n_blue_wcm0
                    n_yellow *= n_yellow_dcct0 / n_yellow_wcm0
            else:
                n_blue = n_blue_dcct0
                n_yellow = n_yellow_dcct0
            lumi = naked_lumi * mb_to_um2 * f_beam * n_blue * n_yellow
            lumi_run['lumis'].append(lumi)

            if lumi_baseline is None and lumi_run['name'] == 'Baseline' and rate_data_exists:
                lumi_baseline, baseline_i = lumi, profile_i
                print(f'Baseline luminosity: {lumi_baseline:.2e} mb^-1 s^-1')

    mbd_rates, zdc_rates = np.array(mbd_rates), np.array(zdc_rates)
    mbd_uncor_rates, zdc_uncor_rates = np.array(mbd_uncor_rates), np.array(zdc_uncor_rates)

    cad_step_0 = cad_df[cad_df['step'] == 0].iloc[0]
    step_0_mask = (np.array(profile_times) >= cad_step_0['start']) & (np.array(profile_times) <= cad_step_0['end'])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(profile_times, mbd_uncor_rates / np.mean(mbd_uncor_rates[step_0_mask]) * lumi_baseline, label='MBD Uncorrected',
            linestyle='--', alpha=0.5, color='blue')
    ax.plot(profile_times, zdc_uncor_rates / np.mean(zdc_uncor_rates[step_0_mask]) * lumi_baseline, label='ZDC Uncorrected',
            linestyle='--', alpha=0.5, color='red')
    ax.plot(profile_times, mbd_rates / np.mean(mbd_rates[step_0_mask]) * lumi_baseline, label='MBD Rate (No z vertex cut)',
            color='blue')
    ax.plot(profile_times, zdc_rates /np.mean( zdc_rates[step_0_mask]) * lumi_baseline, label='ZDC Rate', color='red')
    for lumi_run in lumi_runs:
        ax.plot(profile_times, lumi_run['lumis'], label=lumi_run['name'], color=lumi_run['color'],
                linestyle=lumi_run['ls'], linewidth=lumi_run['lw'])
    ax.set_ylabel(r'Luminosity [$mb^{-1} s^{-1}$] (all quantities normalized to baseline)')
    ax.set_title(f'Luminosity Decay Over Time - {run_name}')
    ax.legend(loc='lower left')
    # Format x-axis to show only hour:minute
    time_format = mdates.DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(time_format)
    ax.grid(zorder=-1)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

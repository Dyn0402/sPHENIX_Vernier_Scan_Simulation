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
import pandas as pd
from datetime import timedelta

from common_logistics import set_base_path
from analyze_ions import read_ions_file
from analyze_emittance import read_emittance_file, parametrize_emittances_vs_time
from bpm_analysis import read_bpm_file
from analyze_sphnx_root_file import get_root_data_time
from z_vertex_fitting_common import get_profile_path
from BunchCollider import BunchCollider


def main():
    base_path = set_base_path()
    scan_path = f'{base_path}Vernier_Scans/auau_oct_16_24/'
    # scan_path = f'{base_path}Vernier_Scans/pp_aug_12_24/'
    plot_lumi_decay(scan_path)
    print('donzo')


def plot_lumi_decay(base_path):
    """
    Plot the luminosity decay over time from the ions data.
    """
    longitudinal_profiles_dir_path = f'{base_path}profiles/'
    combined_cad_step_data_csv_path = f'{base_path}combined_cad_step_data.csv'
    ions_path = f'{base_path}COLOR_ions.dat'
    emittance_file_path = f'{base_path}emittance.dat'
    bpm_file_path = f'{base_path}bpms.dat'
    root_file_name = 'calofit_54733.root'

    ions_data = {'blue': None, 'yellow': None}
    for color in ions_data.keys():
        ions_color_path = ions_path.replace('COLOR_', f'{color}_')
        ion_data = read_ions_file(ions_color_path)
        ions_data[color] = ion_data

    emittance_df = read_emittance_file(emittance_file_path)
    get_root_data_time(base_path, root_file_name, branches=['BCO', 'mbd_raw_count', 'zdc_raw_count'])

    f_beam = 78.4  # kHz
    mb_to_um2 = 1e-19
    ions_avg_window = timedelta(seconds=10)

    cad_df = pd.read_csv(combined_cad_step_data_csv_path)

    profile_paths, profile_times = get_profile_path(
        longitudinal_profiles_dir_path, cad_df['start'].min(), cad_df['end'].max(), True, return_times=True
    )

    bpm_data, blue_xing_h, yellow_xing_h, blue_xing_v, yellow_xing_v, rel_xing_h, rel_xing_v = (
        read_bpm_file(bpm_file_path, cad_df['start'].min(), cad_df['end'].max()))

    emittance_df = parametrize_emittances_vs_time(emittance_df, pd.array(profile_times))
    # Replace column names
    emittance_df = emittance_df.rename(columns={'BlueHoriz_fit': 'blue_horiz_emittance', 'BlueVert_fit': 'blue_vert_emittance',
                                                    'YellowHoriz_fit': 'yellow_horiz_emittance', 'YellowVert_fit': 'yellow_vert_emittance'})

    collider_sim = BunchCollider()
    collider_sim.set_grid_size(31, 31, 101, 31)
    beam_width_x, beam_width_y = 130.0, 130.0
    beta_star = 76.7
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

    lumi_runs = [
        {'angle': False, 'emittance': False, 'profile': False, 'n_protons': False, 'lumis': [], 'name': 'Baseline'},
        {'angle': True, 'emittance': False, 'profile': False, 'n_protons': False, 'lumis': [], 'name': 'Crossing Angles Variation'},
        {'angle': False, 'emittance': True, 'profile': False, 'n_protons': False, 'lumis': [], 'name': 'Emittance Growth'},
        {'angle': False, 'emittance': False, 'profile': True, 'n_protons': False, 'lumis': [], 'name': 'Longitudinal Profiles'},
        {'angle': False, 'emittance': False, 'profile': False, 'n_protons': 'dcct', 'lumis': [], 'name': 'DCCT Proton Burn Off'},
        {'angle': False, 'emittance': False, 'profile': False, 'n_protons': 'wcm', 'lumis': [], 'name': 'WCM Proton Burn Off (Scaled to DCCT)'},
        {'angle': True, 'emittance': True, 'profile': True, 'n_protons': 'dcct', 'lumis': [], 'name': 'All Effects (DCCT)'},
        {'angle': True, 'emittance': True, 'profile': True, 'n_protons': 'wcm', 'lumis': [], 'name': 'All Effects (WCM -- Scaled to DCCT)'},
    ]

    for profile_path, time in zip(profile_paths, profile_times):
        print(f'Time: {time}')
        for lumi_run in lumi_runs:
            if lumi_run['emittance']:
                emittance_time = emittance_df[emittance_df['Time'] == time].iloc[0]
            else:
                emittance_time = emittance_df[emittance_df['Time'] == profile_times[0]].iloc[0]

            em_blue_horiz_nom, em_blue_vert_nom = em_blue_nom
            em_yel_horiz_nom, em_yel_vert_nom = em_yel_nom

            em_blue_horiz, em_blue_vert = emittance_time['blue_horiz_emittance'], emittance_time['blue_vert_emittance']
            em_yel_horiz, em_yel_vert = emittance_time['yellow_horiz_emittance'], emittance_time['yellow_vert_emittance']

            blue_widths = np.array([
                beam_width_x * (em_blue_horiz / em_blue_horiz_nom),
                beam_width_y * (em_blue_vert / em_blue_vert_nom)
            ])
            yellow_widths = np.array([
                beam_width_x * (em_yel_horiz / em_yel_horiz_nom),
                beam_width_y * (em_yel_vert / em_yel_vert_nom)
            ])

            collider_sim.set_bunch_sigmas(blue_widths, yellow_widths)

            if lumi_run['angle']:
                xing_angle_mask = ((bpm_data['Time'] >= time - ions_avg_window / 2) &
                                   (bpm_data['Time'] <= time + ions_avg_window / 2))
            else:
                xing_angle_mask = ((bpm_data['Time'] >= profile_times[0] - ions_avg_window / 2) &
                                   (bpm_data['Time'] <= profile_times[0] + ions_avg_window / 2))

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
                    (ions_data['blue']['Time'] >= time - ions_avg_window / 2) &
                    (ions_data['blue']['Time'] <= time + ions_avg_window / 2)
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
            lumi = naked_lumi * mb_to_um2 * f_beam * 1e3 * n_blue * n_yellow
            lumi_run['lumis'].append(lumi)


    fig, ax = plt.subplots(figsize=(10, 6))
    for lumi_run in lumi_runs:
        ax.plot(profile_times, lumi_run['lumis'], label=lumi_run['name'])
    ax.set_ylabel(r'Luminosity [$mb^{-1} s^{-1}$]')
    ax.set_title('Luminosity Decay Over Time')
    ax.legend()
    ax.grid()
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

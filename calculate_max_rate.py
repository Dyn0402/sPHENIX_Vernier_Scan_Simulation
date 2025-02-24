#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on February 20 17:59 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/calculate_max_rate

@author: Dylan Neff, dn277127
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from BunchCollider import BunchCollider
from Measure import Measure
from vernier_z_vertex_fitting import read_cad_measurement_file
from luminosity_calculation_rcf import get_bw_beta_star_fit_params, get_bw_from_beta_star


def main():
    rates_path = 'cw_errors_rates.txt'
    vertical_rates, horizontal_rates = read_cw_err_rates_from_file(rates_path)
    vertical_head_on_rates = get_head_on_rates(*vertical_rates)
    horizontal_head_on_rates = get_head_on_rates(*horizontal_rates)

    rates_path_corr = 'cw_rates.txt'
    vertical_rates_corr, horizontal_rates_corr = read_cw_err_rates_from_file(rates_path_corr)
    vertical_head_on_rates_corr = get_head_on_rates(*vertical_rates_corr)
    horizontal_head_on_rates_corr = get_head_on_rates(*horizontal_rates_corr)

    # save_path = None
    save_path = 'C:/Users/Dylan/OneDrive - UCLA IT Services/Research/Saclay/sPHENIX/Vernier_Scan/Analysis_Note/rate_measurement/'

    vertical_head_on_rates = [Measure(rc.val, r.err * rc.val / r.val)
                              for r, rc in zip(vertical_head_on_rates, vertical_head_on_rates_corr)]
    horizontal_head_on_rates = [Measure(rc.val, r.err * rc.val / r.val)
                                for r, rc in zip(horizontal_head_on_rates, horizontal_head_on_rates_corr)]
    head_on_rates = {'Vertical': vertical_head_on_rates, 'Horizontal': horizontal_head_on_rates}

    scan_date = 'Aug12'
    cad_measurement_path = f'CAD_Measurements/VernierScan_{scan_date}_combined.dat'
    longitudinal_fit_path = f'CAD_Measurements/VernierScan_{scan_date}_COLOR_longitudinal_fit.dat'
    bw_beta_star_linear_fit_path = f'run_rcf_jobs/output/{scan_date}/bw_opt_vs_beta_star_fits.txt'

    cad_data = read_cad_measurement_file(cad_measurement_path)
    bw_beta_star_fit_params = get_bw_beta_star_fit_params(bw_beta_star_linear_fit_path)

    beta_star = 90
    measured_beta_star = np.array([97., 82., 88., 95.])
    beta_star_scale_factor = beta_star / 90  # Set 90 cm (average of measured) to default values, then scale from there
    beta_star_scaled = measured_beta_star * beta_star_scale_factor

    beam_width_x = get_bw_from_beta_star(beta_star, bw_beta_star_fit_params['x'])
    beam_width_y = get_bw_from_beta_star(beta_star, bw_beta_star_fit_params['y'])

    collider_sim = BunchCollider()
    collider_sim.set_bunch_sigmas(np.array([beam_width_x, beam_width_y]), np.array([beam_width_x, beam_width_y]))
    collider_sim.set_bunch_rs(np.array([0., 0., -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_beta_stars(*beta_star_scaled)
    blue_fit_path = longitudinal_fit_path.replace('_COLOR_', '_blue_')
    yellow_fit_path = longitudinal_fit_path.replace('_COLOR_', '_yellow_')
    collider_sim.set_longitudinal_fit_parameters_from_file(blue_fit_path, yellow_fit_path)

    scan_orientations = ['Horizontal', 'Vertical']
    head_on_steps = [1, 7]
    df = []
    blue_angle_x0, yellow_angle_x0, use_x0 = None, None, True
    for scan_orientation in scan_orientations:
        for step, ho_rate in zip(head_on_steps, head_on_rates[scan_orientation]):
            print(f'Running {scan_orientation} step {step}')
            step_cad_data = cad_data[(cad_data['orientation'] == scan_orientation) & (cad_data['step'] == step)].iloc[0]

            yellow_bunch_len_scaling = step_cad_data['yellow_bunch_length_scaling']
            blue_bunch_len_scaling = step_cad_data['blue_bunch_length_scaling']
            collider_sim.set_longitudinal_fit_scaling(blue_bunch_len_scaling, yellow_bunch_len_scaling)

            blue_angle_xi, yellow_angle_xi = -step_cad_data['bh8_avg'] / 1e3, -step_cad_data['yh8_avg'] / 1e3
            if use_x0:
                if blue_angle_x0 is None:
                    blue_angle_x0, yellow_angle_x0 = blue_angle_xi, yellow_angle_xi
                blue_angle_xi, yellow_angle_xi = blue_angle_x0, yellow_angle_x0
            collider_sim.set_bunch_crossing(blue_angle_xi, 0., yellow_angle_xi, 0.)

            collider_sim.run_sim_parallel()
            lumi = collider_sim.get_naked_luminosity()
            df_entry = {
                'time': step_cad_data['time'],
                'scan_orientation': scan_orientation,
                'step': step,
                'rate': ho_rate.val,
                'rate_err': ho_rate.err,
                'blue_angle_x': blue_angle_xi,
                'yellow_angle_x': yellow_angle_xi,
                'blue_len_scale': blue_bunch_len_scaling,
                'yellow_len_scale': yellow_bunch_len_scaling,
                'luminosity': lumi,
                'color': 'b' if scan_orientation == 'Horizontal' else 'green'
            }
            df.append(df_entry)
    df = pd.DataFrame(df)

    # Calculate lumi scale factors as 1 / max(luminosity)
    df['lumi_scale'] = df['luminosity'].max() / df['luminosity']
    df['scaled_rate'] = df['rate'] * df['lumi_scale']
    df['scaled_rate_err'] = df['rate_err'] * df['lumi_scale']

    # Plot original rates and scaled rates vs time, coloring vertical and horizontal separately
    fig_comp, ax_comp = plt.subplots(figsize=(10, 4))
    ax_lumi = ax_comp.twinx()
    for scan_orientation in scan_orientations:
        scan_df = df[df['scan_orientation'] == scan_orientation]
        times = np.array(pd.to_datetime(scan_df['time']))
        ax_comp.errorbar(times, np.array(scan_df['rate']) / 1e3, yerr=np.array(scan_df['scaled_rate_err']) / 1e3, fmt='o',
                    label=f'{scan_orientation} Rate', color=scan_df['color'].iloc[0])
        ax_comp.errorbar(times, np.array(scan_df['scaled_rate']) / 1e3, yerr=np.array(scan_df['scaled_rate_err']) / 1e3,
                    fmt='s', label=f'{scan_orientation} Scaled Rate', color=scan_df['color'].iloc[0])
    times = np.array(pd.to_datetime(df['time']))
    ax_lumi.plot(times, np.array(df['luminosity']), label=f'Naked Luminosity', color='red', alpha=0.5, zorder=0)
    ax_comp.set_ylabel('Rate [kHz]')
    # Naked luminosity is inverse micron squared
    ax_lumi.set_ylabel('Naked Luminosity [1/µm²]')
    ax_comp.xaxis.set_major_formatter(mdates.DateFormatter('%b %d %H:%M'))
    ax_comp.legend()
    ax_lumi.legend()
    fig_comp.tight_layout()

    # Calculate the error weighted mean of the scaled rate
    mean_rate = np.average(df['scaled_rate'], weights=1 / df['scaled_rate_err'] ** 2)
    mean_rate_err = np.sqrt(1 / np.sum(1 / df['scaled_rate_err'] ** 2))
    rate_std = np.std(df['scaled_rate'])
    mean_rate_meas = Measure(mean_rate, mean_rate_err) / 1e3  # kHz
    mean_rate_std_meas = Measure(mean_rate, rate_std) / 1e3  # kHz

    # Plot just the scaled rates vs time
    fig_cor, ax_cor = plt.subplots(figsize=(10, 4))
    times = np.array(pd.to_datetime(df['time']))
    ax_cor.errorbar(times, np.array(df['scaled_rate']) / 1e3, yerr=np.array(df['scaled_rate_err']) / 1e3, fmt='o',
                label=f'Scaled Rate', color='black')
    ax_cor.axhline(mean_rate / 1e3, color='red', label=f'Mean Rate')
    ax_cor.fill_between(times, (mean_rate - mean_rate_err) / 1e3, (mean_rate + mean_rate_err) / 1e3, color='red', alpha=0.5,
                    label='1σ Error')
    ax_cor.fill_between(times, (mean_rate - rate_std) / 1e3, (mean_rate + rate_std) / 1e3, color='red', alpha=0.2,
                    label=f'1σ Spread')
    mean_str = (f'Mean rate with standard error: {mean_rate_meas} kHz\n'
                f'Mean rate with standard deviation as error: {mean_rate_std_meas} kHz')
    ax_cor.annotate(mean_str, (0.5, 0.08), xycoords='axes fraction', va='bottom', ha='center', fontsize=14,
                bbox=dict(facecolor='wheat', alpha=0.3, boxstyle='round,pad=0.5'))
    ax_cor.set_ylabel('Rate [kHz]')
    ax_cor.xaxis.set_major_formatter(mdates.DateFormatter('%b %d %H:%M'))
    ax_cor.legend()
    fig_cor.tight_layout()

    # Plot the blue and yellow crossing angles vs time
    fig_xing, ax_xing = plt.subplots(figsize=(10, 4))
    times = np.array(pd.to_datetime(df['time']))
    ax_xing.plot(times, np.array(df['blue_angle_x']), marker='o', ls='-', label='Horizontal (X) Blue Angle', color='blue')
    ax_xing.plot(times, np.array(df['yellow_angle_x']), marker='o', ls='-', label='Horizontal (X) Yellow Angle', color='orange')
    ax_xing.set_ylabel('Crossing Angle [mrad]')
    ax_xing.xaxis.set_major_formatter(mdates.DateFormatter('%b %d %H:%M'))
    ax_xing.legend()
    fig_xing.tight_layout()

    # Write mean and std uncertainty to file
    print(f'Mean rate: {mean_rate_std_meas} kHz')
    with open('max_rate.txt', 'w') as file:
        file.write(f'{mean_rate} +- {rate_std} Hz')

    if save_path:
        fig_comp.savefig(f'{save_path}rate_comparison_{scan_date}.png')
        fig_comp.savefig(f'{save_path}rate_comparison_{scan_date}.pdf')
        fig_cor.savefig(f'{save_path}rate_average_{scan_date}.png')
        fig_cor.savefig(f'{save_path}rate_average_{scan_date}.pdf')
        fig_xing.savefig(f'{save_path}crossing_angles_{scan_date}.png')
        fig_xing.savefig(f'{save_path}crossing_angles_{scan_date}.pdf')

    plt.show()

    print('donzo')


def get_head_on_rates(pos, rates, errors):
    """
    Get rates when pos is 0.
    """
    head_on_rates = []
    for i in range(len(pos)):
        if pos[i] == 0:
            if len(errors) == len(pos):
                head_on_rates.append(Measure(rates[i], errors[i]))
            else:
                head_on_rates.append(Measure(rates[i], 0))

    return head_on_rates


def read_cw_err_rates_from_file(file_path):
    vertical_pos = []
    vertical_values = []
    vertical_errors = []
    horizontal_pos = []
    horizontal_values = []
    horizontal_errors = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

        # Determine if we are in the vertical or horizontal section
        current_section = None

        for line in lines:
            line = line.strip()
            if "Vertical pos" in line:
                current_section = "vertical"
                continue
            elif "Horizontal pos" in line:
                current_section = "horizontal"
                continue

            # Extracting values if in the correct section
            if current_section == "vertical":
                if line:
                    parts = line.split()
                    vertical_pos.append(float(parts[0]))
                    vertical_values.append(float(parts[1]))
                    if len(parts) == 3:
                        vertical_errors.append(float(parts[2]))
            elif current_section == "horizontal":
                if line:
                    parts = line.split()
                    horizontal_pos.append(float(parts[0]))
                    horizontal_values.append(float(parts[1]))
                    if len(parts) == 3:
                        horizontal_errors.append(float(parts[2]))

    return (vertical_pos, vertical_values, vertical_errors), (horizontal_pos, horizontal_values, horizontal_errors)


if __name__ == '__main__':
    main()

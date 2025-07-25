#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on July 20 11:47 AM 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/profiles_check.py

@author: Dylan Neff, dylan
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from datetime import datetime, date

from z_vertex_fitting_common import get_profile_path
from longitudinal_profiles import read_longitudinal_profile_data, get_average_longitudinal_profile
from common_logistics import set_base_path


def main():
    base_path = set_base_path()
    # full_check(base_path)
    gain_switch_check(base_path)
    print('donzo')


def full_check(base_path):
    scan_date = 'auau_oct_16_24'  # 'pp_aug_12_24' or 'auau_july_17_25'
    base_path += f'Vernier_Scans/{scan_date}/'
    profiles_path = f'{base_path}profiles'

    start_time = datetime(2024, 10, 16, 22, 14)
    end_time = datetime(2024, 10, 16, 22, 20)
    # start_time = datetime(2025, 7, 17, 13, 33, 25)
    # end_time = datetime(2025, 7, 17, 13, 33, 45)
    colors = ['blue', 'yellow']

    profile_paths = get_profile_path(profiles_path, start_time, end_time, True)

    splits = {'blue': 7.7e-7, 'yellow': 7.5e-7}

    fig_maxes, ax_maxes = plt.subplots(figsize=(10, 5))
    ax_raw_maxes = ax_maxes.twinx()

    fig_envelopes, ax_envelopes = plt.subplots(figsize=(10, 5))

    for color in colors:
        fig, ax = plt.subplots(figsize=(10, 5))
        fig_raw, ax_raw = plt.subplots(figsize=(10, 5))
        fig_norm, ax_norm = plt.subplots(figsize=(10, 5))
        plot_color = 'blue' if color == 'blue' else 'orange'

        maxes, raw_maxes, raw_mins, raw_means, times = [], [], [], [], []
        for profile_path in profile_paths:
            profile_path_color = profile_path.replace('_COLOR_', f'_{color}_')
            profile_data = pd.read_csv(profile_path_color, sep='\t')
            col_i = plot_color if np.max(profile_data['Probability Density']) < splits[color] else 'red'
            ax.plot(profile_data['# z (um)'], profile_data['Probability Density'], color=col_i, alpha=0.3)

            if np.max(profile_data['Probability Density']) < splits[color]:
                shift = 0.2e-9
                data = profile_data['Probability Density'] - shift
                data[data < 0] = 0
                data = data / np.trapezoid(data, profile_data['# z (um)'])
                ax.plot(profile_data['# z (um)'], data, color='green', alpha=0.8)

            date_time = profile_path_color.split("/")[-1].replace('.dat', '').split('_profile_')[-1]
            date_time = datetime.strptime(date_time, '%y_%H_%M_%S')
            maxes.append(profile_data["Probability Density"].max())
            times.append(date_time)

            profile_raw_path_color = profile_path.replace('_COLOR_', f'_{color}_').replace('avg_', '')
            data, date, time, beam_color = read_longitudinal_profile_data(profile_raw_path_color)
            ax_raw.plot(data, color=col_i, alpha=0.8)

            max_avg = data[data > np.percentile(data, 30)].mean()
            min_avg = data[data < data.min() + 3].mean()

            print(f'{color} @ {time} Range: {data.min()} to {data.max()} -- {data.max() - data.min()}, ({min_avg}, {max_avg})')

            data_norm = (max_avg - data) / (max_avg - min_avg)
            ax_norm.plot(data_norm, color=col_i, alpha=0.8)

            raw_maxes.append(data.max())
            raw_mins.append(data.min())
            raw_means.append(data.mean())

        ax.set_xlabel('z (um)')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'{color.capitalize()} Profiles')
        ax.grid()
        fig.tight_layout()

        ax_raw.set_ylabel('Raw Reading')
        fig_raw.tight_layout()

        ax_norm.set_ylabel('Normalized Reading')
        fig_norm.tight_layout()

        ax_maxes.plot(times, maxes, color=plot_color, marker='o')
        ax_raw_maxes.plot(times, raw_maxes, color=plot_color, linestyle='--', marker='x')

        ax_envelopes.plot(times, raw_means, color=plot_color, marker='o')
        ax_envelopes.plot(times, raw_maxes, color=plot_color, marker='x')
        ax_envelopes.plot(times, raw_mins, color=plot_color, marker='x')
    ax_maxes.set_ylabel('Max Probability Density')
    ax_maxes.set_title(f'Max Probability Density Over Time for {color.capitalize()}')
    ax_maxes.grid()
    fig_maxes.tight_layout()

    ax_envelopes.set_ylabel('Max Probability Density')
    ax_envelopes.grid()
    fig_envelopes.tight_layout()

    plt.show()


def gain_switch_check(base_path):
    scan_date = 'auau_oct_16_24'  # 'pp_aug_12_24' or 'auau_july_17_25'
    base_path += f'Vernier_Scans/{scan_date}/'
    profiles_path = f'{base_path}profiles'

    start_time = datetime(2024, 10, 16, 22, 12)
    end_time = datetime(2024, 10, 16, 22, 22)
    # start_time = datetime(2024, 10, 16, 21, 14)
    # end_time = datetime(2024, 10, 16, 23, 50)
    # start_time = datetime(2025, 7, 17, 13, 33, 25)
    # end_time = datetime(2025, 7, 17, 13, 33, 45)
    # start_time = datetime(2025, 7, 17, 13, 20, 25)
    # end_time = datetime(2025, 7, 17, 13, 40, 45)
    colors = ['blue', 'yellow']
    gain_switch_range_threshold = 20

    profile_paths = get_profile_path(profiles_path, start_time, end_time, True)

    fig_envelopes, ax_envelopes = plt.subplots(figsize=(10, 5))
    fig_ranges, ax_ranges = plt.subplots(figsize=(10, 5))

    for color in colors:
        plot_color = 'blue' if color == 'blue' else 'orange'

        maxes, raw_maxes, raw_mins, raw_means, raw_ranges, times = [], [], [], [], [], []
        for profile_path in profile_paths:
            profile_path_color = profile_path.replace('_COLOR_', f'_{color}_')
            profile_data = pd.read_csv(profile_path_color, sep='\t')

            date_time = profile_path_color.split("/")[-1].replace('.dat', '').split('_profile_')[-1]
            date_time = datetime.strptime(date_time, '%y_%H_%M_%S')
            maxes.append(profile_data["Probability Density"].max())
            times.append(date_time)

            profile_raw_path_color = profile_path.replace('_COLOR_', f'_{color}_').replace('avg_', '')
            data, date, time, beam_color = read_longitudinal_profile_data(profile_raw_path_color)

            max_avg = data[data > np.percentile(data, 30)].mean()
            min_avg = data[data < data.min() + 3].mean()

            print(f'{color} @ {time} Range: {data.min()} to {data.max()} -- {data.max() - data.min()}, ({min_avg}, {max_avg})')

            raw_maxes.append(data.max())
            raw_mins.append(data.min())
            raw_means.append(data.mean())
            raw_ranges.append(data.max() - data.min())

        ax_envelopes.plot(times, raw_means, color=plot_color, ls=':')
        ax_envelopes.plot(times, raw_maxes, color=plot_color, marker='.', ls='-')
        ax_envelopes.plot(times, raw_mins, color=plot_color, marker='.', ls='-')

        ax_ranges.plot(times, raw_ranges, color=plot_color, marker='.')

        raw_range_changes = np.diff(np.array(raw_ranges))
        gain_switches = np.where(np.abs(raw_range_changes) > gain_switch_range_threshold)[0] + 1  # +1 to adjust for diff length
        gain_switch_times = [times[i] for i in gain_switches]
        for switch_time in gain_switch_times:
            ax_envelopes.axvline(x=switch_time, color=plot_color, linestyle='--')
            ax_ranges.axvline(x=switch_time, color=plot_color, linestyle='--')

        surrounding_points = 20
        gain_switch_profile_paths, gain_switch_times, after_switches = [], [], []
        for index in gain_switches:
            for offset in range(-surrounding_points, surrounding_points + 1):
                if 0 <= index + offset < len(profile_paths):
                    profile_path = profile_paths[index + offset]
                    profile_path_color = profile_path.replace('_COLOR_', f'_{color}_').replace('avg_', '')
                    gain_switch_profile_paths.append(profile_path_color)
                    gain_switch_times.append(times[index + offset])
                    after_switches.append(True if offset >= 0 else False)

        # baseline_shifts = [0, +0.5e-4, +1e-4, 1.25e-4, 1.5e-4, 1.75e-4, +2e-4]
        # baseline_shifts = [0, +0.5e-4, +1e-4]
        baseline_shifts = [0, 1]
        # baseline_shifts = [6e6, 2e7]
        fig_switch_maxes, ax_switch_maxes = plt.subplots(figsize=(10, 5))
        for baseline_shift in baseline_shifts:
            # fig_switch_profiles, ax_switch_profiles = plt.subplots(figsize=(10, 5))
            switch_maxes, switch_times = [], []
            for time, profile_path, after in zip(gain_switch_times, gain_switch_profile_paths, after_switches):
                bshift_i = baseline_shift if after else 0
                # zs_interp, vals_interp = get_average_longitudinal_profile(profile_path, baseline_shift=bshift_i)
                # zs_interp, vals_interp = get_average_longitudinal_profile(profile_path, left_right_zero=baseline_shift)
                # zs_interp, vals_interp = get_average_longitudinal_profile(profile_path, fixed_z_zero=baseline_shift)
                zs_interp, vals_interp = get_average_longitudinal_profile(profile_path, recalc_baseline=baseline_shift)
                # ax_switch_profiles.plot(zs_interp, vals_interp, label=f'{time.strftime("%H:%M:%S")}')
                switch_times.append(time)
                switch_maxes.append(np.max(vals_interp))

            # ax_switch_profiles.set_xlabel('z (um)')
            # ax_switch_profiles.set_ylabel('Probability Density')
            # ax_switch_profiles.set_title(f'{color.capitalize()} Switch Profiles -- Baseline Shift: {baseline_shift:.1e}')
            # ax_switch_profiles.legend()
            # fig_switch_profiles.tight_layout()

            ax_switch_maxes.plot(switch_times, switch_maxes, marker='.', label=f'Baseline Shift: {baseline_shift:.1e}')
        ax_switch_maxes.set_xlabel('z (um)')
        ax_switch_maxes.set_ylabel('Max Probability Density')
        ax_switch_maxes.set_title(f'{color.capitalize()} Switch Maxes Over Time')
        ax_switch_maxes.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax_switch_maxes.legend()
        fig_switch_maxes.tight_layout()

    ax_envelopes.set_ylabel('Mins, Means, and Maxes of Raw Reading')
    ax_envelopes.set_title(f'Envelopes of WCM Raw Readings Over Scan : {scan_date}')
    ax_envelopes.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax_envelopes.grid()
    fig_envelopes.tight_layout()

    ax_ranges.set_ylabel('Range of Raw Reading')
    ax_ranges.set_title(f'WCM Raw Reading Envelope Ranges Over Scan : {scan_date}')
    ax_ranges.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    fig_ranges.tight_layout()

    plt.show()


if __name__ == '__main__':
    main()

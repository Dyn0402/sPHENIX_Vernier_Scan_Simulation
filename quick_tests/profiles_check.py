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
import pandas as pd
from datetime import datetime, date

from z_vertex_fitting_common import get_profile_path
from longitudinal_profiles import read_longitudinal_profile_data
from common_logistics import set_base_path


def main():
    base_path = set_base_path()
    base_path += 'Vernier_Scans/auau_oct_16_24/'
    # base_path += 'Vernier_Scans/auau_july_17_25/'
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

    print('donzo')


if __name__ == '__main__':
    main()

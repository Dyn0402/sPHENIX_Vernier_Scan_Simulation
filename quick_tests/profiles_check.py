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
    profiles_path = f'{base_path}profiles'

    start_time = datetime(2025, 10, 16, 22, 14)
    end_time = datetime(2025, 10, 16, 22, 20)
    colors = ['blue', 'yellow']

    profile_paths = get_profile_path(profiles_path, start_time, end_time, True)
    print(profile_paths)

    splits = {'blue': 7.25e-7, 'yellow': 8.0e-7}

    fig_maxes, ax_maxes = plt.subplots(figsize=(10, 5))
    ax_raw_maxes = ax_maxes.twinx()
    for color in colors:
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_color = 'blue' if color == 'blue' else 'orange'

        maxes, raw_maxes, times = [], [], []
        for profile_path in profile_paths:
            profile_path_color = profile_path.replace('_COLOR_', f'_{color}_')
            profile_data = pd.read_csv(profile_path_color, sep='\t')
            col_i = plot_color if np.max(profile_data['Probability Density']) < splits[color] else 'red'
            ax.plot(profile_data['# z (um)'], profile_data['Probability Density'], color=col_i, alpha=0.3)
            date_time = profile_path_color.split("/")[-1].replace('.dat', '').split('_profile_')[-1]
            date_time = datetime.strptime(date_time, '%y_%H_%M_%S')
            maxes.append(profile_data["Probability Density"].max())
            times.append(date_time)

            profile_raw_path_color = profile_path.replace('_COLOR_', f'_{color}_').replace('avg_', '')
            data, date, time, beam_color = read_longitudinal_profile_data(profile_raw_path_color)
            raw_maxes.append(data.max())

        ax.set_xlabel('z (um)')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'{color.capitalize()} Profiles')
        ax.grid()
        plt.tight_layout()

        ax_maxes.plot(times, maxes, color=plot_color, marker='o')
        ax_raw_maxes.plot(times, raw_maxes, color=plot_color, linestyle='--', marker='x')
    ax_maxes.set_ylabel('Max Probability Density')
    ax_maxes.set_title(f'Max Probability Density Over Time for {color.capitalize()}')
    ax_maxes.grid()
    fig_maxes.tight_layout()

    plt.show()

    print('donzo')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on February 27 12:13 PM 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/plot_scan_nominal_crossing_angles.py

@author: Dylan Neff, Dylan
"""

import os
import platform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from datetime import datetime
import pytz


def main():
    if platform.system() == 'Windows':
        base_path = 'C:/Users/Dylan/Desktop/pp_crossing_angles/'
    else:  # Linus
        base_path = '/local/home/dn277127/Bureau/pp_crossing_angles/'  # Figure out later
    bpm_dir = f'{base_path}bpm_measurements/'
    run_info_path = 'run_info.csv'
    # month = None  # either month or None to include all months
    month = 'August'
    run = 51195

    run_info_df = get_run_info(run_info_path)
    run_row = run_info_df[run_info_df['Runnumber'] == run].iloc[0]
    print(run_row)

    crossing_angles_df = get_bpm_crossing_angles(bpm_dir, month)
    crossing_angles_run = crossing_angles_df[(crossing_angles_df['time'] >= run_row['Start']) &
                                                (crossing_angles_df['time'] <= run_row['End'])]

    plot_crossing_angles_vs_time(crossing_angles_run)
    plt.show()

    print('donzo')


def get_run_info(run_info_path):
    run_info_df = pd.read_csv(run_info_path)
    run_info_df['Start'] = pd.to_datetime(run_info_df['Start'])
    run_info_df['End'] = pd.to_datetime(run_info_df['End'])

    return run_info_df


def get_bpm_crossing_angles(bpm_dir, month=None):
    """
    Load all crossing angle data from bpm_dir and sort by time.
    :return:
    """
    # Load all crossing angle data and sort by time
    data = []
    for file_name in os.listdir(bpm_dir):
        if month and month.lower() not in file_name.lower():
            continue
        data.append(read_crossing_angle(f'{bpm_dir}{file_name}'))
    data = pd.concat(data)
    data = data.sort_values('time')

    return data


def read_crossing_angle(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Skip the header lines
    data_lines = lines[3:]

    # Lists to store the parsed data
    time_data = []
    bh8_crossing_angle = []
    yh8_crossing_angle = []
    gh8_crossing_angle = []

    # Parse each line
    line_i = 0
    for line in data_lines:
        line_i += 1
        columns = line.strip().split('\t')
        if len(columns) != 4:
            continue

        try:
            time_data.append(datetime.strptime(columns[0].strip(), "%m/%d/%Y %H:%M:%S"))
        except ValueError:
            print(f'Error parsing time at line {line_i}:')
            print(line)
            continue
        bh8_crossing_angle.append(float(columns[1]))
        yh8_crossing_angle.append(float(columns[2]))
        gh8_crossing_angle.append(float(columns[3]))

    df = pd.DataFrame({
        'time': time_data,
        'bh8_crossing_angle': bh8_crossing_angle,
        'yh8_crossing_angle': yh8_crossing_angle,
        'gh8_crossing_angle': gh8_crossing_angle
    })

    # Set the time column to be in New York time
    bnl_tz = pytz.timezone('America/New_York')
    df['time'] = df['time'].dt.tz_localize(bnl_tz)

    # Return the data as a pandas DataFrame
    return df


def plot_crossing_angles_vs_time(crossing_angles_df):
    """
    NOT USEFUL. Too much fluctuation as beams turn off/on
    Plot blue, yellow and relative crossing angles vs time.
    :param crossing_angles_df:
    :return:
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    print(crossing_angles_df.columns)
    print(crossing_angles_df['time'])
    print(crossing_angles_df['bh8_crossing_angle'])
    time, blue, yellow, rel = [np.array(crossing_angles_df[series]) for series in ['time', 'bh8_crossing_angle', 'yh8_crossing_angle', 'gh8_crossing_angle']]
    ax.plot(time, blue, label='Blue', color='b')
    ax.plot(time, yellow, label='Yellow', color='orange')
    ax.plot(time, rel, label='Relative', color='g')
    # ax.plot(crossing_angles_df['time'], crossing_angles_df['bh8_crossing_angle'], label='Blue', color='blue')
    # ax.plot(crossing_angles_df['time'], crossing_angles_df['yh8_crossing_angle'], label='Yellow', color='orange')
    # ax.plot(crossing_angles_df['time'], crossing_angles_df['gh8_crossing_angle'], label='Relative', color='green')
    ax.axhline(0, ls='-', alpha=0.3, color='black')
    ax.set_ylabel('Crossing Angle (mrad)')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M', tz=pytz.timezone('America/New_York')))
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    fig.tight_layout()


if __name__ == '__main__':
    main()

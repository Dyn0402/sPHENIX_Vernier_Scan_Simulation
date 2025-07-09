#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 29 23:44 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/analyze_ions

@author: Dylan Neff, dn277127
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from common_logistics import set_base_path


def main():
    base_path = set_base_path()
    # scan_path = f'{base_path}Vernier_Scans/auau_oct_16_24/'
    scan_path = f'{base_path}Vernier_Scans/pp_aug_12_24/'
    plot_ions(scan_path)
    print('donzo')


def plot_ions(scan_path):
    """
    Analyze the ions data from the scan path.
    """
    ions_path = f'{scan_path}COLOR_ions.dat'
    colors = ['blue', 'yellow']

    plt.figure(figsize=(10, 5))
    for color in colors:
        ions_color_path = ions_path.replace('COLOR_', f'{color}_')
        ion_data = read_ions_file(ions_color_path)
        time = ion_data['Time'].values
        dcct_ions = ion_data[f'{color}_dcct_ions'].values
        wcm_ions = ion_data[f'{color}_wcm_ions'].values


        plt_color = 'blue' if color == 'blue' else 'orange'
        plt.plot(time, dcct_ions, color=plt_color, label=f'{color.capitalize()} DCCT Ions')
        plt.plot(time, wcm_ions, color=plt_color, label=f'{color.capitalize()} WCM Ions', linestyle='--')
        plt.xlabel('Time (s)')
        plt.ylabel('Number of Ions')
        plt.title(f'{color.capitalize()} Ions Over Time')
        plt.legend()
        plt.grid()
        plt.tight_layout()

    plt.show()


def analyze_ions(scan_path, steps_df):
    """
    Analyze the ions data from the ions file and merge with steps dataframe.
    :param scan_path: Path to the directory with the ions files.
    :param steps_df: DataFrame containing the steps information.
    :return: Merged DataFrame with ions data and steps information.
    """
    ions_path = f'{scan_path}COLOR_ions.dat'
    colors = ['blue', 'yellow']

    plt.figure(figsize=(10, 5))
    for color in colors:
        ion_color_path = ions_path.replace('COLOR_', f'{color}_')
        ion_data = read_ions_file(ion_color_path)

        for index, row in steps_df.iterrows():  # Iterate through each step in the steps DataFrame
            start, end = row['start'], row['end']
            step_mask = (ion_data['Time'] >= start) & (ion_data['Time'] <= end)
            step_data = ion_data[step_mask]
            # Get the average number of ions for this step
            avg_dcct_ions = step_data[f'{color}_dcct_ions'].mean()
            avg_wcm_ions = step_data[f'{color}_wcm_ions'].mean()
            # Add the average ions to the steps DataFrame
            steps_df.at[index, f'{color}_dcct_ions'] = avg_dcct_ions
            steps_df.at[index, f'{color}_wcm_ions'] = avg_wcm_ions


def read_ions_file(ions_file_path):
    """
    Read the ions file and return the data.
    :param ions_file_path: Path to the ions file.
    :return: Numpy array with time and ion counts.
    """
    df = pd.read_csv(ions_file_path, skiprows=2, sep='\t')
    df = df.rename(columns={'# Time ': 'Time', 'beamIons ': 'beamIons'})  # Rename columns to remove trailing spaces
    df.drop(columns=['Unnamed: 3'], inplace=True, errors='ignore')  # Drop any unnamed columns if they exist
    df['Time'] = pd.to_datetime(df['Time'], format='%m/%d/%Y %H:%M:%S ', errors='coerce')  # Convert time to datetime

    color = ions_file_path.split('/')[-1].split('_')[0]  # Extract color from file name
    df = df.rename(columns={f'{color[:3]}_TotalIons ': f'{color[:3]}_TotalIons'})

    df[f'{color}_wcm_ions'] = df[f'{color[:3]}_TotalIons'].astype(float) * 10**9  # Convert to number of ions
    df[f'{color}_dcct_ions'] = df['beamIons'].astype(float) * 10**6  # Convert to number of ions

    return df


if __name__ == '__main__':
    main()

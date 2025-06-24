#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 27 03:59 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/analyze_emittance

@author: Dylan Neff, dn277127
"""

import platform
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def main():
    if platform.system() == 'Windows':
        base_path = 'C:/Users/Dylan/Desktop/'
    else:
        base_path = '/local/home/dn277127/Bureau/'
    scan_path = f'{base_path}Vernier_Scans/auau_oct_16_24/'
    emittance_angelika_file_path = f'{scan_path}Emittance_IPM_Fill35240.dat'
    emittance_file_path = f'{scan_path}emittance.dat'
    # df = read_angelika_emittance_file(emittance_angelika_file_path)
    df = read_emittance_file(emittance_file_path)
    compare_with_angelika_emittance(emittance_angelika_file_path, emittance_file_path)

    param_df = parametrize_emittances_vs_time(df)
    df = df.merge(param_df, on='Time', how='left')

    plot_emittances_vs_time(df)

    print('donzo')


def add_emittance_info_to_df(emittance_file_path, times=None):
    """
    Add emittance information to the dataframe.
    :param emittance_file_path: Path to the emittance file.
    :param times: Optional list of times to use for parametrization. If None, uses the times from the emittance file.
    :return: DataFrame with emittance information added.
    """
    emittance_df = read_angelika_emittance_file(emittance_file_path)
    emittance_df = parametrize_emittances_vs_time(emittance_df, times)
    emittance_df = emittance_df.rename(columns={'Time': 'mid_time',
                                                'BlueHoriz_fit': 'blue_horiz_emittance',
                                                'BlueVert_fit': 'blue_vert_emittance',
                                                'YellowHoriz_fit': 'yellow_horiz_emittance',
                                                'YellowVert_fit': 'yellow_vert_emittance'})
    return emittance_df


def parametrize_emittances_vs_time(df, times=None):
    """
    Fit each emittance to a polynomial and return the coefficients.
    """
    if times is None:
        times = df['Time']
    new_df = pd.DataFrame({'Time': pd.Series(times)})
    cols = ['BlueHoriz', 'BlueVert', 'YellowHoriz', 'YellowVert']
    for col in cols:
        # coeffs = np.polyfit(df['Time'].astype(np.int64) // 10**9, df[col], 2)  # Convert time to seconds
        # coeffs = np.polyfit(df['Time'].astype(np.int64), df[col], 2)  # Convert time to seconds
        coeffs = np.polyfit(df['Time'].astype(np.int64), df[col], 2)  # Convert time to seconds
        print(f'Coefficients for {col}: {coeffs}')
        # df[f'{col}_fit'] = np.polyval(coeffs, df['Time'].astype(np.int64) // 10**9)
        # df[f'{col}_fit'] = np.polyval(coeffs, df['Time'].astype(np.int64))
        new_df[f'{col}_fit'] = np.polyval(coeffs, times.astype(np.int64))  # Convert time to seconds
        # new_df.append(np.polyval(coeffs, times.astype(np.int64)))
    # new_df = np.array(new_df).T
    # new_df = pd.DataFrame(new_df, columns=['Time'] + [f'{col}_fit' for col in cols])
    # new_df = pd.to_datetime(new_df['Time'], errors='coerce')

    return new_df


def plot_emittances_vs_time(df, ax=None, make_labels=True):
    """
    Plot the emittances over time.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['Time'], df['BlueHoriz'], marker='o', ls='none', label='Blue Horizontal Emittance', color='blue')
    ax.plot(df['Time'], df['BlueVert'], marker='o', ls='none',  label='Blue Vertical Emittance', color='blue')
    ax.plot(df['Time'], df['YellowHoriz'], marker='o', ls='none', label='Yellow Horizontal Emittance', color='orange')
    ax.plot(df['Time'], df['YellowVert'], marker='o', ls='none', label='Yellow Vertical Emittance', color='orange')

    if 'BlueHoriz_fit' in df.columns:
        ax.plot(df['Time'], df['BlueHoriz_fit'], ls='--', color='blue', label='Blue Horiz Fit')
    if 'BlueVert_fit' in df.columns:
        ax.plot(df['Time'], df['BlueVert_fit'], ls='--', color='blue', label='Blue Vert Fit')
    if 'YellowHoriz_fit' in df.columns:
        ax.plot(df['Time'], df['YellowHoriz_fit'], ls='--', color='orange', label='Yellow Horiz Fit')
    if 'YellowVert_fit' in df.columns:
        ax.plot(df['Time'], df['YellowVert_fit'], ls='--', color='orange', label='Yellow Vert Fit')

    if ax is None or make_labels:
        ax.set_ylabel('Emittance')
        ax.set_title('Emittances Over Time')
        ax.legend()
        plt.tight_layout()
        plt.show()


def compare_with_angelika_emittance(angelika_path, standard_path):
    """
    Compare the emittance values I scraped from CAD myself with the ones from Angelika
    """
    df_angelika = read_angelika_emittance_file(angelika_path)
    df_standard = read_emittance_file(standard_path)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_angelika['Time'], df_angelika['BlueHoriz'], marker='o', ls='none', label='Angelika Blue Horiz', color='blue', alpha=0.5)
    ax.plot(df_angelika['Time'], df_angelika['BlueVert'], marker='o', ls='none', label='Angelika Blue Vert', color='blue', alpha=0.5)
    ax.plot(df_angelika['Time'], df_angelika['YellowHoriz'], marker='o', ls='none', label='Angelika Yellow Horiz', color='orange', alpha=0.5)
    ax.plot(df_angelika['Time'], df_angelika['YellowVert'], marker='o', ls='none', label='Angelika Yellow Vert', color='orange', alpha=0.5)

    ax.plot(df_standard['Time'], df_standard['BlueHoriz'], marker='x', ls='none', label='Standard Blue Horiz', color='blue')
    ax.plot(df_standard['Time'], df_standard['BlueVert'], marker='x', ls='none', label='Standard Blue Vert', color='blue')
    ax.plot(df_standard['Time'], df_standard['YellowHoriz'], marker='x', ls='none', label='Standard Yellow Horiz', color='orange')
    ax.plot(df_standard['Time'], df_standard['YellowVert'], marker='x', ls='none', label='Standard Yellow Vert', color='orange')

    ax.set_ylabel('Emittance')
    ax.set_title('Comparison of Emittances from Angelika and Standardized Script')
    ax.legend()

    plt.tight_layout()



def read_angelika_emittance_file(emittance_file_path):
    """
    Read the emittance file and return the data as a numpy array.
    :param emittance_file_path: Path to the emittance file.
    :return: Numpy array with the emittance data.
    """
    with open(emittance_file_path, 'r') as f:
        lines = f.readlines()

    headers = lines[0].strip().split('\t')
    data = []
    for line in lines[1:]:  # Skip the header
        line = line.strip().split('\t')
        if len(line) == len(headers):
            data.append([float(x) for x in line])
        elif len(line) > 1:
            data_i = [0 for _ in range(len(headers))]
            for i, val in enumerate(line):
                if val:
                    data_i[i] = float(val)
            data.append(data_i)

    data = np.array(data)
    df = pd.DataFrame(data, columns=headers)
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce', unit='s', utc=True)
    df['Time'] = df['Time'].dt.tz_convert('America/New_York').dt.tz_localize(None)

    # Columns to check for zeros
    cols_to_check = ['BlueHoriz', 'BlueVert', 'YellowHoriz', 'YellowVert']

    # Keep rows where NOT all of those columns are zero
    df = df.loc[~(df[cols_to_check] == 0).all(axis=1)].copy()

    # def first_nonzero(series):
    #     # Return the first non-zero value, or 0 if none found
    #     nonzero_vals = series[series != 0]
    #     if not nonzero_vals.empty:
    #         return nonzero_vals.iloc[0]
    #     return 0

    cols_to_check = ['BlueHoriz', 'BlueVert', 'YellowHoriz', 'YellowVert']

    # Drop rows where any of the specified columns have zero
    df = df[(df[cols_to_check] != 0).all(axis=1)]

    return df


def read_emittance_file(emittance_file_path):
    """
    Reads a RHIC emittance data file and returns a pandas DataFrame with columns:
        ['Time', 'BlueHoriz', 'BlueVert', 'YellowHoriz', 'YellowVert']
    :param emittance_file_path: Path to the emittance file.
    :return: Pandas DataFrame with the emittance data.
    """

    records = []

    with open(emittance_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue  # Skip headers and empty lines

            parts = line.split()
            if len(parts) < 6:
                continue  # Skip malformed lines

            # Join date and time, convert to datetime later
            date_str = parts[0] + ' ' + parts[1]
            blue_horiz = float(parts[2])
            blue_vert = float(parts[3])
            yellow_horiz = float(parts[4])
            yellow_vert = float(parts[5])

            records.append([date_str, blue_horiz, blue_vert, yellow_horiz, yellow_vert])

    # Create dataframe
    df = pd.DataFrame(records, columns=['Time', 'BlueHoriz', 'BlueVert', 'YellowHoriz', 'YellowVert'])
    df['Time'] = pd.to_datetime(df['Time'], format='%m/%d/%Y %H:%M:%S')

    return df


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 17 7:26 PM 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/scan_comparison.py

@author: Dylan Neff, Dylan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def main():
    # eyeball()
    from_file()
    print('donzo')


def from_file():
    file_path = 'C:/Users/Dylan/Documents/Vernier_Scan_Raw_Scalars.csv'
    df = pd.read_csv(file_path)
    df['ZDC NS'] = df['ZDC NS'].apply(convert_to_hz)
    df['MBD NS'] = df['MBD NS'].apply(convert_to_hz)
    df['Time'] = pd.to_datetime(df['Time'])

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df['Time'], df['ZDC NS'], '-', label='ZDC NS')
    ax.plot(df['Time'], df['MBD NS'], '-', label='MBD NS')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Rate (Hz)')
    ax.set_title('ZDC NS and MBD NS Rates Over Time')
    ax.legend()
    fig.tight_layout()

    norm_time = pd.to_datetime('2025-07-17 19:17:00')
    # Get the index of the norm_time
    norm_index = df[df['Time'] == norm_time].index[0] if not df[df['Time'] == norm_time].empty else None
    # norm_point = 180
    df['MBD NS Norm'] = df['MBD NS'] / df['MBD NS'].iloc[norm_index] * df['ZDC NS'].iloc[norm_index]

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df['Time'], df['ZDC NS'], '-', label='ZDC NS')
    ax.plot(df['Time'], df['MBD NS Norm'], '-', label='MBD NS Normalized')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Rate (Hz)')
    ax.set_title('ZDC NS and MBD NS Normalized Rates Over Time')
    ax.legend()
    fig.tight_layout()

    plt.show()


# Function to convert values
def convert_to_hz(val):
    if 'kHz' in val:
        return float(val.replace('kHz', '').strip()) * 1000
    elif 'Hz' in val:
        return float(val.replace('Hz', '').strip())
    else:
        return float(val)  # just in case it's already a number


def eyeball():
    auau24_offsets = [0, 100, 250, 450, 700, 1000]
    auau24_rates = [46889.29194, 41768.75915, 23124.79674, 5796.13979, 878.222814, 167]
    auau25_offsets = [0, 150, 300, 400, 500, 600, 700, 800]
    auau25_rates = [66000, 48400, 20400, 9360, 4150, 1800, 900, 500]

    fig, ax = plt.subplots()
    ax.plot(auau24_offsets, auau24_rates, 'o-', label='AuAu 24')
    ax.plot(auau25_offsets, auau25_rates, 'o-', label='AuAu 25')
    ax.set_xlabel('Offset (um)')
    ax.set_ylabel('Rate (Hz)')
    ax.legend()
    fig.tight_layout()

    auau24_norm = auau24_rates / np.max(auau24_rates) * np.max(auau25_rates)

    fig, ax = plt.subplots()
    ax.plot(auau24_offsets, auau24_norm, 'o-', label='AuAu 24 (normalized)')
    ax.plot(auau25_offsets, auau25_rates, 'o-', label='AuAu 25')
    ax.set_xlabel('Offset (um)')
    ax.set_ylabel('Rate (Hz)')
    ax.legend()
    fig.tight_layout()

    plt.show()



if __name__ == '__main__':
    main()

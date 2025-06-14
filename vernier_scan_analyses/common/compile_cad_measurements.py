#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 24 22:19 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/compile_cad_measurements

@author: Dylan Neff, dn277127
"""

import platform
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from bpm_analysis import bpm_analysis, get_start_end_times
from analyze_ions import analyze_ions
from analyze_emittance import add_emittance_info_to_df
from analyze_sphnx_root_file import get_step_rates


def main():
    if platform.system() == 'Windows':
        base_path = 'C:/Users/Dylan/Desktop/'
    else:
        base_path = '/local/home/dn277127/Bureau/'
    scan_path = f'{base_path}Vernier_Scans/auau_oct_16_24/'

    if scan_path.split('/')[-2] == 'auau_oct_16_24':
        pre_run_buffer = 100  # seconds to remove from the start of the scan
    else:
        pre_run_buffer = None

    start_time, end_time = get_start_end_times(scan_path)

    bpm_file_path = f'{scan_path}bpms.dat'
    df = bpm_analysis(bpm_file_path, start_time, end_time, plot=False, pre_scan_buffer_seconds=pre_run_buffer)

    # Add wcm and dcct information to the dataframe
    analyze_ions(scan_path, df)

    # Add offsets to the dataframe
    set_offsets_path = f'{scan_path}set_offsets.txt'
    set_offsets_df = get_set_offsets(set_offsets_path)
    df = df.merge(set_offsets_df, on='step', how='left')

    # Add emittance information to the dataframe
    emittance_file_path = f'{scan_path}Emittance_IPM_Fill35240.dat'
    emittance_df = add_emittance_info_to_df(emittance_file_path, times=df['mid_time'])
    df = df.merge(emittance_df, on='mid_time', how='left')

    # Add rates to the dataframe
    rates_df = get_step_rates(scan_path, df)
    df = df.merge(rates_df, on='step', how='left')

    # Write the dataframe to a CSV file
    df.to_csv(f'{scan_path}combined_cad_step_data.csv', index=False)

    print('donzo')


def get_set_offsets(set_offsets_path):
    """
    Get the set offsets from the set offsets file.
    :param set_offsets_path: Path to the set offsets file.
    :return: Set offsets as a dataframe.
    """
    with open(set_offsets_path, 'r') as f:
        lines = f.readlines()
    offsets = []
    for line in lines[1:]:
        line = line.strip().split('\t')
        if len(line) != 3:
            continue
        offsets.append({'step': int(line[0]), 'orientation': line[1]})
        if line[1] == 'Horizontal':
            offsets[-1].update({'set offset h': float(line[2]), 'set offset v': 0})
        elif line[1] == 'Vertical':
            offsets[-1].update({'set offset h': 0, 'set offset v': float(line[2])})
        else:
            print(f'Invalid orientation: {line[1]}')
            continue

    return pd.DataFrame(offsets)


if __name__ == '__main__':
    main()

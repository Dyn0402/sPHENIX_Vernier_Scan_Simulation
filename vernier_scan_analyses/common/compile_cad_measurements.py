#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 24 22:19 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/compile_cad_measurements

@author: Dylan Neff, dn277127
"""

import pandas as pd

from bpm_analysis import bpm_analysis, get_start_end_times
from analyze_ions import analyze_ions
from analyze_emittance import add_emittance_info_to_df
from analyze_sphnx_root_file import get_step_rates, get_gl1p_bunch_by_bunch_step_rates, get_bco_offset
from rate_corrections import make_rate_corrections, make_gl1p_rate_corrections
from common_logistics import set_base_path


def main():
    base_path = set_base_path()
    # scan_path = f'{base_path}Vernier_Scans/auau_oct_16_24/'
    # root_file_name = 'calofit_54733.root'
    # scan_path = f'{base_path}Vernier_Scans/auau_july_17_25/'
    # root_file_name = '69561.root'
    scan_path = f'{base_path}Vernier_Scans/pp_aug_12_24/'
    # root_file_name = 'calofitting_51195.root'
    root_file_name = 'original_51195.root'

    if '/pp_' in scan_path:
        emittance_poly_order = 0
    else:
        emittance_poly_order = 2

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
    emittance_file_path = f'{scan_path}emittance.dat'
    emittance_df = add_emittance_info_to_df(emittance_file_path, times=df['mid_time'], poly_order=emittance_poly_order)
    df = df.merge(emittance_df, on='mid_time', how='left')

    # Get BCO offset by comparing bpm steps to zdc steps
    bco_offset = get_bco_offset(scan_path, df, root_file_name)
    print(f'BCO offset: {bco_offset} s')

    # Add rates to the dataframe
    rates_df = get_step_rates(scan_path, df, root_file_name, bco_offset=bco_offset)
    df = df.merge(rates_df, on='step', how='left')

    # Apply rate corrections to the dataframe
    df = make_rate_corrections(df)

    # Write the dataframe to a CSV file
    df.to_csv(f'{scan_path}combined_cad_step_data.csv', index=False)

    if scan_path.split('/')[-2] != 'pp_aug_12_24':
        # Get GL1P bunch-by-bunch step rates and put in a separate dataframe
        gl1p_rates_df = get_gl1p_bunch_by_bunch_step_rates(scan_path, df, root_file_name, bco_offset=bco_offset)
        gl1p_rates_df = make_gl1p_rate_corrections(gl1p_rates_df)
        gl1p_rates_df.to_csv(f'{scan_path}gl1p_bunch_by_bunch_step_rates.csv', index=False)

    # For auau24 next run calculate_corrected_raw_rates to deal with mbd background

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

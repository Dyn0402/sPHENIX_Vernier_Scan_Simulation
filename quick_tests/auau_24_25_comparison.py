#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on July 31 11:44 AM 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/auau_24_25_comparison.py

@author: Dylan Neff, dylan
"""

import numpy as np
import matplotlib.pyplot as plt

from analyze_emittance import read_emittance_file, parametrize_emittances_vs_time
from analyze_sphnx_root_file import get_bco_offset, get_root_data_time
from common_logistics import set_base_path


def main():
    base_path = set_base_path()

    scan_path_24 = f'{base_path}Vernier_Scans/auau_oct_16_24/'
    scan_path_25 = f'{base_path}Vernier_Scans/auau_july_17_25/'

    root_file_name_24 = 'calofit_54733.root'
    root_file_name_25 = '69561.root'

    df_24 = read_emittance_file(f'{scan_path_24}emittance.dat')
    df_25 = read_emittance_file(f'{scan_path_25}emittance.dat')

    param_df_24 = parametrize_emittances_vs_time(df_24, poly_order=2)
    param_df_25 = parametrize_emittances_vs_time(df_25, poly_order=2)

    print(param_df_24)

    bco_offset_24 = get_bco_offset(scan_path_24, param_df_24, root_file_name_24)
    bco_offset_25 = get_bco_offset(scan_path_25, param_df_25, root_file_name_25)

    print(f'BCO offset for 2024 scan: {bco_offset_24}')
    print(f'BCO offset for 2025 scan: {bco_offset_25}')

    root_branches = ['BCO', 'GL1_clock_count', 'GL1_live_count', 'mbd_live_count', 'mbd_S_live_count',
                     'mbd_N_live_count', 'zdc_live_count', 'zdc_S_live_count', 'zdc_N_live_count']

    rate_data_24 = get_root_data_time(scan_path_24, root_file_name_24)
    rate_data_25 = get_root_data_time(scan_path_25, root_file_name_25)

    print('donzo')


if __name__ == '__main__':
    main()

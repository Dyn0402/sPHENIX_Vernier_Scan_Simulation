#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on February 12 11:13 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/combine_data_csvs

@author: Dylan Neff, dn277127
"""

import os
import pandas as pd


def main():
    scan_dates = ['Aug12']  # ['July11', 'Aug12']
    orientations = ['Horizontal', 'Vertical']
    path = 'output'

    for scan_date in scan_dates:
        for orientation in orientations:
            set_dir = f'{path}/{scan_date}/{orientation}/'
            set_df = []
            for run_dir in os.listdir(set_dir):
                if not os.path.isdir(f'{set_dir}{run_dir}'):
                    print(f'{set_dir}{run_dir} is not a directory.')
                    continue
                run_params = run_dir.replace('run_', '').split('_')
                # Pair each consecutive item in run_params to make a dictionary of parameter names and values
                run_params = {run_params[i]: float(run_params[i+1]) for i in range(0, len(run_params), 2)}
                run_csv_path = f'{set_dir}{run_dir}/data/scan_data.csv'
                if not os.path.exists(run_csv_path):
                    print(f'No csv found for {run_dir}')
                    continue
                run_df = pd.read_csv(run_csv_path)
                residual_mean = run_df['residuals'].mean()
                set_df.append({**run_params, 'residual_mean': residual_mean})
            set_df = pd.DataFrame(set_df)
            set_df.to_csv(f'{set_dir}combined_scan_residuals.csv', index=False)
    print('donzo')


if __name__ == '__main__':
    main()

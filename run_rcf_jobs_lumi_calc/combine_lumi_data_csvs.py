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
    out_path = 'output'
    combined_csv_name = 'combined_lumis.csv'

    lumi_dfs, job_nums = [], []
    for file_name in os.listdir(out_path):
        if not file_name.endswith('.csv') or file_name == combined_csv_name:
            print(f'Skipping {file_name}')
            continue
        file_path = f'{out_path}/{file_name}'
        set_lumis = pd.read_csv(file_path)
        lumi_dfs.append(set_lumis)
        job_num = file_name.split('_')[1].split('.')[0]
        job_nums.append(job_num)
    lumi_dfs = pd.concat(lumi_dfs)
    lumi_dfs.to_csv(f'{out_path}/{combined_csv_name}', index=False)

    # Print max job num and missing job nums
    max_job_num = max(job_nums)
    job_nums = set(map(int, job_nums))
    missing_jobs = set(range(int(max_job_num))) - job_nums
    print(f'Max job number: {max_job_num}')
    print(f'Missing jobs: {missing_jobs}')

    print('donzo')


if __name__ == '__main__':
    main()

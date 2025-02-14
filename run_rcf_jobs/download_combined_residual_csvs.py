#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on February 14 12:57 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/download_combined_residual_csvs

@author: Dylan Neff, dn277127
"""

import os

from vernier_z_vertex_fitting_clean import create_dir


def main():
    scan_dates = ['Aug12']  # ['July11', 'Aug12']
    orientations = ['Horizontal', 'Vertical']
    out_dir_name = 'output'
    file_name = 'combined_scan_residuals.csv'

    sphenix_sftp_alias = 'sph-sftp'
    sphenix_run_dir = '/sphenix/u/dneffsph/sPHENIX_Vernier_Scan_Simulation/run_rcf_jobs/'

    for scan_date in scan_dates:
        scan_dir = create_dir(f'{out_dir_name}/{scan_date}/')
        for orientation in orientations:
            orientation_dir = create_dir(f'{scan_dir}{orientation}/')
            sphenix_path = f'{sphenix_run_dir}{out_dir_name}/{scan_date}/{orientation}/{file_name}'
            download_file(sphenix_sftp_alias, sphenix_path, orientation_dir)

    print('donzo')


def download_file(sphenix_sftp_alias, sphenix_path, local_path):
    os.system(f'sftp {sphenix_sftp_alias}:{sphenix_path} {local_path}')


if __name__ == '__main__':
    main()

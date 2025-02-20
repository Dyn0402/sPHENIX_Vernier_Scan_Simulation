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
    out_dir_name = 'output'
    file_name = 'combined_lumis.csv'

    sphenix_sftp_alias = 'sph-sftp'
    sphenix_run_dir = '/sphenix/u/dneffsph/sPHENIX_Vernier_Scan_Simulation/run_rcf_jobs_lumi_calc/'

    sphenix_path = f'{sphenix_run_dir}{out_dir_name}/{file_name}'
    local_path = f'{out_dir_name}/{file_name}'
    download_file(sphenix_sftp_alias, sphenix_path, local_path)

    print('donzo')


def download_file(sphenix_sftp_alias, sphenix_path, local_path):
    os.system(f'sftp {sphenix_sftp_alias}:{sphenix_path} {local_path}')


if __name__ == '__main__':
    main()

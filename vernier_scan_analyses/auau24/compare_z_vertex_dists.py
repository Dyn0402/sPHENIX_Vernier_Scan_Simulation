#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 08 17:55 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/compare_z_vertex_dists

@author: Dylan Neff, dn277127
"""

import platform
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from z_vertex_fitting_common import load_vertex_distributions


def main():
    if platform.system() == 'Windows':
        base_path = 'C:/Users/Dylan/Desktop/'
    else:
        base_path = '/local/home/dn277127/Bureau/'

    step = 23
    base_path_auau = f'{base_path}Vernier_Scans/auau_oct_16_24/'

    combined_cad_step_data_csv_path = f'{base_path_auau}combined_cad_step_data.csv'
    cad_df = pd.read_csv(combined_cad_step_data_csv_path)

    directories = ['vertex_data_old', 'vertex_data_less_old', 'vertex_data_calib', 'vertex_data_hold', 'vertex_data']
    # file_name = '54733_vertex_distributions_no_zdc_coinc.root'
    file_name = '54733_vertex_distributions.root'

    fig, ax = plt.subplots()
    for directory in directories:
        path = f'{base_path_auau}{directory}/{file_name}'
        vertex_data = load_vertex_distributions(path, [step], cad_df)
        centers, counts, count_errs = vertex_data[step]
        ax.plot(centers, counts, label=directory)
    ax.set_xlabel('Z Vertex Position (cm)')
    ax.set_yscale('log')
    ax.legend()
    fig.tight_layout()

    plt.show()

    print('donzo')


if __name__ == '__main__':
    main()

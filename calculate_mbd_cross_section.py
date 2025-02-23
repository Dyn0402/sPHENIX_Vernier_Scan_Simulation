#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on February 21 17:13 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/calculate_cross_section

@author: Dylan Neff, dn277127
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from Measure import Measure
from vernier_z_vertex_fitting import read_cad_measurement_file


def main():
    lumi_path = 'lumi_vs_beta_star.csv'
    max_rate_path = 'max_rate.txt'
    cad_measurement_path = 'CAD_Measurements/VernierScan_Aug12_combined.dat'
    lumi_data = pd.read_csv(lumi_path)
    max_rate = read_max_rate(max_rate_path)
    cad_data = read_cad_measurement_file(cad_measurement_path)

    n_bunch = 111

    max_rate_per_bunch = max_rate / n_bunch

    f_beam = 78.4  # kHz
    n_blue, n_yellow = get_nblue_nyellow(cad_data, orientation='Horizontal', step=1, n_bunch=n_bunch)  # n_protons
    print(f'N Blue: {n_blue:.2e}, N Yellow: {n_yellow:.2e}')
    # n_blue = 1.636e11  # n_protons
    # n_yellow = 1.1e11  # n_protons
    mb_to_um2 = 1e-19

    print(f'Max Rate Per Bunch: {max_rate_per_bunch} Hz')
    for index, row in lumi_data.iterrows():
        beta_star = row['beta_star']
        naked_lumi = Measure(row['luminosity'], row['luminosity_err'])
        lumi = naked_lumi * mb_to_um2 * f_beam * 1e3 * n_blue * n_yellow
        cross_section = max_rate_per_bunch / lumi
        print(f'Beta Star: {beta_star} cm, Luminosity: {lumi} mb⁻¹s⁻¹, Cross Section: {cross_section} mb')

    print('donzo')


def read_max_rate(path):
    with open(path, 'r') as file:
        max_rate = file.readline().split()
        max_rate = Measure(float(max_rate[0]), float(max_rate[2]))
    return max_rate


def get_nblue_nyellow(cad_data, orientation='Horizontal', step=1, n_bunch=111):
    cad_step = cad_data[(cad_data['orientation'] == orientation) & (cad_data['step'] == step)].iloc[0]
    print(cad_step)
    wcm_blue, wcm_yellow = cad_step['dcct_blue'], cad_step['dcct_yellow']
    n_blue, n_yellow = wcm_blue * 1e9 / n_bunch, wcm_yellow * 1e9 / n_bunch
    return n_blue, n_yellow


if __name__ == '__main__':
    main()

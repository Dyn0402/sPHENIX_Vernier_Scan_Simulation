#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on July 02 6:03 PM 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/rate_corrections.py

@author: Dylan Neff, dylan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from common_logistics import set_base_path


def main():
    base_path = set_base_path()
    scan_path = f'{base_path}Vernier_Scans/auau_oct_16_24/'
    combined_cad_step_data_csv_path = f'{scan_path}combined_cad_step_data.csv'
    cad_df = pd.read_csv(combined_cad_step_data_csv_path)

    f_beam = 78.4 * 1e3  # Hz
    n_bunch = 111

    zdc, zdc_n, zdc_s = cad_df["zdc_cor_rate"], cad_df['zdc_N_cor_rate'], cad_df['zdc_S_cor_rate']
    by = zdc_n * zdc_s / (f_beam * n_bunch)
    zdc_cor = n_bunch * f_beam * (-np.log(1 - (zdc - by) / (n_bunch * f_beam + zdc - zdc_n - zdc_s)))
    print(f'ZDC cor rates: {zdc.iloc[0]}, zdc_by: {by.iloc[0]}, zdc_cor: {zdc_cor.iloc[0]}')

    fig, ax = plt.subplots()
    ax.plot(cad_df['step'], zdc, marker='o', label='ZDC Raw Rate')
    ax.plot(cad_df['step'], zdc - by, marker='o', label='ZDC Accidental Corrected')
    ax.plot(cad_df['step'], zdc_cor, marker='o', label='ZDC Full Corrected Rate')
    ax.set_xlabel('Step')
    ax.set_ylabel('ZDC Rate (Hz)')
    ax.set_title('ZDC Rate Corrections')
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.tight_layout()

    # Save the corrected rates back to the DataFrame
    cad_df['zdc_acc_multi_cor_rate'] = zdc_cor
    cad_df.to_csv(combined_cad_step_data_csv_path, index=False)

    print(cad_df.columns)
    plt.show()

    print('donzo')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 19 10:02 AM 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/trigger_rate_vs_offset_comparison.py

@author: Dylan Neff, Dylan
"""

import platform
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as cf
import pandas as pd

from vernier_z_vertex_fitting import read_cad_measurement_file, get_cw_rates
from analyses.phenix_style_hourglass_analysis import vernier_scan_fit_bkg
from Measure import Measure


def main():
    if platform.system() == 'Linux':
        base_path = '/local/home/dn277127/Bureau/vernier_scan/'
    else:  # Windows
        base_path = 'C:/Users/Dylan/Desktop/vernier_scan/'

    # Gaussian approximation with bad beam width luminosity
    f_beam = 78.4  # kHz
    n_blue = 1.636e11  # n_protons
    n_yellow = 1.1e11  # n_protons
    mb_to_um2 = 1e-19

    vernier_scan_dates = ['Aug12']
    orientations = ['Horizontal', 'Vertical']
    # orientations = ['Horizontal']
    # beta_star, bws = 90, {'Horizontal': 161.8, 'Vertical': 157.1}
    beta_star, bws = 105, {'Horizontal': 165.3, 'Vertical': 163.6}
    # bw_file_post = '_nofit'
    bw_file_post = ''
    for scan_date in vernier_scan_dates:
        cad_measurement_path = f'{base_path}CAD_Measurements/VernierScan_{scan_date}_combined.dat'

        cad_data = read_cad_measurement_file(cad_measurement_path)
        cw_rates = get_cw_rates(cad_data)

        for orientation in orientations:
            print(f'\n{orientation}')
            cad_orientation = cad_data[cad_data['orientation'] == orientation]
            cad_orientation = cad_orientation.dropna(subset=['offset_set_val', 'cw_rate'])

            offsets = cad_orientation['offset_set_val'].to_numpy() * 1000
            rates = cad_orientation['cw_rate'].to_numpy()

            bw_fit_path = (f'{base_path}Analysis/bw_fitting/bstar{beta_star}{bw_file_post}/{scan_date}/'
                           f'{orientation.lower()}/{bws[orientation]}/scan_fit.csv')
            bw_fit_df = pd.read_csv(bw_fit_path)
            orientation_xy = 'x' if orientation == 'Horizontal' else 'y'
            bw_df_offsets = bw_fit_df[f'bunch1_offset_{orientation_xy}']
            bw_df_naked_lumis = bw_fit_df[f'naked_luminosity'].to_numpy()
            # bw_df_lumis = bw_df_naked_lumis * mb_to_um2 * (f_beam * 1e3) * n_blue * n_yellow
            bw_df_amps = bw_fit_df[f'amplitude'].to_numpy()
            bw_df_lumis = bw_df_naked_lumis * np.max(rates) / np.max(bw_df_naked_lumis)

            # Sort bw_df_offsets and bw_df_lumis together by bw_df_offsets
            bw_df_offsets, bw_df_lumis = zip(*sorted(zip(bw_df_offsets, bw_df_lumis)))

            # Sort offsets and rates together by offset
            offsets, rates = zip(*sorted(zip(offsets, rates)))

            popt_cw, pcov_cw = cf(vernier_scan_fit_bkg, offsets, rates, p0=[np.max(rates), 200, 0, 0])
            perr_cw = np.sqrt(np.diag(pcov_cw))
            pmeas_cw = [Measure(val, err) for val, err in zip(popt_cw, perr_cw)]
            print('\nCW Fit:')
            for meas in pmeas_cw:
                print(meas)

            popt_sim, pcov_sim = cf(vernier_scan_fit_bkg, bw_df_offsets, bw_df_lumis, p0=[np.max(bw_df_lumis), 200, 0, 0])
            perr_sim = np.sqrt(np.diag(pcov_sim))
            pmeas_sim = [Measure(val, err) for val, err in zip(popt_sim, perr_sim)]
            print('\nSim Fit:')
            for meas in pmeas_sim:
                print(meas)

            xs_plot = np.linspace(np.min(offsets), np.max(offsets), 1000)

            width_str = f'CW Width: {pmeas_cw[1]} μm\nSim Width: {pmeas_sim[1]} μm'

            # Plot
            fig, ax = plt.subplots()
            ax.plot(offsets, rates, marker='o', c='k', ls='none', label='CW Rates')
            ax.plot(xs_plot, vernier_scan_fit_bkg(xs_plot, *popt_cw), '--', c='k', label='CW Fit')
            ax.plot(bw_df_offsets, bw_df_lumis, marker='o', c='r', ls='none', label='Sim Rates')
            ax.plot(xs_plot, vernier_scan_fit_bkg(xs_plot, *popt_sim), '--', c='r', label='Sim Fit')
            ax.set_xlabel('Offset (um)')
            ax.set_ylabel('Rate (Hz)')
            ax.annotate(width_str, (0.02, 0.97), xycoords='axes fraction', ha='left', va='top', fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.5, boxstyle='round'))
            ax.legend()
            ax.set_ylim(bottom=0)
            ax.set_title(f'{scan_date} {orientation} Offset vs Rate -- Default CAD Parameters (no fit)')
            fig.tight_layout()
        plt.show()

    print('donzo')


if __name__ == '__main__':
    main()

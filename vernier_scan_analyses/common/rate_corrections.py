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
from scipy.optimize import root_scalar

from common_logistics import set_base_path


def main():
    base_path = set_base_path()
    scan_path = f'{base_path}Vernier_Scans/auau_oct_16_24/'
    combined_cad_step_data_csv_path = f'{scan_path}combined_cad_step_data.csv'
    cad_df = pd.read_csv(combined_cad_step_data_csv_path)

    f_beam = 78.4 * 1e3  # Hz
    n_bunch = 111
    detectors = ['mbd', 'zdc']

    for detector in detectors:
        r_ns, r_n, r_s = cad_df[f"{detector}_cor_rate"], cad_df[f'{detector}_N_cor_rate'], cad_df[f'{detector}_S_cor_rate']
        by = r_n * r_s / (f_beam * n_bunch)
        r_ns_cor = n_bunch * f_beam * (-np.log(1 - (r_ns - by) / (n_bunch * f_beam + r_ns - r_n - r_s)))
        print(f'{detector.upper()} cor rates: {r_ns.iloc[0]}, by: {by.iloc[0]}, r_ns_cor: {r_ns_cor.iloc[0]}')
        print(f'Step 0 rates: r_ns: {r_ns.iloc[0]}, r_n: {r_n.iloc[0]}, r_s: {r_s.iloc[0]}')
        ks = (r_s - r_ns) / r_ns
        kn = (r_n - r_ns) / r_ns
        print(f'Step 0: ks: {ks.iloc[0]}, kn: {kn.iloc[0]}')
        print(f'Bunch crossing rate: {f_beam * n_bunch} Hz')

        cad_df[f'{detector}_sasha_cor_rate'] = r_ns
        for i, step in enumerate(cad_df['step']):
            r_ns_sasha_cor = solve_sasha_equation(r_n.iloc[i], r_s.iloc[i], r_ns.iloc[i], n_bunch * f_beam, plot=False)
            cad_df.at[i, f'{detector}_sasha_cor_rate'] = r_ns_sasha_cor

        fig, ax = plt.subplots()
        ax.plot(cad_df['step'], r_ns, marker='o', label=f'{detector.upper()} Raw Rate')
        ax.plot(cad_df['step'], r_ns - by, marker='o', label=f'{detector.upper()} Accidental Corrected')
        ax.plot(cad_df['step'], r_ns_cor, marker='o', label=f'{detector.upper()} Full Corrected Rate')
        ax.plot(cad_df['step'], cad_df[f'{detector}_sasha_cor_rate'], marker='o', label=f'{detector.upper()} Sasha Corrected Rate')
        ax.set_xlabel('Step')
        ax.set_ylabel(f'{detector.upper()} Rate (Hz)')
        ax.set_title(f'{detector.upper()} Rate Corrections')
        ax.set_ylim(bottom=0)
        ax.legend()
        fig.tight_layout()

        # Save the corrected rates back to the DataFrame
        cad_df[f'{detector}_acc_multi_cor_rate'] = r_ns_cor

    cad_df.to_csv(combined_cad_step_data_csv_path, index=False)
    print(cad_df.columns)
    plt.show()

    print('donzo')


def solve_sasha_equation(n_eff, s_eff, ns_eff, n_crossings, plot=False):
    """
    Solve the Sasha equation for accidental corrections.

    Parameters:
        n_eff: effective number of times N fired
        s_eff: effective number of times S fired
        ns_eff: effective number of times both N and S fired together
        n_crossings: total number of bunch crossings
        plot: whether to plot the minimization

    Returns:
        ns_acc_cor: corrected accidental rate for NS
        ns_acc_mc_cor: Monte Carlo corrected accidental rate for NS
    """
    n_0s = n_eff - ns_eff
    s_0n = s_eff - ns_eff
    ks = s_0n / ns_eff
    kn = n_0s / ns_eff

    ns_trues = np.linspace(0, ns_eff * 2, 1000)
    sasha_res = sasha_eq(ns_trues / n_crossings, ns_eff / n_crossings, ks, kn)

    root_result = root_scalar(
        sasha_eq,
        args=(ns_eff / n_crossings, ks, kn),
        bracket=[0, ns_eff / n_crossings * 2],
        method='bisect'
    )

    if plot:
        fig, ax = plt.subplots()
        ax.plot(ns_trues, sasha_res, label='Sasha Equation')
        ax.axhline(0, color='red', linestyle='--', label='Zero Line')
        ax.axvline(ns_eff, color='blue', linestyle='--', label='Effective NS Rate')
        ax.axvline(root_result.root * n_crossings, color='orange', linestyle='--', label='Root of Sasha Equation')
        ax.set_xlabel('ns_true')
        ax.set_ylabel('Sasha Equation Result')
        ax.set_title('Sasha Equation Results')
        ax.legend()
        fig.tight_layout()

    return root_result.root * n_crossings


def sasha_eq(r_true, r_eff, ks, kn):
    return 1 - r_eff - np.exp(-r_true * (1 + ks)) - np.exp(-r_true * (1 + kn)) + np.exp(-r_true * (1 + ks + kn))


if __name__ == '__main__':
    main()

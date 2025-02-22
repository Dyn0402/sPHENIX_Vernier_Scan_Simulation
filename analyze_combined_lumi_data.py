#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on February 20 09:23 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/analyze_combined_lumi_data

@author: Dylan Neff, dn277127
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as cf
from scipy.special import erf
import pandas as pd

from Measure import Measure


def main():
    err_type = 'conservative'  # 'best'  ''
    combined_lumi_path = f'run_rcf_jobs_lumi_calc/output/{err_type}_err_combined_lumis.csv'
    combined_lumis = pd.read_csv(combined_lumi_path)

    def_lumi_path = 'lumi_vs_beta_star.csv'
    def_lumis = pd.read_csv(def_lumi_path)
    print(def_lumis)
    lumi_bs_90 = def_lumis[def_lumis['beta_star'] == 90.0]['luminosity'].iloc[0]
    lumi_bs_105 = def_lumis[def_lumis['beta_star'] == 105.0]['luminosity'].iloc[0]
    lumi_gaus = def_lumis[pd.isna(def_lumis['beta_star'])]['luminosity'].iloc[0]
    print(lumi_gaus)
    print(lumi_bs_90)
    print(lumi_bs_105)

    # Plot histogram of luminosity
    luminosities = combined_lumis['luminosity']
    fig_lumi_hist, ax_lumi_hist = plt.subplots(figsize=(8, 6))
    hist, bin_edges, _ = ax_lumi_hist.hist(luminosities, bins=100)

    # Calculate standard deviation and plot a Gaussian with same standard deviation
    std = np.std(luminosities)
    x = np.linspace(min(bin_edges), max(bin_edges), 1000)
    y = gaus(x, amp=max(hist), loc=lumi_bs_90, scale=std)
    ax_lumi_hist.plot(x, y, color='r', ls='--', label='Normal Approximation')

    lumi_bs_90_meas = Measure(lumi_bs_90, std)
    lumi_bs_105_meas = Measure(lumi_bs_105, std)

    title_str = r'$\mathcal{L}_{Naked}$ [$\mu$m$^{-2}$]:'
    lumi_bs_90_str = f'{lumi_bs_90_meas}' + r' for $\beta^*=90$ cm'
    lumi_bs_105_str = f'{lumi_bs_105_meas}' + r' for $\beta^*=105$ cm'
    lumi_gaus_str = f'{lumi_gaus:.2e}' + r' for Gaussian Approximation'
    std_err_str = f'{std:.1e}' + ' Uncertainty (from std)'
    full_str = f'{title_str}\n{lumi_bs_90_str}\n{lumi_bs_105_str}\n{lumi_gaus_str}\n{std_err_str}'
    ax_lumi_hist.annotate(full_str, (0.02, 0.3), xycoords='axes fraction', ha='left', va='bottom', fontsize=12,
                          bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.5'))

    ax_lumi_hist.axvline(lumi_bs_90, color='r', label='Beta Star 90 cm')
    ax_lumi_hist.axvline(lumi_bs_105, color='g', label='Beta Star 105 cm')
    ax_lumi_hist.axvline(lumi_gaus, color='b', label='Gaussian Approximation')
    ax_lumi_hist.set_xlabel('Naked Luminosity [1/µm²]')
    ax_lumi_hist.set_ylabel('Counts')
    ax_lumi_hist.legend()
    fig_lumi_hist.tight_layout()
    plt.show()

    # Plot correlation plot of luminosity vs beta star
    beta_stars = combined_lumis['beta_star']
    fig_betastar_corr, ax_betastar_corr = plt.subplots()
    ax_betastar_corr.scatter(beta_stars, luminosities, alpha=0.5)
    ax_betastar_corr.set_title('Naked Luminosity vs Beta Star')
    ax_betastar_corr.set_xlabel('Beta Star [cm]')
    ax_betastar_corr.set_ylabel('Naked Luminosity [1/µm²]')
    fig_betastar_corr.tight_layout()

    # Plot bwx and bwy vs beta star
    bwx = combined_lumis['bw_x']
    bwy = combined_lumis['bw_y']
    fig_bw_betastar, ax_bw_betastar = plt.subplots()
    ax_bw_betastar.scatter(beta_stars, bwx, alpha=0.5, label='Beam Width X')
    ax_bw_betastar.scatter(beta_stars, bwy, alpha=0.5, label='Beam Width Y')
    ax_bw_betastar.set_title('Beam Width vs Beta Star')
    ax_bw_betastar.set_xlabel('Beta Star [cm]')
    ax_bw_betastar.set_ylabel('Beam Width [cm]')
    ax_bw_betastar.legend()
    fig_bw_betastar.tight_layout()

    # Write lumis to file
    def_lumis['luminosity_err'] = std
    def_lumis.to_csv(f'lumi_vs_beta_star.csv', index=False)

    plt.show()

    print('donzo')


def skew_normal_pdf(x, amp=1, loc=0, scale=1, alpha=0):
    """
    Compute the skew-normal probability density function (PDF) using NumPy.

    Parameters:
    x : float or array-like
        Input values where the PDF is evaluated.
    amp : float
        Amplitude parameter (overall scaling factor).
    loc : float
        Location parameter (mean).
    scale : float
        Scale parameter (standard deviation).
    alpha : float
        Skewness parameter (controls asymmetry; alpha=0 gives a normal distribution).

    Returns:
    pdf : float or array-like
        Skew-normal PDF values.
    """
    z = (x - loc) / scale
    phi = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * z**2)  # Standard normal PDF
    Phi = 0.5 * (1 + erf(alpha * z / np.sqrt(2)))      # Standard normal CDF
    return amp * (2 / scale) * phi * Phi


def gaus(x, amp=1, loc=0, scale=1):
    return amp * np.exp(-(x - loc)**2 / (2 * scale**2))


if __name__ == '__main__':
    main()

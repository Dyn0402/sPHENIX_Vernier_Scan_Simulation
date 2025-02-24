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
from scipy.stats.distributions import norm
import pandas as pd

from Measure import Measure
from vernier_z_vertex_fitting import read_cad_measurement_file


def main():
    err_type = 'conservative'  # 'best'  ''
    combined_lumi_path = f'run_rcf_jobs_lumi_calc/output/{err_type}_err_combined_lumis.csv'
    combined_lumis = pd.read_csv(combined_lumi_path)

    # save_path = None
    save_path = 'C:/Users/Dylan/OneDrive - UCLA IT Services/Research/Saclay/sPHENIX/Vernier_Scan/Analysis_Note/Cross_Section/'

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
    hist, bin_edges, _ = ax_lumi_hist.hist(luminosities, bins=100, density=True, color='k')

    # Calculate standard deviation and plot a Gaussian with same standard deviation
    std = np.std(luminosities)
    asym_left, asym_right = np.percentile(luminosities, 16), np.percentile(luminosities, 84)
    x = np.linspace(min(bin_edges), max(bin_edges), 1000)
    y = norm.pdf(x, lumi_bs_90, std)
    ax_lumi_hist.plot(x, y, color='r', ls='--', label='Symmetric Approximation')

    lumi_bs_90_meas = Measure(lumi_bs_90, std)
    lumi_bs_105_meas = Measure(lumi_bs_105, std)

    title_str = r'$\mathcal{L}_{Naked}$ [$\mu$m$^{-2}$]:'
    lumi_bs_90_str = f'{lumi_bs_90_meas}' + r' for $\beta^*=90$ cm'
    lumi_bs_105_str = f'{lumi_bs_105_meas}' + r' for $\beta^*=105$ cm'
    lumi_gaus_str = f'{lumi_gaus:.2e}' + r' for Gaussian Approximation'
    std_err_str = f'{std:.1e}' + ' Uncertainty (from std)'
    asym_err_str = f'{asym_left:.2e} - {asym_right:.2e}' + ' Asymmetric 68% CI'
    full_str = f'{title_str}\n{lumi_bs_90_str}\n{lumi_bs_105_str}\n{lumi_gaus_str}\n{std_err_str}\n{asym_err_str}'
    ax_lumi_hist.annotate(full_str, (0.02, 0.4), xycoords='axes fraction', ha='left', va='bottom', fontsize=12,
                          bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.5'))

    ax_lumi_hist.axvline(lumi_bs_90, color='r', label=r'$\beta^* =$ 90 cm')
    ax_lumi_hist.axvline(lumi_bs_105, color='g', label=r'$\beta^* =$ 105 cm')
    ax_lumi_hist.axvline(lumi_gaus, color='b', label='Gaussian Approximation')
    ax_lumi_hist.fill_betweenx([0, max(hist)], asym_left, asym_right, color='yellow', alpha=0.2, label='68% CI')
    ax_lumi_hist.set_xlabel('Naked Luminosity [1/µm²]')
    ax_lumi_hist.set_ylabel('Probability')
    ax_lumi_hist.legend()
    fig_lumi_hist.tight_layout()
    # plt.show()

    # # Plot correlation plot of luminosity vs beta star
    # beta_stars = combined_lumis['beta_star']
    # fig_betastar_corr, ax_betastar_corr = plt.subplots()
    # ax_betastar_corr.scatter(beta_stars, luminosities, alpha=0.5)
    # ax_betastar_corr.set_title('Naked Luminosity vs Beta Star')
    # ax_betastar_corr.set_xlabel('Beta Star [cm]')
    # ax_betastar_corr.set_ylabel('Naked Luminosity [1/µm²]')
    # fig_betastar_corr.tight_layout()
    #
    # # Plot bwx and bwy vs beta star
    # bwx = combined_lumis['bw_x']
    # bwy = combined_lumis['bw_y']
    # fig_bw_betastar, ax_bw_betastar = plt.subplots()
    # ax_bw_betastar.scatter(beta_stars, bwx, alpha=0.5, label='Beam Width X')
    # ax_bw_betastar.scatter(beta_stars, bwy, alpha=0.5, label='Beam Width Y')
    # ax_bw_betastar.set_title('Beam Width vs Beta Star')
    # ax_bw_betastar.set_xlabel('Beta Star [cm]')
    # ax_bw_betastar.set_ylabel('Beam Width [cm]')
    # ax_bw_betastar.legend()
    # fig_bw_betastar.tight_layout()

    # Write lumis to file
    def_lumis['luminosity_err'] = std
    def_lumis.to_csv(f'lumi_vs_beta_star.csv', index=False)

    # MBD Cross Section
    max_rate_path = 'max_rate.txt'
    cad_measurement_path = 'CAD_Measurements/VernierScan_Aug12_combined.dat'
    max_rate = read_max_rate(max_rate_path)
    cad_data = read_cad_measurement_file(cad_measurement_path)

    n_bunch = 111

    max_rate_per_bunch = max_rate / n_bunch

    f_beam = 78.4  # kHz
    n_blue, n_yellow = get_nblue_nyellow(cad_data, orientation='Horizontal', step=1, n_bunch=n_bunch)  # n_protons
    print(f'N Blue: {n_blue:.2e}, N Yellow: {n_yellow:.2e}')
    mb_to_um2 = 1e-19

    lumis = luminosities * mb_to_um2 * f_beam * 1e3 * n_blue * n_yellow
    max_rate_samples = np.random.normal(max_rate_per_bunch.val, max_rate_per_bunch.err, len(lumis))
    cross_sections = max_rate_samples / lumis

    cross_section_bs_90 = max_rate_per_bunch.val / (lumi_bs_90 * mb_to_um2 * f_beam * 1e3 * n_blue * n_yellow)
    cross_section_bs_105 = max_rate_per_bunch.val / (lumi_bs_105 * mb_to_um2 * f_beam * 1e3 * n_blue * n_yellow)
    cross_section_gaus = max_rate_per_bunch.val / (lumi_gaus * mb_to_um2 * f_beam * 1e3 * n_blue * n_yellow)

    std_cross_section = np.std(cross_sections)
    asym_cross_sec_left, asym_cross_sec_right = np.percentile(cross_sections, 16), np.percentile(cross_sections, 84)

    cross_section_bs_90_meas = Measure(cross_section_bs_90, std_cross_section)
    cross_section_bs_105_meas = Measure(cross_section_bs_105, std_cross_section)

    cross_title_str = 'MBD Cross Section [mb]:'
    cross_90_str = rf'{cross_section_bs_90_meas} for $\beta^* = $ 90 cm'
    cross_105_str = rf'{cross_section_bs_105_meas} for $\beta^* = $ 105 cm'
    cross_gaus_str = f'{cross_section_gaus:.1f} for Gaussian Approximation'
    cross_std_err_str = f'{std_cross_section:.1f} Uncertainty (from std)'
    asym_err_str = f'{asym_cross_sec_left:.1f} - {asym_cross_sec_right:.1f} Asymmetric 68% CI'
    full_cross_str = (f'{cross_title_str}\n{cross_90_str}\n{cross_105_str}\n{cross_gaus_str}\n{cross_std_err_str}\n'
                      f'{asym_err_str}')

    fig_cross_section_hist, ax_cross_section_hist = plt.subplots(figsize=(8, 6))
    hist_cross_section, bin_edges_cross_section, _ = ax_cross_section_hist.hist(cross_sections, bins=100, density=True,
                                                                                color='k')

    x_cross_sec = np.linspace(min(bin_edges_cross_section), max(bin_edges_cross_section), 1000)
    y_cross_sec = norm.pdf(x_cross_sec, cross_section_bs_90, std_cross_section)
    ax_cross_section_hist.plot(x_cross_sec, y_cross_sec, color='r', ls='--', label='Symmetric Approximation')

    ax_cross_section_hist.axvline(cross_section_bs_90, color='red', label=r'$\beta^* =$ 90 cm')
    ax_cross_section_hist.axvline(cross_section_bs_105, color='green', label=r'$\beta^* =$ 105 cm')
    ax_cross_section_hist.fill_betweenx([0, max(hist_cross_section)], asym_cross_sec_left, asym_cross_sec_right,
                                        color='yellow', alpha=0.2, label='68% CI')
    ax_cross_section_hist.set_xlabel('MBD Cross Section [mb]')
    ax_cross_section_hist.set_ylabel('Probability')
    ax_cross_section_hist.annotate(full_cross_str, (0.4, 0.4), xycoords='axes fraction', ha='left', va='bottom',
                                      fontsize=12, bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.5'))
    ax_cross_section_hist.legend()
    fig_cross_section_hist.tight_layout()

    left_lumi_err90, right_lumi_err90 = asym_left - lumi_bs_90, asym_right - lumi_bs_90
    left_lumi_err105, right_lumi_err105 = asym_left - lumi_bs_105, asym_right - lumi_bs_105

    left_err90, right_err90 = asym_cross_sec_left - cross_section_bs_90, asym_cross_sec_right - cross_section_bs_90
    left_err105, right_err105 = asym_cross_sec_left - cross_section_bs_105, asym_cross_sec_right - cross_section_bs_105

    print(f'Naked Luminosity bs90: {lumi_bs_90_meas}, {left_lumi_err90} +{right_lumi_err90}')
    print(f'Naked Luminosity bs105: {lumi_bs_105_meas}, {left_lumi_err105} +{right_lumi_err105}')

    print(f'MBD Cross-Section bs90: {cross_section_bs_90_meas}, {left_err90} +{right_err90}')
    print(f'MBD Cross-Section bs105: {cross_section_bs_105_meas}, {left_err105} +{right_err105}')

    # print(f'Max Rate Per Bunch: {max_rate_per_bunch} Hz')
    # for index, row in lumi_data.iterrows():
    #     beta_star = row['beta_star']
    #     naked_lumi = Measure(row['luminosity'], row['luminosity_err'])
    #     lumi = naked_lumi * mb_to_um2 * f_beam * 1e3 * n_blue * n_yellow
    #     cross_section = max_rate_per_bunch / lumi
    #     print(f'Beta Star: {beta_star} cm, Luminosity: {lumi} mb⁻¹s⁻¹, Cross Section: {cross_section} mb')

    if save_path:
        fig_lumi_hist.savefig(f'{save_path}luminosity_histogram.png')
        fig_lumi_hist.savefig(f'{save_path}luminosity_histogram.pdf')
        fig_cross_section_hist.savefig(f'{save_path}cross_section_histogram.png')
        fig_cross_section_hist.savefig(f'{save_path}cross_section_histogram.pdf')

    # Write cross_section histogram to file as csv
    bin_centers = (bin_edges_cross_section[1:] + bin_edges_cross_section[:-1]) / 2
    hist_df = pd.DataFrame({'bin_center': bin_centers, 'hist': hist_cross_section})
    hist_df.to_csv('mbd_cross_section_distribution.csv', index=False)

    plt.show()

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


def gaus_pdf(x, loc=0, scale=1):
    pass


if __name__ == '__main__':
    main()

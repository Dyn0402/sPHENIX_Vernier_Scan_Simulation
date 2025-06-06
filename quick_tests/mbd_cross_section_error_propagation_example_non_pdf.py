#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on February 28 6:24 PM 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/mbd_cross_section_error_propagation_example.py

@author: Dylan Neff, Dylan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
from scipy.interpolate import interp1d


def main():
    """
    Example showing a method for numerically propagating errors from the MBD cross-section.
    :return:
    """
    # Enter function here
    def function(your_number, mbd_cross_section):
        return your_number * mbd_cross_section

    # Enter your number here
    your_number_value = 10.0
    your_number_error = 0.5

    # Load the MBD cross section pdf
    cross_section_path = '../run_rcf_jobs_lumi_calc_old/output/conservative_err_combined_lumis.csv'
    cross_section_df = pd.read_csv(cross_section_path)

    n_samples = 100  # If None use all samples, otherwise use n_samples

    if n_samples is not None:  # Get first n_samples rows from cross_section_df
        cross_section_df = cross_section_df.head(n_samples)
    else:
        n_samples = len(cross_section_df)

    print(cross_section_df)

    # Sample your value from a normal distribution
    your_number_samples = norm.rvs(your_number_value, your_number_error, size=n_samples)

    # Calulate your function for each sample
    function_samples = function(your_number_samples, cross_section_df['luminosity'])

    # Plot the MBD cross section pdf
    fig, ax = plt.subplots()
    ax.hist(cross_section_df['luminosity'], bins=100, density=True, histtype='step')
    ax.set_title('MBD Cross Section PDF')
    ax.set_xlabel('Cross Section [mb]')
    ax.set_ylabel('Probability Density')
    fig.tight_layout()

    # Plot the function samples
    fig_your_function, ax_your_function = plt.subplots()
    ax_your_function.hist(function_samples, bins=100, density=True, histtype='step')
    ax_your_function.set_title('Your Function PDF')
    ax_your_function.set_xlabel('Function Value')
    ax_your_function.set_ylabel('Probability Density')
    fig_your_function.tight_layout()

    plt.show()


    print('donzo')


if __name__ == '__main__':
    main()

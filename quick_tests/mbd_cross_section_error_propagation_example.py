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
    cross_section_path = 'mbd_cross_section_distribution.csv'
    cross_section_df = pd.read_csv(cross_section_path)

    n_samples = 100000  # Number of samples to take for error propagation

    # Sample your value from a normal distribution
    your_number_samples = norm.rvs(your_number_value, your_number_error, size=n_samples)

    # Sample the MBD cross section from the pdf
    bin_width = cross_section_df['bin_center'][1] - cross_section_df['bin_center'][0]
    cdf_values = np.cumsum(cross_section_df['hist'] * bin_width)
    inverse_cdf = interp1d(cdf_values, cross_section_df['bin_center'], kind='linear', fill_value='extrapolate')
    random_uniform_samples = np.random.uniform(0, 1, n_samples)
    cross_section_samples = inverse_cdf(random_uniform_samples)

    # Plot the MBD cross section pdf
    fig, ax = plt.subplots()
    ax.bar(cross_section_df['bin_center'], cross_section_df['hist'], width=bin_width, align='center')
    ax.set_title('MBD Cross Section PDF')
    ax.set_xlabel('Cross Section [mb]')
    ax.set_ylabel('Probability Density')
    fig.tight_layout()

    # Calculate your function for each sample
    function_samples = function(your_number_samples, cross_section_samples)

    # Plot the function samples
    fig_your_function, ax_your_function = plt.subplots()
    ax_your_function.hist(function_samples, bins=100, density=True, histtype='step')
    ax_your_function.set_title('Your Function PDF')
    ax_your_function.set_xlabel('Function Value')
    ax_your_function.set_ylabel('Probability Density')
    fig_your_function.tight_layout()

    # Print 16 and 84 percentiles
    print(f'16th percentile: {np.percentile(function_samples, 16)}')
    print(f'84th percentile: {np.percentile(function_samples, 84)}')

    plt.show()

    print('donzo')


if __name__ == '__main__':
    main()

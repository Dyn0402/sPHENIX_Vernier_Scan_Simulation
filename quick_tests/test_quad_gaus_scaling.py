#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on February 21 13:42 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/test_quad_gaus_scaling

@author: Dylan Neff, dn277127
"""

import numpy as np
import matplotlib.pyplot as plt

from BunchCollider import BunchCollider
from BunchDensity import quad_gaus_pdf


def main():
    scan_date = 'Aug12'
    longitudinal_fit_path = f'../CAD_Measurements/VernierScan_{scan_date}_COLOR_longitudinal_fit.dat'

    param_holder = BunchCollider()
    blue_fit_path = longitudinal_fit_path.replace('_COLOR_', f'_blue_')
    yellow_fit_path = longitudinal_fit_path.replace('_COLOR_', f'_yellow_')
    param_holder.set_longitudinal_fit_parameters_from_file(blue_fit_path, yellow_fit_path)
    print(param_holder.bunch1.longitudinal_params)
    print(param_holder.bunch2.longitudinal_params)

    # Plot quad gaus pdf for blue and yellow
    x = np.linspace(-5e6, 5e6, 1000)
    pdf1 = quad_gaus_pdf(x, *param_holder.bunch1.longitudinal_params.values())
    pdf2 = quad_gaus_pdf(x, *param_holder.bunch2.longitudinal_params.values())
    fig, ax = plt.subplots()
    ax.plot(x, pdf1, label='Blue')
    ax.plot(x, pdf2, label='Yellow')
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.tight_layout()

    # Scale blue in x by 1.5 and plot original and scaled
    scaled_blue_params = param_holder.bunch1.longitudinal_params.copy()
    for key in scaled_blue_params:
        if 'mu' in key or 'sigma' in key:
            scaled_blue_params[key] *= 1.2
    pdf1_scaled = quad_gaus_pdf(x, *scaled_blue_params.values())
    fig, ax = plt.subplots()
    ax.plot(x, pdf1, label='Blue')
    ax.plot(x, pdf1_scaled, label='Blue Scaled')
    ax.set_ylim(bottom=0)
    # Calculate sum of each and annotate on plot
    sum1 = np.sum(pdf1)
    sum1_scaled = np.sum(pdf1_scaled)
    ax.annotate(f'Original Sum: {sum1:.2e}', (0, 0), (0, 0), textcoords='offset points')
    ax.annotate(f'Scaled Sum: {sum1_scaled:.2e}', (0, 0), (0, 15), textcoords='offset points')

    ax.legend()
    fig.tight_layout()

    plt.show()

    print('donzo')


if __name__ == '__main__':
    main()

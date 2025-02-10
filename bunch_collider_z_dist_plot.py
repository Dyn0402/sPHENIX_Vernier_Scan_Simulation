#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on December 16 10:14 2024
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/bunch_collider_z_dist_plot

@author: Dylan Neff, dn277127
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as cf

from BunchCollider import BunchCollider
from Measure import Measure

mu = '\u03BC'
sigma = '\u03C3'


def main():
    plot_z_dist()
    print('donzo')


def plot_z_dist():
    """
    Define collision geometry and plot resulting z-vertex distribution.
    """
    beam_width = 170  # um
    beta_star = 85  # cm
    bkg = 0.0e-17  # Background rate
    blue_angle_x = 0.0e-3  # rad
    blue_angle_y = -0.0e-3  # rad
    # blue_angle_y = -0.75e-3  # rad
    # blue_angle_y = -1.e-3  # rad
    yellow_angle_x = 0.0e-3  # rad
    yellow_angle_y = +0.0e-3  # rad
    # yellow_angle_y = +0.75e-3  # rad
    # yellow_angle_y = +1.e-3  # rad
    z_init = 6.0e6  # cm

    collider_sim = BunchCollider()
    collider_sim.set_bunch_rs(np.array([0., 0., -z_init]), np.array([0., 0., z_init]))
    collider_sim.set_bunch_beta_stars(beta_star, beta_star)
    collider_sim.set_bunch_sigmas(np.array([beam_width, beam_width]), np.array([beam_width, beam_width]))
    collider_sim.set_bkg(bkg)
    collider_sim.set_bunch_crossing(blue_angle_x, blue_angle_y, yellow_angle_x, yellow_angle_y)

    collider_sim.run_sim_parallel()

    zs, z_dist = collider_sim.get_z_density_dist()

    fit_indices = np.where(abs(zs) < 80)
    zs_fit, z_dist_fit = zs[fit_indices], z_dist[fit_indices]

    # Fit gaussian to z distribution
    p0 = [np.max(z_dist_fit), 0, 20.]
    popt, pcov = cf(gaus, zs_fit, z_dist_fit, p0=p0)
    meases = [Measure(val, err) for val, err in zip(popt, np.sqrt(np.diag(pcov)))]
    print(f'Gaussian fit: a={popt[0]}, b={popt[1]}, c={popt[2]}')

    fig, ax = plt.subplots()
    ax.plot(zs, z_dist)
    ax.plot(zs, gaus(zs, *p0), color='gray', alpha=0.6, label='Initial Guess')
    ax.plot(zs, gaus(zs, *popt), 'r-', label='Fit')
    ax.set_xlabel('z (cm)')
    ax.set_ylabel('Density')
    ax.set_title('Z-Vertex Distribution')
    ax.set_ylim(bottom=0)
    ax.annotate(f'A={meases[0]}\n{mu}={meases[1]}\n{sigma}={meases[2]}', xy=(0.05, 0.95), xycoords='axes fraction',
                ha='left', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.grid()
    ax.legend()
    fig.tight_layout()
    plt.show()


def gaus(x, a, b, c):
    return a * np.exp(-(x - b) ** 2 / (2 * c ** 2))


if __name__ == '__main__':
    main()

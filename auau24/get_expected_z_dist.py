#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 15 18:37 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/get_expected_z_dist

@author: Dylan Neff, dn277127
"""

import numpy as np
import matplotlib.pyplot as plt

from BunchCollider import BunchCollider


def main():
    base_path_auau = '/local/home/dn277127/Bureau/vernier_scan_AuAu24/'
    longitudinal_fit_path_auau = f'{base_path_auau}CAD_Measurements/VernierScan_AuAu_COLOR_longitudinal_fit_22.dat'
    base_path_pp = '/local/home/dn277127/Bureau/vernier_scan/'
    longitudinal_fit_path_pp = f'{base_path_pp}CAD_Measurements/VernierScan_Aug12_COLOR_longitudinal_fit.dat'

    beam_width = 110  # um
    beta_star = 85  # cm
    bkg = 0.0e-17  # Background rate
    blue_angle_x = 0.0e-3  # rad
    blue_angle_y = -0.05e-3  # rad
    yellow_angle_x = 0.0e-3  # rad
    yellow_angle_y = +0.0e-3  # rad
    z_init = 6.0e6  # cm

    fig, ax = plt.subplots()

    for longitudinal_fit_path in [longitudinal_fit_path_auau, longitudinal_fit_path_pp]:
        collider_sim = BunchCollider()
        collider_sim.set_bunch_rs(np.array([0., 0., -z_init]), np.array([0., 0., z_init]))
        collider_sim.set_bunch_beta_stars(beta_star, beta_star)
        collider_sim.set_bunch_sigmas(np.array([beam_width, beam_width]), np.array([beam_width, beam_width]))
        collider_sim.set_bkg(bkg)
        collider_sim.set_bunch_crossing(blue_angle_x, blue_angle_y, yellow_angle_x, yellow_angle_y)
        blue_fit_path = longitudinal_fit_path.replace('_COLOR_', '_blue_')
        yellow_fit_path = longitudinal_fit_path.replace('_COLOR_', '_yellow_')
        collider_sim.set_longitudinal_fit_parameters_from_file(blue_fit_path, yellow_fit_path)

        collider_sim.run_sim_parallel()

        zs, z_dist = collider_sim.get_z_density_dist()

        label = 'AuAu' if longitudinal_fit_path == longitudinal_fit_path_auau else 'pp'
        ax.plot(zs, z_dist, label=label, linewidth=2)
    ax.set_xlabel('z (cm)')
    ax.set_ylabel('Density')
    ax.set_title('Z-Vertex Distribution')
    ax.set_ylim(bottom=0)
    ax.grid()
    ax.legend()
    fig.tight_layout()
    plt.show()
    print('donzo')


if __name__ == '__main__':
    main()

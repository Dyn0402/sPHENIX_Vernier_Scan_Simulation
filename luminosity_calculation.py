#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 10 12:33 PM 2024
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/luminosity_calculation.py

@author: Dylan Neff, Dylan
"""

import numpy as np
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize

from BunchCollider import BunchCollider
from Measure import Measure


def main():
    vernier_scan_date = 'Aug12'
    # vernier_scan_date = 'Jul11'
    # base_path = '/local/home/dn277127/Bureau/vernier_scan/'
    base_path = 'C:/Users/Dylan/Desktop/vernier_scan/'
    cad_measurement_path = f'{base_path}CAD_Measurements/VernierScan_{vernier_scan_date}_combined.dat'
    longitudinal_fit_path = f'{base_path}CAD_Measurements/VernierScan_{vernier_scan_date}_COLOR_longitudinal_fit.dat'
    # head_on_luminosity(longitudinal_fit_path)
    # head_on_luminosity_simple_gaus()
    head_on_luminosity_plot()
    print('donzo')


def head_on_luminosity(longitudinal_fit_path):
    # step_to_use = 'vertical_0'
    step_to_use = 'vertical_6'

    all_params_dict = {
        'vertical_0': {
            'hist_num': 0,
            'angle_y_blue': -0.071e-3,
            'angle_y_yellow': -0.113e-3,
            'blue_len_scaling': 1.001144,
            'yellow_len_scaling': 1.00105,
            'rate': 845710.0067
        },
        'vertical_6': {
            'hist_num': 6,
            'angle_y_blue': -0.074e-3,
            'angle_y_yellow': -0.110e-3,
            'blue_len_scaling': 1.003789,
            'yellow_len_scaling': 1.004832,
            'rate': 843673.7041
        },
    }
    params = all_params_dict[step_to_use]

    # Important parameters
    bw_nom = 160
    beta_star_nom = 85.
    mbd_online_resolution_nom = 5  # cm MBD resolution on trigger level
    # mbd_online_resolution_nom = 25.0  # cm MBD resolution on trigger level
    z_eff_width = 500.  # cm
    y_offset_nom = -0.
    x_offset_nom = 0.
    yellow_length_scaling, blue_length_scaling = params['yellow_len_scaling'], params['blue_len_scaling']
    angle_y_blue, angle_y_yellow = params['angle_y_blue'], params['angle_y_yellow']
    angle_x_blue, angle_x_yellow = 0., 0.

    collider_sim = BunchCollider()
    collider_sim.set_bunch_rs(np.array([x_offset_nom, y_offset_nom, -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_beta_stars(beta_star_nom, beta_star_nom)
    collider_sim.set_bunch_sigmas(np.array([bw_nom, bw_nom]), np.array([bw_nom, bw_nom]))
    collider_sim.set_bunch_crossing(angle_x_blue, angle_y_blue, angle_x_yellow, angle_y_yellow)
    collider_sim.set_gaus_smearing_sigma(mbd_online_resolution_nom)
    collider_sim.set_gaus_z_efficiency_width(z_eff_width)
    collider_sim.set_longitudinal_fit_scaling(blue_length_scaling, yellow_length_scaling)
    blue_fit_path = longitudinal_fit_path.replace('_COLOR_', '_blue_')
    yellow_fit_path = longitudinal_fit_path.replace('_COLOR_', '_yellow_')
    collider_sim.set_longitudinal_fit_parameters_from_file(blue_fit_path, yellow_fit_path)

    collider_sim.run_sim_parallel()
    zs_og, z_dist_og = collider_sim.get_z_density_dist()

    # Calculate density integral
    z_int = np.sum(z_dist_og)
    grid_info = collider_sim.get_grid_info()
    rho_int = z_int * grid_info['dz'] * grid_info['dy'] * grid_info['dx'] * grid_info['n_points_t'] * grid_info['dt']
    print(f'Integral of density: {rho_int}')

    # Plot the original z distribution
    fig, ax = plt.subplots()
    ax.plot(zs_og, z_dist_og, label='Original')
    ax.set_xlabel('z (cm)')
    ax.set_ylabel('Density')
    ax.legend()
    fig.tight_layout()

    plt.show()


def head_on_luminosity_simple_gaus():
    # Important parameters
    bw_nom = 180.  # um Width of bunch
    bunch_length = 1.1e6  # um Length of bunch
    beta_star_nom = None
    y_offset_nom = -0.
    x_offset_nom = 0.
    angle_y_blue, angle_y_yellow = 0., 0.
    angle_x_blue, angle_x_yellow = 0., 0.

    collider_sim = BunchCollider()
    collider_sim.set_bunch_rs(np.array([x_offset_nom, y_offset_nom, -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_beta_stars(beta_star_nom, beta_star_nom)
    collider_sim.set_bunch_sigmas(np.array([bw_nom, bw_nom, bunch_length]), np.array([bw_nom, bw_nom, bunch_length]))
    collider_sim.set_bunch_crossing(angle_x_blue, angle_y_blue, angle_x_yellow, angle_y_yellow)

    collider_sim.n_points_t = 80
    collider_sim.n_points_z = 100
    collider_sim.n_points_y = 100

    collider_sim.run_sim_parallel()
    zs_og, z_dist_og = collider_sim.get_z_density_dist()

    # Calculate density integral
    z_int = np.sum(z_dist_og)
    grid_info = collider_sim.get_grid_info()
    rho_int = z_int * grid_info['dz'] * grid_info['dy'] * grid_info['dx'] * grid_info['n_points_t'] * grid_info['dt']
    rho_int *= 2 * collider_sim.bunch1.c
    print(f'Grid info: {grid_info}')
    print(f'Number of z points: {collider_sim.n_points_z}')
    print(f'Integral of density: {rho_int}')

    # Plot the original z distribution
    fig, ax = plt.subplots()
    ax.plot(zs_og, z_dist_og, label='Original')
    ax.set_xlabel('z (cm)')
    ax.set_ylabel('Density')
    ax.legend()
    fig.tight_layout()

    plt.show()


def head_on_luminosity_plot():
    # Beam widths and bunch lengths
    beam_widths = [160., 180.]  # um
    bunch_lengths = [1.1e6, 1.3e6]  # um

    width_colors = dict(zip(beam_widths, ['r', 'b']))
    length_line_styles = dict(zip(bunch_lengths, ['-', '--']))

    mu = u'\u03BC'

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Stacking position for annotations
    annotation_offset = 0.25
    y_offset = 0.98  # Start at 100% of the plot height

    sim_lumis = {}

    for bw in beam_widths:
        # Calculate analytical luminosity
        analytical_lum = analytical_luminosity(bw)
        annotate_str = f'Luminosity for {bw:.0f}{mu}m beam width\nAnalytical = {analytical_lum:.2e}'

        for bl in bunch_lengths:
            # Simulate luminosity
            print(f'Simulating beam width: {bw} {mu}m, bunch length: {bl / 1e6:.1e} m')
            zs, z_dist, simulated_luminosity = simulate_luminosity(bw, bl)

            # Plot z distribution
            label = f'Beam width {bw:.0f}{mu}m, length: {bl / 1e6:.1f}m'
            ax.plot(zs, z_dist, color=width_colors[bw], ls=length_line_styles[bl], label=label)

            # Calculate percent difference
            percent_diff = 100 * (simulated_luminosity - analytical_lum) / analytical_lum
            annotate_str += f'\nLength {bl / 1e6:.1f}m = {simulated_luminosity:.2e} ({percent_diff:.2f}%)'

        # Annotate percent difference for both bunch lengths
        ax.annotate(annotate_str, xy=(0.99, y_offset), xycoords='axes fraction', ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        y_offset -= annotation_offset

    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=-349, right=349)
    ax.set_xlabel('z (cm)')
    ax.set_ylabel('Single Particle Luminosity')
    ax.legend(loc='upper left')
    fig.tight_layout()

    plt.show()


def simulate_luminosity(bw_nom, bunch_length):
    collider_sim = BunchCollider()
    collider_sim.set_bunch_rs(np.array([0., -0., -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_beta_stars(None, None)
    collider_sim.set_bunch_sigmas(np.array([bw_nom, bw_nom, bunch_length]), np.array([bw_nom, bw_nom, bunch_length]))
    collider_sim.set_bunch_crossing(0., 0., 0., 0.)
    collider_sim.set_z_bounds((-345 * 1e4, 345 * 1e4))

    collider_sim.run_sim_parallel()

    zs, z_dist = collider_sim.get_z_density_dist()

    luminosity = collider_sim.get_naked_luminosity()

    return zs, z_dist, luminosity


def analytical_luminosity(width):
    return 1 / (4 * np.pi * width**2)


if __name__ == '__main__':
    main()

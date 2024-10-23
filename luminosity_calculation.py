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

from BunchCollider import BunchCollider
from Measure import Measure
from vernier_z_vertex_fitting import read_cad_measurement_file


def main():
    vernier_scan_date = 'Aug12'
    # vernier_scan_date = 'Jul11'
    # base_path = '/local/home/dn277127/Bureau/vernier_scan/'
    base_path = 'C:/Users/Dylan/Desktop/vernier_scan/'
    cad_measurement_path = f'{base_path}CAD_Measurements/VernierScan_{vernier_scan_date}_combined.dat'
    longitudinal_fit_path = f'{base_path}CAD_Measurements/VernierScan_{vernier_scan_date}_COLOR_longitudinal_fit.dat'
    # head_on_luminosity(longitudinal_fit_path)
    # head_on_luminosity_simple_gaus()
    # head_on_luminosity_plot()
    # angle_luminosity_plot()
    # moller_factor_test()
    cad_parameters_luminosity(cad_measurement_path, longitudinal_fit_path)
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


def moller_factor_test():
    # Important parameters
    y_offset_nom = -0.
    x_offset_nom = 0.
    angle_y_blue, angle_y_yellow = 0., 2e-3
    angle_x_blue, angle_x_yellow = 0., 0.

    collider_sim = BunchCollider()
    collider_sim.set_bunch_rs(np.array([x_offset_nom, y_offset_nom, -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_crossing(angle_x_blue, angle_y_blue, angle_x_yellow, angle_y_yellow)
    collider_sim.bunch1.calculate_r_and_beta()
    collider_sim.bunch2.calculate_r_and_beta()

    print(f'v1: {collider_sim.bunch1.beta * collider_sim.bunch1.c}')
    print(f'v2: {collider_sim.bunch2.beta * collider_sim.bunch2.c}')

    print(f'Moller factor: {collider_sim.get_relativistic_moller_factor() / collider_sim.bunch1.c}')
    print(f'Moller factor difference from head on: '
          f'{((collider_sim.get_relativistic_moller_factor() / collider_sim.bunch1.c) - 2) * 100}%')


def head_on_luminosity_plot():
    """
    Plot luminosity z-densities for combinations of beam widths and bunch lengths. Compare to analytical luminosity.
    :return:
    """
    # Beam widths and bunch lengths
    beam_widths = [160., 180.]  # um
    bunch_lengths = [1.1e6, 1.3e6]  # um

    width_colors = dict(zip(beam_widths, ['r', 'b']))
    length_line_styles = dict(zip(bunch_lengths, ['-', '--']))

    mu = u'\u03BC'

    fig, ax = plt.subplots(figsize=(8, 4))

    # Stacking position for annotations
    annotation_offset = 0.24
    y_offset = 0.88  # Start at 100% of the plot height

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
    ax.set_ylabel('Naked Luminosity z Density')

    # Add an annotation with the analytical luminosity equation
    ax.annotate('Analytical Luminosity = 1 / 4πσ²', xy=(0.99, 0.97), xycoords='axes fraction', ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='salmon', alpha=0.3))

    ax.legend(loc='upper left')
    fig.tight_layout()

    plt.show()


def angle_luminosity_plot():
    """
    Plot luminosity z-densities for different crossing angles. Compare to analytical luminosity.
    :return:
    """
    # bunch_2_y_angles = [0., 1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3, 10e-3]  # rad
    bunch_1_y_angles = [0., 1e-3, 10e-3]  # rad
    bunch_2_y_angle = 0.  # rad
    bw = 160.  # um
    bl = 1.1e6  # um

    mu = u'\u03BC'

    fig, ax = plt.subplots(figsize=(8, 4))

    # Stacking position for annotations
    annotation_offset = 0.18
    y_offset = 0.88  # Start at 100% of the plot height

    for angle in bunch_1_y_angles:
        # Calculate analytical luminosity
        analytical_lum = analytical_luminosity(bw, bl, angle)
        annotate_str = f'Luminosity for {angle * 1e3:.0f} mrad crossing angle\nAnalytical = {analytical_lum:.2e}'

        # Simulate luminosity
        print(f'Simulating crossing angle: {angle * 1e3:.0f} mrad')
        print(f'Analytical luminosity: {analytical_lum:.2e}')
        zs, z_dist, simulated_luminosity = simulate_luminosity(bw, bl, angle, bunch_2_y_angle)

        # Plot z distribution
        label = f'Crossing angle {angle * 1e3:.0f} mrad'
        ax.plot(zs, z_dist, label=label)

        # Calculate percent difference
        percent_diff = 100 * (simulated_luminosity - analytical_lum) / analytical_lum
        annotate_str += f'\nSimulated = {simulated_luminosity:.2e} ({percent_diff:.2f}%)'

        # Annotate percent difference for both bunch lengths
        ax.annotate(annotate_str, xy=(0.99, y_offset), xycoords='axes fraction', ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        y_offset -= annotation_offset

    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=-349, right=349)
    ax.set_xlabel('z (cm)')
    ax.set_ylabel('Naked Luminosity z Density')

    # Add an annotation with the analytical luminosity equation
    ax.annotate('Analytical Luminosity = S / 4πσ²', xy=(0.99, 0.97), xycoords='axes fraction', ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='salmon', alpha=0.3))

    ax.legend(loc='upper left')
    fig.tight_layout()

    plt.show()


def cad_parameters_luminosity(cad_measurement_path, longitudinal_fit_path):
    """
    Simulate the bunch collision given CAD parameters and calculate the luminosity.
    :param cad_measurement_path:
    :param longitudinal_fit_path:
    :return:
    """
    cad_data = read_cad_measurement_file(cad_measurement_path)
    print(cad_data)
    scan_orientation = 'Horizontal'
    head_on_cad_data = cad_data[cad_data['orientation'] == scan_orientation].iloc[0]

    # Important parameters
    bw_nom = 160.  # um Width of bunch
    beta_star_nom = 85.
    mbd_online_resolution = 2  # cm MBD resolution on trigger level
    yellow_length_scaling, blue_length_scaling = head_on_cad_data['yellow_bunch_length_scaling'], head_on_cad_data['blue_bunch_length_scaling']

    # Will be overwritten by CAD values
    y_offset_nom = 0.
    angle_nom = +0.14e-3

    collider_sim = BunchCollider()
    collider_sim.set_bunch_rs(np.array([0., y_offset_nom, -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_beta_stars(beta_star_nom, beta_star_nom)
    collider_sim.set_bunch_sigmas(np.array([bw_nom, bw_nom]), np.array([bw_nom, bw_nom]))
    collider_sim.set_bunch_crossing(0, angle_nom, 0, 0)
    collider_sim.set_gaus_smearing_sigma(mbd_online_resolution)
    blue_fit_path = longitudinal_fit_path.replace('_COLOR_', '_blue_')
    yellow_fit_path = longitudinal_fit_path.replace('_COLOR_', '_yellow_')
    collider_sim.set_longitudinal_fit_parameters_from_file(blue_fit_path, yellow_fit_path)
    collider_sim.set_longitudinal_fit_scaling(blue_length_scaling, yellow_length_scaling)

    offset = head_on_cad_data['offset_set_val'] * 1e3  # mm to um
    blue_angle, yellow_angle = -head_on_cad_data['bh8_avg'] / 1e3, -head_on_cad_data['yh8_avg'] / 1e3  # mrad to rad

    if scan_orientation == 'Horizontal':
        collider_sim.set_bunch_offsets(np.array([offset, 0.]), np.array([0., 0.]))
    elif scan_orientation == 'Vertical':
        collider_sim.set_bunch_offsets(np.array([0., offset]), np.array([0., 0.]))

    collider_sim.set_bunch_crossing(blue_angle, 0, yellow_angle, 0)

    collider_sim.run_sim_parallel()
    zs, z_dist = collider_sim.get_z_density_dist()

    luminosity = collider_sim.get_naked_luminosity()
    print(f'Luminosity: {luminosity:.2e}')

    # analytic_lumi = analytical_luminosity(bw_nom)
    analytic_lumi = 2.25534e-6
    print(f'Analytical luminosity: {analytic_lumi:.2e}')

    percent_differs = 100 * (luminosity - analytic_lumi) / analytic_lumi
    print(f'Percent difference: {percent_differs:.2f}%')

    fig, ax = plt.subplots()
    ax.plot(zs, z_dist)
    ax.set_xlabel('z (cm)')
    ax.set_ylabel('Naked Luminosity z Density')
    fig.tight_layout()

    plt.show()


def simulate_luminosity(bw_nom=None, bunch_length=None, angle_y_blue=None, angle_y_yellow=None):
    if bw_nom is None:
        bw_nom = 160.
    if bunch_length is None:
        bunch_length = 1.1e6
    if angle_y_blue is None:
        angle_y_blue = 0.
    if angle_y_yellow is None:
        angle_y_yellow = 0.

    collider_sim = BunchCollider()
    collider_sim.set_bunch_rs(np.array([0., -0., -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_beta_stars(None, None)
    collider_sim.set_bunch_sigmas(np.array([bw_nom, bw_nom, bunch_length]), np.array([bw_nom, bw_nom, bunch_length]))
    collider_sim.set_bunch_crossing(0., angle_y_blue, 0., angle_y_yellow)
    z_lim = 345 / (angle_y_blue * 1e3 + 1)
    print(f'Z limit: {z_lim}')
    collider_sim.set_z_bounds((-z_lim * 1e4, z_lim * 1e4))

    collider_sim.run_sim_parallel()

    zs, z_dist = collider_sim.get_z_density_dist()

    luminosity = collider_sim.get_naked_luminosity()

    return zs, z_dist, luminosity


def analytical_luminosity(width, length=None, angle=None):
    lumi = 1 / (4 * np.pi * width**2)  # Head on luminosity
    if length is not None and angle is not None:
        lumi *= 1. / np.sqrt(1 + (width / length * np.tan(angle / 2))**2) / np.sqrt(1 + (length / width * np.tan(angle / 2))**2)
    return lumi


if __name__ == '__main__':
    main()

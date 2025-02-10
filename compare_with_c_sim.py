#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on November 04 16:56 2024
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/compare_with_c_sim

@author: Dylan Neff, dn277127
"""

import numpy as np
import matplotlib.pyplot as plt

import uproot


from BunchCollider import BunchCollider


def main():
    base_path = '/local/home/dn277127/Bureau/vernier_scan/'
    compare_to_c_sim(base_path)
    # compare_to_mbd24(base_path)

    print('donzo')


def compare_to_mbd24(base_path):
    c_root_path = 'sasha_simulation/profile.root'
    with uproot.open(c_root_path) as c_root:
        # Get hz;1 histogram from file
        hz = c_root['hz;1']
        bin_edges, bin_vals = hz.axis().edges(), hz.values()
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    bin_centers /= 10  # Convert from mm to cm

    mbd_24_dists_path = f'{base_path}/vertex_data/vernier_scan_Aug12_mbd_vertex_z_distributions.root'
    with uproot.open(mbd_24_dists_path) as mbd_24_dists_root:
        head_on_step0 = mbd_24_dists_root['MBDvtxZ_Vertical_scan_step_0;1']
        head_on_step0_edges, head_on_step0_vals = head_on_step0.axis().edges(), head_on_step0.values()
        head_on_step0_centers = (head_on_step0_edges[1:] + head_on_step0_edges[:-1]) / 2

    # Important parameters
    # bw_x, bw_y = 139, 143
    # bunch_length = 1.250e6  # m
    # beta_star_nom = 70.

    bw_x, bw_y = 155, 155
    bunch_length = 1.250e6  # m
    beta_star_nom = 85.

    parameter_str = (fr'$\sigma_x = {bw_x} \mu m$' + '\n' +
                     fr'$\sigma_y = {bw_y} \mu m$' + '\n' +
                     fr'$\beta^* = {beta_star_nom} cm$' + '\n' +
                     fr'$L = {bunch_length / 1e6} m$' + '\n' +
                     'Head On')

    collider_sim = BunchCollider()
    collider_sim.set_bunch_rs(np.array([0., 0., -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_beta_stars(beta_star_nom, beta_star_nom)
    collider_sim.set_bunch_sigmas(np.array([bw_x, bw_y, bunch_length]), np.array([bw_x, bw_y, bunch_length]))

    collider_sim.run_sim_parallel()
    zs, z_dist = collider_sim.get_z_density_dist()

    # Scale z_dist and bin_vals to match mbd24
    z_dist = z_dist * head_on_step0_vals.max() / z_dist.max()
    bin_vals = bin_vals * head_on_step0_vals.max() / bin_vals.max()

    print(f'Bunch 1 beam length: {collider_sim.bunch1.get_beam_length() / 1e6}m')
    print(f'Bunch 2 beam length: {collider_sim.bunch2.get_beam_length() / 1e6}m')

    fig, ax = plt.subplots()
    bin_widths_mbd24 = head_on_step0_centers[1] - head_on_step0_centers[0]
    ax.step(head_on_step0_centers, head_on_step0_vals, color='blue', label='MBD24', where='mid', linewidth=1.5)
    ax.plot(bin_centers, bin_vals, color='orange', label='C Simulation')
    ax.plot(zs, z_dist, color='red', label='Python Simulation')
    ax.annotate(parameter_str, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, va='top', ha='left',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    ax.set_xlabel('z (cm)')
    ax.set_ylabel('Luminosity Density (arb. units, scaled to match height)')
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.tight_layout()
    plt.show()


def compare_to_c_sim(base_path):
    c_root_path = 'sasha_simulation/profile.root'
    with uproot.open(c_root_path) as c_root:
        # Get hz;1 histogram from file
        hz = c_root['hz;1']
        bin_edges, bin_vals = hz.axis().edges(), hz.values()
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    bin_centers /= 10  # Convert from mm to cm

    # Important parameters
    bw_x, bw_y = 139, 143
    bunch_length = 1.250e6  # m
    beta_star_nom = 70.

    parameter_str = (fr'$\sigma_x = {bw_x} \mu m$' + '\n' +
                     fr'$\sigma_y = {bw_y} \mu m$' + '\n' +
                     fr'$\beta^* = {beta_star_nom} cm$' + '\n' +
                     fr'$L = {bunch_length / 1e6} m$' + '\n' +
                     'Head On')

    collider_sim = BunchCollider()
    collider_sim.set_bunch_rs(np.array([0., 0., -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_beta_stars(beta_star_nom, beta_star_nom)
    collider_sim.set_bunch_sigmas(np.array([bw_x, bw_y, bunch_length]), np.array([bw_x, bw_y, bunch_length]))

    collider_sim.run_sim_parallel()
    zs, z_dist = collider_sim.get_z_density_dist()

    # Scale z_dist and bin_vals to match c sim
    z_dist = z_dist * bin_vals.max() / z_dist.max()

    fig, ax = plt.subplots()
    ax.plot(zs, z_dist, color='red', label='Python Simulation', alpha=0.9)
    ax.plot(bin_centers, bin_vals, color='orange', label='C Simulation', alpha=0.9)
    ax.set_ylim(bottom=0)
    ax.annotate(parameter_str, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, va='top', ha='left',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    ax.set_xlabel('z (cm)')
    ax.set_ylabel('Luminosity Density (arb. units, scaled to match height)')
    ax.legend()
    fig.tight_layout()

    fig.savefig(f'{base_path}c_sim_comparison.png')
    fig.savefig(f'{base_path}c_sim_comparison.pdf')

    plt.show()


if __name__ == '__main__':
    main()

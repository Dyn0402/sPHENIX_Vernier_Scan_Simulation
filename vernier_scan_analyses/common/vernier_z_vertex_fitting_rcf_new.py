#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on February 10 12:25 PM 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/vernier_z_vertex_fitting_rcf.py

@author: Dylan Neff, Dylan
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

from vernier_z_vertex_fitting import read_cad_measurement_file, get_cw_rates, get_mbd_z_dists
from vernier_z_vertex_fitting_clean import fit_sim_to_mbd_step, plot_mbd_and_sim_dist, print_status, update_bw_plot_dict
from vernier_z_vertex_fitting_clean import plot_bw_dict, create_dir

from BunchCollider import BunchCollider


def main():
    """
    Need to pass vernier_z_vertex_fitting, vernier_z_vertex_fitting_clean, BunchCollider, BunchDensity,
    bunch_density.cpp, bunch_density_cpp.cpython-310-x86_64-linux-gnu.so and Measure to this script.
    :return:
    """
    if len(sys.argv) != 6:  # Get 5 sys argv: scan_date, scan_orientation, beam width x, beam width y, beta star
        print('Invalid number of arguments.')
        return
    print(f'System argvs: {sys.argv}')
    scan_date, scan_orientation = sys.argv[1], sys.argv[2]
    beam_width_x, beam_width_y, beta_star = float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
    save_plots = False

    # z_vertex_root_path = f'../vertex_data/vernier_scan_{scan_date}_mbd_vertex_z_distributions.root'
    # cad_measurement_path = f'../CAD_Measurements/VernierScan_{scan_date}_combined.dat'
    # longitudinal_fit_path = f'../CAD_Measurements/VernierScan_{scan_date}_COLOR_longitudinal_fit.dat'
    longitudinal_profiles_path = '../CAD_Measurements/profiles/'
    z_vertex_root_path = '../vertex_data/54733_vertex_distributions.root'
    cad_measurement_path = '../CAD_Measurements/combined_cad_step_data.csv'

    out_dir = create_dir('output/')
    out_dir = create_dir(f'{out_dir}{scan_date}/')
    out_dir = create_dir(f'{out_dir}{scan_orientation}/')

    fit_crossing_angles_to_mbd_dists(z_vertex_root_path, cad_measurement_path, longitudinal_profiles_path, scan_orientation,
                                     scan_date, beam_width_x, beam_width_y, beta_star, out_dir, save_plots)
    print('donzo')


def fit_crossing_angles_to_mbd_dists(z_vertex_root_path, cad_measurement_path, longitudinal_profiles_path, orientation,
                                     scan_date, beam_width_x, beam_width_y, beta_star, out_dir, save_plots):
    """
    For a list of bunch widths, fits the crossing angle to the z-vertex distributions for each bunch width.
    :param z_vertex_root_path: Path to root file with z-vertex distributions.
    :param cad_measurement_path: Path to cad measurements file.
    :param longitudinal_profiles_path: Path to longitudinal profiles.
    :param orientation: 'Horizontal' or 'Vertical'.
    :param scan_date: Date of the scan.
    :param beam_width_x: Beam width in x direction.
    :param beam_width_y: Beam width in y direction.
    :param beta_star: Beta star to use for the simulation.
    :param out_dir: Directory to output plots and data.
    :param save_plots: If True, save plots to out_dir, if False, display plots.
    """
    # Important parameters
    mbd_resolution = 2.0  # cm MBD resolution
    gauss_eff_width = 500  # cm Gaussian efficiency width
    bkg = 0.0  # Background level
    n_points_xy, n_points_z, n_points_t = 61, 151, 61
    fit_range = [-230, 230]

    run_dir = create_dir(f'{out_dir}run_bwx_{beam_width_x:.1f}_bwy_{beam_width_y:.1f}_betastar_{beta_star:.1f}/')
    plot_out_dir = create_dir(f'{run_dir}plots/')
    data_out_dir = create_dir(f'{run_dir}data/')

    cad_data = pd.read_csv(cad_measurement_path)
    cw_rates = get_cw_rates(cad_data)
    z_vertex_hists = get_mbd_z_dists(z_vertex_root_path, first_dist=False, norms=cw_rates, abs_norm=True)

    measured_beta_star = np.array([72., 72., 72., 72.])
    beta_star_scale_factor = beta_star / 72  # Set 72 cm (average of measured) to default values, then scale from there
    beta_star_scaled = measured_beta_star * beta_star_scale_factor

    collider_sim = BunchCollider()
    collider_sim.set_bunch_sigmas(np.array([beam_width_x, beam_width_y]), np.array([beam_width_x, beam_width_y]))
    collider_sim.set_grid_size(n_points_xy, n_points_xy, n_points_z, n_points_t)
    collider_sim.set_bunch_rs(np.array([0., 0., -6.e6]), np.array([0., 0., +6.e6]))
    # collider_sim.set_bunch_beta_stars(beta_star, beta_star)
    collider_sim.set_bunch_beta_stars(*beta_star_scaled)
    collider_sim.set_gaus_smearing_sigma(mbd_resolution)
    collider_sim.set_gaus_z_efficiency_width(gauss_eff_width)
    collider_sim.set_bkg(bkg)
    # blue_fit_path = longitudinal_profiles_path.replace('_COLOR_', '_blue_')
    # yellow_fit_path = longitudinal_profiles_path.replace('_COLOR_', '_yellow_')
    # collider_sim.set_longitudinal_fit_parameters_from_file(blue_fit_path, yellow_fit_path)

    # Get nominal dcct ions and emittances
    step_0 = cad_data[cad_data['step'] == 0].iloc[0]
    dcct_blue_nom, dcct_yellow_nom = step_0['blue_dcct_ions'], step_0['yellow_dcct_ions']
    em_blue_horiz_nom, em_blue_vert_nom = step_0['blue_horiz_emittance'], step_0['blue_vert_emittance']
    em_yel_horiz_nom, em_yel_vert_nom = step_0['yellow_horiz_emittance'], step_0['yellow_vert_emittance']

    z_vertex_hists_orient = [hist for hist in z_vertex_hists if hist['scan_axis'] == orientation]

    start_time, n_fits, total_fits = datetime.now(), 1, len(z_vertex_hists_orient)
    title_base = f'{scan_date} {orientation} Scan'
    bws_str = f'{beam_width_x:.1f}, {beam_width_y:.1f}'
    title_bw = f'{title_base}\nBeam Width {bws_str} microns, beta-star {beta_star} cm'

    first_step_hist = z_vertex_hists_orient[0]  # Fit the first step, which is head on
    fit_sim_to_mbd_step(collider_sim, first_step_hist, cad_data, True)

    bw_plot_dict = {'steps': [], 'angles': [[], [], [], []], 'residuals': [], 'dist_plot_data': []}
    for hist_data in z_vertex_hists_orient:
        print(f'\nStarting {scan_date} Beam Widths {bws_str} µm, Step {hist_data["scan_step"]}')

        fit_sim_to_mbd_step(collider_sim, hist_data, cad_data, fit_crossing_angles=True)  # All the fitting

        title = f'{title_bw}, Step: {hist_data["scan_step"]}'
        plot_data_dict = plot_mbd_and_sim_dist(collider_sim, hist_data, title=title, out_dir=None)
        plt.close()
        bw_plot_dict['dist_plot_data'].append(plot_data_dict)
        n_fits = print_status(bws_str, hist_data["scan_step"], start_time, n_fits, total_fits)
        update_bw_plot_dict(bw_plot_dict, hist_data, collider_sim)  # Update dict for plotting
    if save_plots:
        plot_bw_dict(bw_plot_dict, title_bw, out_dir=plot_out_dir)
    write_bw_dict_to_file(bw_plot_dict, data_out_dir)
    write_collider_info_to_file(collider_sim, data_out_dir)


def write_bw_dict_to_file(bw_plot_dict, out_dir):
    """
    Write the bw_plot_dict to a file in out_dir.
    :param bw_plot_dict: Dictionary of data to write.
    :param out_dir: Directory to write output file.
    """
    df = pd.DataFrame({
        'step': bw_plot_dict['steps'],
        'blue_horizontal': bw_plot_dict['angles'][0],
        'blue_vertical': bw_plot_dict['angles'][1],
        'yellow_horizontal': bw_plot_dict['angles'][2],
        'yellow_vertical': bw_plot_dict['angles'][3],
        'residuals': bw_plot_dict['residuals']
    })
    df.to_csv(f'{out_dir}scan_data.csv', index=False)


def write_collider_info_to_file(collider_sim, out_dir):
    """
    Write collider_sim info to a file in out_dir.
    :param collider_sim: BunchCollider object.
    :param out_dir: Directory to write output file.
    """
    with open(f'{out_dir}collider_sim_info.txt', 'w') as file:
        file.write(str(collider_sim))


if __name__ == '__main__':
    main()

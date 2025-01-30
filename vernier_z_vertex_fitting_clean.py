#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on January 28 12:39 PM 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/vernier_z_vertex_fitting_clean.py

@author: Dylan Neff, dylan
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar
from scipy.optimize import curve_fit as cf
from scipy.interpolate import interp1d
import pandas as pd
from datetime import datetime, timedelta

from vernier_z_vertex_fitting import read_cad_measurement_file, get_cw_rates, get_mbd_z_dists
from BunchCollider import BunchCollider
from Measure import Measure


def main():
    # base_path = '/local/home/dn277127/Bureau/vernier_scan/'
    base_path = '/home/dylan/Desktop/vernier_scan/'
    # base_path = 'C:/Users/Dylan/Desktop/vernier_scan/'

    bw_fitting_path = f'{base_path}Analysis/bw_fitting/'
    create_dir(bw_fitting_path)

    default_bws = {'Horizontal': 162, 'Vertical': 151}
    orientations_beam_widths = {'Horizontal': np.arange(156.0, 168.5, 0.5), 'Vertical': np.arange(145.0, 157.5, 0.5)}
    # orientations_beam_widths = {'Horizontal': np.array([162])}

    # vernier_scan_dates = ['Aug12', 'Jul11']  # No CAD_Measurements/VernierScan_Jul11_combined.dat
    vernier_scan_dates = ['Aug12']
    for vernier_scan_date in vernier_scan_dates:
        dist_root_file_name = f'vernier_scan_{vernier_scan_date}_mbd_vertex_z_distributions.root'
        z_vertex_root_path = f'{base_path}vertex_data/{dist_root_file_name}'
        cad_measurement_path = f'{base_path}CAD_Measurements/VernierScan_{vernier_scan_date}_combined.dat'
        longitudinal_fit_path = f'{base_path}CAD_Measurements/VernierScan_{vernier_scan_date}_COLOR_longitudinal_fit.dat'

        vernier_date_path = f'{bw_fitting_path}{vernier_scan_date}/'
        create_dir(vernier_date_path)

        for orientation, beam_widths in orientations_beam_widths.items():
            pdf_out_path = f'{vernier_date_path}{orientation.lower()}/'
            create_dir(pdf_out_path)
            fit_crossing_angles_for_bw_variations(z_vertex_root_path, cad_measurement_path, longitudinal_fit_path,
                                                  pdf_out_path, orientation, vernier_scan_date, beam_widths, default_bws)

        # # Run horizontal
        # orientation = 'Horizontal'
        # # beam_widths = np.arange(156, 167.5, 0.5)
        # beam_widths = np.array([162])
        # pdf_out_path = f'{vernier_date_path}{orientation.lower()}_test/'
        # create_dir(pdf_out_path)
        # fit_crossing_angles_for_bw_variations(z_vertex_root_path, cad_measurement_path, longitudinal_fit_path, pdf_out_path,
        #                                       orientation, vernier_scan_date, beam_widths, default_bws)
        #
        # # Run vertical
        # orientation = 'Vertical'
        # # beam_widths = np.arange(145, 156, 0.5)
        # beam_widths = np.array([156])
        # pdf_out_path = f'{vernier_date_path}{orientation.lower()}_test/'
        # create_dir(pdf_out_path)
        # fit_crossing_angles_for_bw_variations(z_vertex_root_path, cad_measurement_path, longitudinal_fit_path, pdf_out_path,
        #                                       orientation, vernier_scan_date, beam_widths, default_bws)

    # Fit residual sum vs beam width to estimate minimum. Then run the minimum and make combined plot of distributions.
    # orientation = 'Horizontal'
    # pdf_out_path = f'{base_path}Analysis/new_bw_opt/{orientation.lower()}/'
    # get_min_bw_and_run(z_vertex_root_path, cad_measurement_path, longitudinal_fit_path, pdf_out_path, orientation,
    #                    vernier_scan_date)

    print('donzo')


def fit_crossing_angles_for_bw_variations(z_vertex_root_path, cad_measurement_path, longitudinal_fit_path, out_path,
                                          orientation, scan_date, beam_widths=None, default_bws=None):
    """
    For a list of bunch widths, fits the crossing angle to the z-vertex distributions for each bunch width.
    :param z_vertex_root_path: Path to root file with z-vertex distributions.
    :param cad_measurement_path: Path to cad measurements file.
    :param longitudinal_fit_path: Path to longitudinal fit file.
    :param out_path: Path to output pdf.
    :param orientation: 'Horizontal' or 'Vertical'.
    :param scan_date: Date of the scan.
    :param beam_widths: List of beam widths to fit.
    :param default_bws: Default beam widths to use for the opposite orientation of the scan.
    """

    if beam_widths is None:
        beam_widths = np.arange(150, 170, 5)

    # Important parameters
    beta_star_nom = 85.
    mbd_resolution = 2.0  # cm MBD resolution
    gauss_eff_width = 500  # cm Gaussian efficiency width
    # bkg = 0.4e-16  # Background level
    bkg = 0.0  # Background level
    n_points_xy, n_points_z, n_points_t = 61, 151, 61

    cad_data = read_cad_measurement_file(cad_measurement_path)
    cw_rates = get_cw_rates(cad_data)
    z_vertex_hists = get_mbd_z_dists(z_vertex_root_path, first_dist=False, norms=cw_rates, abs_norm=True)

    collider_sim = BunchCollider()
    collider_sim.set_grid_size(n_points_xy, n_points_xy, n_points_z, n_points_t)
    collider_sim.set_bunch_rs(np.array([0., 0., -6.e6]), np.array([0., 0., +6.e6]))
    collider_sim.set_bunch_beta_stars(beta_star_nom, beta_star_nom)
    collider_sim.set_gaus_smearing_sigma(mbd_resolution)
    collider_sim.set_gaus_z_efficiency_width(gauss_eff_width)
    collider_sim.set_bkg(bkg)
    blue_fit_path = longitudinal_fit_path.replace('_COLOR_', '_blue_')
    yellow_fit_path = longitudinal_fit_path.replace('_COLOR_', '_yellow_')
    collider_sim.set_longitudinal_fit_parameters_from_file(blue_fit_path, yellow_fit_path)

    z_vertex_hists_orient = [hist for hist in z_vertex_hists if hist['scan_axis'] == orientation]

    start_time, n_fits, total_fits = datetime.now(), 1, len(z_vertex_hists_orient) * len(beam_widths)
    title_base = f'{scan_date} {orientation} Scan'
    plot_dict = {'bws': beam_widths, 'residual_means': [], 'steps': [], 'residuals': []}
    for bw in beam_widths:
        title_bw = f'{title_base} Beam Width {bw} µm'
        bw_out_path = create_dir(f'{out_path}{bw}/')
        bw_xy = [bw, bw]
        if default_bws is not None:
            if orientation == 'Horizontal':
                bw_xy = [bw, default_bws['Vertical']]
            elif orientation == 'Vertical':
                bw_xy = [default_bws['Horizontal'], bw]
        collider_sim.set_bunch_sigmas(np.array(bw_xy), np.array(bw_xy))  # Set beam width for this iteration

        # Fit the first step, which is head on
        first_step_hist = z_vertex_hists_orient[0]
        fit_sim_to_mbd_step(collider_sim, first_step_hist, cad_data, True)
        title = f'{title_bw}, Step {first_step_hist["scan_step"]} Original'
        plot_mbd_and_sim_dist(collider_sim, first_step_hist, title=title, out_dir=bw_out_path)

        bw_plot_dict = {'steps': [], 'angles': [[], [], [], []], 'residuals': [], 'dist_plot_data': []}
        for hist_data in z_vertex_hists_orient:
            print(f'\nStarting {scan_date} Beam Width {bw} µm, Step {hist_data["scan_step"]}')
            fit_sim_to_mbd_step(collider_sim, hist_data, cad_data, fit_crossing_angles=True)
            title = f'{title_bw}, Step: {hist_data["scan_step"]}'
            plot_data_dict = plot_mbd_and_sim_dist(collider_sim, hist_data, title=title, out_dir=bw_out_path)
            bw_plot_dict['dist_plot_data'].append(plot_data_dict)
            n_fits = print_status(bw, hist_data["scan_step"], start_time, n_fits, total_fits)
            update_bw_plot_dict(bw_plot_dict, hist_data, collider_sim)  # Update dict for plotting
        plot_bw_dict(bw_plot_dict, title_bw, out_dir=bw_out_path)
        update_plot_dict(plot_dict, bw_plot_dict)  # Update dict for plotting
    plot_plot_dict(plot_dict, f'{title_base} Residuals', out_dir=out_path)

    plt.show()


def get_min_bw_and_run(z_vertex_root_path, cad_measurement_path, longitudinal_fit_path, out_path, orientation, scan_date):
    """
    Get the residuals from file and fit to a polynomial to find the minimum. Then run the simulation with the minimum
    beam width and plot the distributions.
    """
    resid_vs_bw_csv_path = f'{out_path}{scan_date}_{orientation}_Scan_Residuals_residual_means.csv'
    resid_vs_bw_df = pd.read_csv(resid_vs_bw_csv_path)

    # Plot
    bws = np.array(resid_vs_bw_df['Beam Width'])
    resids = np.array(resid_vs_bw_df['Mean Residual'])

    min_res_bw, min_res = bws[resids.argmin()], resids.min()

    print(f'Minimum Residual: {min_res_bw} µm at {min_res} µm')

    fit_filter = abs(bws - min_res_bw) < 2.5
    print(f'Fitting to {bws[fit_filter]} µm')
    bws_fit, resids_fit = bws[fit_filter], resids[fit_filter]

    popt, pcov = cf(poly2, bws_fit, resids_fit, p0=[1, 1, 1])
    bws_plot = np.linspace(bws_fit.min(), bws_fit.max(), 500)

    fig, ax = plt.subplots()
    ax.plot(bws, resids, marker='o', ls='none')
    ax.plot(bws_plot, poly2(bws_plot, *popt), color='red')
    ax.set_xlabel('Beam Width [µm]')
    ax.set_ylabel('Mean of Residuals for All Steps')
    ax.set_title(f'{scan_date} {orientation} Scan Residuals vs Beam Width')
    fig.tight_layout()

    plt.show()


def poly2(x, a, b, c):
    return a * x ** 2 + b * x + c


def update_bw_plot_dict(bw_plot_dict, hist_data, collider_sim):
    """
    Update bw_plot_dict with the fit results from collider_sim.
    - Final residual
    - Final crossing angles
    :param bw_plot_dict: Dictionary to update.
    :param hist_data: Dictionary of histogram data.
    :param collider_sim: BunchCollider object.
    """

    bw_plot_dict['steps'].append(hist_data['scan_step'])
    bw_plot_dict['residuals'].append(get_residual(collider_sim, hist_data))
    angles = collider_sim.get_bunch_crossing_angles()
    for i in range(4):
        bw_plot_dict['angles'][i].append(angles[i] * 1e3)  # Convert to mrad


def plot_bw_dict(bw_plot_dict, title, out_dir=None):
    """
    For all steps in a beam width, plot the crossing angles and residuals.
    """
    fig_angles_horiz, ax_angles_horiz = plt.subplots()
    ax_angles_horiz.plot(bw_plot_dict['steps'], bw_plot_dict['angles'][0], label='Blue Horizontal', color='blue')
    ax_angles_horiz.plot(bw_plot_dict['steps'], bw_plot_dict['angles'][2], label='Yellow Horizontal', color='orange')
    ax_angles_horiz.axhline(0, color='black', alpha=0.3, zorder=0, linestyle='-')
    ax_angles_horiz.set_xlabel('Step')
    ax_angles_horiz.set_ylabel('Angle [mrad]')
    ax_angles_horiz.set_title(f'{title} Horizontal Crossing Angles')
    ax_angles_horiz.legend()
    fig_angles_horiz.tight_layout()

    fig_angles_vert, ax_angles_vert = plt.subplots()
    ax_angles_vert.plot(bw_plot_dict['steps'], bw_plot_dict['angles'][1], label='Blue Vertical', color='blue')
    ax_angles_vert.plot(bw_plot_dict['steps'], bw_plot_dict['angles'][3], label='Yellow Vertical', color='orange')
    ax_angles_vert.axhline(0, color='black', alpha=0.3, zorder=0, linestyle='-')
    ax_angles_vert.set_xlabel('Step')
    ax_angles_vert.set_ylabel('Angle [mrad]')
    ax_angles_vert.set_title(f'{title} Vertical Crossing Angles')
    ax_angles_vert.legend()
    fig_angles_vert.tight_layout()

    fig_res, ax_res = plt.subplots()
    ax_res.plot(bw_plot_dict['steps'], bw_plot_dict['residuals'])
    ax_res.set_xlabel('Step')
    ax_res.set_ylabel('Residual')
    ax_res.set_title(f'{title} Residuals')
    ax_res.set_ylim(bottom=0)
    fig_res.tight_layout()

    # Combined dist plot
    fig_dists, axs = plt.subplots(nrows=3, ncols=4, figsize=(22, 10), sharex='all')
    axs = axs.flatten()
    fig_dists.subplots_adjust(hspace=0.0, wspace=0.0, top=0.995, bottom=0.045, left=0.01, right=0.995)

    for i, (ax, data_i) in enumerate(zip(axs, bw_plot_dict['dist_plot_data'])):
        # data_i = bw_plot_dict['dist_plot_data'][i]
        step = bw_plot_dict['steps'][i]
        if i >= 8:
            ax.set_xlabel('Z Vertex (cm)')
        if i % 4 == 0:
            ax.set_ylabel('Counts (scaled)')
        ax.annotate(f'Step {step}', xy=(0.05, 0.75), xycoords='axes fraction', fontsize=15, va='top', ha='left')
        max_y = int(max(max(data_i['mbd_dist']), max(data_i['sim_dist'])))
        ax.axhline(max_y, color='black', alpha=0.3, zorder=0, linestyle='-')
        ax.annotate(f'{max_y}', xy=(250, max_y * 0.995), xycoords='data', fontsize=10, alpha=0.5,
                    va='top', ha='right')
        ax.set_yticks([])  # Removes both the tick labels and the ticks
        width = data_i['mbd_zs'][1] - data_i['mbd_zs'][0]
        ax.bar(data_i['mbd_zs'], data_i['mbd_dist'], width=width, label='MBD')
        ax.plot(data_i['sim_zs'], data_i['sim_dist'], label='Simulation', color='red')
        collider_params = data_i['collider_params']

        if i > 0:  # Remove first two lines of collider_params string
            collider_params = '\n'.join(collider_params.split('\n')[2:])
        ax.annotate(collider_params, xy=(0.02, 0.98), xycoords='axes fraction', fontsize=8, va='top', ha='left',
                    bbox=dict(facecolor='white', alpha=0.1))
    axs[0].legend()


    if out_dir is not None:
        out_path = f'{out_dir}{title.replace(" ","_").replace(":","").replace(",","").replace("__","_")}'
        fig_angles_horiz.savefig(f'{out_path}_horizontal_angles.pdf', format='pdf')
        fig_angles_horiz.savefig(f'{out_path}_horizontal_angles.png', format='png')
        fig_angles_vert.savefig(f'{out_path}_vertical_angles.pdf', format='pdf')
        fig_angles_vert.savefig(f'{out_path}_vertical_angles.png', format='png')
        fig_res.savefig(f'{out_path}_residuals.pdf', format='pdf')
        fig_res.savefig(f'{out_path}_residuals.png', format='png')
        fig_dists.savefig(f'{out_path}_dists.pdf', format='pdf')
        fig_dists.savefig(f'{out_path}_dists.png', format='png')
        plt.close(fig_angles_horiz)
        plt.close(fig_angles_vert)
        plt.close(fig_res)
        plt.close(fig_dists)


def update_plot_dict(plot_dict, bw_plot_dict):
    """
    Update plot_dict with the fit results from bw_plot_dict.
    :param plot_dict: Dictionary to update.
    :param bw_plot_dict: Dictionary of fit results for a single beam width.
    """
    plot_dict['residual_means'].append(np.mean(bw_plot_dict['residuals']))
    plot_dict['steps'].append(bw_plot_dict['steps'])
    plot_dict['residuals'].append(bw_plot_dict['residuals'])


def plot_plot_dict(plot_dict, title, out_dir=None):
    """
    Plot the residuals for each beam width.
    """
    fig, ax = plt.subplots()
    ax.plot(plot_dict['bws'], plot_dict['residual_means'], marker='o')
    ax.set_xlabel('Beam Width [µm]')
    ax.set_ylabel('Mean of Residuals for All Steps')
    ax.set_ylim(bottom=0)
    ax.set_title(title)
    fig.tight_layout()

    fig_res_vs_steps, ax_res_vs_steps = plt.subplots()
    for i, bw in enumerate(plot_dict['bws']):
        ax_res_vs_steps.plot(plot_dict['steps'][i], plot_dict['residuals'][i], label=f'{bw} µm')
    ax_res_vs_steps.set_xlabel('Step')
    ax_res_vs_steps.set_ylabel('Residual')
    ax_res_vs_steps.set_title(f'{title} Residuals vs Step')
    ax_res_vs_steps.set_ylim(bottom=0)
    ax_res_vs_steps.legend()
    fig_res_vs_steps.tight_layout()

    if out_dir is not None:
        out_path = f'{out_dir}{title.replace(" ","_").replace(":","").replace(",","").replace("__","_")}'
        fig.savefig(f'{out_path}_residuals_vs_bw.pdf', format='pdf')
        fig.savefig(f'{out_path}_residuals_vs_bw.png', format='png')
        fig_res_vs_steps.savefig(f'{out_path}_residuals_vs_steps.pdf', format='pdf')
        fig_res_vs_steps.savefig(f'{out_path}_residuals_vs_steps.png', format='png')
        plt.close(fig)
        plt.close(fig_res_vs_steps)

        # Save residual_means to a file
        res_df = pd.DataFrame({'Beam Width': plot_dict['bws'], 'Mean Residual': plot_dict['residual_means']})
        res_df.to_csv(f'{out_path}_residual_means.csv', index=False)


def print_status(bw, step, start_time, n_fits, total_fits):
    """
    Print the status of the fitting.
    :param bw: Beam width of the current fit.
    :param step: Step of the current fit.
    :param start_time: Time the fitting started.
    :param n_fits: Number of fits completed.
    :param total_fits: Total number of fits to complete.
    """
    time_elapsed = datetime.now() - start_time
    time_per_fit = time_elapsed / n_fits
    time_remaining = time_per_fit * (total_fits - n_fits)
    time_elapsed_str = str(time_elapsed).split('.')[0]  # Truncate the microseconds
    time_remaining_str = str(time_remaining).split('.')[0]
    print(f'BW {bw} step {step} complete, fit {n_fits}/{total_fits}, Time Elapsed: {time_elapsed_str}, Time Remaining: {time_remaining_str}')
    return n_fits + 1


def fit_sim_to_mbd_step(collider_sim, hist_data, cad_data, fit_amp_shift_flag=False, fit_crossing_angles=False):
    """
    Fit simulation crossing angles then amplitude/shift to the input z-vertex histogram data (hist_data).
    :param collider_sim: BunchCollider object to fit.
    :param hist_data: Dictionary of histogram data.
    :param cad_data: Dictionary of cad data.
    :param fit_amp_shift_flag: Flag to fit amplitude and shift.
    :param fit_crossing_angles: Flag to fit crossing angles.
    """

    scan_orientation = hist_data['scan_axis']
    step_cad_data = cad_data[(cad_data['orientation'] == scan_orientation) &
                             (cad_data['step'] == hist_data['scan_step'])].iloc[0]

    yellow_bunch_len_scaling = step_cad_data['yellow_bunch_length']
    blue_bunch_len_scaling = step_cad_data['blue_bunch_length']
    collider_sim.set_longitudinal_fit_scaling(blue_bunch_len_scaling, yellow_bunch_len_scaling)

    offset = step_cad_data['offset_set_val'] * 1e3  # mm to um
    if scan_orientation == 'Horizontal':
        collider_sim.set_bunch_offsets(np.array([offset, 0.]), np.array([0., 0.]))
    elif scan_orientation == 'Vertical':
        collider_sim.set_bunch_offsets(np.array([0., offset]), np.array([0., 0.]))

    blue_angle_x, yellow_angle_x = -step_cad_data['bh8_avg'] / 1e3, -step_cad_data['yh8_avg'] / 1e3  # mrad to rad
    if scan_orientation == 'Horizontal':
        collider_sim.set_bunch_crossing(blue_angle_x, 0., yellow_angle_x, 0.1e-3)
    elif scan_orientation == 'Vertical':
        collider_sim.set_bunch_crossing(blue_angle_x, 0., yellow_angle_x, 0.)

    print(f'Offset: {offset}, Blue Angle X: {blue_angle_x * 1e3:.3f} mrad, Yellow Angle X: {yellow_angle_x * 1e3:.3f} mrad')

    if fit_amp_shift_flag:
        collider_sim.set_amplitude(1.0)
        collider_sim.set_z_shift(0.0)

    collider_sim.run_sim_parallel()

    if fit_amp_shift_flag:
        fit_amp_shift(collider_sim, hist_data['counts'], hist_data['centers'])

    if fit_crossing_angles:  # Fit crossing angles in nested minimization.
        fit_crossing_angles_func(collider_sim, hist_data, scan_orientation)
        # fit_crossing_angles_2var(collider_sim, hist_data, scan_orientation)  # Do 2 var min after --> didn't change
        # fit_crossing_angle_single(collider_sim, hist_data, scan_orientation)

    collider_sim.run_sim_parallel()  # Run simulation with optimized angles
    fit_shift(collider_sim, hist_data['counts'], hist_data['centers'])


def fit_crossing_angles_func(collider_sim, hist_data, scan_orientation):
    """
    Outer minimizer for the crossing angles. Here iterate over the parallel yellow beam crossing angle.
    Keep parallel blue beam crossing angle fixed at CAD value. Iterate over perpendicular crossing angle in inner
    minimizer. Return the best parallel yellow and perpendicular crossing angles.
    :param collider_sim: BunchCollider object to fit.
    :param hist_data: Dictionary of histogram data.
    :param scan_orientation: 'Horizontal' or 'Vertical'.
    """
    xing_bounds = (-1.0e-3, 1.0e-3)
    res = minimize_scalar(fit_perp_crossing_angle, args=(collider_sim, hist_data, scan_orientation, xing_bounds),
                          bounds=xing_bounds, method='bounded')

    best_parallel_yellow_angle = res.x
    min_residual = res.fun

    set_xing_angles(collider_sim, scan_orientation, parallel_yellow=best_parallel_yellow_angle)

    # Perform a final minimization for the perpendicular crossing angle with the optimized parallel crossing angle.
    res = minimize_scalar(get_res_perp_xing,
                          args=(collider_sim, hist_data, scan_orientation),
                          bounds=xing_bounds, method='bounded')

    best_perp_yellow_angle = res.x
    min_residual2 = res.fun
    get_res_perp_xing(best_perp_yellow_angle, collider_sim, hist_data, scan_orientation)  # Final time to set

    print(f'Best Parallel Yellow Angle: {best_parallel_yellow_angle * 1e3:.3f} mrad, Best Perpendicular Yellow Angle: {best_perp_yellow_angle * 1e3:.3f} mrad')
    print(f'Min Residual: {min_residual:.3f} ({min_residual2:.3f})')  # Double check to make sure minimum is stable


def fit_crossing_angle_single(collider_sim, hist_data, scan_orientation):
    """
    Outer minimizer for the crossing angles. Here iterate over the parallel yellow beam crossing angle.
    Keep parallel blue beam crossing angle fixed at CAD value. Iterate over perpendicular crossing angle in inner
    minimizer. Return the best parallel yellow and perpendicular crossing angles.
    :param collider_sim: BunchCollider object to fit.
    :param hist_data: Dictionary of histogram data.
    :param scan_orientation: 'Horizontal' or 'Vertical'.
    """
    xing_bounds = (-0.5e-3, 0.5e-3)
    res = minimize_scalar(get_res_parallel_xing, args=(collider_sim, hist_data, scan_orientation),
                          bounds=xing_bounds, method='bounded')

    best_parallel_yellow_angle = res.x
    min_residual = res.fun

    print(f'Best Parallel Yellow Angle: {best_parallel_yellow_angle * 1e3:.3f} mrad')
    print(f'Min Residual: {min_residual:.3f}')  # Double check to make sure minimum is stable


def fit_crossing_angles_2var(collider_sim, hist_data, scan_orientation):
    """
    Fit both together.
    :param collider_sim: BunchCollider object to fit.
    :param hist_data: Dictionary of histogram data.
    :param scan_orientation: 'Horizontal' or 'Vertical'.
    """
    x0 = get_yellow_xing_angles(collider_sim, scan_orientation)
    res = minimize(get_res_perp_parallel_xings, args=(collider_sim, hist_data, scan_orientation),
                     x0=x0, method='Powell')

    best_parallel_yellow_angle, best_perp_yellow_angle = res.x
    min_residual = res.fun

    min_res2 = get_res_perp_parallel_xings(res.x, collider_sim, hist_data, scan_orientation)  # Final time to set

    print(f'Best Parallel Yellow Angle: {best_parallel_yellow_angle * 1e3:.3f} mrad, Best Perpendicular Yellow Angle: {best_perp_yellow_angle * 1e3:.3f} mrad')
    print(f'Min Residual: {min_residual:.3f} ({min_res2:.3f})')  # Double check to make sure minimum is stable


def fit_perp_crossing_angle(crossing_angle_parallel, collider_sim, hist_data, scan_orientation, xing_bounds):
    """
    Pass
    """
    set_xing_angles(collider_sim, scan_orientation, parallel_yellow=crossing_angle_parallel)

    res = minimize_scalar(get_res_perp_xing, args=(collider_sim, hist_data, scan_orientation),
                            bounds=xing_bounds, method='bounded')

    print(f'Parallel Yellow Angle: {crossing_angle_parallel * 1e3:.3f} mrad, Perp Angle: {abs(res.x) * 1e3:.3f} mrad, Residual: {res.fun:.3f}')

    return res.fun


def get_res_perp_xing(crossing_angle_perp, collider_sim, hist_data, scan_orientation):
    """
    Pass
    """
    set_xing_angles(collider_sim, scan_orientation, perp_yellow=crossing_angle_perp)

    collider_sim.run_sim_parallel()
    fit_shift(collider_sim, hist_data['counts'], hist_data['centers'])
    return get_residual(collider_sim, hist_data)


def get_res_parallel_xing(crossing_angle_parallel, collider_sim, hist_data, scan_orientation):
    """
    Pass
    """
    set_xing_angles(collider_sim, scan_orientation, parallel_yellow=crossing_angle_parallel)

    collider_sim.run_sim_parallel()
    fit_shift(collider_sim, hist_data['counts'], hist_data['centers'])
    return get_residual(collider_sim, hist_data)


def get_res_perp_parallel_xings(crossing_angles, collider_sim, hist_data, scan_orientation):
    """
    Pass
    """
    set_xing_angles(collider_sim, scan_orientation, parallel_yellow=crossing_angles[0], perp_yellow=crossing_angles[1])

    collider_sim.run_sim_parallel()
    fit_shift(collider_sim, hist_data['counts'], hist_data['centers'])
    return get_residual(collider_sim, hist_data)


def set_xing_angles(collider_sim, scan_orientation, parallel_yellow=None, perp_yellow=None):
    """
    Set crossing angles of collider_sim
    """
    xings = list(collider_sim.get_bunch_crossing_angles())
    if scan_orientation == 'Horizontal':
        if perp_yellow is not None:
            xings[3] = perp_yellow
        if parallel_yellow is not None:
            xings[2] = parallel_yellow
    elif scan_orientation == 'Vertical':
        if perp_yellow is not None:
            xings[2] = perp_yellow
        if parallel_yellow is not None:
            xings[3] = parallel_yellow
    collider_sim.set_bunch_crossing(*xings)


def get_yellow_xing_angles(collider_sim, scan_orientation):
    """
    Get crossing angles of collider_sim
    """
    xings = list(collider_sim.get_bunch_crossing_angles())
    if scan_orientation == 'Horizontal':
        return xings[2], xings[3]
    elif scan_orientation == 'Vertical':
        return xings[3], xings[2]


def get_residual(collider_sim, hist_data):
    sim_zs, sim_z_dist = collider_sim.get_z_density_dist()
    sim_interp = interp1d(sim_zs, sim_z_dist)
    return np.sqrt(np.sum((hist_data['counts'] - sim_interp(hist_data['centers'])) ** 2)) / np.mean(hist_data['counts'])


def fit_shift(collider_sim, z_dist_data, zs_data):
    sim_zs, sim_z_dist = collider_sim.get_z_density_dist()
    sim_interp = interp1d(sim_zs, sim_z_dist)
    upper_bound = min(zs_data) - min(sim_zs) - 0.1
    lower_bound = max(zs_data) - max(sim_zs) + 0.1
    res = minimize(shift_residual, np.array([collider_sim.z_shift]),
                   args=(sim_interp, z_dist_data, zs_data),
                   bounds=((lower_bound, upper_bound),))
    collider_sim.set_z_shift(collider_sim.z_shift - res.x[0] * 1e4)


def shift_residual(x, sim_interp, z_dist_data, zs_data):
    return np.sqrt(np.sum((z_dist_data - sim_interp(zs_data - x)) ** 2)) / np.mean(z_dist_data)


def fit_amp_shift(collider_sim, z_dist_data, zs_data):
    zs, z_dist = collider_sim.get_z_density_dist()
    scale = max(z_dist_data) / max(z_dist)
    z_max_sim = zs[np.argmax(z_dist)]
    z_max_hist = zs_data[np.argmax(z_dist_data)]
    shift = z_max_sim - z_max_hist  # microns

    res = minimize(amp_shift_residual, np.array([1.0, 0.0]),
                   args=(collider_sim, scale, shift, z_dist_data, zs_data),
                   bounds=((0.0, 2.0), (-10e4, 10e4)))
    scale = res.x[0] * scale
    shift = res.x[1] + shift

    collider_sim.set_amplitude(scale)
    collider_sim.set_z_shift(shift)


def amp_shift_residual(x, collider_sim, scale_0, shift_0, z_dist_data, zs_data):
    collider_sim.set_amplitude(x[0] * scale_0)
    collider_sim.set_z_shift(x[1] + shift_0)
    sim_zs, sim_z_dist = collider_sim.get_z_density_dist()
    sim_interp = interp1d(sim_zs, sim_z_dist)
    return np.sqrt(np.sum((z_dist_data - sim_interp(zs_data)) ** 2)) / np.mean(z_dist_data)


def plot_mbd_and_sim_dist(collider_sim, hist_data, title=None, out_dir=None):
    sim_zs, sim_z_dist = collider_sim.get_z_density_dist()
    collider_params = collider_sim.get_param_string()

    fig, ax = plt.subplots()
    ax.set_xlabel('Z Vertex [cm]')
    ax.set_ylabel('Counts')
    ax.set_title(title)

    hist_data['width'] = hist_data['centers'][1] - hist_data['centers'][0]
    ax.bar(hist_data['centers'], hist_data['counts'], width=hist_data['width'], label='MBD')
    ax.plot(sim_zs, sim_z_dist, label='Simulation', color='red')
    ax.annotate(collider_params, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=8, va='top', ha='left',
                   bbox=dict(facecolor='white', alpha=0.1))

    ax.legend(loc='upper left')
    fig.tight_layout()

    if out_dir is not None:
        out_path = f'{out_dir}{title.replace(" ", "_").replace(":", "").replace(",", "").replace("__", "_")}'
        plt.savefig(f'{out_path}.pdf', format='pdf')
        plt.savefig(f'{out_path}.png', format='png')
        plt.close()

    plot_data_dict = {'mbd_zs': hist_data['centers'], 'mbd_dist': hist_data['counts'], 'sim_zs': sim_zs,
                     'sim_dist': sim_z_dist, 'collider_params': collider_params}

    return plot_data_dict

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


if __name__ == '__main__':
    main()
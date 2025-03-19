#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on February 04 3:16 PM 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/phenix_style_hourglass_analysis.py

@author: Dylan Neff, Dylan
"""

import platform
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit as cf

from BunchCollider import BunchCollider
from Measure import Measure
from vernier_z_vertex_fitting import read_cad_measurement_file, get_cw_rates, get_mbd_z_dists


def main():
    if platform.system() == 'Linux':
        base_path = '/local/home/dn277127/Bureau/vernier_scan/'
        # base_path = '/home/dylan/Desktop/vernier_scan/'
    else:
        base_path = 'C:/Users/Dylan/Desktop/vernier_scan/'
    # simulate_vernier_scan(base_path)
    plot_distorted_width_vs_bw(base_path)
    print('donzo')


def plot_distorted_width_vs_bw(base_path):
    vernier_scan_date = 'Aug12'
    scan_orientation = 'Horizontal'
    out_csv_path = f'{base_path}simulated_phenix_scan/'

    fit_func = vernier_scan_fit_bkg
    # fit_func = vernier_scan_fit_nobkg
    n_params = fit_func.__code__.co_argcount - 1  # Get number of parameters in fit function
    fit_pars_dict = {'p0': [1, 150, 0], 'names': ['Amp', 'Sigma', 'x0']}
    if n_params == 4:
        fit_pars_dict['p0'].append(0)
        fit_pars_dict['names'].append('b')

    bws, distorted_bws, distorted_bw_errs, correction_percents, true_bw_lumi_factor = [], [], [], [], []
    for file_name in os.listdir(out_csv_path):
        if 'first_simple' in file_name:
            if '_bw' in file_name:
                bw = int(file_name.split('_')[-2])
            else:
                bw = 160
            file_name_second = file_name.replace('_first_simple', '_second_simple')

            max_lumi_real, max_lumi_gaus, max_lumi_hg_off_bw_true = None, None, None
            beta_stars = ['On', 'Off']
            for i, file_name_i in enumerate([file_name, file_name_second]):
                for beta_star_on_off in beta_stars:
                    if i == 1 and beta_star_on_off == 'On':
                        continue
                    data_df = pd.read_csv(f'{out_csv_path}{file_name_i}')
                    offsets = np.array(data_df['Offsets'])
                    # on_off = 'On' if i == 0 else 'Off'
                    # lumis = np.array(data_df[on_off])
                    lumis = np.array(data_df[beta_star_on_off])
                    p0 = fit_pars_dict['p0']
                    p0[0] = max(lumis)
                    p0[1] = bw
                    popt, pcov = cf(fit_func, offsets, lumis, p0=p0)
                    popt[1] = abs(popt[1])  # Ensure sigma is positive
                    perr = np.sqrt(np.diag(pcov))
                    # For perrs that are inf, set to 0. Where abs(err/val) < 1e-6, set to 0
                    perr[np.isinf(perr)] = 0
                    perr[np.abs(perr / popt) < 1e-6] = 0
                    pmeas = [Measure(popt[i], perr[i]) if perr[i] > 0 else popt[i] for i in range(len(popt))]
                    if i == 0 and beta_star_on_off == 'On':
                        bws.append(bw)
                        distorted_bws.append(popt[1])
                        distorted_bw_errs.append(perr[1])
                        max_lumi_real = max(lumis)
                    elif i == 0 and beta_star_on_off == 'Off':
                        max_lumi_hg_off_bw_true = max(lumis)
                    elif i == 1 and beta_star_on_off == 'Off':
                        max_lumi_gaus = max(lumis)
            correction_percent = (max_lumi_real - max_lumi_gaus) / max_lumi_gaus
            correction_percents.append(correction_percent)
            print(f'BW_true: {bw} Max_lumi_real: {max_lumi_real}, max_lumi_gaus: {max_lumi_gaus}, correction: {correction_percent}')
            true_bw_lumi_factor.append(max_lumi_real / max_lumi_hg_off_bw_true)

    # Sort bws, distorted bws, and correction_percents by bw
    bws, distorted_bws, correction_percents = zip(*sorted(zip(bws, distorted_bws, correction_percents)))

    popt, pcov = cf(lin, bws, distorted_bws, p0=[1.1])

    fig, ax = plt.subplots()
    ax.errorbar(bws, distorted_bws, yerr=distorted_bw_errs, marker='o', ls='none')
    ax.plot(bws, lin(np.array(bws), *popt), color='r', ls='--')
    ax.set_xlabel('True Beam Width [μm]')
    ax.set_ylabel('Measured (Distorted) Beam Width [μm]')
    ax.grid()
    fig.tight_layout()

    fig_dev, ax_dev = plt.subplots()
    ax_dev.errorbar(bws, distorted_bws - lin(np.array(bws), *popt), yerr=distorted_bw_errs, marker='o', ls='none')
    ax_dev.axhline(0, color='black', zorder=0)
    ax_dev.set_xlabel('True Beam Width [μm]')
    ax_dev.set_ylabel('Measured (Distorted) Beam Width [μm]')
    # ax_dev.grid()
    fig_dev.tight_layout()

    fig_cor_per, ax_cor_per = plt.subplots()
    ax_cor_per.scatter(bws, correction_percents, zorder=10)
    ax_cor_per.axhline(0, color='black', zorder=0)
    ax_cor_per.set_xlabel('True Beam Width [μm]')
    ax_cor_per.set_ylabel('Correction Percentage')
    ax_cor_per.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    fig_cor_per.tight_layout()

    fig_true_bw_lumi_scale, ax_true_bw_lumi_scale = plt.subplots()
    ax_true_bw_lumi_scale.scatter(bws, true_bw_lumi_factor, zorder=10)
    ax_true_bw_lumi_scale.set_xlabel('True Beam Width [μm]')
    ax_true_bw_lumi_scale.set_ylabel('Hourglass On Lumi / Hourglass Off Lumi')
    ax_true_bw_lumi_scale.axhline(1, color='black', zorder=1)
    fig_true_bw_lumi_scale.tight_layout()

    plt.show()


def lin(x, m):
    return x * m


def simulate_vernier_scan(base_path):
    run_scan = False  # If, false, read in data from file

    # If simple just use beta star and bunch width, if realistic use all realistic parameters,
    # if crossing use crossing angles with realistic beam width and beta star
    realistic = 'simple'
    # realistic = 'crossing'
    # realistic = 'realistic'
    # realistic = 'very_realistic'
    # file_post = ''
    file_post = '_70_bw'

    if realistic == 'very_realistic':
        bunch_width_truth = np.array([161.79, 157.08])  # microns Transverse Gaussian bunch width
        beta_star_actual = np.array([97., 82., 88., 95.])  # cm
    else:
        # bunch_width_truth = 160.  # microns Transverse Gaussian bunch width
        bunch_width_truth = 70.  # microns Transverse Gaussian bunch width
        beta_star_actual = 90.  # cm
    beta_star_off = 9e11  # cm Effectively turned off

    mbd_resolution_real = 2.0  # cm MBD resolution
    mbd_resolution_simple = 0.0  # cm MBD resolution
    gauss_eff_width_real = 500  # cm Gaussian efficiency width
    gauss_eff_width_simple = None  # cm Gaussian efficiency width
    bkg_real = 0.0  # Background level
    bkg_simple = 0.0  # Background level
    z_bunch_len_simple = 1.1e6  # mm Bunch length in z direction

    fit_func = vernier_scan_fit_bkg
    # fit_func = vernier_scan_fit_nobkg
    n_params = fit_func.__code__.co_argcount - 1  # Get number of parameters in fit function
    fit_pars_dict = {'p0': [1, 150, 0], 'names': ['Amp', 'Sigma', 'x0']}
    if n_params == 4:
        fit_pars_dict['p0'].append(0)
        fit_pars_dict['names'].append('b')

    vernier_scan_date = 'Aug12'
    scan_orientation = 'Horizontal'
    cad_measurement_path = f'{base_path}CAD_Measurements/VernierScan_{vernier_scan_date}_combined.dat'
    longitudinal_fit_path = f'{base_path}CAD_Measurements/VernierScan_{vernier_scan_date}_COLOR_longitudinal_fit.dat'
    out_csv_path = f'{base_path}simulated_phenix_scan/VernierScan_{vernier_scan_date}_{scan_orientation}'

    if run_scan:
        collider_sim = BunchCollider()
        collider_sim.set_bunch_rs(np.array([0., 0., -6.e6]), np.array([0., 0., +6.e6]))
        if realistic == 'very_realistic' or realistic == 'realistic':
            collider_sim.set_gaus_smearing_sigma(mbd_resolution_real)
            collider_sim.set_gaus_z_efficiency_width(gauss_eff_width_real)
            collider_sim.set_bkg(bkg_real)
        else:
            collider_sim.set_gaus_smearing_sigma(mbd_resolution_simple)
            collider_sim.set_gaus_z_efficiency_width(gauss_eff_width_simple)
            collider_sim.set_bkg(bkg_simple)
        if realistic == 'very_realistic' or realistic == 'realistic' or realistic == 'crossing':
            blue_fit_path = longitudinal_fit_path.replace('_COLOR_', '_blue_')
            yellow_fit_path = longitudinal_fit_path.replace('_COLOR_', '_yellow_')
            collider_sim.set_longitudinal_fit_parameters_from_file(blue_fit_path, yellow_fit_path)

    beta_stars = {'Off': beta_star_off, 'On': beta_star_actual}

    bunch_widths = [bunch_width_truth, None]   # To be populated with measured bunch width of iteration
    for bunch_width in bunch_widths:
        if run_scan:
            print(f'Starting Scan with Bunch Width: {bunch_width}')
            if type(bunch_width) == np.ndarray:
                collider_sim.set_bunch_sigmas(np.array([bunch_width[0], bunch_width[1]]), np.array([bunch_width[0], bunch_width[1]]))
            else:
                collider_sim.set_bunch_sigmas(np.array([bunch_width, bunch_width]), np.array([bunch_width, bunch_width]))

            if realistic == 'simple':
                collider_sim.set_bunch_sigmas(np.array([bunch_width, bunch_width, z_bunch_len_simple]),
                                              np.array([bunch_width, bunch_width, z_bunch_len_simple]))

            cad_data = read_cad_measurement_file(cad_measurement_path)
            cad_data_orientation = cad_data[cad_data['orientation'] == scan_orientation]

            data_dict = {'Offsets': []}
            data_dict.update({beta_star_name: [] for beta_star_name in beta_stars.keys()})
            for row_i, step_cad_data in cad_data_orientation.iterrows():
                print(f'Starting Step {row_i + 1} of {len(cad_data_orientation)}')
                offset = step_cad_data['offset_set_val'] * 1e3  # mm to um
                if scan_orientation == 'Horizontal':
                    collider_sim.set_bunch_offsets(np.array([offset, 0.]), np.array([0., 0.]))
                elif scan_orientation == 'Vertical':
                    collider_sim.set_bunch_offsets(np.array([0., offset]), np.array([0., 0.]))

                if realistic == 'very_realistic' or realistic == 'realistic' or realistic == 'crossing':
                    blue_angle_x, yellow_angle_x = -step_cad_data['bh8_avg'] / 1e3, -step_cad_data['yh8_avg'] / 1e3
                else:
                    blue_angle_x, yellow_angle_x = 0., 0.
                collider_sim.set_bunch_crossing(blue_angle_x, 0., yellow_angle_x, 0.0)
                print(f'Offset: {offset}, Blue Angle X: {blue_angle_x * 1e3:.3f} mrad, Yellow Angle X: {yellow_angle_x * 1e3:.3f} mrad')

                if realistic == 'very_realistic' or realistic == 'realistic':
                    yellow_bunch_len_scaling = step_cad_data['yellow_bunch_length_scaling']
                    blue_bunch_len_scaling = step_cad_data['blue_bunch_length_scaling']
                    collider_sim.set_longitudinal_fit_scaling(blue_bunch_len_scaling, yellow_bunch_len_scaling)
                # elif realistic == 'simple':
                #     collider_sim.set_longitudinal_fit_scaling(0., 0.)

                data_dict['Offsets'].append(offset)
                for beta_star_name, beta_star in beta_stars.items():
                    if type(beta_star) is np.ndarray:
                        collider_sim.set_bunch_beta_stars(*beta_star)
                    else:
                        collider_sim.set_bunch_beta_stars(beta_star, beta_star)
                    collider_sim.run_sim_parallel()
                    lumi = collider_sim.get_naked_luminosity()
                    print(f'Beta star {beta_star_name} Luminosity: {lumi:.3e}')
                    data_dict[beta_star_name].append(lumi)

            # Write data_dict to file
            data_dict['Offsets'] = np.array(data_dict['Offsets'])
            for beta_star_name in beta_stars.keys():
                data_dict[beta_star_name] = np.array(data_dict[beta_star_name])
            data_df_first = pd.DataFrame(data_dict)

            out_sufx = 'second'
            if bunch_widths[-1] is None:  # Get bunch width from first iteration's beta-star on width measurement
                lumis = np.array(data_df_first['On'])  # Fit to vernier_scan fit
                p0 = fit_pars_dict['p0']
                p0[0] = max(lumis)
                if type(bunch_widths[-1]) is np.ndarray:
                    if scan_orientation == 'Horizontal':
                        p0[1] = bunch_width[0]
                    else:
                        p0[1] = bunch_width[1]
                else:
                    p0[1] = bunch_width
                popt, pcov = cf(fit_func, data_df_first['Offsets'], lumis, p0=p0)
                if type(bunch_width) == np.ndarray:
                    bunch_widths[-1] = bunch_width
                    if scan_orientation == 'Horizontal':
                        bunch_widths[-1][0] = abs(popt[1])  # Ensure sigma is positive
                    else:
                        bunch_widths[-1][1] = abs(popt[1])  # Ensure sigma is positive
                else:
                    bunch_widths[-1] = abs(popt[1])  # Ensure sigma is positive

                with open(f'{out_csv_path}_first_measured_bunch_width_{realistic}{file_post}.txt', 'w') as file:
                    file.write(f'{abs(popt[1])}')
                out_sufx = 'first'

            data_df_first.to_csv(f'{out_csv_path}_{out_sufx}_{realistic}{file_post}.csv',
                                 index=False)

    data_df_first = pd.read_csv(f'{out_csv_path}_first_{realistic}{file_post}.csv')
    data_df_second = pd.read_csv(f'{out_csv_path}_second_{realistic}{file_post}.csv')
    with open(f'{out_csv_path}_first_measured_bunch_width_{realistic}{file_post}.txt', 'r') as file:
        measured_bunch_width = float(file.read())
    if type(bunch_width_truth) == np.ndarray:
        bunch_widths[-1] = bunch_width_truth.copy()
        if scan_orientation == 'Horizontal':
            bunch_widths[-1][0] = measured_bunch_width
        else:
            bunch_widths[-1][1] = measured_bunch_width
    else:
        bunch_widths[-1] = measured_bunch_width

    fig_together, ax_together = plt.subplots(figsize=(8, 4))
    # fig_2col, ax_grid = plt.subplots(2, 2, figsize=(10, 4), sharey='all', sharex='all')
    fig_3row, ax_3row = plt.subplots(3, 1, figsize=(8, 7), sharey='all', sharex='all')

    print(f'Bunch widths: {bunch_widths}')
    both_fit_results = []
    for j, data_df in enumerate([data_df_first, data_df_second]):
        # ax_2col = ax_grid[j]
        offsets = np.array(data_df['Offsets'])
        fit_results = {beta_star_name: {} for beta_star_name in beta_stars.keys()}
        for i, beta_star_name in enumerate(beta_stars.keys()):
            lumis = np.array(data_df[beta_star_name])  # Fit to vernier_scan fit

            orient_bw = bunch_widths[j]
            if type(orient_bw) == np.ndarray:
                orient_bw = orient_bw[0] if scan_orientation == 'Horizontal' else orient_bw[1]

            p0 = fit_pars_dict['p0']
            p0[0] = max(lumis)
            p0[1] = orient_bw
            popt, pcov = cf(fit_func, offsets, lumis, p0=p0)
            popt[1] = abs(popt[1])  # Ensure sigma is positive
            perr = np.sqrt(np.diag(pcov))
            # For perrs that are inf, set to 0. Where abs(err/val) < 1e-6, set to 0
            perr[np.isinf(perr)] = 0
            perr[np.abs(perr / popt) < 1e-6] = 0
            pmeas = [Measure(popt[i], perr[i]) if perr[i] > 0 else popt[i] for i in range(len(popt))]
            for fit_par_i, fit_par_name in enumerate(fit_pars_dict['names']):
                fit_results[beta_star_name][fit_par_name] = pmeas[fit_par_i]

            x_fit = np.linspace(min(offsets), max(offsets), 1000)

            fig, ax = plt.subplots()
            ax.scatter(offsets, lumis, label=beta_star_name)
            ax.plot(x_fit, fit_func(x_fit, *popt), color='red', label='Fit')
            fit_str = f'σ={orient_bw:.0f} μm Hourglass {beta_star_name}\n'
            for fit_par_name, fit_par_meas in zip(fit_pars_dict['names'], pmeas):
                if fit_par_name == 'Amp' or fit_par_name == 'b':
                    fit_par_meas = fit_par_meas * 1e6
                fit_unit = ''
                if fit_par_name == 'Amp' or fit_par_name == 'b':
                    fit_unit = r'mm$^{-2}$'
                elif fit_par_name == 'Sigma' or fit_par_name == 'x0':
                    fit_unit = 'μm'
                if type(fit_par_meas) == Measure:
                    print(fit_par_name, fit_par_meas)
                    fit_str += f'{fit_par_name}: {fit_par_meas} {fit_unit}\n'
                else:
                    # fit_str += f'{fit_par_name}: {fit_par_meas:.3e} {fit_unit}\n'
                    fit_str += f'{fit_par_name}: {fit_par_meas:.5} {fit_unit}\n'

            fit_str = fit_str.strip()
            ax.annotate(fit_str, (0.05, 0.95), xycoords='axes fraction', ha='left', va='top', fontsize=12,
                        bbox=dict(facecolor='wheat', alpha=0.5, boxstyle='round'))
            ax.set_ylim(bottom=0)
            ax.legend()
            ax.set_xlabel('Offset [μm]')
            ax.set_ylabel('Naked Luminosity')
            ax.set_title(f'Vernier Scan Hourglass {beta_star_name}')
            fig.tight_layout()

            line, = ax_together.plot(offsets, lumis, label=f'{beta_star_name} Data', marker='o', ls='none', zorder=10)
            color = line.get_color()
            ax_together.plot(x_fit, fit_func(x_fit, *popt), color=color, zorder=5, label=f'{beta_star_name} Fit')
            ax_together.annotate(fit_str, (0.02 + j * 0.3, 0.97 - i * 0.35), xycoords='axes fraction', ha='left', va='top',
                                 fontsize=11, bbox=dict(facecolor=color, alpha=0.2, boxstyle='round'))

            if j * 2 + i < 3:
                ax_3row[j * 2 + i].plot(offsets, lumis * 1e6, color=color, label=f'Simulated Data', marker='o', ls='none', zorder=10)
                ax_3row[j * 2 + i].plot(x_fit, fit_func(x_fit, *popt) * 1e6, color=color, label=f'Fit', zorder=5)
                ax_3row[j * 2 + i].annotate(fit_str, (0.03, 0.9), xycoords='axes fraction', ha='left', va='top',
                                            fontsize=10, bbox=dict(facecolor=color, alpha=0.2, boxstyle='round', pad=0.5))
                if j * 2 + i == 2:
                    ax_3row[j * 2 + i].set_xlabel('Offset [μm]')
                ax_3row[j * 2 + i].set_ylabel(r'Naked Luminosity [mm$^{-2}$]')
                ax_3row[j * 2 + i].legend()
                max_naked_lumi_str = r'Maximum $\mathbf{L}_{naked}= ' + str(round(max(lumis) * 1e6, 3)) + '$'
                ax_3row[j * 2 + i].annotate(max_naked_lumi_str, (0.03, 0.3), xycoords='axes fraction', ha='left', va='top',
                                            fontsize=10, bbox=dict(facecolor=color, alpha=0.05, boxstyle='round', pad=0.5))
                fit_eq_str = r'$\mathbf{L} = \mathbf{A} \exp\left(-\frac{(\mathbf{x} - \mathbf{x_0})^2}{4 \mathbf{\sigma}^2}\right) + \mathbf{B}$'

            # ax_2col[i].plot(offsets, lumis, color=color, label=f'Data', marker='o', ls='none', zorder=10)
            # ax_2col[i].plot(x_fit, fit_func(x_fit, *popt), color=color, label=f'Fit', zorder=5)
            # ax_2col[i].annotate(fit_str, (0.02, 0.96), xycoords='axes fraction', ha='left', va='top', fontsize=9,
            #                     bbox=dict(facecolor=color, alpha=0.2, boxstyle='round'))
            # ax_2col[i].set_xlabel('Offset [μm]')
            # if i == 0:
            #     ax_2col[i].set_ylabel('Naked Luminosity')
            # ax_2col[i].legend()

        print(f'\nBeam Width: {bunch_widths[j]}')
        # Neglecting hourglass overestimates luminosity. Need to multiply by hourglass on / off, smaller than 1
        beta_star_amp_ratio = fit_results['On']['Amp'] / fit_results['Off']['Amp']  # On / Off

        # Neglecting hourglass overestimates σ, therefore underestimates lumi. Need to multiply by (on / off)**2, > 1
        beta_star_sigma_ratio = fit_results['On']['Sigma'] / fit_results['Off']['Sigma']  # On / Off

        lumi_correction = beta_star_amp_ratio * beta_star_sigma_ratio ** 2

        print(f'Ratio of beta star amps: {beta_star_amp_ratio}')
        print(f'Ratio of beta star sigmas: {beta_star_sigma_ratio}')
        print(f'Luminosity Correction: {lumi_correction}')
        print(f'Percent Correction: {(lumi_correction - 1) * 100}%')

        print(f'Max luminosity Hourglass On: {fit_results["On"]["Amp"]}')
        print(f'Max luminosity Hourglass Off: {fit_results["Off"]["Amp"]}')
        print(f'Max luminosity Hourglass Corrected: {fit_results["Off"]["Amp"] * lumi_correction}')

        both_fit_results.append(fit_results)

    true_lumi = both_fit_results[0]['On']['Amp']
    uncorrected_lumi = both_fit_results[1]['Off']['Amp']
    amp_correction = both_fit_results[0]['On']['Amp'] / both_fit_results[0]['Off']['Amp']
    sigma_correction = both_fit_results[0]['On']['Sigma'] / both_fit_results[0]['Off']['Sigma']
    corrected_lumi = uncorrected_lumi * amp_correction * sigma_correction**2

    print(f'True Luminosity: {true_lumi}')
    print(f'Uncorrected Luminosity: {uncorrected_lumi}')
    print(f'Amp Correction: {amp_correction}')
    print(f'Sigma Correction: {sigma_correction}')
    print(f'True Luminosity: {true_lumi}')
    print(f'Corrected Luminosity: {corrected_lumi}')
    print(f'Percent Off: {(corrected_lumi - true_lumi) / true_lumi * 100}%')
    print(f'Percent Correction: {(corrected_lumi - uncorrected_lumi) / uncorrected_lumi * 100}%')

    ax_together.set_ylim(bottom=0)
    ax_together.set_xlabel('Offset [μm]')
    ax_together.set_ylabel('Naked Luminosity')
    ax_together.legend()
    fig_together.tight_layout()

    ax_3row[0].set_ylim(bottom=0)
    fig_3row.subplots_adjust(wspace=0, hspace=0, top=0.995, bottom=0.065, left=0.075, right=0.995)
    # ax_2col[0].set_ylim(bottom=0)
    # fig_2col.tight_layout()
    # fig_2col.subplots_adjust(wspace=0.0, hspace=0.0, top=0.96, bottom=0.11, left=0.04, right=0.995)

    plt.show()


def vernier_scan_fit_bkg(x, a, sigma, x0, b):
    return a * np.exp(-(x - x0) ** 2 / (4 * sigma ** 2)) + b


def vernier_scan_fit_nobkg(x, a, sigma, x0):
    return a * np.exp(-(x - x0) ** 2 / (4 * sigma ** 2))


if __name__ == '__main__':
    main()

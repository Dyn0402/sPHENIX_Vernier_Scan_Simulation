#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on August 29 11:01 AM 2024
Created in PyCharm
Created as sphenix_polarimetry/plot_cad_measurements.py

@author: Dylan Neff, Dylan
"""

import numpy as np
from scipy.optimize import curve_fit as cf
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import pytz
import gzip

from Measure import Measure


def main():
    vernier_scan_date = 'Aug12'
    # vernier_scan_date = 'Jul11'
    # cad_measurements_path = 'C:/Users/Dylan/Desktop/vernier_scan/CAD_Measurements/'
    cad_measurements_path = '/local/home/dn277127/Bureau/vernier_scan_AuAu24/CAD_Measurements/'
    # crossing_angle(cad_measurements_path, vernier_scan_date)
    # bunch_length(cad_measurements_path, vernier_scan_date)
    # beam_offset_and_intensity(cad_measurements_path, vernier_scan_date)
    plot_beam_longitudinal_measurements(cad_measurements_path, vernier_scan_date, False)
    # plot_individual_beam_longitudinal_measurements(cad_measurements_path, vernier_scan_date, True)
    # combine_cad_measurements(cad_measurements_path, vernier_scan_date, cw_rate_path)
    plt.show()
    print('donzo')


def plot_beam_longitudinal_measurements(cad_measurements_path, vernier_scan_date, write_out=False):
    beam_colors = ['blue', 'yellow']
    plot_colors = ['blue', 'orange']

    p0s_quad = {'blue': [52.3, 1.16, 2.07, 48.8, 2.5, 3.6, 54.5, 3.5, 0.3, 52.3, 20],
                'yellow': [51.7, 2, 0.235, 47.335, 1.18, 0.235, 56.15, 1.25, 0.1, 51.7, 20]}
    bnds_quad = ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [100, 10, 10, 100, 10, 10, 100, 10, 100, 100, 100])

    min_time, max_time = 30, 75
    min_val = 1.5
    max_pdf_val = 0.125 if vernier_scan_date == 'Aug12' else 0.1

    plot_weird_ones = False

    fig, ax = plt.subplots(figsize=(12, 6))
    fig2, ax2 = plt.subplots(figsize=(12, 6))

    for beam_color, plot_color in zip(beam_colors, plot_colors):
        file_path = f'{cad_measurements_path}VernierScan_{vernier_scan_date}_{beam_color}_longitudinal.dat.gz'
        fit_out_path = f'{cad_measurements_path}VernierScan_{vernier_scan_date}_{beam_color}_longitudinal_fit.dat'
        fit_plots_out_path = (f'{cad_measurements_path}Longitudinal_Bunch_Profile_Fits/'
                              f'{vernier_scan_date}_{beam_color}_longitudinal_fit_plots.png')
        with gzip.open(file_path, 'rt') as file:
            file_content = file.read()
        lines = file_content.split('\n')
        times, values = [[]], [[]]
        for line in lines[1:]:
            if line == '':
                continue
            columns = line.split('\t')
            time, value = float(columns[0]), float(columns[1])
            if len(times[-1]) > 0 and time < times[-1][-1]:
                times.append([])
                values.append([])
            times[-1].append(time)
            values[-1].append(value)

        # For times less than 30 ns and greater than 75 ns set to 0
        for bunch_i, (bunch_times, bunch_vals) in enumerate(zip(times, values)):
            bunch_vals = np.array(bunch_vals)
            bunch_vals[(np.array(bunch_times) < min_time) | (np.array(bunch_times) > max_time)] = 0
            values[bunch_i] = bunch_vals

        full_time = 0
        for bunch_i, (bunch_times, bunch_vals) in enumerate(zip(times, values)):
            bunch_vals = np.array(bunch_vals)
            max_bunch_val = np.max(bunch_vals)
            times_increasing = np.array(bunch_times)
            times_increasing += full_time
            full_time += bunch_times[-1]
            if max_bunch_val > min_val:
                ax.plot(bunch_times, bunch_vals / np.max(bunch_vals), color=plot_color)
            ax2.plot(times_increasing, bunch_vals, color=plot_color)

        # Fit all the bunches superimposed
        fit_times, fit_vals, weird_one_times, weird_one_vals, weird_ones = [], [], [], [], 0
        for bunch_i, (bunch_times, bunch_vals) in enumerate(zip(times, values)):
            if np.max(bunch_vals) < min_val:
                continue
            bin_width = bunch_times[1] - bunch_times[0]
            vals = list(np.array(bunch_vals) / np.sum(bunch_vals) / bin_width)
            if beam_color == 'yellow' and np.max(vals) < max_pdf_val:
                weird_one_times.extend(bunch_times)
                weird_one_vals.extend(vals)
                weird_ones += 1
            else:
                fit_times.extend(bunch_times)
                fit_vals.extend(vals)

        print(f'{beam_color} {weird_ones} weird ones of {len(times)} bunches')

        popt, pcov = cf(quad_gaus_pdf, fit_times, fit_vals, p0=p0s_quad[beam_color], bounds=bnds_quad)
        perr = np.sqrt(np.diag(pcov))
        pmeas = [Measure(p, e) for p, e in zip(popt, perr)]
        fit_str = [rf'$\mu_1$ = {pmeas[0]} ns', rf'$\sigma_1$ = {pmeas[1]} ns',
                   rf'$\mu_2$ = {pmeas[3]} ns', rf'$\sigma_2$ = {pmeas[4]} ns', rf'$a_2$ = {pmeas[2]}',
                   rf'$\mu_3$ = {pmeas[6]} ns', rf'$\sigma_3$ = {pmeas[7]} ns', rf'$a_3$ = {pmeas[5]}',
                   rf'$\mu_4$ = {pmeas[9]} ns', rf'$\sigma_4$ = {pmeas[10]} ns', rf'$a_4$ = {pmeas[8]}']
        fit_str = '\n'.join(fit_str)

        # Write out quad gaussian fit equation
        fit_eq = (r'$p(t) = \frac{\frac{1}{\sigma_1 \sqrt{2 \pi}} \exp\left(-\frac{(t - \mu_1)^2}{2\sigma_1^2}\right)'
                  r'+ \frac{a_2}{\sigma_2 \sqrt{2 \pi}} \exp\left(-\frac{(t - \mu_2)^2}{2\sigma_2^2}\right)'
                  r'+ \frac{a_3}{\sigma_3 \sqrt{2 \pi}} \exp\left(-\frac{(t - \mu_3)^2}{2\sigma_3^2}\right)'
                  r'+ \frac{a_4}{\sigma_4 \sqrt{2 \pi}} \exp\left(-\frac{(t - \mu_4)^2}{2\sigma_4^2}\right)}'
                  r'{1 + a_2 + a_3 + a_4}$')

        fig_all, ax_all = plt.subplots(figsize=(8, 6))
        x_plot = np.linspace(fit_times[0], fit_times[-1], 1000)
        ax_all.plot(fit_times, fit_vals, color=plot_color, alpha=0.7, label='CAD Profiles')
        if len(weird_one_times) > 0 and plot_weird_ones:
            ax_all.plot(weird_one_times, weird_one_vals, color='purple', alpha=0.7, label='Weird Ones')
        ax_all.plot(x_plot, quad_gaus_pdf(x_plot, *popt), color='red', ls='-', label='Fit')
        ax_all.axvline(popt[0], color='green', ls='--', alpha=0.5, zorder=0)
        ax_all.set_xlabel('Time (ns)')
        ax_all.set_ylabel('Probability Density')
        ax_all.set_title(f'{vernier_scan_date} Vernier Scan {beam_color.capitalize()} Beam Longitudinal Bunch Density')
        ax_all.annotate(fit_str, (0.01, 0.98), xycoords='axes fraction', ha='left', va='top',
                        bbox=dict(boxstyle='round', fc='white', alpha=0.8), fontsize=12)
        ax_all.annotate(fit_eq, (0.02, 0.02), xycoords='axes fraction', ha='left', va='bottom',
                        bbox=dict(boxstyle='round', fc='white', alpha=1.0), fontsize=14.5)
        ax_all.set_ylim(-0.029, 0.145)
        ax_all.set_xlim(min_time, max_time)
        ax_all.legend(loc='upper right', fontsize=14)
        ax_all.grid(True)
        fig_all.tight_layout()

        if write_out:  # Write out fit parameters
            fig_all.savefig(fit_plots_out_path)
            write_longitudinal_beam_profile_fit_parameters(fit_out_path, beam_color, fit_eq, pmeas)

    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Value')
    ax.set_title('Longitudinal Beam Measurements vs Time')
    ax.grid(True)

    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('Value')
    ax2.set_title('Longitudinal Beam Measurements vs Index')
    ax2.grid(True)

    fig.tight_layout()
    fig2.tight_layout()


def plot_individual_beam_longitudinal_measurements(cad_measurements_path, vernier_scan_date, write_out=False):
    beam_colors = ['blue', 'yellow']
    plot_colors = ['blue', 'orange']

    p0s_quad = {'blue': [52.3, 1.16, 2.07, 48.8, 2.5, 3.6, 54.5, 3.5, 0.3, 52.3, 20],
                'yellow': [51.7, 2, 0.235, 47.335, 1.18, 0.235, 56.15, 1.25, 0.1, 51.7, 20]}
    bnds_quad = ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [100, 10, 10, 100, 10, 10, 100, 10, 100, 100, 100])

    min_time, max_time = 30, 75
    min_val = 1.5

    fig, ax = plt.subplots(figsize=(12, 6))
    fig2, ax2 = plt.subplots(figsize=(12, 6))

    for beam_color, plot_color in zip(beam_colors, plot_colors):
        file_path = f'{cad_measurements_path}VernierScan_{vernier_scan_date}_{beam_color}_longitudinal.dat.gz'
        with gzip.open(file_path, 'rt') as file:
            file_content = file.read()
        lines = file_content.split('\n')
        times, values = [[]], [[]]
        for line in lines[1:]:
            if line == '':
                continue
            columns = line.split('\t')
            time, value = float(columns[0]), float(columns[1])
            if len(times[-1]) > 0 and time < times[-1][-1]:
                times.append([])
                values.append([])
            times[-1].append(time)
            values[-1].append(value)

        # For times less than 30 ns and greater than 75 ns set to 0
        for bunch_i, (bunch_times, bunch_vals) in enumerate(zip(times, values)):
            bunch_vals = np.array(bunch_vals)
            bunch_vals[(np.array(bunch_times) < min_time) | (np.array(bunch_times) > max_time)] = 0
            values[bunch_i] = bunch_vals

        full_time = 0
        for bunch_i, (bunch_times, bunch_vals) in enumerate(zip(times, values)):
            bunch_vals = np.array(bunch_vals)
            max_bunch_val = np.max(bunch_vals)
            times_increasing = np.array(bunch_times)
            times_increasing += full_time
            full_time += bunch_times[-1]
            if max_bunch_val > min_val:
                ax.plot(bunch_times, bunch_vals / np.max(bunch_vals), color=plot_color)
            ax2.plot(times_increasing, bunch_vals, color=plot_color)

            # Fit the individual bunch
            if np.max(bunch_vals) >= min_val:
                bin_width = bunch_times[1] - bunch_times[0]
                vals = list(np.array(bunch_vals) / np.sum(bunch_vals) / bin_width)

                try:
                    popt, pcov = cf(quad_gaus_pdf, bunch_times, vals, p0=p0s_quad[beam_color], bounds=bnds_quad)
                    perr = np.sqrt(np.diag(pcov))
                    pmeas = [Measure(p, e) for p, e in zip(popt, perr)]
                except RuntimeError:
                    print(f"Fit failed for bunch {bunch_i} of {beam_color} beam.")
                    continue

                # Create the fit string for annotation
                fit_str = [rf'$\mu_1$ = {pmeas[0]} ns', rf'$\sigma_1$ = {pmeas[1]} ns',
                           rf'$\mu_2$ = {pmeas[3]} ns', rf'$\sigma_2$ = {pmeas[4]} ns', rf'$a_2$ = {pmeas[2]}',
                           rf'$\mu_3$ = {pmeas[6]} ns', rf'$\sigma_3$ = {pmeas[7]} ns', rf'$a_3$ = {pmeas[5]}',
                           rf'$\mu_4$ = {pmeas[9]} ns', rf'$\sigma_4$ = {pmeas[10]} ns', rf'$a_4$ = {pmeas[8]}']
                fit_str = '\n'.join(fit_str)

                # Fit equation for annotation
                fit_eq = (
                    r'$p(t) = \frac{\frac{1}{\sigma_1 \sqrt{2 \pi}} \exp\left(-\frac{(t - \mu_1)^2}{2\sigma_1^2}\right)'
                    r'+ \frac{a_2}{\sigma_2 \sqrt{2 \pi}} \exp\left(-\frac{(t - \mu_2)^2}{2\sigma_2^2}\right)'
                    r'+ \frac{a_3}{\sigma_3 \sqrt{2 \pi}} \exp\left(-\frac{(t - \mu_3)^2}{2\sigma_3^2}\right)'
                    r'+ \frac{a_4}{\sigma_4 \sqrt{2 \pi}} \exp\left(-\frac{(t - \mu_4)^2}{2\sigma_4^2}\right)}'
                    r'{1 + a_2 + a_3 + a_4}$')

                # Plot individual fit
                fig_bunch, ax_bunch = plt.subplots(figsize=(8, 6))
                x_plot = np.linspace(bunch_times[0], bunch_times[-1], 1000)
                ax_bunch.plot(bunch_times, vals, color=plot_color, alpha=0.7, label='CAD Profile')
                ax_bunch.plot(x_plot, quad_gaus_pdf(x_plot, *popt), color='red', ls='-', label='Fit')
                ax_bunch.axvline(popt[0], color='green', ls='--', alpha=0.5, zorder=0)
                ax_bunch.set_ylim(-0.024, 0.143)
                ax_bunch.set_xlim(21, 79)
                ax_bunch.set_xlabel('Time (ns)')
                ax_bunch.set_ylabel('Probability Density')
                ax_bunch.set_title(f'{vernier_scan_date} Vernier Scan {beam_color.capitalize()} Beam Bunch {bunch_i}')
                ax_bunch.annotate(fit_str, (0.01, 0.98), xycoords='axes fraction', ha='left', va='top',
                                  bbox=dict(boxstyle='round', fc='white', alpha=0.8), fontsize=12)
                ax_bunch.annotate(fit_eq, (0.02, 0.02), xycoords='axes fraction', ha='left', va='bottom',
                                  bbox=dict(boxstyle='round', fc='white', alpha=1.0), fontsize=14.5)
                ax_bunch.legend(loc='upper right', fontsize=14)
                ax_bunch.grid(True)
                fig_bunch.tight_layout()

                # Save individual plot and parameters
                if write_out:
                    plot_out_path = (f'{cad_measurements_path}Individual_Longitudinal_Bunch_Profile_Fits/'
                                     f'{vernier_scan_date}_{beam_color}_bunch_{bunch_i}_fit_plot.png')
                    fit_out_path = (f'{cad_measurements_path}Individual_Longitudinal_Bunch_Profile_Fits/'
                                    f'VernierScan_{vernier_scan_date}_{beam_color}_bunch_{bunch_i}_longitudinal_fit.dat')
                    fig_bunch.savefig(plot_out_path)
                    write_longitudinal_beam_profile_fit_parameters(fit_out_path, beam_color, fit_eq, pmeas)
                    plt.close(fig_bunch)

    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Value')
    ax.set_title('Longitudinal Beam Measurements vs Time')
    ax.grid(True)

    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('Value')
    ax2.set_title('Longitudinal Beam Measurements vs Index')
    ax2.grid(True)

    fig.tight_layout()
    fig2.tight_layout()


def combine_cad_measurements(cad_measurements_path, vernier_scan_date, cw_rate_path):
    crossing_angle_path = f'{cad_measurements_path}VernierScan_{vernier_scan_date}_crossing_angle.dat'
    bunch_length_path = f'{cad_measurements_path}VernierScan_{vernier_scan_date}_bunch_length.dat'
    beam_offset_intensity_x_path = f'{cad_measurements_path}VernierScan_{vernier_scan_date}_intensity_position_x.dat'
    beam_offset_intensity_y_path = f'{cad_measurements_path}VernierScan_{vernier_scan_date}_intensity_position_y.dat'

    bunch_length_data = read_bunch_length(bunch_length_path)
    convert_bunch_length_to_distance(bunch_length_data)
    calculate_bunch_length_scaling(bunch_length_data, vernier_scan_date)

    vertical_cw_rates, horizontal_cw_rates = read_cw_rates_from_file(cw_rate_path)

    beam_offset_intensity_x_data = read_beam_offset_and_intensity(beam_offset_intensity_x_path)
    beam_offset_intensity_x_data['orientation'] = 'Horizontal'
    beam_offset_intensity_x_data = append_relative_and_set_offsets(beam_offset_intensity_x_data)
    beam_offset_intensity_x_data = calculate_relative_intensity(beam_offset_intensity_x_data)
    beam_offset_intensity_x_data = append_cw_rates(beam_offset_intensity_x_data, horizontal_cw_rates)

    beam_offset_intensity_y_data = read_beam_offset_and_intensity(beam_offset_intensity_y_path)
    beam_offset_intensity_y_data['orientation'] = 'Vertical'
    beam_offset_intensity_y_data = append_relative_and_set_offsets(beam_offset_intensity_y_data)
    beam_offset_intensity_y_data = calculate_relative_intensity(beam_offset_intensity_y_data)
    beam_offset_intensity_y_data = append_cw_rates(beam_offset_intensity_y_data, vertical_cw_rates)

    boi_data = pd.concat([beam_offset_intensity_x_data, beam_offset_intensity_y_data], axis=0, ignore_index=True)

    crossing_angle_data = read_crossing_angle(crossing_angle_path)
    summarized_crossing_angle_data = average_crossing_angles(crossing_angle_data, bunch_length_data['time'])

    print(summarized_crossing_angle_data['time'])
    print(bunch_length_data['time'])

    summarized_crossing_angle_data = summarized_crossing_angle_data.set_index('time')
    bunch_length_data = bunch_length_data.set_index('time')
    boi_data = boi_data.set_index('time')

    combined_data = pd.concat([summarized_crossing_angle_data, bunch_length_data, boi_data], axis=1)
    combined_data = combined_data.reset_index()
    print(combined_data)
    print(combined_data.columns)
    print(combined_data.iloc[0])

    combined_data.to_csv(f'{cad_measurements_path}VernierScan_{vernier_scan_date}_combined.dat', sep='\t')


def crossing_angle(cad_measurements_path, vernier_scan_date):
    file_path = f'{cad_measurements_path}VernierScan_{vernier_scan_date}_crossing_angle.dat'
    data = read_crossing_angle(file_path)
    plot_crossing_angle(data)


def bunch_length(cad_measurements_path, vernier_scan_date):
    file_path = f'{cad_measurements_path}VernierScan_{vernier_scan_date}_bunch_length.dat'
    data = read_bunch_length(file_path)
    convert_bunch_length_to_distance(data)

    bnl_tz = pytz.timezone('America/New_York')
    profile_times = {
        'Jul11': bnl_tz.localize(datetime(2024, 7, 11, 14, 3)),
        'Aug12': bnl_tz.localize(datetime(2024, 8, 12, 14, 29))
    }
    profile_time = profile_times[vernier_scan_date]

    plot_bunch_length(data, profile_time)


def beam_offset_and_intensity(cad_measurements_path, vernier_scan_date):
    file_path_x = f'{cad_measurements_path}VernierScan_{vernier_scan_date}_intensity_position_x.dat'
    file_path_y = f'{cad_measurements_path}VernierScan_{vernier_scan_date}_intensity_position_y.dat'
    data_x = read_beam_offset_and_intensity(file_path_x)
    plot_beam_offset_and_intensity(data_x, 'Horizontal')
    data_y = read_beam_offset_and_intensity(file_path_y)
    plot_beam_offset_and_intensity(data_y, 'Vertical')


def plot_crossing_angle(data):
    plt.figure(figsize=(12, 6))

    plt.plot(list(data['time']), list(data['bh8_crossing_angle']), label='Blue', color='blue')
    plt.plot(list(data['time']), list(data['yh8_crossing_angle']), label='Yellow', color='orange')
    plt.plot(list(data['time']), list(data['gh8_crossing_angle']), label='Blue - Yellow', color='green')
    plt.axhline(0, color='gray', linestyle='-', alpha=0.7)

    plt.xlabel('Time')
    plt.ylabel('Angle (mrad)')
    plt.title('Crossing Angles vs Time')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()


def plot_bunch_length(data, profile_time=None):
    plt.figure(figsize=(12, 6))

    plt.plot(list(data['time']), list(data['blue_bunch_length']), label='Blue', color='blue', marker='o')
    plt.plot(list(data['time']), list(data['yellow_bunch_length']), label='Yellow', color='orange', marker='o')
    if profile_time:
        plt.axvline(profile_time, color='red', linestyle='--', alpha=1.0, label='Z-Profile Measurement')

    plt.ylabel('Bunch Length (m)')
    plt.annotate('Bunch Length Expansion over the Vernier Scan', (0.03, 0.65), xycoords='axes fraction',
                 ha='left', va='top', fontsize=16, bbox=dict(facecolor='white', boxstyle='round,pad=0.5', ec='none'))
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=0)
    # Set format of x-axis to be time and display month/day hour:minute
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %d %H:%M'))
    plt.tight_layout()


def plot_beam_offset_and_intensity(data, orientation):
    fig_intensity, ax_intensity = plt.subplots(figsize=(12, 6))

    ax_intensity.plot(list(data['time']), list(data['wcm_blue']), label='WCM Blue', color='blue', marker='o')
    ax_intensity.plot(list(data['time']), list(data['wcm_yellow']), label='WCM Yellow', color='orange', marker='o')
    ax_intensity.plot(list(data['time']), list(data['dcct_blue']), label='DCCT Blue', color='blue', ls='--', marker='o')
    ax_intensity.plot(list(data['time']), list(data['dcct_yellow']), label='DCCT Yellow', color='orange', ls='--',
                      marker='o')

    ax_intensity.set_xlabel('Time')
    ax_intensity.set_ylabel('Intensity (10^9 protons)')
    ax_intensity.set_title(f'{orientation} Scan Beam Intensity vs Time')
    ax_intensity.legend()
    ax_intensity.grid(True)
    ax_intensity.tick_params(axis='x', rotation=45)
    fig_intensity.tight_layout()

    fig_offset, ax_offset = plt.subplots(figsize=(12, 6))

    ax_offset.plot(list(data['time']), list(data['bpm_south_ir']), label='BPM South IR', color='black', marker='o')
    ax_offset.plot(list(data['time']), list(data['bpm_north_ir']), label='BPM North IR', color='green', marker='o')
    ax_offset.axhline(data['bpm_south_ir'][0], color='black', linestyle='-', alpha=0.7)
    ax_offset.axhline(data['bpm_north_ir'][0], color='green', linestyle='-', alpha=0.7)

    ax_offset.set_xlabel('Time')
    ax_offset.set_ylabel('Offset (mm)')
    ax_offset.set_title(f'{orientation} Scan Beam Offset vs Time')
    ax_offset.legend()
    ax_offset.grid(True)
    ax_offset.tick_params(axis='x', rotation=45)
    fig_offset.tight_layout()

    fig_offset_from_baseline, ax_offset_from_baseline = plt.subplots(figsize=(12, 6))

    ax_offset_from_baseline.plot(list(data['time']), list(data['bpm_south_ir'] - data['bpm_south_ir'][0]),
                                 label='BPM South IR', color='black', marker='o')
    ax_offset_from_baseline.plot(list(data['time']), list(data['bpm_north_ir'] - data['bpm_north_ir'][0]),
                                 label='BPM North IR', color='green', marker='o')
    ax_offset_from_baseline.axhline(0, color='gray', linestyle='-', alpha=0.7)

    ax_offset_from_baseline.set_xlabel('Time')
    ax_offset_from_baseline.set_ylabel('Offset from Baseline (mm)')
    ax_offset_from_baseline.set_title(f'{orientation} Scan Beam Offset from Baseline vs Time')
    ax_offset_from_baseline.legend()
    ax_offset_from_baseline.grid(True)
    ax_offset_from_baseline.tick_params(axis='x', rotation=45)
    fig_offset_from_baseline.tight_layout()


def read_crossing_angle(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Skip the header lines
    data_lines = lines[3:]

    # Lists to store the parsed data
    time_data = []
    bh8_crossing_angle = []
    yh8_crossing_angle = []
    gh8_crossing_angle = []

    # Parse each line
    for line in data_lines:
        columns = line.strip().split('\t')
        if len(columns) != 4:
            continue

        time_data.append(datetime.strptime(columns[0].strip(), "%m/%d/%Y %H:%M:%S"))
        bh8_crossing_angle.append(float(columns[1]))
        yh8_crossing_angle.append(float(columns[2]))
        gh8_crossing_angle.append(float(columns[3]))

    df = pd.DataFrame({
        'time': time_data,
        'bh8_crossing_angle': bh8_crossing_angle,
        'yh8_crossing_angle': yh8_crossing_angle,
        'gh8_crossing_angle': gh8_crossing_angle
    })

    # Set the time column to be in New York time
    bnl_tz = pytz.timezone('America/New_York')
    df['time'] = df['time'].dt.tz_localize(bnl_tz)

    # Return the data as a pandas DataFrame
    return df

def read_bunch_length(file_path):
    df = pd.read_csv(file_path, sep='\t', header=0)
    # Split the DataFrame into two parts: first 4 columns and last 4 columns
    n_actual_columns = df.shape[1] // 2
    first_part = df.iloc[:, :n_actual_columns]
    last_part = df.iloc[:, n_actual_columns:]

    # Rename the columns of the last part to match the first part
    last_part.columns = first_part.columns

    # Concatenate the two DataFrames vertically
    df = pd.concat([last_part, first_part], axis=0).reset_index(drop=True)

    df.columns = ['time', 'blue_bunch_length', 'yellow_time', 'yellow_bunch_length']
    df = df.drop(columns='yellow_time')
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    bnl_tz = pytz.timezone('America/New_York')
    df['time'] = df['time'].dt.tz_convert(bnl_tz)

    return df


def read_beam_offset_and_intensity(file_path):
    durations = [40, 40, 45, 45, 60, 120, 40, 40, 45, 45, 60, 120, 40]

    # Read the file into a DataFrame
    df = pd.read_csv(file_path, sep='\t', header=0)

    # Rename the columns
    cols = [
        'time',             # Time in Unix epoch seconds
        'wcm_blue',         # Wall Current Monitor for Blue beam [10^9 protons]
        'wcm_yellow',       # Wall Current Monitor for Yellow beam [10^9 protons]
        'dcct_blue',        # Direct Current-Current Transformer for Blue beam [10^9 protons]
        'dcct_yellow',      # Direct Current-Current Transformer for Yellow beam [10^9 protons]
        'bpm_south_ir',     # BPM Position in South side of IR [mm]
        'bpm_north_ir'      # BPM Position in North side of IR [mm]
    ]

    if len(df.columns) == 9:
        cols.insert(6, 'bpm_south_ir_sd')
        cols.insert(8, 'bpm_north_ir_sd')

    df.columns = cols

    # Convert the time from Unix epoch seconds to New York time
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    bnl_tz = pytz.timezone('America/New_York')
    df['time'] = df['time'].dt.tz_convert(bnl_tz)

    # Add step number starting from 1
    df['step'] = range(1, len(df) + 1)

    # Add duration of each step
    df['duration'] = durations[:len(df)]

    return df



def average_crossing_angles(crossing_angle_data, times):
    """
    Average crossing angle data over a given time range. Return averages along with standard deviations and mins/maxes.
    :param crossing_angle_data: DataFrame of crossing angle data.
    :param times: Pandas Series of time points to average around.
    :return: DataFrame of summarized crossing angle data.
    """
    # Get time start and end points from times Series using midpoints between points
    time_starts = times - (times - times.shift()) / 2
    time_ends = times + (times - times.shift()) / 2
    time_starts.iloc[0] = times.iloc[0] - (times.iloc[1] - times.iloc[0]) / 2
    time_ends.iloc[0] = times.iloc[0] + (times.iloc[1] - times.iloc[0]) / 2
    durations = (time_ends - time_starts).dt.total_seconds()

    # Initialize lists to store averages, standard deviations, mins, and maxes
    avg_bh8, avg_yh8, avg_gh8, std_bh8, std_yh8, std_gh8 = [], [], [], [], [], []
    min_bh8, min_yh8, min_gh8, max_bh8, max_yh8, max_gh8 = [], [], [], [], [], []

    # Loop through time ranges and calculate averages, standard deviations, mins, and maxes
    for start, end in zip(time_starts, time_ends):
        data = crossing_angle_data[(crossing_angle_data['time'] >= start) & (crossing_angle_data['time'] <= end)]
        avg_bh8.append(data['bh8_crossing_angle'].mean())
        avg_yh8.append(data['yh8_crossing_angle'].mean())
        avg_gh8.append(data['gh8_crossing_angle'].mean())
        std_bh8.append(data['bh8_crossing_angle'].std())
        std_yh8.append(data['yh8_crossing_angle'].std())
        std_gh8.append(data['gh8_crossing_angle'].std())
        min_bh8.append(data['bh8_crossing_angle'].min())
        min_yh8.append(data['yh8_crossing_angle'].min())
        min_gh8.append(data['gh8_crossing_angle'].min())
        max_bh8.append(data['bh8_crossing_angle'].max())
        max_yh8.append(data['yh8_crossing_angle'].max())
        max_gh8.append(data['gh8_crossing_angle'].max())

    # Return the summarized data as a DataFrame
    return pd.DataFrame({
        'time': times,
        'time_start': time_starts,
        'time_end': time_ends,
        'duration_2': durations,
        'bh8_avg': avg_bh8,
        'yh8_avg': avg_yh8,
        'gh8_avg': avg_gh8,
        'bh8_std': std_bh8,
        'yh8_std': std_yh8,
        'gh8_std': std_gh8,
        'bh8_min': min_bh8,
        'yh8_min': min_yh8,
        'gh8_min': min_gh8,
        'bh8_max': max_bh8,
        'yh8_max': max_yh8,
        'gh8_max': max_gh8
    })


def convert_bunch_length_to_distance(data):
    # Convert bunch length from fwhm in ns to 1 sigma in meters
    c = 299792458  # m/s
    data['blue_bunch_length'] = data['blue_bunch_length'] * c * 1e-9 / (2 * np.sqrt(2 * np.log(2)))
    data['yellow_bunch_length'] = data['yellow_bunch_length'] * c * 1e-9 / (2 * np.sqrt(2 * np.log(2)))


def calculate_bunch_length_scaling(data, vernier_scan_date):
    """
    Calculate the scaling factor for the bunch length data.
    Currently use fits of beam profiles for simulation. These fits are for specific times during each scan.
    Use bunch length measurements through the scan to scale the widths of these parameterized profiles.
    :param data: DataFrame of bunch length data.
    :param vernier_scan_date: Date of the Vernier scan.
    """
    bnl_tz = pytz.timezone('America/New_York')
    profile_times = {
        'Jul11': bnl_tz.localize(datetime(2024, 7, 11, 14, 3)),
        'Aug12': bnl_tz.localize(datetime(2024, 8, 12, 14, 29))
    }
    profile_time = profile_times[vernier_scan_date]

    # Estimate the bunch length at profile_time using linear interpolation from two bracketing measurements
    # Find two bracketing measurements
    before_time = data['time'][data['time'] < profile_time].max()
    after_time = data['time'][data['time'] > profile_time].min()
    dt = (after_time - before_time).total_seconds()
    d = (profile_time - before_time).total_seconds()

    # Linearly interpolate the bunch length at profile_time
    before_blue_bunch_length = data['blue_bunch_length'][data['time'] == before_time].values[0]
    after_blue_bunch_length = data['blue_bunch_length'][data['time'] == after_time].values[0]
    dl_blue = after_blue_bunch_length - before_blue_bunch_length
    profile_blue_bunch_length = before_blue_bunch_length + dl_blue * d / dt

    before_yellow_bunch_length = data['yellow_bunch_length'][data['time'] == before_time].values[0]
    after_yellow_bunch_length = data['yellow_bunch_length'][data['time'] == after_time].values[0]
    dl_yellow = after_yellow_bunch_length - before_yellow_bunch_length
    profile_yellow_bunch_length = before_yellow_bunch_length + dl_yellow * d / dt

    # Calculate the blue and yellow scaling factors and add them to the DataFrame
    data['blue_bunch_length_scaling'] = data['blue_bunch_length'] / profile_blue_bunch_length
    data['yellow_bunch_length_scaling'] = data['yellow_bunch_length'] / profile_yellow_bunch_length


def append_relative_and_set_offsets(boi_data):
    """
    Append the relative and intended offsets to the beam offset and intensity data DataFrame.
    The relative offsets are the offsets relative to the first measured offset.
    The intended offsets are the offsets that are set in the control room. Use the relative offsets and match to the
    closest set_vals.
    """
    set_vals = [0.0, 0.1, 0.25, 0.4, 0.6, 0.9]

    # Create a copy of the DataFrame
    new_boi_data = boi_data.copy()

    # Calculate the relative offsets
    new_boi_data['bpm_south_ir_rel'] = new_boi_data['bpm_south_ir'] - new_boi_data['bpm_south_ir'].iloc[0]
    new_boi_data['bpm_north_ir_rel'] = new_boi_data['bpm_north_ir'] - new_boi_data['bpm_north_ir'].iloc[0]

    # Initialize new columns for the average and set values
    new_boi_data['offset_avg_val'] = new_boi_data['bpm_south_ir']
    new_boi_data['offset_set_val'] = new_boi_data['bpm_south_ir']

    # Calculate the average measured offsets and match to the closest set values
    for i in range(0, len(new_boi_data)):
        avg_meas_offset = (new_boi_data['bpm_south_ir_rel'].iloc[i] + new_boi_data['bpm_north_ir_rel'].iloc[i]) / 2
        closest_set_val = min(set_vals, key=lambda x: abs(x - abs(avg_meas_offset)))

        if closest_set_val != 0.0:
            closest_set_val *= np.sign(avg_meas_offset)

        new_boi_data.at[i, 'offset_avg_val'] = avg_meas_offset
        new_boi_data.at[i, 'offset_set_val'] = closest_set_val

    return new_boi_data


def calculate_relative_intensity(boi_data):
    """
    Calculate the beam intensity at each point (N_y * N_b) relative to the first point.
    """
    new_boi_data = boi_data.copy()

    # Calculate the relative intensities
    new_boi_data['intensity_rel_wcm'] = (new_boi_data['wcm_yellow'] * new_boi_data['wcm_blue']) / (
            new_boi_data['wcm_yellow'].iloc[0] * new_boi_data['wcm_blue'].iloc[0])
    new_boi_data['intensity_rel_dcct'] = (new_boi_data['dcct_yellow'] * new_boi_data['dcct_blue']) / (
            new_boi_data['dcct_yellow'].iloc[0] * new_boi_data['dcct_blue'].iloc[0])

    return new_boi_data


def write_longitudinal_beam_profile_fit_parameters(fit_out_path, beam_color, fit_eq, fit_parameters):
    c = 299792458. * 1e6 / 1e9  # um/ns Speed of light
    with open(fit_out_path, 'w') as file:
        file.write(f'Fit Parameters for {beam_color.capitalize()} Beam Longitudinal Profile\n')
        file.write(f'Fit Equation: {fit_eq}\n')
        file.write(f'Fit Parameters:\n')
        for i, (param, meas) in enumerate(zip(['mu1', 'sigma1', 'a2', 'mu2', 'sigma2', 'a3', 'mu3', 'sigma3',
                                               'a4', 'mu4', 'sigma4'], fit_parameters)):
            if 'mu' in param:  # Shift all mu value by the first mu value
                meas = meas - fit_parameters[0].val
            if 'mu' in param or 'sigma' in param:  # Convert from ns to um with speed of light
                meas = meas * c
            file.write(f'{param}: {meas.val}\n')


def read_cw_rates_from_file(file_path):
    vertical_pos = []
    vertical_values = []
    horizontal_pos = []
    horizontal_values = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

        # Determine if we are in the vertical or horizontal section
        current_section = None

        for line in lines:
            line = line.strip()
            if "Vertical pos" in line:
                current_section = "vertical"
                continue
            elif "Horizontal pos" in line:
                current_section = "horizontal"
                continue

            # Extracting values if in the correct section
            if current_section == "vertical":
                if line:
                    parts = line.split()
                    vertical_pos.append(float(parts[0]))
                    vertical_values.append(float(parts[1]))
            elif current_section == "horizontal":
                if line:
                    parts = line.split()
                    horizontal_pos.append(float(parts[0]))
                    horizontal_values.append(float(parts[1]))

    return (vertical_pos, vertical_values), (horizontal_pos, horizontal_values)


def append_cw_rates(data, cw_rates):
    """
    Append cw_rates to data based on set position. cw_rates is a tuple of two lists: set_pos and rates.
    data is a DataFrame with a 'set_pos' column.
    :param data:
    :param cw_rates:
    :return:
    """
    set_pos, rates = cw_rates
    set_pos.append(0.0)
    rates.append(np.nan)

    if not np.all(data['offset_set_val'] == set_pos):
        print(f'Set positions do not match!\n{data["offset_set_val"]}\n{set_pos}')
        return

    new_data = data.copy()

    new_data['cw_rate'] = rates

    return new_data


def gaus(x, a, b, c):
    return a * np.exp(-(x - b)**2 / (2 * c**2))


def gaus_pdf(x, b, c):
    return np.exp(-(x - b)**2 / (2 * c**2)) / (c * np.sqrt(2 * np.pi))


def triple_gaus(x, a1, b1, c1, a2, b2, c2, a3, b3, c3):
    return (gaus(x, a1, b1, c1) + gaus(x, a2, b2, c2) + gaus(x, a3, b3, c3)) / 3


def triple_gaus_pdf(x, b1, c1, a2, b2, c2, a3, b3, c3):
    return (gaus_pdf(x, b1, c1) + a2 * gaus_pdf(x, b2, c2) + a3 * gaus_pdf(x, b3, c3)) / (1 + a2 + a3)


def quad_gaus_pdf(x, b1, c1, a2, b2, c2, a3, b3, c3, a4, b4, c4):
    return (gaus_pdf(x, b1, c1) + a2 * gaus_pdf(x, b2, c2) + a3 * gaus_pdf(x, b3, c3) + a4 * gaus_pdf(x, b4, c4)) / (
            1 + a2 + a3 + a4)


if __name__ == '__main__':
    main()

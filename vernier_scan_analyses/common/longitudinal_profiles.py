#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 15 11:20 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/longitudinal_profiles

@author: Dylan Neff, dn277127
"""

import os
import platform
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as cf
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

from plot_cad_measurements import gaus_pdf
from Measure import Measure


def main():
    # write_avg_longitudinal_profiles()
    # fit_longitudinal_profiles()
    # plot_profiles_vs_time()
    # compare_direct_profiles()
    plot_abort_gaps()
    print('donzo')


def fit_longitudinal_profiles_old():
    if platform.system() == 'Windows':
        base_path = 'C:/Users/Dylan/Desktop/'
    else:
        base_path = '/local/home/dn277127/Bureau/'
    cad_measurements_path = f'{base_path}vernier_scan_AuAu24/CAD_Measurements/'

    # min_time, max_time = 30, 75
    min_time, max_time = 0, 106
    # min_val = 1.5
    min_val = 0.2
    fit_range = [35, 70]
    # max_pdf_val = 0.125 if vernier_scan_date == 'Aug12' else 0.1
    beam_colors = ['blue', 'yellow']
    plot_colors = ['blue', 'orange']
    # beam_colors = ['blue']
    # plot_colors = ['blue']
    # sufx = '_21:33'
    sufx = '_22'
    peak_list = define_peaks()

    write_out = True
    p0 = None

    fig, ax = plt.subplots(figsize=(12, 6))
    fig2, ax2 = plt.subplots(figsize=(12, 6))

    for beam_color, plot_color in zip(beam_colors, plot_colors):
        file_path = f'{cad_measurements_path}VernierScan_AuAu_longitudinal_{beam_color}{sufx}.dat'
        fit_out_path = f'{cad_measurements_path}VernierScan_AuAu_{beam_color}_longitudinal_fit{sufx}.dat'
        with open(file_path, 'r') as f:
            file_content = f.read()
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
            bunch_times, bunch_vals = np.array(bunch_times), np.array(bunch_vals)
            fit_mask = (bunch_times > fit_range[0]) & (bunch_times < fit_range[1])
            vals = bunch_vals[fit_mask] / np.sum(bunch_vals[fit_mask]) / bin_width
            fit_times.extend(list(bunch_times[fit_mask]))
            fit_vals.extend(list(vals))

            # if p0 is None:  # Get guess from first blue bunch
                # p0, bounds, peak_xs = make_initial_guess_from_data(np.array(bunch_times), np.array(vals))
                # p0, bounds, peak_xs = make_initial_guess_from_expectation(np.array(bunch_times), np.array(vals))
        p0, bounds = build_manual_initial_guess(peak_list[beam_color])

        print(f'{beam_color} {weird_ones} weird ones of {len(times)} bunches')

        popt, pcov = cf(multi_gaus_pdf, fit_times, fit_vals, p0=p0, bounds=bounds)
        perr = np.sqrt(np.diag(pcov))
        pmeas = [Measure(p, e) for p, e in zip(popt, perr)]

        fig_all, ax_all = plt.subplots(figsize=(8, 6))
        x_plot = np.linspace(fit_times[0], fit_times[-1], 1000)
        ax_all.plot(fit_times, fit_vals, color=plot_color, alpha=0.01, ls='none', marker='.', label='CAD Profiles')
        ax_all.plot(x_plot, multi_gaus_pdf(x_plot, *p0), color='green', ls='-', label='Guess')
        ax_all.plot(x_plot, p0[-3] / (1 + np.sum(p0[2::3])) * gaus_pdf(x_plot, *p0[-2:]), color='green', ls='--')
        ax_all.plot(x_plot, multi_gaus_pdf(x_plot, *popt), color='red', ls='-', label='Fit')
        ax_all.plot(x_plot, popt[-3] / (1 + np.sum(popt[2::3])) * gaus_pdf(x_plot, *popt[-2:]), color='red', ls='--')
        ax_all.set_xlabel('Time (ns)')
        ax_all.set_ylabel('Probability Density')
        ax_all.set_title(
            f'AuAu24 Vernier Scan {beam_color.capitalize()} Beam Longitudinal Bunch Density')
        ax_all.set_xlim(min_time, max_time)
        ax_all.legend(loc='upper right', fontsize=14)
        ax_all.grid(True)
        ax_all.set_xlim(30, 75)
        fig_all.tight_layout()

        if write_out:  # Write out fit parameters
            # fig_all.savefig(fit_plots_out_path)
            write_longitudinal_beam_profile_fit_parameters(fit_out_path, beam_color, pmeas)

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

    plt.show()


def plot_profiles_vs_time():
    if platform.system() == 'Windows':
        base_path = 'C:/Users/Dylan/Desktop/'
    else:
        base_path = '/local/home/dn277127/Bureau/'
    cad_measurements_path = f'{base_path}vernier_scan_AuAu24/CAD_Measurements/profiles/'

    min_val = 0.2
    plot_colors = {'blue': 'blue', 'yellow': 'orange'}

    fig, ax = plt.subplots(figsize=(12, 6))
    fig2, ax2 = plt.subplots(figsize=(12, 6))

    for file_name in os.listdir(cad_measurements_path):
        if not file_name.endswith('.dat'):
            continue
        file_name_parts = file_name.split('_')
        beam_color = file_name_parts[0]
        hour, minute, second = file_name_parts[1:]
        file_path = f'{cad_measurements_path}{file_name}'
        with open(file_path, 'r') as f:
            file_content = f.read()
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

        full_time = 0
        for bunch_i, (bunch_times, bunch_vals) in enumerate(zip(times, values)):
            bunch_vals = np.array(bunch_vals)
            max_bunch_val = np.max(bunch_vals)
            times_increasing = np.array(bunch_times)
            times_increasing += full_time
            full_time += bunch_times[-1]
            if max_bunch_val > min_val:
                ax.plot(bunch_times, bunch_vals / np.max(bunch_vals), color=plot_colors[beam_color], alpha=0.01)
            ax2.plot(times_increasing, bunch_vals, color=plot_colors[beam_color])

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

    plt.show()


def compare_direct_profiles():
    if platform.system() == 'Windows':
        base_path = 'C:/Users/Dylan/Desktop/'
    else:
        base_path = '/local/home/dn277127/Bureau/'
    cad_measurements_path = f'{base_path}Vernier_Scans/auau_oct_16_24/profiles/'
    # cad_measurements_path = f'{base_path}Vernier_Scans/pp_aug_12_24/profiles/'
    beam_color = 'blue'
    blue_profiles_test_path = f'{cad_measurements_path}{beam_color}_profile_24_22_22_10.dat'
    # blue_profiles_test_path = f'{cad_measurements_path}{beam_color}_profile_24_14_26_00.dat'
    # blue_profiles_test_path = f'{cad_measurements_path}profiles_test/{beam_color}_profile_24_14_12.dat'
    # blue_profile_wcm_path = f'{cad_measurements_path}profiles/{beam_color}_22_14_12.dat'

    with open(blue_profiles_test_path, 'r') as f:
        lines = f.readlines()
    print(len(lines))
    print(lines[:-1])
    print(lines[-1][:200].split())
    data_str_list = lines[-1].split()
    date = data_str_list[0]
    time = data_str_list[1]
    data = np.array([int(x) for x in data_str_list[3:]])
    print(f'Date: {date}, Time: {time}')
    print(f'Data: {data[:100]}')
    print(f'Data length: {len(data)}')
    print(f'Sample frequency: {len(data) * 1e-6} MHz')
    print(f'Sample period: {1 / len(data) * 1e9} ns')
    print(f'Max value: {np.max(data)}')

    fig, ax = plt.subplots()
    bins = np.arange(70.5, 85.5)
    ax.hist(data, bins=bins, color='blue', alpha=0.5, label='Blue Profile')

    # Flip
    baseline = np.percentile(data, 90)
    print(f'Baseline: {baseline}')
    data = baseline - data

    # with open(blue_profile_wcm_path, 'r') as f:
    #     file_content = f.read()
    # lines = file_content.split('\n')
    # times, values = [[]], [[]]
    # for line in lines[1:]:
    #     if line == '':
    #         continue
    #     columns = line.split('\t')
    #     time, value = float(columns[0]), float(columns[1])
    #     if len(times[-1]) > 0 and time < times[-1][-1]:
    #         times.append([])
    #         values.append([])
    #     times[-1].append(time)
    #     values[-1].append(value)

    # fig, ax = plt.subplots(figsize=(12, 6))
    # ax.plot(data[:len(values[0])], color='blue')
    # ax.plot(np.array(values[0]) / np.max(values[0]) * np.max(data[:len(values[0])]), color='red')
    #
    # fig, ax = plt.subplots(figsize=(12, 6))
    # ax.plot(data, color='blue')
    # flat_vals = np.concatenate(values)
    # ax.plot(flat_vals / np.max(flat_vals) * np.max(data), color='red')
    # print([len(x) for x in values])
    #
    # print(f'Length of data: {len(values[0])}')
    #
    # print(f'Number of bunches: {len(data) / len(values[0])}')
    #
    # fig, ax = plt.subplots(figsize=(12, 6))
    # for i in range(50):
    #     ax.plot(data[i * len(values[0]):(i + 1) * len(values[0])], color='blue', alpha=0.2)

    # Parameters
    segment_size = 2131
    peak_threshold = 60
    fit_window = 20  # points to the left and right of the peak

    # Gaussian function for fitting
    def gaussian(x, a, mu, sigma, c):
        return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + c

    # Prepare segments
    num_full_segments = len(data) // segment_size
    data = data[:num_full_segments * segment_size]  # truncate the end
    segments = np.split(data, num_full_segments)

    # Store the Gaussian means
    gaussian_means = []

    for seg_i, segment in enumerate(segments):
        peak_index = np.argmax(segment)
        peak_value = segment[peak_index]

        if peak_value > peak_threshold:
            left = max(0, peak_index - fit_window)
            right = min(len(segment), peak_index + fit_window + 1)
            x = np.arange(left, right) + seg_i * segment_size
            y = segment[left:right]

            # Initial guess for fitting
            a_guess = y.max() - y.min()
            mu_guess = x[np.argmax(y)]
            sigma_guess = 5
            c_guess = y.min()
            try:
                popt, _ = cf(gaussian, x, y, p0=[a_guess, mu_guess, sigma_guess, c_guess])
                gaussian_means.append(popt[1])
            except RuntimeError:
                continue  # skip if the fit fails

    # Compute differences between successive Gaussian means
    mean_differences = np.diff(gaussian_means)
    mean_differences = mean_differences[mean_differences < 5000]

    mean_differences *= 0.05  # Convert to ns

    # Plot histogram
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(mean_differences, bins=30, edgecolor='black')
    ax.set_xlabel("Difference between successive peak positions")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of Gaussian Peak Differences")

    print(f'Mean of differences: {np.mean(mean_differences)} +- {np.std(mean_differences)} ns')

    time_step = 0.05  # ns
    segment_time = 106.573785  # ns
    bunch_min_peak = np.max(data) * 0.1
    times = np.arange(len(data)) * time_step

    abort_gap = False
    bunch_vals, bunch_times = [], []
    while not abort_gap:
        n_segs = len(bunch_times)
        new_seg_mask = (times >= n_segs * segment_time) & (times < (n_segs + 1) * segment_time)
        new_segment_times, new_segment_values = times[new_seg_mask], data[new_seg_mask]
        new_segment_times = new_segment_times - (n_segs * segment_time)
        if np.max(new_segment_values) < bunch_min_peak:
            abort_gap = True
        else:
            bunch_times.append(new_segment_times)
            bunch_vals.append(new_segment_values)

    fig, ax = plt.subplots(figsize=(12, 6))
    for times_i, vals_i in zip(bunch_times, bunch_vals):
        ax.plot(times_i, vals_i / np.max(vals_i), color='blue', alpha=0.1)

    print(f'Number of bunches: {len(bunch_times)}')

    plt.show()


def plot_abort_gaps():
    if platform.system() == 'Windows':
        base_path = 'C:/Users/Dylan/Desktop/'
    else:
        base_path = '/local/home/dn277127/Bureau/'
    cad_measurements_path = f'{base_path}Vernier_Scans/auau_oct_16_24/profiles/'
    # cad_measurements_path = f'{base_path}Vernier_Scans/pp_aug_12_24/profiles/'
    beam_color = 'blue'
    blue_profiles_test_path = f'{cad_measurements_path}{beam_color}_profile_24_22_22_10.dat'
    # blue_profiles_test_path = f'{cad_measurements_path}{beam_color}_profile_24_14_26_00.dat'
    # blue_profiles_test_path = f'{cad_measurements_path}profiles_test/{beam_color}_profile_24_14_12.dat'
    # blue_profile_wcm_path = f'{cad_measurements_path}profiles/{beam_color}_22_14_12.dat'

    with open(blue_profiles_test_path, 'r') as f:
        lines = f.readlines()
    print(len(lines))
    print(lines[:-1])
    print(lines[-1][:200].split())
    data_str_list = lines[-1].split()
    date = data_str_list[0]
    time = data_str_list[1]
    data = np.array([int(x) for x in data_str_list[3:]])
    print(f'Date: {date}, Time: {time}')
    print(f'Data: {data[:100]}')
    print(f'Data length: {len(data)}')
    print(f'Sample frequency: {len(data) * 1e-6} MHz')
    print(f'Sample period: {1 / len(data) * 1e9} ns')
    print(f'Max value: {np.max(data)}')

    fig, ax = plt.subplots()
    bins = np.arange(70.5, 85.5)
    ax.hist(data, bins=bins, color='blue', alpha=0.5, label='Blue Profile')

    # Flip
    baseline = np.percentile(data, 90)
    print(f'Baseline: {baseline}')
    data = baseline - data

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data, color='blue', label='Profile Data')
    ax.set_xlabel('Index')

    # Parameters
    segment_size = 2131
    peak_threshold = 60
    fit_window = 20  # points to the left and right of the peak

    # Prepare segments
    num_full_segments = len(data) // segment_size
    data = data[:num_full_segments * segment_size]  # truncate the end
    segments = np.split(data, num_full_segments)

    fig, ax = plt.subplots(figsize=(12, 6))
    for seg_i, segment in enumerate(segments):
        peak_index = np.argmax(segment)
        peak_value = segment[peak_index]

        if peak_value < peak_threshold:
            left = max(0, peak_index - fit_window)
            right = min(len(segment), peak_index + fit_window + 1)
            x = np.arange(left, right) + seg_i * segment_size
            y = segment[left:right]

            ax.plot(x -  np.min(x), y, color='blue', alpha=0.1, label=f'Segment {seg_i + 1}')

    plt.show()


def fit_longitudinal_profiles():
    if platform.system() == 'Windows':
        base_path = 'C:/Users/Dylan/Desktop/'
    else:
        base_path = '/local/home/dn277127/Bureau/'
    profiles_path = f'{base_path}vernier_scan_AuAu24/CAD_Measurements/profiles_test/'

    peak_list = define_peaks()

    for file_name in os.listdir(profiles_path):
        if not file_name.endswith('.dat'):
            continue
        with open(f'{profiles_path}{file_name}', 'r') as f:
            lines = f.readlines()
        data_str_list = lines[-1].split()
        date = data_str_list[0]
        time = data_str_list[1]
        data = np.array([int(x) for x in data_str_list[3:]])
        beam_color = 'blue' if 'bo2' in lines[2] else 'yellow'
        print(f'{beam_color} from {date} at {time}')

        baseline = np.percentile(data, 90)
        print(f'Baseline: {baseline}')
        data = baseline - data

        time_step = 0.05  # ns
        segment_time = 106.573785  # ns
        fit_range = [35, 70]  # ns
        bunch_min_peak = np.max(data) * 0.1
        times_flat = np.arange(len(data)) * time_step

        abort_gap = False
        values, times = [], []
        while not abort_gap:
            n_segs = len(times)
            new_seg_mask = (times_flat >= n_segs * segment_time) & (times_flat < (n_segs + 1) * segment_time)
            new_segment_times, new_segment_values = times_flat[new_seg_mask], data[new_seg_mask]
            new_segment_times = new_segment_times - (n_segs * segment_time)
            if np.max(new_segment_values) < bunch_min_peak:
                abort_gap = True
            else:
                times.append(new_segment_times)
                values.append(new_segment_values)

        print(f'Number of bunches: {len(times)}')

        # Fit all the bunches superimposed
        fit_times, fit_vals, weird_one_times, weird_one_vals, weird_ones = [], [], [], [], 0
        for bunch_i, (bunch_times, bunch_vals) in enumerate(zip(times, values)):
            bin_width = bunch_times[1] - bunch_times[0]
            bunch_times, bunch_vals = np.array(bunch_times), np.array(bunch_vals)
            fit_mask = (bunch_times > fit_range[0]) & (bunch_times < fit_range[1])
            vals = bunch_vals[fit_mask] / np.sum(bunch_vals[fit_mask]) / bin_width
            fit_times.extend(list(bunch_times[fit_mask]))
            fit_vals.extend(list(vals))

        p0, bounds = build_manual_initial_guess(peak_list[beam_color])
        popt, pcov = cf(multi_gaus_pdf, fit_times, fit_vals, p0=p0, bounds=bounds)
        perr = np.sqrt(np.diag(pcov))
        pmeas = [Measure(p, e) for p, e in zip(popt, perr)]

        fig_all, ax_all = plt.subplots(figsize=(8, 6))
        x_plot = np.linspace(fit_times[0], fit_times[-1], 1000)
        plot_color = 'blue' if beam_color == 'blue' else 'orange'
        ax_all.plot(fit_times, fit_vals, color=plot_color, alpha=0.01, ls='none', marker='.', label='CAD Profiles')
        ax_all.plot(x_plot, multi_gaus_pdf(x_plot, *p0), color='green', ls='-', label='Guess')
        ax_all.plot(x_plot, p0[-3] / (1 + np.sum(p0[2::3])) * gaus_pdf(x_plot, *p0[-2:]), color='green', ls='--')
        ax_all.plot(x_plot, multi_gaus_pdf(x_plot, *popt), color='red', ls='-', label='Fit')
        ax_all.plot(x_plot, popt[-3] / (1 + np.sum(popt[2::3])) * gaus_pdf(x_plot, *popt[-2:]), color='red', ls='--')
        ax_all.set_xlabel('Time (ns)')
        ax_all.set_ylabel('Probability Density')
        ax_all.set_title(
            f'AuAu24 Vernier Scan {beam_color.capitalize()} Beam Longitudinal Bunch Density')
        ax_all.set_xlim(*fit_range)
        ax_all.legend(loc='upper right', fontsize=14)
        ax_all.grid(True)
        ax_all.set_xlim(30, 75)
        fig_all.tight_layout()

        plt.show()


def write_avg_longitudinal_profiles():
    if platform.system() == 'Windows':
        base_path = 'C:/Users/Dylan/Desktop/'
    else:
        base_path = '/local/home/dn277127/Bureau/'
    # profiles_path = f'{base_path}vernier_scan_AuAu24/CAD_Measurements/profiles/'
    profiles_path = f'{base_path}Vernier_Scans/auau_oct_16_24/profiles/'
    plot = False

    for file_name in os.listdir(profiles_path):
        if not file_name.endswith('.dat') or file_name.startswith('avg_'):
            continue
        with open(f'{profiles_path}{file_name}', 'r') as f:
            lines = f.readlines()
        print(file_name)
        data_str_list = lines[-1].split()
        date = data_str_list[0]
        time = data_str_list[1]
        data = np.array([int(x) for x in data_str_list[3:]])
        beam_color = 'blue' if 'bo2' in lines[2] else 'yellow'
        print(f'{beam_color} from {date} at {time}')

        baseline = np.percentile(data, 90)
        print(f'Baseline: {baseline}')
        data = baseline - data

        time_step = 0.05  # ns
        segment_time = 106.573785  # ns
        bunch_min_peak = np.max(data) * 0.1
        times_flat = np.arange(len(data)) * time_step

        abort_gap = False
        values, times = [], []
        while not abort_gap:
            n_segs = len(times)
            new_seg_mask = (times_flat >= n_segs * segment_time) & (times_flat < (n_segs + 1) * segment_time)
            new_segment_times, new_segment_values = times_flat[new_seg_mask], data[new_seg_mask]
            new_segment_times = new_segment_times - (n_segs * segment_time)
            if np.max(new_segment_values) < bunch_min_peak:
                abort_gap = True
            else:
                times.append(new_segment_times)
                values.append(new_segment_values)

        n_bunches = len(times)
        if n_bunches != 111:
            print(f'Number of bunches not 111: {n_bunches}!!!!!!')

        norm_vals = []
        for bunch_i, (bunch_times, bunch_vals) in enumerate(zip(times, values)):
            bin_width = bunch_times[1] - bunch_times[0]
            bunch_times, bunch_vals = np.array(bunch_times), np.array(bunch_vals)
            vals = bunch_vals / np.sum(bunch_vals) / bin_width
            norm_vals.extend(list(vals))

        times_flat = np.concatenate(times)
        sorted_idx = np.argsort(times_flat)
        times_flat = times_flat[sorted_idx]
        norm_vals = np.array(norm_vals)[sorted_idx]

        mv_avg_times, mv_avg_vals = moving_average(times_flat, norm_vals, n_bunches)

        if plot:
            fig_all, ax_all = plt.subplots(figsize=(8, 6))
            plot_color = 'blue' if beam_color == 'blue' else 'orange'
            ax_all.plot(times_flat, norm_vals, color=plot_color, alpha=0.01, ls='none', marker='.', label='CAD Profiles')
            ax_all.plot(mv_avg_times, mv_avg_vals, ls='-', label=f'Moving Average ({n_bunches} points)', color='red')
            ax_all.set_xlabel('Time (ns)')
            ax_all.set_ylabel('Probability Density')
            ax_all.set_title(
                f'AuAu24 Vernier Scan {beam_color.capitalize()} Beam Longitudinal Bunch Density')
            ax_all.legend(loc='upper right', fontsize=14)
            ax_all.grid(True)
            fig_all.tight_layout()

        interp_func = interp1d(mv_avg_times, mv_avg_vals, kind='linear', bounds_error=False)
        times_interp = np.arange(0, segment_time + 0.01, 0.01)
        min_interp = np.nanmin(interp_func(times_interp))

        # Normalize
        interp_func = interp1d(mv_avg_times, mv_avg_vals - min_interp, kind='linear', bounds_error=False, fill_value=0)
        vals_interp = interp_func(times_interp)
        vals_integral = np.trapz(vals_interp, times_interp)
        print(f'Integral: {vals_integral}')
        vals_interp /= vals_integral

        if plot:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(times_flat, norm_vals, color=plot_color, alpha=0.01, ls='none', marker='.', label='CAD Profiles')
            ax.plot(times_interp, vals_interp, ls='-', label=f'Interpolated ({n_bunches} points)', color='red')
            ax.set_xlabel('z (um)')
            ax.set_ylabel('Probability Density')
            ax.legend()
            fig.tight_layout()

        c = 299792458. * 1e6 / 1e9  # um/ns Speed of light
        zs_interp = c * (times_interp - segment_time / 2)  # Convert to um

        if beam_color == 'blue':
            zs_interp = -zs_interp[::-1]  # Reverse for blue beam
            vals_interp = vals_interp[::-1]

        if plot:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(zs_interp, vals_interp, ls='-', label=f'Interpolated ({n_bunches} points)', color='red')
            ax.set_xlabel('z (um)')
            ax.set_ylabel('Probability Density')
            fig.tight_layout()
            plt.show()

        # Write times_interp and vals_interp to file with numpy
        out_path = f'{profiles_path}avg_{file_name}'
        np.savetxt(out_path, np.column_stack((zs_interp, vals_interp)), header='z (um)\tProbability Density', delimiter='\t')
        print()



def moving_average(xs, ys, window_size):
    """
    Calculate the moving average of ys with respect to xs.
    """
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    if window_size < 1 or window_size > len(ys):
        raise ValueError("window_size must be between 1 and the length of the input arrays")

    # Calculate the moving average for ys
    kernel = np.ones(window_size) / window_size
    ys_avg = np.convolve(ys, kernel, mode='valid')

    # Calculate the corresponding average for xs
    xs_avg = np.convolve(xs, kernel, mode='valid')

    return xs_avg, ys_avg


def define_peaks():
    spacing = 3.3
    sigma = 0.9
    center_blue = 52.55
    center_yellow = 52.36
    manual_peaks = {
        'blue' : [
            {'a': 1.0, 'mu': center_blue, 'sigma': sigma},
            {'a': 0.2, 'mu': center_blue + spacing + 0.1, 'sigma': sigma},
            {'a': 0.2, 'mu': center_blue - spacing - 0.3, 'sigma': sigma},
            {'a': 0.2, 'mu': center_blue + 2 * spacing - 0.3, 'sigma': sigma},
            {'a': 0.2, 'mu': center_blue - 2 * spacing + 0.25, 'sigma': sigma},
            {'a': 0.07, 'mu': center_blue + 3 * spacing - 1.1, 'sigma': sigma},
            {'a': 0.01, 'mu': center_blue - 3 * spacing + 1.5, 'sigma': sigma},
            {'a': 0.04, 'mu': center_blue + 4 * spacing - 1.8, 'sigma': sigma},
            {'a': 0.05, 'mu': center_blue - 4 * spacing + 1.8, 'sigma': sigma},
            {'a': 0.01, 'mu': center_blue, 'sigma': sigma * 5},
        ],
        'yellow' : [
            {'a': 1.0, 'mu': center_yellow, 'sigma': sigma},
            {'a': 0.2, 'mu': center_yellow + spacing, 'sigma': sigma},
            {'a': 0.2, 'mu': center_yellow - spacing - 0.4, 'sigma': sigma},
            {'a': 0.2, 'mu': center_yellow + 2 * spacing, 'sigma': sigma},
            {'a': 0.2, 'mu': center_yellow - 2 * spacing + 0.5, 'sigma': sigma},
            {'a': 0.05, 'mu': center_yellow + 3 * spacing - 1, 'sigma': sigma},
            {'a': 0.05, 'mu': center_yellow - 3 * spacing + 1.5, 'sigma': sigma},
            {'a': 0.05, 'mu': center_yellow + 4 * spacing - 1.5, 'sigma': sigma},
            {'a': 0.05, 'mu': center_yellow - 4 * spacing + 1.8, 'sigma': sigma},
            {'a': 0.05, 'mu': center_yellow - 4 * spacing + 1.8, 'sigma': sigma},
            {'a': 0.01, 'mu': center_yellow, 'sigma': sigma * 5},
        ]
    }

    return manual_peaks


def multi_gaus_pdf(x, *params):
    """
    Sum of N Gaussians, where the first Gaussian has fixed amplitude = 1
    and the rest have variable amplitudes.

    Parameters layout:
    [b1, c1, a2, b2, c2, a3, b3, c3, ..., aN, bN, cN]

    Returns:
        Sum of Gaussians normalized so that total amplitude = 1.
    """
    # First Gaussian (amplitude implicitly 1)
    b1, c1 = params[0], params[1]
    y = gaus_pdf(x, b1, c1)
    total_amp = 1.0

    # Remaining Gaussians
    i = 2
    while i + 2 < len(params):
        a = params[i]
        b = params[i + 1]
        c = params[i + 2]
        y += a * gaus_pdf(x, b, c)
        total_amp += a
        i += 3

    return y / total_amp


def make_initial_guess_from_data(x_data, y_data, n_peaks=8, sigma_guess=1.2):
    # Find all peaks
    peaks, properties = find_peaks(y_data, prominence=0.01)

    # Sort peaks by prominence (highest first)
    prominences = properties["prominences"]
    top_peak_indices = np.array(np.argsort(prominences)[-n_peaks:])
    top_peaks = peaks[top_peak_indices]
    peak_xs = x_data[top_peaks]
    peak_heights = y_data[top_peaks]

    # Sort peak_xs largest to smallest
    peak_heights, peak_xs = zip(*sorted(zip(peak_heights, peak_xs), reverse=True))

    # Generate initial guess parameters
    print(f'peak_xs[0]: {peak_xs[0]}, peak_heights[0]: {peak_heights[0]}')
    p0 = [peak_xs[0], sigma_guess]  # b1, c1
    bounds_lower = [peak_xs[0] - 4, 0]
    bounds_upper = [peak_xs[0] + 4, 10]

    for b, h in zip(peak_xs[1:], peak_heights[1:]):
        a = h / peak_heights[0]  # Amplitude
        c = np.abs(sigma_guess)
        p0 += [a, b, c]
        bounds_lower += [0, b - 4, 0]
        bounds_upper += [np.inf, b + 4, 10]

        for i, (p, l, u) in enumerate(zip(p0, bounds_lower, bounds_upper)):  # Check if any p0 outside bounds
            if p < l or p > u:
                print(f'p0[{i}] = {p} is outside bounds [{l}, {u}]')

    # Add final underlying wide Gaussian
    p0 += [0, peak_xs[0], 5]
    bounds_lower += [0, peak_xs[0] - 4, 0]
    bounds_upper += [np.inf, peak_xs[0] + 4, 20]

    return p0, (bounds_lower, bounds_upper), peak_xs


def make_initial_guess_from_expectation(x_data, y_data, n_peaks=8, sigma_guess=1.2):
    # Find all peaks
    peaks, properties = find_peaks(y_data, prominence=0.01)

    # Sort peaks by prominence (highest first)
    prominences = properties["prominences"]
    top_peak_indices = np.array(np.argsort(prominences)[-n_peaks:])
    top_peaks = peaks[top_peak_indices]
    peak_xs = x_data[top_peaks]
    peak_heights = y_data[top_peaks]

    # Sort peak_xs largest to smallest
    peak_heights, peak_xs = zip(*sorted(zip(peak_heights, peak_xs), reverse=True))

    # Generate initial guess parameters
    print(f'peak_xs[0]: {peak_xs[0]}, peak_heights[0]: {peak_heights[0]}')
    p0 = [peak_xs[0], sigma_guess]  # b1, c1
    bounds_lower = [peak_xs[0] - 4, 0]
    bounds_upper = [peak_xs[0] + 4, 10]

    for i in [-4, -3, -2, -1, 1, 2, 3, 4]:
        a = peak_heights[0] / 4  # Amplitude
        c = sigma_guess
        b = peak_xs[0] + i * 3.2
        p0 += [a, b, c]
        bounds_lower += [0, b - 4, 0]
        bounds_upper += [np.inf, b + 4, 10]

        for i, (p, l, u) in enumerate(zip(p0, bounds_lower, bounds_upper)):  # Check if any p0 outside bounds
            if p < l or p > u:
                print(f'p0[{i}] = {p} is outside bounds [{l}, {u}]')

    # Add final underlying wide Gaussian
    p0 += [peak_heights[0] / 20, peak_xs[0], 5]
    bounds_lower += [0, peak_xs[0] - 4, 0]
    bounds_upper += [np.inf, peak_xs[0] + 4, 20]

    return p0, (bounds_lower, bounds_upper), peak_xs


def build_manual_initial_guess(peak_list):
    """
    peak_list: list of dictionaries like:
        [{'a': 1.0, 'mu': 52.3, 'sigma': 1.0}, ...]
        First peak doesn't use 'a' (it's absorbed into normalization)
    """
    mu_range = 3
    p0 = [peak_list[0]['mu'], peak_list[0]['sigma']]
    bounds_lower = [peak_list[0]['mu'] - mu_range, 0]
    bounds_upper = [peak_list[0]['mu'] + mu_range, 5]

    for peak in peak_list[1:]:
        a = peak.get('a', 1.0)
        mu = peak['mu']
        sigma = peak['sigma']
        p0 += [a, mu, sigma]
        bounds_lower += [0, mu - mu_range, 0]
        bounds_upper += [np.inf, mu + mu_range, 5]

    bounds_upper[-1] = 50  # Last sigma is wide

    return p0, (bounds_lower, bounds_upper)



def write_longitudinal_beam_profile_fit_parameters(fit_out_path, beam_color, fit_parameters):
    """
    Write longitudinal fit parameters to a file and include the dynamic LaTeX equation.
    """
    c = 299792458. * 1e6 / 1e9  # um/ns Speed of light

    # Infer number of Gaussians from parameter count
    n_params = len(fit_parameters)
    assert (n_params - 2) % 3 == 0, "Fit parameters do not match expected format"
    n_gaussians = 1 + (n_params - 2) // 3

    # Generate dynamic fit equation string
    fit_eq = generate_fit_equation_string(n_gaussians)

    # Create parameter name list
    param_names = ['mu1', 'sigma1']
    for i in range(2, n_gaussians + 1):
        param_names.extend([f'a{i}', f'mu{i}', f'sigma{i}'])

    with open(fit_out_path, 'w') as file:
        file.write(f'Fit Parameters for {beam_color.capitalize()} Beam Longitudinal Profile\n')
        file.write(f'Fit Equation: {fit_eq}\n')
        file.write(f'Fit Parameters:\n')
        mu1_val = fit_parameters[0].val
        for param, meas in zip(param_names, fit_parameters):
            val = meas.val
            if 'mu' in param:
                val -= mu1_val
            if 'mu' in param or 'sigma' in param:
                val *= c  # Convert from ns to um
            file.write(f'{param}: {val:.6g}\n')



def generate_fit_equation_string(n_gaussians):
    """
    Dynamically generate the LaTeX fit equation for n Gaussians.
    """
    terms = []
    for i in range(1, n_gaussians + 1):
        if i == 1:
            term = r'\frac{1}{\sigma_1 \sqrt{2 \pi}} \exp\left(-\frac{(t - \mu_1)^2}{2\sigma_1^2}\right)'
        else:
            term = (fr'\frac{{a_{i}}}{{\sigma_{i} \sqrt{{2 \pi}}}}'
                    fr' \exp\left(-\frac{{(t - \mu_{i})^2}}{{2\sigma_{i}^2}}\right)')
        terms.append(term)
    numerator = ' + '.join(terms)
    denominator = '1' + ''.join([f' + a_{i}' for i in range(2, n_gaussians + 1)])
    return fr'$p(t) = \frac{{{numerator}}}{{{denominator}}}$'



if __name__ == '__main__':
    main()

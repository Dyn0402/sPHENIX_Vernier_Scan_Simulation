#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 26 02:08 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/analyze_sphnx_root_file

@author: Dylan Neff, dn277127
"""

import platform
import numpy as np
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
import pandas as pd
import uproot


def main():
    if platform.system() == 'Windows':
        base_path = 'C:/Users/Dylan/Desktop/'
    else:
        base_path = '/local/home/dn277127/Bureau/'

    sub_dir = 'vertex_data/'
    base_path_auau = f'{base_path}Vernier_Scans/auau_oct_16_24/'
    out_root_file_path = f'{base_path_auau}{sub_dir}54733_vertex_distributions.root'
    out_root_file_path_no_zdc_coinc = f'{base_path_auau}{sub_dir}54733_vertex_distributions_no_zdc_coinc.root'
    out_root_file_path_bbb = f'{base_path_auau}{sub_dir}54733_vertex_distributions_bunch_by_bunch.root'
    out_root_file_path_no_zdc_coinc_bbb = f'{base_path_auau}{sub_dir}54733_vertex_distributions_no_zdc_coinc_bunch_by_bunch.root'
    cad_data_path = f'{base_path_auau}combined_cad_step_data.csv'

    cad_df = pd.read_csv(cad_data_path, sep=',')
    print(cad_df)

    compare_avg_total_step_rates(base_path_auau, cad_df, sub_dir)
    plt.show()

    data, time = get_root_data_time(base_path_auau, root_file_name='54733_slimmed.root', tree_name='calo_tree', sub_dir=sub_dir,
                                    branches=['BCO', 'mbd_zvtx', 'mbd_SN_trigger', 'zdc_SN_trigger',
                                              'mbd_SN_live_trigger', 'zdc_SN_live_trigger',
                                              'GL1_clock_count', 'GL1_live_count',
                                              'mbd_raw_count', 'zdc_raw_count', 'mbd_live_count', 'zdc_live_count',
                                              'bunch'])
    print(data.columns)

    # plot_rates(data, time, cad_df)
    # more_rate_plotting(data, time, cad_df)
    # check_unreconstructed_z_vtx(data, time, cad_df, zdc_coincidence=False)
    # compare_scaled_live_triggers(data, time, cad_df)
    # compare_scaled_live_triggers_steps(data, time, cad_df)
    # compare_new_old_root_file(base_path, data, time)
    plt.show()

    print('donzo')


def get_root_data_time(scan_path, root_file_name='54733_slimmed.root', tree_name='calo_tree', branches=None, sub_dir='vertex_data/'):
    """
    Get the data and time from the ROOT file.
    :param scan_path: Path to the directory with the ROOT file.
    :param root_file_name: Name of the ROOT file.
    :param tree_name: Name of the tree in the ROOT file.
    :param branches: List of branches to extract from the tree.
    :return: DataFrame with the data and time.
    """
    if branches is None:
        branches = ['BCO', 'mbd_zvtx', 'mbd_SN_trigger', 'zdc_SN_trigger', 'mbd_raw_count', 'zdc_raw_count',
                    'GL1_clock_count', 'GL1_live_count', 'mbd_live_count', 'zdc_live_count']

    root_file_path = f'{scan_path}{sub_dir}{root_file_name}'

    with uproot.open(root_file_path) as root_file:
        tree = root_file[tree_name]
        print(f'Keys in tree: {tree.keys()}')
        data = tree.arrays(branches, library='pd')

    bco_step = 106.57377e-9  # BCO step in ns
    constant_offset = 2.0  # Constant offset in s

    time = (data['BCO'] - data['BCO'][0]) * bco_step + constant_offset  # Convert BCO to time in seconds

    return data, time


def plot_rates(data, time, cad_df):
    mbd_diff = data['mbd_raw_count'].diff().fillna(0)
    bco_diff = data['BCO'].diff().fillna(0)
    time_diff = time.diff().fillna(0)
    mbd_change = mbd_diff / time_diff

    time_avgs, mbd_rates = average_counts_in_time_windows(np.array(time), np.array(mbd_change), window_size=1)
    fig_1s, ax_1s = plt.subplots()
    cad_df['start'] = pd.to_datetime(cad_df['start'])
    cad_df['end'] = pd.to_datetime(cad_df['end'])
    start_datetime = cad_df.iloc[0]['start']
    for index, row in cad_df.iterrows():
        ax_1s.axvline(x=(row['start'] - start_datetime).total_seconds(), color='r', linestyle='-')
        ax_1s.axvline(x=(row['end'] - start_datetime).total_seconds(), color='r', linestyle='-')
        avg_time = (row['start'] + (row['end'] - row['start']) / 2 - start_datetime).total_seconds()
        ax_1s.annotate(row['step'], xy=(avg_time, 20000), ha='center',
                       xytext=(0, 10), textcoords='offset points', color='black', fontsize=8)
    ax_1s.plot(time_avgs, mbd_rates)
    plt.show()

    # Do moving average of mbd_change
    window_size = 10000
    mbd_change_ma = mbd_change.rolling(window=window_size, min_periods=1).mean()
    bco_avg = data['BCO'].rolling(window=window_size, min_periods=1).mean()
    time_avg = time.rolling(window=window_size, min_periods=1).mean()

    fig_mv_avg, ax_mv_avg = plt.subplots()
    ax_mv_avg.plot(time_avg[::1000], mbd_change_ma[::1000], 'o', markersize=1, label='MBD Z Vertex Change (Moving Avg)')

    # Take derivative of mbd_change_ma
    mbd_change_derivative = mbd_change_ma.diff().fillna(0) / time_avg.diff().fillna(0)
    fig_mv_avg_derivative, ax_mv_avg_derivative = plt.subplots()
    ax_mv_avg_derivative.plot(time_avg[::1000], mbd_change_derivative[::1000], 'o', markersize=1, label='MBD Z Vertex Change Derivative (Moving Avg)')

    plt.show()

    # fig, ax = plt.subplots()
    # ax.plot(data['BCO'][:10000], data['mbd_raw_count'][:10000], 'o', markersize=1, label='mbd_raw_count')
    # ax.plot(data['BCO'][:10000], np.cumsum(data['mbd_SN_trigger'][:10000]), 'o', markersize=1, label='mbd_SN_trigger')
    # ax.plot(data['BCO'][:5000], data['mbd_live_count'][:5000], 'o', markersize=1, label='mbd_live_count')
    # ax.legend()
    # plt.show()

    # fig2, ax2 = plt.subplots()
    # ax2.hist(bco_diff, bins=100, alpha=0.5, label='BCO Diff')

    # fig3, ax3 = plt.subplots()
    # ax3.plot(data['BCO'], mbd_diff / bco_diff, 'o', markersize=1, label='MBD Z Vertex Diff')

    plt.show()


def more_rate_plotting(data, time, cad_df):
    detectors = ['zdc', 'mbd']
    types = ['raw', 'live']

    fig, ax = plt.subplots()

    rate_avgs = {det: {} for det in detectors}
    avg_time_diffs = None
    for detector in detectors:
        for count_type in types:
            col_name = f'{detector}_{count_type}_count'
            # rate_name = f'{detector}_{count_type}_rate'

            counts = data[col_name].diff().fillna(0)
            time_diffs = time.diff().fillna(0)
            rates = counts / time_diffs
            avg_times, avg_rates = average_counts_in_time_windows(
                np.array(time),
                np.array(rates),
                window_size=1
            )
            rate_avgs[detector][count_type] = avg_rates
            avg_time_diffs = avg_times if avg_time_diffs is None else avg_time_diffs

            label = f'{detector.upper()} {count_type.capitalize()} Rate'
            ax.plot(avg_times, avg_rates, 'o-', markersize=1, label=label)

    ax.set_ylim(bottom=0)
    ax.legend()

    # Plot ZDC to MBD rate ratio for Live and Raw
    fig, ax = plt.subplots()
    for count_type in types:
        zdc_rates = rate_avgs['zdc'][count_type]
        mbd_rates = rate_avgs['mbd'][count_type]
        ratio = mbd_rates / zdc_rates
        label = f'MBD to ZDC {count_type.capitalize()} Rate Ratio'
        ax.plot(avg_time_diffs, ratio, 'o-', markersize=1, label=label)

    plt.show()


# def get_steps(data, time, cad_df, plot=True):
#     """
#     Get the steps from the data and cad_df.
#     :param data: Data from the root file.
#     :param time: Time array.
#     :param cad_df: Dataframe with CAD step data.
#     :return: None
#     """
#     step_time_cushion = 1.0  # Time cushion in seconds to avoid transitions
#     binning = np.linspace(-300, 300, 201)
#
#     # Get the start and end times of each step
#     cad_df['start'] = pd.to_datetime(cad_df['start'])
#     cad_df['end'] = pd.to_datetime(cad_df['end'])
#     run_start = cad_df.iloc[0]['start']
#
#     for index, row in cad_df.iterrows():
#         start_time = (row['start'] - run_start).total_seconds() + step_time_cushion
#         end_time = (row['end'] - run_start).total_seconds() - step_time_cushion
#         mask = (time >= start_time) & (time <= end_time)
#         step_data = data[mask]
#         step_duration = end_time - start_time
#
#         raw_mbd_zvtx = step_data['mbd_zvtx']
#         zdc_ns_trigger = step_data['zdc_SN_trigger']
#         mbd_ns_trigger = step_data['mbd_SN_trigger']
#         mbd_zvtx_mbd_coinc = raw_mbd_zvtx[(mbd_ns_trigger == 1)]
#         mbd_zvtx_mbd_zdc_coinc = raw_mbd_zvtx[(zdc_ns_trigger == 1) & (mbd_ns_trigger == 1)]
#
#         hist_counts, _ = np.histogram(mbd_zvtx_mbd_zdc_coinc, bins=binning)
#         hist_counts = np.divide(hist_counts, step_duration)  # Normalize by step duration to get rate
#
#         print(f"Step {index}: Start Time: {start_time}, End Time: {end_time}, Duration: {step_duration} s, "
#               f"Data Points: {len(step_data)}, Rate: {len(step_data) / step_duration:.2f} Hz")
#
#         if plot:
#             fig, ax = plt.subplots()
#             ax.hist(raw_mbd_zvtx, bins=binning, alpha=0.5, label='Raw MBD Z Vertex')
#             ax.hist(mbd_zvtx_mbd_coinc, bins=binning, alpha=0.5, label='MBD Z Vertex (MBD Coincidence)')
#             ax.hist(mbd_zvtx_mbd_zdc_coinc, bins=binning, alpha=0.5, label='MBD Z Vertex (MBD+ZDC Coincidence)')
#             ax.set_xlabel('MBD Z Vertex (cm)')
#             ax.legend()
#
#             fig, ax = plt.subplots()
#             ax.bar(binning[:-1], hist_counts, width=np.diff(binning), align='center', alpha=0.5,
#                    label='MBD Z Vertex (MBD+ZDC Coincidence Rate)')
#             ax.set_xlabel('MBD Z Vertex (cm)')
#             ax.set_ylabel('Rate (Hz)')
#             ax.legend()
#             plt.show()


def check_unreconstructed_z_vtx(data, time, cad_df, zdc_coincidence=True):
    """
    Get the steps from the data and cad_df, plot and save histograms to a ROOT file.
    :param data: Data from the root file.
    :param time: Time array.
    :param cad_df: Dataframe with CAD step data.
    :param zdc_coincidence: Whether to include ZDC coincidence in the histograms.
    :return: None
    """
    step_time_cushion = 1.0  # Time cushion in seconds to avoid transitions
    binning = np.linspace(-300, 300, 201)

    # Get the start and end times of each step
    cad_df['start'] = pd.to_datetime(cad_df['start'])
    cad_df['end'] = pd.to_datetime(cad_df['end'])
    run_start = cad_df.iloc[0]['start']

    for index, row in cad_df.iterrows():
        start_time = (row['start'] - run_start).total_seconds() + step_time_cushion
        end_time = (row['end'] - run_start).total_seconds() - step_time_cushion
        mask = (time >= start_time) & (time <= end_time)
        step_data = data[mask]

        raw_mbd_zvtx = step_data['mbd_zvtx']
        zdc_ns_trigger = step_data['zdc_SN_trigger']
        mbd_ns_trigger = step_data['mbd_SN_trigger']
        mbd_zvtx_mbd_coinc = raw_mbd_zvtx[(mbd_ns_trigger == 1)]
        mbd_zvtx_mbd_zdc_coinc = raw_mbd_zvtx[(zdc_ns_trigger == 1) & (mbd_ns_trigger == 1)]

        # Raw counts for error calculation
        if zdc_coincidence:
            raw_counts, _ = np.histogram(mbd_zvtx_mbd_zdc_coinc, bins=binning)
        else:
            raw_counts, _ = np.histogram(mbd_zvtx_mbd_coinc, bins=binning)
            print(f'Step {index}:')
            print(f'Sum of raw counts: {raw_counts.sum()}')
            print(f'Length of data: {len(mbd_zvtx_mbd_coinc)}')
            print(f'Number of values within binning: {np.sum((mbd_zvtx_mbd_coinc >= binning[0]) & (mbd_zvtx_mbd_coinc <= binning[-1]))}')
            print(f'Number of values outside binning: {np.sum((mbd_zvtx_mbd_coinc < binning[0]) | (mbd_zvtx_mbd_coinc > binning[-1]))}')
            print(f'Fraction of values outside binning: {np.sum((mbd_zvtx_mbd_coinc < binning[0]) | (mbd_zvtx_mbd_coinc > binning[-1])) / len(mbd_zvtx_mbd_coinc) * 100:.2f}%')
            print(f'Fraction MBD + ZDC outside binning: {np.sum((mbd_zvtx_mbd_zdc_coinc < binning[0]) | (mbd_zvtx_mbd_zdc_coinc > binning[-1])) / len(mbd_zvtx_mbd_zdc_coinc) * 100:.2f}%')
            print()


# def write_step_raw_rates(data, time, cad_df, out_path='step_raw_rates.csv'):
#     """
#     Get the mean and standard deviations of the raw rates at each step from the data, using step boundaries from cad_df.
#     """
#     step_time_cushion = 1.0  # Time cushion in seconds to avoid transitions
#     cad_df['start'] = pd.to_datetime(cad_df['start'])
#     cad_df['end'] = pd.to_datetime(cad_df['end'])
#     run_start = cad_df.iloc[0]['start']
#
#     rates = []
#     fig, ax = plt.subplots()
#     for index, row in cad_df.iterrows():
#         start_time = (row['start'] - run_start).total_seconds() + step_time_cushion
#         end_time = (row['end'] - run_start).total_seconds() - step_time_cushion
#         mask = (time >= start_time) & (time <= end_time)
#         step_data = data[mask]
#         step_times = time[mask]
#
#         detectors = ['zdc', 'mbd']
#         types = ['raw', 'live']
#
#         step_rates = {'step': row['step']}
#         for detector in detectors:
#             for count_type in types:
#                 col_name = f'{detector}_{count_type}_count'
#
#                 step_counts = step_data[col_name].diff().fillna(0)
#                 step_time_diffs = step_times.diff().fillna(0)
#                 step_rate = step_counts / step_time_diffs
#                 avg_step_times, avg_step_rates = average_counts_in_time_windows(
#                     np.array(step_times),
#                     np.array(step_rate),
#                     window_size=1
#                 )
#                 if detector == 'zdc' and count_type == 'raw':
#                     ax.plot(avg_step_times, avg_step_rates, 'o-', markersize=1, label=f'Step {row["step"]}')
#
#                 # Count number of nans in avg_step_rates
#                 nan_entries = np.sum(np.isnan(avg_step_rates))
#
#                 avg_step_rate = np.nanmean(avg_step_rates)
#                 std_step_rate = np.nanstd(avg_step_rates)
#
#                 step_rates[f'{detector}_{count_type}_rate_mean'] = avg_step_rate
#                 step_rates[f'{detector}_{count_type}_rate_std'] = std_step_rate
#
#         rates.append(step_rates)
#
#     # Save rates to CSV
#     rates_df = pd.DataFrame(rates)
#     rates_df.to_csv(out_path, index=False)
#     print(f'Saved step raw rates to {out_path}')


def compare_avg_total_step_rates(base_path, cad_df, sub_dir='vertex_data/', out_path='step_raw_rates.csv'):
    """
    Get the mean and standard deviations of the raw rates at each step from the data, using step boundaries from cad_df.
    """
    data, time = get_root_data_time(base_path, root_file_name='54733_slimmed.root', tree_name='calo_tree',
                                    sub_dir=sub_dir,
                                    branches=['BCO',
                                              'GL1_clock_count', 'GL1_live_count',
                                              'mbd_raw_count', 'mbd_live_count'])
                                    # branches=['BCO', 'mbd_zvtx', 'mbd_SN_trigger', 'zdc_SN_trigger',
                                    #           'mbd_SN_live_trigger', 'zdc_SN_live_trigger',
                                    #           'GL1_clock_count', 'GL1_live_count',
                                    #           'mbd_raw_count', 'zdc_raw_count', 'mbd_live_count', 'zdc_live_count',
                                    #           'bunch'])

    fig, ax = plt.subplots()
    ax.plot(time, data['GL1_clock_count'] / 1e7, label='Gl1_clock_count')
    ax.plot(time, data['GL1_live_count'] / 1e7, label='Gl1_live_count')
    ax.legend()

    # fig, ax = plt.subplots()
    # mbd_live_diffs = data['mbd_live_count'].diff().fillna(0)
    # clock_live_diffs = data['GL1_live_count'].diff().fillna(0)
    # clock_raw_diffs = data['GL1_clock_count'].diff().fillna(0)
    # mbd_live_rate = mbd_live_diffs / clock_raw_diffs
    # mbd_raw_rate_gl1 = mbd_live_rate / (clock_live_diffs / clock_raw_diffs)
    # clock_bins = np.arange(data['GL1_clock_count'].min(), data['GL1_clock_count'].max(), 1e7)
    # avg_mbd_live_rate = []
    # for i in range(len(clock_bins) - 1):
    #     bin_mask = (data['GL1_clock_count'] >= clock_bins[i]) & (data['GL1_clock_count'] < clock_bins[i + 1])
    #     bin_mbd_live_counts = data['mbd_live_count'][bin_mask]
    #     bin_mbd_raw_counts = data['mbd_raw_count'][bin_mask]
    #     bin_clock_raw_counts = data['GL1_clock_count'][bin_mask]
    #     bin_clock_live_counts = data['GL1_clock_count'][bin_mask]
    #     print(f'Bin {i}: MBD Live Counts: {bin_mbd_live_counts.values}, Clock Counts: {bin_clock_counts.values}')
    #     avg_mbd_live_rate.append((bin_mbd_live_counts.iloc[-1] - bin_mbd_live_counts.iloc[0]) /
    #                              (bin_clock_counts.iloc[1] - bin_clock_counts.iloc[0]) if len(bin_mbd_live_counts) > 1 else 0)
    # clock_bin_centers = (clock_bins[:-1] + clock_bins[1:]) / 2
    # ax.plot(clock_bin_centers / 1e7, avg_mbd_live_rate, 'o-', markersize=1, label='Avg MBD Live Rate (Binned by GL1 Clock Count)')

    # def compute_rate_sliding_window(data, clock_col, count_col, clock_freq_hz=10_000_000):
    #     clock_counts = data[clock_col].values
    #     detector_counts = data[count_col].values
    #
    #     start_idx = 0
    #     result = []
    #
    #     while True:
    #         # Current clock and count
    #         start_clock = clock_counts[start_idx]
    #         target_clock = start_clock + clock_freq_hz
    #
    #         # Find the first index where the clock exceeds the 1-second mark
    #         end_idx = np.searchsorted(clock_counts[start_idx:], target_clock, side='left') + start_idx
    #
    #         if end_idx >= len(clock_counts):
    #             break  # no more data beyond 1 second
    #
    #         delta_clock = clock_counts[end_idx] - clock_counts[start_idx]
    #         delta_count = detector_counts[end_idx] - detector_counts[start_idx]
    #
    #         rate = delta_count / (delta_clock / clock_freq_hz)
    #         mid_time = (clock_counts[start_idx] + clock_counts[end_idx]) / 2 / clock_freq_hz  # optional, for x-axis
    #
    #         result.append((mid_time, rate))
    #         start_idx = end_idx
    #
    #     # Convert to DataFrame
    #     return pd.DataFrame(result, columns=['time_sec', 'rate_hz'])

    # mbd_live_rate_df = compute_rate_sliding_window(data, 'GL1_clock_count', 'mbd_live_count')
    # mbd_raw_rate_df = compute_rate_sliding_window(data, 'GL1_clock_count', 'mbd_raw_count')
    # clock_live_rate_df = compute_rate_sliding_window(data, 'GL1_clock_count', 'GL1_live_count')
    mbd_live_rate_df = compute_rate_sliding_window_bco(data, time, 'mbd_live_count')
    mbd_raw_rate_df = compute_rate_sliding_window_bco(data, time,'mbd_raw_count')
    clock_live_rate_df = compute_rate_sliding_window_bco(data, time, 'GL1_live_count')
    clock_raw_rate_df = compute_rate_sliding_window_bco(data, time, 'GL1_clock_count')
    cor_mbd_rate = mbd_live_rate_df['rate_hz'] / (clock_live_rate_df['rate_hz'] / clock_raw_rate_df['rate_hz'])
    fig, ax = plt.subplots()
    ax.plot(mbd_raw_rate_df['time_sec'], mbd_raw_rate_df['rate_hz'], 'o-', markersize=1, color='blue', label='MBD Raw Rate')
    ax.plot(mbd_live_rate_df['time_sec'], mbd_live_rate_df['rate_hz'], 'o-', markersize=1, color='orange', label='MBD Live Rate')
    ax.plot(clock_live_rate_df['time_sec'], cor_mbd_rate, 'o-', markersize=1, color='red', label='MBD Live->Raw Rate')
    ax.set_xlabel('Time (s)')
    ax.legend()

    # plt.show()

    step_time_cushion = 1.0  # Time cushion in seconds to avoid transitions
    cad_df['start'] = pd.to_datetime(cad_df['start'])
    cad_df['end'] = pd.to_datetime(cad_df['end'])
    run_start = cad_df.iloc[0]['start']

    mid_times, avg_raw_rates, avg_cor_rates, se_raw_rates, se_cor_rates = [], [], [], [], []
    for index, row in cad_df.iterrows():
        start_time = (row['start'] - run_start).total_seconds() + step_time_cushion
        end_time = (row['end'] - run_start).total_seconds() - step_time_cushion
        mask = (time >= start_time) & (time <= end_time)
        step_duration = end_time - start_time
        step_data = data[mask]
        step_times = time[mask]

        mbd_live_rate_step_df = compute_rate_sliding_window_bco(step_data, step_times, 'mbd_live_count')
        mbd_raw_rate_step_df = compute_rate_sliding_window_bco(step_data, step_times, 'mbd_raw_count')
        clock_live_rate_step_df = compute_rate_sliding_window_bco(step_data, step_times, 'GL1_live_count')
        clock_raw_rate_step_df = compute_rate_sliding_window_bco(step_data, step_times, 'GL1_clock_count')
        cor_mbd_step_rate = mbd_live_rate_step_df['rate_hz'] / (clock_live_rate_step_df['rate_hz'] / clock_raw_rate_step_df['rate_hz'])

        mid_step_time = start_time + step_duration / 2
        avg_mbd_live_step_rate = np.mean(mbd_live_rate_step_df['rate_hz'])
        avg_mbd_raw_step_rate = np.mean(mbd_raw_rate_step_df['rate_hz'])
        avg_cor_mbd_step_rate = np.mean(cor_mbd_step_rate)

        se_raw_rate = ((step_data['mbd_raw_count'].iloc[-1] - step_data['mbd_raw_count'].iloc[0]) /
                       (step_times.iloc[-1] - step_times.iloc[0]))
        se_live_rate = ((step_data['mbd_live_count'].iloc[-1] - step_data['mbd_live_count'].iloc[0]) /
                       (step_times.iloc[-1] - step_times.iloc[0]))
        se_clock_raw_rate = ((step_data['GL1_clock_count'].iloc[-1] - step_data['GL1_clock_count'].iloc[0]) /
                             (step_times.iloc[-1] - step_times.iloc[0]))
        se_clock_live_rate = ((step_data['GL1_live_count'].iloc[-1] - step_data['GL1_live_count'].iloc[0]) /
                              (step_times.iloc[-1] - step_times.iloc[0]))
        se_cor_rate = se_live_rate / (se_clock_live_rate / se_clock_raw_rate)

        ax.scatter([mid_step_time], [avg_mbd_live_step_rate], marker='s', s=100, zorder=2, color='orange')
        ax.scatter([mid_step_time], [avg_mbd_raw_step_rate], marker='s', s=100, zorder=2, color='blue')
        ax.scatter([mid_step_time], [avg_cor_mbd_step_rate], marker='s', s=100, zorder=2, color='red')
        ax.scatter([mid_step_time], [se_raw_rate], marker='^', s=100, zorder=2, color='blue')
        ax.scatter([mid_step_time], [se_cor_rate], marker='^', s=100, zorder=2, color='red')


        mid_times.append(mid_step_time)
        avg_raw_rates.append(avg_mbd_raw_step_rate)
        avg_cor_rates.append(avg_cor_mbd_step_rate)
        se_raw_rates.append(se_raw_rate)
        se_cor_rates.append(se_cor_rate)

    fig, ax = plt.subplots()
    ax.scatter(mid_times, np.array(avg_raw_rates) - np.array(avg_cor_rates))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('MBD Raw - Corrected Rate (Hz)')
    fig.tight_layout()

    fig, ax = plt.subplots()
    ax.scatter(mid_times, np.array(avg_raw_rates) / np.array(avg_cor_rates))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('MBD Raw / Corrected Rate (Hz)')
    fig.tight_layout()

    plt.show()


def compare_scaled_live_triggers_steps(data, time, cad_df):
    """
    Compare the scaled live triggers of ZDC and MBD.
    """
    step_time_cushion = 1.0  # Time cushion in seconds to avoid transitions
    cad_df['start'] = pd.to_datetime(cad_df['start'])
    cad_df['end'] = pd.to_datetime(cad_df['end'])
    run_start = cad_df.iloc[0]['start']

    for index, row in cad_df.iterrows():
        start_time = (row['start'] - run_start).total_seconds() + step_time_cushion
        end_time = (row['end'] - run_start).total_seconds() - step_time_cushion
        mask = (time >= start_time) & (time <= end_time)
        step_data = data[mask]
        step_times = time[mask]

        step_zdc_live_trigger = step_data['zdc_SN_live_trigger']
        step_mbd_live_trigger = step_data['mbd_SN_live_trigger']
        step_mbd_scaled_trigger = step_data['mbd_SN_trigger']
        step_zdc_scaled_trigger = step_data['zdc_SN_trigger']

        print(f'Step {row["step"]}:')
        print(f'Number of ZDC live triggers: {step_zdc_live_trigger.sum()}')
        print(f'Number of scaled ZDC triggers: {step_zdc_scaled_trigger.sum()}')
        print(f'Number of MBD live triggers: {step_mbd_live_trigger.sum()}')
        print(f'Number of scaled MBD triggers: {step_mbd_scaled_trigger.sum()}')
        print(f'ZDC scaled to live trigger ratio: {step_zdc_scaled_trigger.sum() / step_zdc_live_trigger.sum() * 100}%')
        print(f'MBD scaled to live trigger ratio: {step_mbd_scaled_trigger.sum() / step_mbd_live_trigger.sum() * 100}%')


def compare_scaled_live_triggers(data, time, cad_df):
    """
    Compare the scaled live triggers of ZDC and MBD.
    """
    # Iterate through data:
    for i in range(100):
        zdc_scaled_trigger = data['zdc_SN_trigger'].iloc[i]
        zdc_live_trigger = data['zdc_SN_live_trigger'].iloc[i]
        mbd_scaled_trigger = data['mbd_SN_trigger'].iloc[i]
        mbd_live_trigger = data['mbd_SN_live_trigger'].iloc[i]
        print(f'MBD Scaled: {mbd_scaled_trigger}, MBD Live: {mbd_live_trigger}, ZDC Scaled: {zdc_scaled_trigger}, ZDC Live: {zdc_live_trigger}')


def compare_new_old_root_file(base_path, data, time):

    data_old, time_old = get_root_data_time(f'{base_path}Vernier_Scans/auau_oct_16_24/',
                                            root_file_name='54733_slimmed.root', tree_name='calo_tree',
                                            sub_dir='vertex_data_old/',
                                            branches=['BCO', 'mbd_zvtx', 'mbd_SN_trigger', 'zdc_SN_trigger',
                                                      'bunch'])

    indices = np.arange(159751, 159762)
    last_vtx = 0
    for i in indices:
        print(f'Index {i}:')
        print(f'Data New: BCO: {data.iloc[i]["BCO"]}, MBD Z Vtx: {data.iloc[i]["mbd_zvtx"]}, MBD Scaled: {data.iloc[i]["mbd_SN_trigger"]}, ZDC Scaled: {data.iloc[i]["zdc_SN_trigger"]}, Bunch: {data.iloc[i]["bunch"]}')
        print(f'Data Old: BCO: {data_old.iloc[i]["BCO"]}, MBD Z Vtx: {data_old.iloc[i]["mbd_zvtx"]}, MBD Scaled: {data_old.iloc[i]["mbd_SN_trigger"]}, ZDC Scaled: {data_old.iloc[i]["zdc_SN_trigger"]}, Bunch: {data_old.iloc[i]["bunch"]}\n')
        print(f'Is the same vertex? {data.iloc[i]["mbd_zvtx"] == last_vtx}')
        last_vtx = data.iloc[i]["mbd_zvtx"]
    input('Press Enter to continue...')

    last_mvtx, streak, streak_indices, ring_streak_index, ring_streaks = None, 0, [], [], []
    for i in range(len(data)):
        if not data.iloc[i]['mbd_SN_trigger']:
            continue
        if data.iloc[i]['mbd_zvtx'] == last_mvtx:
            streak += 1
        else:
            if streak > 0:
                ring_streak_index.append(i - streak)
                ring_streaks.append(streak)
                print(f'Streak of {streak} at index {i - streak} of {len(data)}')
                for j in streak_indices:  # Print the previous streak entries
                    print(f'Index {j}: BCO: {data.iloc[j]["BCO"]}, MBD Z Vtx: {data.iloc[j]["mbd_zvtx"]}, MBD Scaled: {data.iloc[j]["mbd_SN_trigger"]}, ZDC Scaled: {data.iloc[j]["zdc_SN_trigger"]}, Bunch: {data.iloc[j]["bunch"]}')
                streak = 0
            streak_indices = []
        last_mvtx = data.iloc[i]['mbd_zvtx']
        streak_indices.append(i)

    print(f'Found {len(ring_streaks)} streaks')
    print(f'Longest streak: {max(ring_streaks)}')
    print(f'Average streak length: {np.mean(ring_streaks)}')
    print(f'Streaks: {ring_streaks}')
    print(f'Streak indices: {ring_streak_index}')

    for i in range(len(data)):
        if i % 1000000 == 0:
            print(f'Index {i}/{len(data)}')
        mbd_scaled_trigger = data['mbd_SN_trigger'].iloc[i]
        mbd_scaled_trigger_old = data_old['mbd_SN_trigger'].iloc[i]
        mbd_z_vtx = data['mbd_zvtx'].iloc[i]
        mbd_z_vtx_old = data_old['mbd_zvtx'].iloc[i]
        if mbd_scaled_trigger != mbd_scaled_trigger_old:
            print(f'Index {i}: New MBD Scaled: {mbd_scaled_trigger}, Old MBD Scaled: {mbd_scaled_trigger_old}')
        if mbd_z_vtx != mbd_z_vtx_old:
            print(f'Index {i}: New MBD Z Vtx: {mbd_z_vtx}, Old MBD Z Vtx: {mbd_z_vtx_old}')


def get_step_rates(scan_path, cad_df, root_file_name=None):
    """
    Get the rates at each step from the data, using step boundaries defined in cad_df.
    """
    detectors = ['zdc', 'mbd']
    types = ['raw', 'live', 'cor']

    if root_file_name is None:
        data, time = get_root_data_time(scan_path)
    else:
        data, time = get_root_data_time(scan_path, root_file_name)

    step_time_cushion = 1.0  # Time cushion in seconds to avoid transitions
    cad_df['start'] = pd.to_datetime(cad_df['start'])
    cad_df['end'] = pd.to_datetime(cad_df['end'])
    run_start = cad_df.iloc[0]['start']

    rates = []
    for index, row in cad_df.iterrows():
        start_time = (row['start'] - run_start).total_seconds() + step_time_cushion
        end_time = (row['end'] - run_start).total_seconds() - step_time_cushion
        mask = (time >= start_time) & (time <= end_time)
        step_data = data[mask]
        # step_duration = end_time - start_time

        step_times = time[mask]
        step_fine_dur = step_times.iloc[-1] - step_times.iloc[0]
        clock_raw_rate = (step_data['GL1_clock_count'].iloc[-1] - step_data['GL1_clock_count'].iloc[0]) / step_fine_dur
        clock_live_rate = (step_data['GL1_live_count'].iloc[-1] - step_data['GL1_live_count'].iloc[0]) / step_fine_dur
        clock_scale = clock_raw_rate / clock_live_rate

        step_rates = {'step': row['step']}
        for detector in detectors:
            for count_type in types:
                col_name = f'{detector}_{count_type}_count' if count_type != 'cor' else f'{detector}_live_count'
                rate = (step_data[col_name].iloc[-1] - step_data[col_name].iloc[0]) / step_fine_dur

                if count_type == 'cor':
                    rate *= clock_scale

                # counts = step_data[col_name].diff().fillna(0)
                # rate = counts.sum() / step_duration  # Total counts divided by duration gives rate
                step_rates[f'{detector}_{count_type}_rate'] = rate

        rates.append(step_rates)

    return pd.DataFrame(rates)


def average_counts_in_time_windows(time_array, counts_array, window_size, bin_stat='mean'):
    """
    Compute average counts in fixed-width time windows using vectorized numpy/scipy.

    Parameters:
        time_array (np.ndarray): Monotonically increasing time values.
        counts_array (np.ndarray): Count values (same shape as time_array).
        window_size (float): Size of each time bin.
        bin_stat (str): Statistic to compute in each bin ('mean', 'sum', etc.).

    Returns:
        window_centers (np.ndarray): Center of each time bin.
        window_averages (np.ndarray): Average counts in each bin.
    """
    start_time = time_array[0]
    end_time = time_array[-1]

    # Define bin edges
    bin_edges = np.arange(start_time, end_time + window_size, window_size)

    # Compute average in each bin
    averages, _, _ = binned_statistic(time_array, counts_array, statistic=bin_stat, bins=bin_edges)

    # Compute bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return bin_centers, averages


def compute_rate_sliding_window_bco(data, time, count_col, time_step=1):
    detector_counts = data[count_col].values
    time = time.values

    start_idx = 0
    result = []

    while True:
        # Current clock and count
        start_time = time[start_idx]
        target_time = start_time + time_step

        # Find the first index where the clock exceeds the 1-second mark
        end_idx = np.searchsorted(time[start_idx:], target_time, side='left') + start_idx

        if end_idx >= len(time):
            break  # no more data beyond 1 second

        delta_time = time[end_idx] - time[start_idx]
        delta_count = detector_counts[end_idx] - detector_counts[start_idx]

        rate = delta_count / delta_time
        mid_time = (time[start_idx] + time[end_idx]) / 2

        result.append((mid_time, rate))
        start_idx = end_idx

    # Convert to DataFrame
    return pd.DataFrame(result, columns=['time_sec', 'rate_hz'])


if __name__ == '__main__':
    main()

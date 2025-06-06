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
import ROOT


def main():
    if platform.system() == 'Windows':
        base_path = 'C:/Users/Dylan/Desktop/'
    else:
        base_path = '/local/home/dn277127/Bureau/'

    base_path_auau = f'{base_path}Vernier_Scans/auau_oct_16_24/'
    out_root_file_path = f'{base_path_auau}vertex_data/54733_vertex_distributions.root'
    out_root_file_path_no_zdc_coinc = f'{base_path_auau}vertex_data/54733_vertex_distributions_no_zdc_coinc.root'
    out_root_file_path_bbb = f'{base_path_auau}vertex_data/54733_vertex_distributions_bunch_by_bunch.root'
    out_root_file_path_no_zdc_coinc_bbb = f'{base_path_auau}vertex_data/54733_vertex_distributions_no_zdc_coinc_bunch_by_bunch.root'
    cad_data_path = f'{base_path_auau}combined_cad_step_data.csv'

    cad_df = pd.read_csv(cad_data_path, sep=',')
    print(cad_df)

    data, time = get_root_data_time(base_path_auau, root_file_name='54733_slimmed.root', tree_name='calo_tree',
                                    branches=['BCO', 'mbd_zvtx', 'mbd_SN_trigger', 'zdc_SN_trigger',
                                              'mbd_raw_count', 'zdc_raw_count', 'mbd_live_count', 'zdc_live_count',
                                              'bunch'])

    # plot_rates(data, time, cad_df)
    # more_rate_plotting(data, time, cad_df)
    # get_step_z_vertex_dists(data, time, cad_df, plot=False, out_path=out_root_file_path)
    # get_step_z_vertex_dists(data, time, cad_df, plot=False, out_path=out_root_file_path_no_zdc_coinc, zdc_coincidence=False)
    get_bunch_by_bunch_step_z_vertex_dists(data, time, cad_df, out_path=out_root_file_path_bbb)
    get_bunch_by_bunch_step_z_vertex_dists(data, time, cad_df, out_path=out_root_file_path_no_zdc_coinc_bbb, zdc_coincidence=False)

    print('donzo')


def get_root_data_time(scan_path, root_file_name='54733_slimmed.root', tree_name='calo_tree', branches=None):
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
                    'mbd_live_count', 'zdc_live_count']

    root_file_path = f'{scan_path}vertex_data/{root_file_name}'

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


def get_step_z_vertex_dists(data, time, cad_df, plot=True, out_path="output_histograms.root", zdc_coincidence=True):
    """
    Get the steps from the data and cad_df, plot and save histograms to a ROOT file.
    :param data: Data from the root file.
    :param time: Time array.
    :param cad_df: Dataframe with CAD step data.
    :param plot: Whether to plot the histograms.
    :param out_path: Name of the output ROOT file.
    :param zdc_coincidence: Whether to include ZDC coincidence in the histograms.
    :return: None
    """
    step_time_cushion = 1.0  # Time cushion in seconds to avoid transitions
    binning = np.linspace(-300, 300, 201)
    nbins = len(binning) - 1

    root_file = ROOT.TFile(out_path, "RECREATE")

    # Get the start and end times of each step
    cad_df['start'] = pd.to_datetime(cad_df['start'])
    cad_df['end'] = pd.to_datetime(cad_df['end'])
    run_start = cad_df.iloc[0]['start']

    for index, row in cad_df.iterrows():
        start_time = (row['start'] - run_start).total_seconds() + step_time_cushion
        end_time = (row['end'] - run_start).total_seconds() - step_time_cushion
        mask = (time >= start_time) & (time <= end_time)
        step_data = data[mask]
        step_duration = end_time - start_time

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
        hist_counts = raw_counts / step_duration
        bin_errors = np.sqrt(raw_counts) / step_duration

        # Create and fill ROOT histogram
        hist_name = f"step_{index}"
        hist = ROOT.TH1F(hist_name, f"Step {index} MBD+ZDC Coincidence Rate;MBD Z Vertex (cm);Rate (Hz)",
                         nbins, binning[0], binning[-1])

        for i in range(nbins):
            hist.SetBinContent(i + 1, hist_counts[i])
            hist.SetBinError(i + 1, bin_errors[i])

        hist.Write()  # Write to file

        print(f"Step {index}: Start Time: {start_time}, End Time: {end_time}, Duration: {step_duration:.2f} s, "
              f"Data Points: {len(step_data)}, Rate: {len(step_data) / step_duration:.2f} Hz")

        if plot:
            fig, ax = plt.subplots()
            ax.hist(raw_mbd_zvtx, bins=binning, alpha=0.5, label='Raw MBD Z Vertex')
            ax.hist(mbd_zvtx_mbd_coinc, bins=binning, alpha=0.5, label='MBD Coincidence')
            ax.hist(mbd_zvtx_mbd_zdc_coinc, bins=binning, alpha=0.5, label='MBD+ZDC Coincidence')
            ax.set_xlabel('MBD Z Vertex (cm)')
            ax.legend()

            fig, ax = plt.subplots()
            ax.bar(binning[:-1], hist_counts, yerr=bin_errors, width=np.diff(binning), align='edge',
                   alpha=0.5, label='Rate with Error Bars')
            ax.set_xlabel('MBD Z Vertex (cm)')
            ax.set_ylabel('Rate (Hz)')
            ax.legend()
            plt.show()

    root_file.Close()


def get_bunch_by_bunch_step_z_vertex_dists(data, time, cad_df, out_path="output_histograms.root", zdc_coincidence=True):
    """
    Get the steps from the data and cad_df, plot and save histograms to a ROOT file.
    :param data: Data from the root file.
    :param time: Time array.
    :param cad_df: Dataframe with CAD step data.
    :param plot: Whether to plot the histograms.
    :param out_path: Name of the output ROOT file.
    :param zdc_coincidence: Whether to include ZDC coincidence in the histograms.
    :return: None
    """
    step_time_cushion = 1.0  # Time cushion in seconds to avoid transitions
    binning = np.linspace(-300, 300, 201)
    nbins = len(binning) - 1

    root_file = ROOT.TFile(out_path, "RECREATE")

    # Get the start and end times of each step
    cad_df['start'] = pd.to_datetime(cad_df['start'])
    cad_df['end'] = pd.to_datetime(cad_df['end'])
    run_start = cad_df.iloc[0]['start']

    bunch_numbers = np.sort(data['bunch'].unique())

    for index, row in cad_df.iterrows():
        start_time = (row['start'] - run_start).total_seconds() + step_time_cushion
        end_time = (row['end'] - run_start).total_seconds() - step_time_cushion
        mask = (time >= start_time) & (time <= end_time)
        step_data = data[mask]
        step_duration = end_time - start_time

        raw_mbd_zvtx = step_data['mbd_zvtx']
        zdc_ns_trigger = step_data['zdc_SN_trigger']
        mbd_ns_trigger = step_data['mbd_SN_trigger']
        mbd_zvtx_mbd_coinc = raw_mbd_zvtx[(mbd_ns_trigger == 1)]
        mbd_zvtx_mbd_zdc_coinc = raw_mbd_zvtx[(zdc_ns_trigger == 1) & (mbd_ns_trigger == 1)]

        for bunch in bunch_numbers:
            bunch_mask = step_data['bunch'] == bunch
            # Raw counts for error calculation
            if zdc_coincidence:
                raw_counts, _ = np.histogram(mbd_zvtx_mbd_zdc_coinc[bunch_mask], bins=binning)
            else:
                raw_counts, _ = np.histogram(mbd_zvtx_mbd_coinc[bunch_mask], bins=binning)
            hist_counts = raw_counts / step_duration
            bin_errors = np.sqrt(raw_counts) / step_duration

            # Create and fill ROOT histogram
            hist_name = f"step_{index}_bunch_{bunch}"
            hist = ROOT.TH1F(hist_name, f"Step {index} Bunch {bunch} MBD+ZDC Coincidence Rate;MBD Z Vertex (cm);Rate (Hz)",
                             nbins, binning[0], binning[-1])

            for i in range(nbins):
                hist.SetBinContent(i + 1, hist_counts[i])
                hist.SetBinError(i + 1, bin_errors[i])

            hist.Write()  # Write to file

        print(f"Step {index}: Start Time: {start_time}, End Time: {end_time}, Duration: {step_duration:.2f} s, "
              f"Data Points: {len(step_data)}, Rate: {len(step_data) / step_duration:.2f} Hz")

    root_file.Close()


def get_step_rates(scan_path, cad_df):
    """
    Get the rates at each step from the data, using step boundaries defined in cad_df.
    """
    data, time = get_root_data_time(scan_path)

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
        step_duration = end_time - start_time

        detectors = ['zdc', 'mbd']
        types = ['raw', 'live']

        step_rates = {'step': row['step']}
        for detector in detectors:
            for count_type in types:
                col_name = f'{detector}_{count_type}_count'

                counts = step_data[col_name].diff().fillna(0)
                rate = counts.sum() / step_duration  # Total counts divided by duration gives rate
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



if __name__ == '__main__':
    main()

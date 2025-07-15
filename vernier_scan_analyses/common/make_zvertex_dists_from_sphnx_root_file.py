#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 23 15:39 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/make_zvertex_dists_from_sphnx_root_file

@author: Dylan Neff, dn277127
"""

import platform
import numpy as np
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
import pandas as pd
import uproot
import ROOT

from analyze_sphnx_root_file import get_root_data_time


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
    file_name = 'calofit_54733.root'

    cad_df = pd.read_csv(cad_data_path, sep=',')
    print(cad_df)

    data, time = get_root_data_time(base_path_auau, root_file_name=file_name, tree_name='calo_tree', sub_dir=sub_dir,
                                    branches=['BCO', 'mbd_zvtx', 'mbd_SN_trigger', 'zdc_SN_trigger',
                                              'mbd_SN_live_trigger', 'zdc_SN_live_trigger',
                                              'GL1_clock_count', 'GL1_live_count',
                                              'mbd_raw_count', 'zdc_raw_count', 'mbd_live_count', 'zdc_live_count',
                                              'bunch'])
    print(data.columns)

    get_step_z_vertex_dists(data, time, cad_df, plot=False, out_path=out_root_file_path)
    get_step_z_vertex_dists(data, time, cad_df, plot=False, out_path=out_root_file_path_no_zdc_coinc, zdc_coincidence=False)
    get_bunch_by_bunch_step_z_vertex_dists(data, time, cad_df, out_path=out_root_file_path_bbb)
    get_bunch_by_bunch_step_z_vertex_dists(data, time, cad_df, out_path=out_root_file_path_no_zdc_coinc_bbb, zdc_coincidence=False)
    plt.show()
    print('donzo')


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
        step_num = row['step']

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
        hist_name = f"step_{step_num}"
        hist = ROOT.TH1F(hist_name, f"Step {step_num} MBD+ZDC Coincidence Rate;MBD Z Vertex (cm);Rate (Hz)",
                         nbins, binning[0], binning[-1])

        for i in range(nbins):
            hist.SetBinContent(i + 1, hist_counts[i])
            hist.SetBinError(i + 1, bin_errors[i])

        hist.Write()  # Write to file

        print(f"Step {step_num}: Start Time: {start_time}, End Time: {end_time}, Duration: {step_duration:.2f} s, "
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
        step_num = row['step']

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
            hist_name = f"step_{step_num}_bunch_{bunch}"
            hist = ROOT.TH1F(hist_name, f"Step {step_num} Bunch {bunch} MBD+ZDC Coincidence Rate;MBD Z Vertex (cm);Rate (Hz)",
                             nbins, binning[0], binning[-1])

            for i in range(nbins):
                hist.SetBinContent(i + 1, hist_counts[i])
                hist.SetBinError(i + 1, bin_errors[i])

            hist.Write()  # Write to file

        print(f"Step {step_num}: Start Time: {start_time}, End Time: {end_time}, Duration: {step_duration:.2f} s, "
              f"Data Points: {len(step_data)}, Rate: {len(step_data) / step_duration:.2f} Hz")

    root_file.Close()


if __name__ == '__main__':
    main()

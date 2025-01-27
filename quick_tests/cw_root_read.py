#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on January 27 08:54 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/cw_root_read

@author: Dylan Neff, dn277127
"""

import numpy as np
import matplotlib.pyplot as plt

import awkward as ak
import uproot
from matplotlib.style import library


def main():
    root_test_path = '/local/home/dn277127/Bureau/vernier_scan/hold/ntuple_run48029_00022.root'
    with uproot.open(root_test_path) as root:
        print(root.keys())
        tree = root['tree']
        print(tree.keys())
        evtID = tree['evtID'].array()
        live_trig_dec = tree['LiveTrigger_Decimal'].array(library='np')
        print(live_trig_dec)
        print([bin(i) for i in live_trig_dec[:10]])  # convert first few live_trig_dec to binary and print
        trigger_input_vec = tree['TriggerInput_Vec'].array()
        print('Trigger input vec:')
        print(trigger_input_vec)
        live_trigger_vec = tree['LiveTrigger_Vec'].array()
        print('Live trigger vec:')
        print(live_trigger_vec)
        scaled_trigger_vec = tree['ScaledTrigger_Vec'].array()
        print('Scaled trigger vec:')
        print(scaled_trigger_vec)
        gl1_scalers_mbdns_raw = tree['GL1Scalers_MBDNS_raw'].array(library='np')
        print('GL1 Scalers MBDNS raw:')
        print(gl1_scalers_mbdns_raw)
        gl1_scalers_clock_raw = tree['GL1Scalers_clock_raw'].array(library='np')
        print('GL1 Scalers Clock raw:')
        print(gl1_scalers_clock_raw)
        mbd_z_vtx = tree['mbd_z_vtx'].array()
        print(mbd_z_vtx)

    total_events = len(live_trig_dec)
    mbd_ns_trigger_id, zdc_coinc_trigger_id = 10, 3
    mbd_ns_triggers = np.sum(live_trig_dec & (1 << mbd_ns_trigger_id) != 0)
    zdc_coinc_triggers = np.sum(live_trig_dec & (1 << zdc_coinc_trigger_id) != 0)
    mbd_and_zdc_triggers = np.sum((live_trig_dec & (1 << mbd_ns_trigger_id) != 0) & (live_trig_dec & (1 << zdc_coinc_trigger_id) != 0))

    print(f'Total events: {total_events}')
    print(f'MBD NS triggers: {mbd_ns_triggers}')
    print(f'ZDC Coincidence triggers: {zdc_coinc_triggers}')
    print(f'MBD NS and ZDC Coincidence triggers: {mbd_and_zdc_triggers}')

    # Plot mbd_z_vtx distribution in two cases: When MBD NS trigger and when MBD NS + ZDC Coincidence trigger
    # Overlay the two plots
    mbd_live_z_vtx = mbd_z_vtx[(live_trig_dec & (1 << mbd_ns_trigger_id) != 0) & (mbd_z_vtx != -999)]
    mbd_zdc_live_z_vtx = mbd_z_vtx[(live_trig_dec & (1 << mbd_ns_trigger_id) != 0) &
                                   (live_trig_dec & (1 << zdc_coinc_trigger_id) != 0) & (mbd_z_vtx != -999)]

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.hist(mbd_live_z_vtx, bins=100, alpha=0.5, label='MBD NS Trigger')
    ax.hist(mbd_zdc_live_z_vtx, bins=100, alpha=0.5, label='MBD NS + ZDC Coincidence Trigger')
    ax.set_xlabel('Z Vertex')
    ax.set_ylabel('Counts')
    ax.set_title(f'MBD Z Vertex Distribution for {root_test_path}')
    ax.legend()
    fig.tight_layout()

    # Next show the same but with the distributions normalized
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.hist(mbd_live_z_vtx, bins=100, alpha=0.5, label='MBD NS Trigger', density=True)
    ax.hist(mbd_zdc_live_z_vtx, bins=100, alpha=0.5, label='MBD NS + ZDC Coincidence Trigger', density=True)
    ax.set_xlabel('Z Vertex')
    ax.set_ylabel('Density')
    ax.set_title(f'MBD Z Vertex Distribution for {root_test_path}')
    ax.legend()
    fig.tight_layout()

    # Plot MBD scalers and mbd ns scalars rate
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(gl1_scalers_mbdns_raw)
    ax.set_xlabel('Event')
    ax.set_ylabel('MBD NS Scaler')
    ax.set_title('MBD NS Scaler Raw Values')
    fig.tight_layout()

    # Get MBD scalers rate using clock
    clock_freq = 250e6
    clock_period = 1 / clock_freq
    clock_diff = np.diff(gl1_scalers_clock_raw)
    clock_diff_centers = (gl1_scalers_clock_raw[1:] + gl1_scalers_clock_raw[:-1]) / 2
    mbd_scalar_diff = np.diff(gl1_scalers_mbdns_raw)
    gl1_scalers_mbdns_rate = mbd_scalar_diff / clock_diff / clock_period

    # Do an n point moving average on the rate
    n = 100
    gl1_scalers_mbdns_rate_avg = np.convolve(gl1_scalers_mbdns_rate, np.ones(n) / n, mode='valid')
    clock_diff_centers_avg = np.convolve(clock_diff_centers, np.ones(n) / n, mode='valid')

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(clock_diff_centers, gl1_scalers_mbdns_rate)
    ax.plot(clock_diff_centers_avg, gl1_scalers_mbdns_rate_avg, color='red')
    ax.set_xlabel('Clock')
    ax.set_ylabel('MBD NS Scaler Rate (Hz)')
    ax.set_title('MBD NS Scaler Rate')
    fig.tight_layout()

    plt.show()

    print('donzo')


if __name__ == '__main__':
    main()

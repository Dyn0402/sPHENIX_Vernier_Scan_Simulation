#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on December 17 7:10 PM 2024
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/get_test_mbd_zdists.py

@author: Dylan Neff, Dylan
"""

import platform
import os

import numpy as np
import matplotlib.pyplot as plt

import uproot



def main():
    if platform.system() == 'Windows':
        root_dir = 'C:/Users/Dylan/Desktop/vernier_scan/hold/'
    else:
        print('Don\'t have a root directory for this system.')
        return

    root_name_flag = 'run_'
    tree_name = 'mbd_vertex_tree'
    event_vars = ['bunchnumber', 'mbd_z_vtx', 'mbd_z_vtx_err', 'mbd_t0', 'mbd_t0_err']

    for file_name in os.listdir(root_dir):
        if not file_name.endswith('.root') or root_name_flag not in file_name:
            continue
        file_path = root_dir + file_name
        with uproot.open(file_path) as file:
            tree = file[tree_name]
            events = tree.arrays(event_vars)

            mbd_z_vtx = events['mbd_z_vtx']

            # Count and remove z_vertices of -999 error code
            num_z_vtx = len(mbd_z_vtx)
            mbd_z_vtx = mbd_z_vtx[mbd_z_vtx != -999]
            num_z_vtx_good = len(mbd_z_vtx)
            num_z_vtx_bad = num_z_vtx - num_z_vtx_good

            # Plot z vertex distribution
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            ax.hist(mbd_z_vtx, bins=100)
            ax.set_xlabel('Z Vertex')
            ax.set_ylabel('Counts')
            ax.set_title(f'Z Vertex Distribution for {file_name}\n{num_z_vtx_good} good, {num_z_vtx_bad} bad')
    plt.show()


    print('donzo')


if __name__ == '__main__':
    main()

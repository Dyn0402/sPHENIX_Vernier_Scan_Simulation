#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on February 26 18:00 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/plot_raw_mbd_z_vertex_distributions

@author: Dylan Neff, dn277127
"""

import platform
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from vernier_z_vertex_fitting import get_mbd_z_dists


def main():
    if platform.system() == 'Linux':
        base_path = '/local/home/dn277127/Bureau/vernier_scan/'
    else:  # Windows
        base_path = 'C:/Users/Dylan/Research/sPHENIX_Vernier_Scan_Simulation/'
    vernier_scan_date = 'Aug12'
    dist_root_file_name = f'vernier_scan_{vernier_scan_date}_mbd_vertex_z_distributions.root'
    z_vertex_root_path = f'{base_path}vertex_data/{dist_root_file_name}'
    mbd_hists = get_mbd_z_dists(z_vertex_root_path, first_dist=False)

    for hists in mbd_hists:
        print(hists['scan_axis'], hists['scan_step'])

    mbd_hists_df = pd.DataFrame(mbd_hists)
    print(mbd_hists_df)

    orientations = mbd_hists_df['scan_axis'].unique()
    for orientation in orientations:
        orientation_hists = mbd_hists_df[mbd_hists_df['scan_axis'] == orientation]
        # fig, ax = plt.subplots(3, 4, sharex='all')
        # ax = ax.flatten()
        # for i, (_, row) in enumerate(orientation_hists.iterrows()):
        #     counts = row['counts']
        #     centers = row['centers']
        #     bin_width = centers[1] - centers[0]
        #     ax[i].bar(centers, counts, width=bin_width, align='center')
        # fig.tight_layout()
        # Combined distribution plot
        fig_dists, axs = plt.subplots(nrows=3, ncols=4, figsize=(22, 10), sharex='all')
        axs = axs.flatten()
        fig_dists.subplots_adjust(hspace=0.0, wspace=0.0, top=0.995, bottom=0.045, left=0.01, right=0.995)

        for i, (ax, (_, row)) in enumerate(zip(axs, orientation_hists.iterrows())):
            if i >= 8:
                ax.set_xlabel('Z Vertex (cm)')
            if i % 4 == 0:
                ax.set_ylabel('Counts (scaled)')
            step = row['scan_step']
            counts = row['counts']
            centers = row['centers']

            ax.annotate(f'Step {step}', xy=(0.05, 0.75), xycoords='axes fraction', fontsize=15, va='top', ha='left')
            max_y = int(max(counts))
            ax.axhline(max_y, color='black', alpha=0.3, zorder=0, linestyle='-')
            ax.annotate(f'{max_y}', xy=(250, max_y * 0.995), xycoords='data', fontsize=10, alpha=0.5, va='top',
                        ha='right')
            ax.set_yticks([])

            width = centers[1] - centers[0]
            ax.bar(centers, counts, width=width)

        axs[0].legend()
    plt.show()

    print('donzo')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 06 03:36 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/abort_gap_check

@author: Dylan Neff, dn277127
"""

import platform
import numpy as np
import matplotlib.pyplot as plt
import uproot


def main():
    if platform.system() == 'Windows':
        base_path = 'C:/Users/Dylan/Desktop/'
    else:
        base_path = '/local/home/dn277127/Bureau/'

    base_path = f'{base_path}Vernier_Scans/auau_oct_16_24/'
    out_root_file_path_no_zdc_coinc_bbb = f'{base_path}vertex_data/54733_vertex_distributions_no_zdc_coinc_bunch_by_bunch.root'

    fig, ax = plt.subplots(figsize=(10, 6))
    hist_total = None
    with uproot.open(out_root_file_path_no_zdc_coinc_bbb) as f:
        for bunch_num in range(110, 120):
            hist = f[f'step_11_bunch_{bunch_num}']
            centers = hist.axis().centers()
            counts = hist.counts()
            count_errs = hist.errors()
            count_errs[count_errs == 0] = 1

            ax.bar(centers, counts, yerr=count_errs, align='center', width=centers[1] - centers[0], label=f'Bunch {bunch_num}', alpha=0.5)
            if hist_total is None:
                hist_total = counts
            else:
                hist_total += counts
    ax.set_xlabel('Z Vertex Position (cm)')

    fig_sum, ax_sum = plt.subplots()
    ax_sum.bar(centers, hist_total, align='center', width=centers[1] - centers[0], label='Total', alpha=0.5)
    ax_sum.set_yscale('log')
    ax_sum.set_xlabel('Z Vertex Position (cm)')

    plt.show()

    print('donzo')


if __name__ == '__main__':
    main()

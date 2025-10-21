#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 20 3:28 PM 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/compare_bbb_rate_only_with_original.py

@author: Dylan Neff, Dylan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from common_logistics import set_base_path


def main():
    base_path = set_base_path()
    og_analysis_mbd_path = '../../mbd_cross_section_distribution.csv'
    bunch_by_bunch_path = f'{base_path}Vernier_scans/pp_aug_12_24/Figures/Beam_Param_Inferences/Bunch_By_Bunch_Rate_Only/cross_sections_bunch_by_bunch_rate_only.csv'

    og_analysis_mbd_df = pd.read_csv(og_analysis_mbd_path)
    bunch_by_bunch_df = pd.read_csv(bunch_by_bunch_path)

    detector = 'MBD'
    bunch_by_bunch_df = bunch_by_bunch_df[bunch_by_bunch_df['absolute_rate_name'] == detector]

    og_most_prob = 25.2
    og_left_err, og_right_err = 1.7, 2.3

    fig, ax = plt.subplots(figsize=(8, 5))
    centers = og_analysis_mbd_df['bin_center'].values
    hist_vals = og_analysis_mbd_df['hist'].values
    width = centers[1] - centers[0]
    edges = np.concatenate([centers - width/2, [centers[-1] + width/2]])
    ax.step(edges, np.append(hist_vals, hist_vals[-1]), where='post',
            color='k', label=r'Original Analysis $\sigma_{\text{MBD}}$ PDF')
    # Fill gray under the error region using a histogram
    ci_filter = (og_analysis_mbd_df['bin_center'] >= (og_most_prob - og_left_err)) & (og_analysis_mbd_df['bin_center'] <= (og_most_prob + og_right_err))
    ax.bar(og_analysis_mbd_df['bin_center'][ci_filter], og_analysis_mbd_df['hist'][ci_filter],
           width=og_analysis_mbd_df['bin_center'][1] - og_analysis_mbd_df['bin_center'][0],
           color='gray', alpha=0.5, label='Original Analysis 68% CI', align='center')

    # steps = pd.unique(bunch_by_bunch_df['step'])
    steps = [0]
    for step in steps:
        bunch_by_bunch_df_step = bunch_by_bunch_df[bunch_by_bunch_df['step'] == step]

        ax.hist(bunch_by_bunch_df_step['cross_section'], bins=10, density=True,
                alpha=0.5, label=f'Bunch-by-Bunch Rate Step {step}', histtype='stepfilled')

    ax.set_xlim(right=35)
    ax.set_xlabel('Cross Section [mb]')
    ax.set_ylabel('Probability')
    ax.set_title('Comparison of MBD Cross Section Distributions')
    ax.legend()

    fig.tight_layout()
    plt.show()

    print('donzo')


if __name__ == '__main__':
    main()

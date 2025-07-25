#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on July 24 5:27 PM 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/log_stacked_bar.py

@author: Dylan Neff, dylan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def main():
    # Create DataFrame
    df = pd.DataFrame({
        'Group 1': [5, 20, 30],
        'Group 2': [5, 15, 25],
        'Group 3': [5, 8, 12]
    }, index=['A', 'B', 'C'])

    total = df.sum(axis=1)

    # Normalize each group to itself
    df_normalized = df.div(total, axis=0) * 100  # Convert to percentage

    # Plot original values
    fig, ax = plt.subplots()
    df.plot(kind='bar', stacked=True, ax=ax)
    fig.tight_layout()

    # Plot original values in log
    fig, ax = plt.subplots()
    df.plot(kind='bar', stacked=True, ax=ax)
    ax.set_yscale('log')
    ax.set_ylim(bottom=0.5)
    fig.tight_layout()

    # Plot
    fig, axs = plt.subplots(nrows=2, sharex=True)
    df_normalized.plot(kind='bar', stacked=True, ax=axs[1])
    axs[1].set_ylabel('Percent Contribution %')

    # Sum the values across the columns
    axs[0].plot(['Category A', 'Category B', 'Category C'], total, marker='o', color='black', label='Total')
    axs[0].set_ylabel('Magnitude')

    plt.legend()

    plt.tight_layout()
    fig.subplots_adjust(hspace=0)
    plt.show()

    print('donzo')


if __name__ == '__main__':
    main()

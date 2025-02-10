#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 18 4:40 PM 2024
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/plot_cw_rates.py

@author: Dylan Neff, Dylan
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    file_path = 'cw_rates.txt'
    vertical_data, horizontal_data = read_cw_rates_from_file(file_path)
    plot_data(vertical_data, horizontal_data)

    print('donzo')


def read_cw_rates_from_file(file_path):
    vertical_pos = []
    vertical_values = []
    horizontal_pos = []
    horizontal_values = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

        # Determine if we are in the vertical or horizontal section
        current_section = None

        for line in lines:
            line = line.strip()
            if "Vertical pos" in line:
                current_section = "vertical"
                continue
            elif "Horizontal pos" in line:
                current_section = "horizontal"
                continue

            # Extracting values if in the correct section
            if current_section == "vertical":
                if line:
                    parts = line.split()
                    vertical_pos.append(float(parts[0]))
                    vertical_values.append(float(parts[1]))
            elif current_section == "horizontal":
                if line:
                    parts = line.split()
                    horizontal_pos.append(float(parts[0]))
                    horizontal_values.append(float(parts[1]))

    return (vertical_pos, vertical_values), (horizontal_pos, horizontal_values)


def plot_data(vertical_data, horizontal_data):
    # Sort data by position
    vertical_pos, vertical_values = zip(*sorted(zip(vertical_data[0], vertical_data[1])))
    horizontal_pos, horizontal_values = zip(*sorted(zip(horizontal_data[0], horizontal_data[1])))

    plt.figure(figsize=(12, 6))

    # Vertical Plot
    plt.subplot(1, 2, 1)
    plt.plot(vertical_pos, vertical_values, marker='o', color='b', label='Vertical Position')
    plt.title('Vertical Position vs Values')
    plt.xlabel('Position')
    plt.ylabel('Values (MBDNS_ana_ttft)')
    plt.grid(True)
    plt.legend()

    # Horizontal Plot
    plt.subplot(1, 2, 2)
    plt.plot(horizontal_pos, horizontal_values, marker='o', color='r', label='Horizontal Position')
    plt.title('Horizontal Position vs Values')
    plt.xlabel('Position')
    plt.ylabel('Values (MBDNS_ana_ttft)')
    plt.grid(True)
    plt.legend()

    # Show plots
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

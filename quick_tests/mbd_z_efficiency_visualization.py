#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on September 26 3:46 PM 2024
Created in PyCharm
Created as sphenix_polarimetry/mbd_z_efficiency_visualization.py

@author: Dylan Neff, Dylan
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import alpha


def main():
    # gaussian_model()
    quadratic_exponential_model()
    print('donzo')


def quadratic_exponential_model():
    # Generate z values
    amp = 0.57
    z_vals = np.linspace(-300, 300, 500)

    # Calculate efficiency values for different parameter sets
    efficiency_vals1 = efficiency(z_vals, z_quad=700, z_switch=200, steepness=-0.1) * amp
    efficiency_vals2 = efficiency(z_vals, z_quad=700, z_switch=200, steepness=-0.2) * amp

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.plot(z_vals, efficiency_vals1, label=r'$k=0.1$', color='blue', alpha=0.7)
    plt.plot(z_vals, efficiency_vals2, label=r'$k=0.2$', color='red', alpha=0.7)
    plt.axhline(amp, color='gray', linestyle='--', label='Amplitude')
    plt.axhline(0.52, color='gray', linestyle='--', label='Minimum Efficiency')
    plt.axvline(250, color='gray', linestyle='-', label='MBD Edge')
    plt.axvline(-250, color='gray', linestyle='-')

    plt.xlabel(r'$z_{\mathrm{truth}}$ [cm]')
    plt.ylabel('Efficiency')
    plt.legend(loc='best')
    plt.grid(True)
    plt.title('Efficiency vs z')
    plt.show()


def gaussian_model():
    widths = [175, 500, 1000]  # cm
    amp = 0.57

    x_lim = (-219, 219)

    x = np.linspace(x_lim[0], x_lim[1], 1000)
    fig, ax = plt.subplots()
    fig_norm, ax_norm = plt.subplots()

    for width in widths:
        y = gaus(x, amp, 0, width)
        ax.plot(x, y, label=f'Width: {width} cm')
        y_norm = gaus(x, 1, 0, width)
        if width == 500:
            ax_norm.plot(x, y_norm, color='orange', label=f'Width: {width} cm')

    ax.axhline(amp, color='gray', linestyle='--', label='Amplitude')
    ax.axhline(0.52, color='gray', linestyle='--', label='Minimum Efficiency')
    ax.set_xlim(x_lim)
    ax.set_ylim(0, 1.09)
    ax.set_xlabel(r'$v_z^{truth}$')
    ax.set_ylabel('Efficiency')
    ax.legend()
    fig.tight_layout()

    ax_norm.axhline(1, color='gray', linestyle='--')
    ax_norm.set_xlim(x_lim)
    ax_norm.set_ylim(0, 1.09)
    ax_norm.set_xlabel(r'$v_z^{truth}$')
    ax_norm.set_ylabel('Relative Efficiency')
    ax_norm.legend()
    fig_norm.tight_layout()

    plt.show()


def gaus(x, a, b, c):
    return a * np.exp(-(x - b) ** 2 / (2 * c ** 2))


# Define the efficiency function
# def efficiency(z, z0=200, z1=200, alpha=0.05):
#     # Quadratic in the middle, exponential decay after |z1|
#     quadratic_part = 1 - (z / z0)**2
#     exponential_decay = np.where(np.abs(z) > z1, np.exp(-alpha * (np.abs(z) - z1)), 1)
#     return np.clip(quadratic_part * exponential_decay, 0, 1)


# def efficiency(z, z_quad=700, z_switch=200, z_end=250):
#     # Quadratic part
#     quadratic_part = np.where(np.abs(z) <= z_switch, 1 - (z / z_quad) ** 2, 0)
#
#     # Linear part
#     linear_part = np.where(np.abs(z) > z_switch,
#                            (1 - (z_switch / z_quad) ** 2 - (np.abs(z) - z_switch) / (z_end - z_switch)),
#                            0)
#
#     # Combine both parts
#     efficiency_value = quadratic_part + linear_part
#
#     # Clip the values to ensure they stay within [0, 1]
#     return np.clip(efficiency_value, 0, 1)


def efficiency(z, z_quad=700, z_switch=200, steepness=-0.1):
    # Quadratic part
    quadratic_part = np.where(np.abs(z) <= z_switch, 1 - (z / z_quad) ** 2, 0)

    # Calculate the value and derivative of the quadratic part at z_switch
    q_sw = 1 - (z_switch / z_quad) ** 2
    dq_sw = -2 * z_switch / z_quad ** 2

    if not 0 < q_sw < 1:
        raise ValueError('Quadratic part must be between 0 and 1 at z_switch.')

    # Sigmoid parameters
    # We set the steepness based on how quickly we want the transition
    print(dq_sw / q_sw)

    sigmoid_z0 = z_switch - np.log(steepness * q_sw / dq_sw - 1) / steepness
    sigmoid_a = q_sw * (1 + np.exp(-steepness * (z_switch - sigmoid_z0)))

    # Sigmoid part
    sigmoid_part = np.where(np.abs(z) > z_switch,
                            sigmoid_a / (1 + np.exp(-steepness * (np.abs(z) - sigmoid_z0))),
                            0)

    # Combine both parts
    efficiency_value = quadratic_part + sigmoid_part

    # Clip the values to ensure they stay within [0, 1]
    return np.clip(efficiency_value, 0, 1)


if __name__ == '__main__':
    main()

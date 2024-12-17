#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on September 12 12:49 2024
Created in PyCharm
Created as sphenix_polarimetry/hourglass_visualization
@author: Dylan Neff, dn277127
"""

import numpy as np
import matplotlib.pyplot as plt

from BunchCollider import BunchCollider

mu = '\u03BC'

def main():
    visualize_two_beams()
    # visualize_one_beam()
    plt.show()
    print('donzo')


def visualize_one_beam():
    beta_star = 85  # cm  Scaling factor for x^2/a^2 term
    sigma = 170  # um  Full beam width at collision point
    z = np.linspace(-200, 200, 400)  # cm  z values
    y1 = sigma * f(z, beta_star, 0)  # um  show the 1 sigma line
    y1_reflected = -y1  # um  reflected 1 sigma line

    # Plot the original and reflected curves
    fig, ax = plt.subplots(figsize=(7, 3), dpi=144)
    ax.plot(z, y1, color='blue')
    ax.plot(z, y1_reflected, color='blue')
    ax.fill_between(z, y1, y1_reflected, color='blue', alpha=0.1)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='-', alpha=0.5)

    # Draw a vertical line at the collision point from y=0 to y=1 sigma, label this as 1 sigma
    y1_z0 = sigma * f(0, beta_star, 0)
    ax.plot([0, 0], [0, y1_z0], color='black', linestyle='-', alpha=1)
    ax.text(0, y1_z0 / 2, r'$\sigma_0$', verticalalignment='center', horizontalalignment='right')

    # Draw a vertical line at 1 beta* from y=0 to y1 at this point, label this as 2 sigma
    y1_z_beta = sigma * f(beta_star, beta_star, 0)
    ax.plot([beta_star, beta_star], [0, y1_z_beta], color='black', linestyle='-', alpha=1)
    ax.text(beta_star, y1_z_beta / 2, r'$\sqrt{2}\sigma_0$', verticalalignment='center', horizontalalignment='left')
    # Draw a horizontal line between 0 and beta* at y=1 sigma, label this as beta*
    ax.plot([0, beta_star], [y1_z0 / 5, y1_z0 / 5], color='black', linestyle='-', alpha=1)
    ax.text(beta_star / 2, y1_z0 / 5, r'$\beta^*$', verticalalignment='bottom', horizontalalignment='center')
    ax.set_xlabel('z (cm)')
    ax.set_ylabel(f'y ({mu}m)')
    # ax.grid(True)
    fig.tight_layout()
    fig.subplots_adjust(left=0.1, right=0.995, top=0.99, bottom=0.14)


def visualize_two_beams():
    beta_star = 85  # cm
    sigma = 170  # um
    z = np.linspace(-200, 200, 400)  # cm
    y_offset_blue = 500  # um, vertical offset
    y_offset_yellow = 0  # um, vertical offset
    theta_yellow = np.deg2rad(0)  # rotation angle in degrees
    # theta_blue = np.deg2rad(0)  # rotation angle in degrees
    theta_blue = np.deg2rad(45)  # rotation angle in degrees

    # Blue beam parameters
    y1_blue = sigma * f(z, beta_star, 0)  # um
    y2_blue = -y1_blue  # reflected curve

    # Yellow beam parameters
    y1_yellow = sigma * f(z, beta_star, 0)  # um
    y2_yellow = -y1_yellow  # reflected curve

    # Rotate and offset the blue and yellow beams
    y1_blue_rot = rotate_curve_fixed_x(z, y1_blue, theta_blue, y_offset_blue)
    y2_blue_rot = rotate_curve_fixed_x(z, y2_blue, theta_blue, y_offset_blue)
    y1_yellow_rot = rotate_curve_fixed_x(z, y1_yellow, theta_yellow, y_offset_yellow)
    y2_yellow_rot = rotate_curve_fixed_x(z, y2_yellow, theta_yellow, y_offset_yellow)

    # Simulate to get z distribution
    collider = BunchCollider()
    collider.set_bunch_beta_stars(beta_star, beta_star)
    collider.set_bunch_offsets([0, y_offset_blue], [0, y_offset_yellow])
    collider.set_bunch_sigmas([sigma, sigma], [sigma, sigma])
    collider.set_bunch_crossing(0, theta_blue, 0, theta_yellow)

    fig, axs = plt.subplots(nrows=2, figsize=(8, 5), dpi=144, sharex='all')
    ax, ax_sim = axs

    # Plot blue beam
    ax.plot(z, y1_blue_rot, color='blue', label='Blue Beam')
    ax.plot(z, y2_blue_rot, color='blue')
    ax.fill_between(z, y1_blue_rot, y2_blue_rot, color='blue', alpha=0.1)

    # Plot yellow beam
    ax.plot(z, y1_yellow_rot, color='orange', label='Yellow Beam')
    ax.plot(z, y2_yellow_rot, color='orange')
    ax.fill_between(z, y1_yellow_rot, y2_yellow_rot, color='orange', alpha=0.3)

    ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='-', alpha=0.5)
    # ax.set_xlabel('z (cm)')
    ax.set_ylabel(f'y ({mu}m)')
    ax.legend()

    # Simulate and plot z distribution
    collider.run_sim_parallel()
    z_vals, z_dist = collider.get_z_density_dist()

    ax_sim.plot(z_vals, z_dist, color='black', label='Z Distribution')
    ax_sim.axhline(0, color='gray', linestyle='-', alpha=0.5)
    ax_sim.set_ylim(bottom=0)
    ax_sim.set_xlim(z[0], z[-1])
    ax_sim.set_xlabel('z (cm)')
    ax_sim.set_ylabel('Naked Luminosity Density')

    fig.tight_layout()

    fig_top, ax_top = plt.subplots(figsize=(7, 3), dpi=144)
    # Plot blue beam
    ax_top.plot(z, y1_blue_rot, color='blue', label='Blue Beam')
    ax_top.plot(z, y2_blue_rot, color='blue')
    ax_top.fill_between(z, y1_blue_rot, y2_blue_rot, color='blue', alpha=0.1)

    # Plot yellow beam
    ax_top.plot(z, y1_yellow_rot, color='orange', label='Yellow Beam')
    ax_top.plot(z, y2_yellow_rot, color='orange')
    ax_top.fill_between(z, y1_yellow_rot, y2_yellow_rot, color='orange', alpha=0.3)

    ax_top.axhline(0, color='gray', linestyle='-', alpha=0.5)
    ax_top.axvline(0, color='gray', linestyle='-', alpha=0.5)
    # ax.set_xlabel('z (cm)')
    ax_top.set_ylabel(f'y ({mu}m)')
    ax_top.set_xlabel('z (cm)')
    ax_top.legend()
    fig_top.tight_layout()

    # fig.subplots_adjust(left=0.1, right=0.995, top=0.98, bottom=0.14)

    # Calculate and plot overlap region
    # overlap = np.minimum(y1_blue_rot, y1_yellow_rot) - np.maximum(y2_blue_rot, y2_yellow_rot)
    # fig_overlap, ax_overlap = plt.subplots(figsize=(7, 3), dpi=144)
    # ax_overlap.plot(z, overlap, color='red', label='Overlap Region')
    # ax_overlap.axhline(0, color='gray', linestyle='-', alpha=0.5)
    # ax_overlap.set_xlabel('z (cm)')
    # ax_overlap.set_ylabel(f'y ({mu}m)')
    # ax_overlap.legend()
    # fig_overlap.tight_layout()


# Function for the original curve
def f(x, a, b):
    return np.sqrt(1 + (x ** 2 / a ** 2)) + b


# Function for the reflected curve
def reflected_f(x, a, b):
    return -f(x, a, b)


# Function to rotate y-values while keeping x-values fixed
def rotate_curve_fixed_x(x, y, theta, y_offset=0):
    # Rotate only the y-values, keep x fixed
    y_rot = y * np.cos(theta) - x * np.sin(theta) + y_offset
    return y_rot


if __name__ == '__main__':
    main()
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
    # visualize_two_beams()
    # visualize_one_beam()
    visualize_two_beams_3_angles()
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
    sigma = 150  # um
    z = np.linspace(-250, 250, 500)  # cm
    y_offset_blue = 900  # um, vertical offset
    y_offset_yellow = 0  # um, vertical offset
    theta_yellow = np.deg2rad(0)  # rotation angle in degrees
    # theta_blue = np.deg2rad(0)  # rotation angle in degrees
    theta_blue = 0.1e-3  # rotation angle in radians

    # Blue beam parameters
    y1_blue = sigma * f(z, beta_star, 0)  # um
    y2_blue = -y1_blue  # reflected curve

    # Yellow beam parameters
    y1_yellow = sigma * f(z, beta_star, 0)  # um
    y2_yellow = -y1_yellow  # reflected curve

    # Rotate and offset the blue and yellow beams
    y1_blue_rot = rotate_curve_fixed_x(z * 1e4, y1_blue, theta_blue, y_offset_blue)
    y2_blue_rot = rotate_curve_fixed_x(z * 1e4, y2_blue, theta_blue, y_offset_blue)
    y1_yellow_rot = rotate_curve_fixed_x(z * 1e4, y1_yellow, theta_yellow, y_offset_yellow)
    y2_yellow_rot = rotate_curve_fixed_x(z * 1e4, y2_yellow, theta_yellow, y_offset_yellow)

    # Simulate to get z distribution
    collider = BunchCollider()
    collider.set_bunch_beta_stars(beta_star, beta_star)
    collider.set_bunch_offsets([0, y_offset_blue], [0, y_offset_yellow])
    collider.set_bunch_sigmas([sigma, sigma], [sigma, sigma])
    collider.set_bunch_crossing(0, theta_blue, 0, theta_yellow)

    fig, axs = plt.subplots(nrows=2, figsize=(8, 3.5), dpi=144, sharex='all')
    ax, ax_sim = axs

    # Plot blue beam
    ax.plot(z, y1_blue_rot * 1e-3, color='blue', label='Blue Beam')
    ax.plot(z, y2_blue_rot * 1e-3, color='blue')
    ax.fill_between(z, y1_blue_rot * 1e-3, y2_blue_rot * 1e-3, color='blue', alpha=0.1)

    # Plot yellow beam
    ax.plot(z, y1_yellow_rot * 1e-3, color='orange', label='Yellow Beam')
    ax.plot(z, y2_yellow_rot * 1e-3, color='orange')
    ax.fill_between(z, y1_yellow_rot * 1e-3, y2_yellow_rot * 1e-3, color='orange', alpha=0.3)

    ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='-', alpha=0.5)
    # ax.set_xlabel('z (cm)')
    # ax.set_ylabel(f'y ({mu}m)')
    ax.set_ylabel(f'y (mm)')
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

    fig.subplots_adjust(top=0.995, bottom=0.08, right=0.995, left=0.08, hspace=0.1)

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


def visualize_two_beams_3_angles():
    beta_star = 85  # cm
    sigma = 150  # um
    z = np.linspace(-250, 250, 500)  # cm
    # y_offset_blue = 900  # um, vertical offset
    y_offsets_blue = [900, 900, -900]
    y_offset_yellow = 0  # um, vertical offset
    theta_yellow = 0.  # rotation angle in radians
    theta_blues = [-0.1e-3, 0.0, -0.1e-3]

    # Blue beam parameters
    y1_blue = sigma * f(z, beta_star, 0)  # um
    y2_blue = -y1_blue  # reflected curve

    # Yellow beam parameters
    y1_yellow = sigma * f(z, beta_star, 0)  # um
    y2_yellow = -y1_yellow  # reflected curve

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(8, 4), dpi=144, sharex='all', sharey='row')
    for theta_i, (theta_blue, y_offset_blue) in enumerate(zip(theta_blues, y_offsets_blue)):
        # Rotate and offset the blue and yellow beams
        y1_blue_rot = rotate_curve_fixed_x(z * 1e4, y1_blue, theta_blue, y_offset_blue)
        y2_blue_rot = rotate_curve_fixed_x(z * 1e4, y2_blue, theta_blue, y_offset_blue)
        y1_yellow_rot = rotate_curve_fixed_x(z * 1e4, y1_yellow, theta_yellow, y_offset_yellow)
        y2_yellow_rot = rotate_curve_fixed_x(z * 1e4, y2_yellow, theta_yellow, y_offset_yellow)

        # Simulate to get z distribution
        collider = BunchCollider()
        collider.set_bunch_beta_stars(beta_star, beta_star)
        collider.set_bunch_offsets([0, y_offset_blue], [0, y_offset_yellow])
        collider.set_bunch_sigmas([sigma, sigma], [sigma, sigma])
        collider.set_bunch_crossing(0, -theta_blue, 0, theta_yellow)

        ax = axs[0, theta_i]
        ax_sim = axs[1, theta_i]

        # Plot blue beam
        ax.plot(z, y1_blue_rot * 1e-3, color='blue', label='Blue Beam')
        ax.plot(z, y2_blue_rot * 1e-3, color='blue')
        ax.fill_between(z, y1_blue_rot * 1e-3, y2_blue_rot * 1e-3, color='blue', alpha=0.1)

        # Plot yellow beam
        ax.plot(z, y1_yellow_rot * 1e-3, color='orange', label='Yellow Beam')
        ax.plot(z, y2_yellow_rot * 1e-3, color='orange')
        ax.fill_between(z, y1_yellow_rot * 1e-3, y2_yellow_rot * 1e-3, color='orange', alpha=0.3)

        ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
        ax.axvline(0, color='gray', linestyle='-', alpha=0.5)
        if theta_i == 0:
            ax.set_ylabel(f'y (mm)')
        if theta_i == 1:
            ax.legend()

        # Simulate and plot z distribution
        collider.run_sim_parallel()
        z_vals, z_dist = collider.get_z_density_dist()

        ax_sim.plot(z_vals, z_dist, color='black', label='Z Distribution')
        ax_sim.axhline(0, color='gray', linestyle='-', alpha=0.5)
        ax_sim.set_ylim(bottom=0)
        ax_sim.set_xlim(z[0], z[-1])
        ax_sim.set_xlabel('z (cm)')
        if theta_i == 0:
            ax_sim.set_ylabel('Naked Luminosity Density')

        param_str = r'$\theta_{blue}=$' + f'{theta_blue * 1e3:.1f} mrad\nBlue offset = {y_offset_blue} {mu}m'
        if theta_i == 0:
            ax.annotate(param_str, xy=(0.5, 0.11), fontsize=12, xycoords='axes fraction', ha='center', va='bottom',
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))
        elif theta_i == 1:
            ax_sim.annotate(param_str, xy=(0.5, 0.9), fontsize=12, xycoords='axes fraction', ha='center', va='top',
                            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))
        else:
            ax.annotate(param_str, xy=(0.5, 0.9), fontsize=12, xycoords='axes fraction', ha='center', va='top',
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))

    fig.subplots_adjust(top=0.995, bottom=0.105, right=0.995, left=0.065, wspace=0.025, hspace=0.)


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
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 13 7:13 PM 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/vernier_scan_animation.py

@author: Dylan Neff, Dylan
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec


def main():
    create_gif()
    print('donzo')


def gaussian(x, mean, sigma):
    return np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def update(frame, blue_circles, yellow_circles, gaussian_lines, red_points, mean_start, mean_end, sigmas, gaus_max):
    phase = frame // frames_per_phase  # Determine which beam width phase we are in
    frame_in_phase = frame % frames_per_phase  # Frame index within the current phase

    if phase >= len(blue_circles):  # Stop if we've completed all phases
        return []

    # Move the corresponding blue circle
    blue_center_x = mean_start + frame_in_phase * (mean_end - mean_start) / frames_per_phase
    blue_circles[phase].set_center((blue_center_x, 0))
    yellow_circles[phase].set_center((0, 0))

    # Hide all other circles
    for i in range(len(blue_circles)):
        if i != phase:
            blue_circles[i].set_center((mean_start * 2, 0))  # Move off screen
            yellow_circles[i].set_center((mean_start * 2, 0))

    # Update the Gaussian curve
    x = np.linspace(mean_start, blue_center_x, 100)
    y = gaus_max * gaussian(x, 0, sigmas[phase])
    gaussian_lines[phase].set_xdata(x)
    gaussian_lines[phase].set_ydata(y)

    # Update red point
    red_points[phase].set_data([blue_center_x], [y[-1]])

    return blue_circles[phase], gaussian_lines[phase], red_points[phase]


def create_gif():
    global frames_per_phase
    beam_widths = [150, 100, 75]  # List of beam widths to animate
    beam_width_colors = ['white', 'green', 'orange']
    num_phases = len(beam_widths)
    frames_per_phase = 200  # Frames per beam width phase
    total_frames = frames_per_phase * num_phases

    x_bounds = [-1000, 1000]
    max_trigger_rate = 800

    fig = plt.figure(figsize=(10, 5), facecolor='#37474f')
    plt.rcParams['text.color'] = 'white'  # Set text color to white
    plt.rcParams['axes.labelcolor'] = 'white'  # Set axis labels color to white
    plt.rcParams['xtick.color'] = 'white'  # Set x-axis tick color to white
    plt.rcParams['ytick.color'] = 'white'  # Set y-axis tick color to white
    plt.rcParams['axes.edgecolor'] = 'white'  # Set axis border color to white
    plt.rcParams['axes.titlecolor'] = 'white'  # Set title color to white
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['white'])  # Set default line color to white

    # Define gridspec
    gs = gridspec.GridSpec(2, 1, height_ratios=[7, 3], hspace=0.0)

    ax0 = fig.add_subplot(gs[0], position=[0.07, 0.38, 0.86, 0.6])
    ax1 = fig.add_subplot(gs[1], position=[0.07, 0.0, 0.86, 0.3])
    ax0.set_facecolor('#37474f')

    ax0.set_xlim(*x_bounds)
    ax1.set_xlim(*x_bounds)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_frame_on(False)
    ax1.set_ylim(-beam_widths[0] * 1.2, beam_widths[0] * 1.2)
    ax0.set_ylim(0, max_trigger_rate * 1.1)
    ax0.set_xlabel('Blue Beam Offset [\u03bcm]')
    ax0.set_ylabel('Trigger Rate [kHz]')

    mean_start, mean_end = x_bounds
    sigmas = [np.sqrt(2) * bw for bw in beam_widths]

    # Initialize lists for animation objects
    blue_circles = []
    yellow_circles = []
    gaussian_lines = []
    red_points = []

    # Create plots for each beam width
    for i, (beam_width, color) in enumerate(zip(beam_widths, beam_width_colors)):
        x = np.linspace(*x_bounds, 100)
        y = max_trigger_rate * gaussian(x, mean_start, sigmas[i])

        gaussian_line, = ax0.plot([], [], lw=2, label=f'Beam Width: {beam_width} \u03bcm',
                                  color=color)  # Different color for each
        red_point, = ax0.plot([], [], 'ro', markersize=5)

        yellow_circle = plt.Circle((0, 0), beam_width, color='yellow', alpha=0.5)
        blue_circle = plt.Circle((mean_start, 0), beam_width, color='blue', alpha=0.5)

        ax1.add_patch(yellow_circle)
        ax1.add_patch(blue_circle)

        blue_circles.append(blue_circle)
        yellow_circles.append(yellow_circle)
        gaussian_lines.append(gaussian_line)
        red_points.append(red_point)

    ax0.legend(loc='upper right', facecolor='black')

    ani = animation.FuncAnimation(
        fig, update, frames=total_frames,
        fargs=(blue_circles, yellow_circles, gaussian_lines, red_points, mean_start, mean_end, sigmas, max_trigger_rate),
        interval=50
    )

    ani.save("circles_gaussian_multi.gif", writer="pillow", fps=30)
    plt.close()


if __name__ == '__main__':
    main()

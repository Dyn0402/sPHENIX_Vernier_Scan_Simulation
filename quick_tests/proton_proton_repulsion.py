#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on April 22 12:29 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/proton_proton_repulsion

@author: Dylan Neff, dn277127
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Constants
e = 1.602e-19  # Elementary charge (C)
epsilon_0 = 8.854e-12  # Vacuum permittivity (F/m)
m_p = 1.673e-27  # Mass of proton (kg)
c = 3e8  # Speed of light (m/s)


def simulate_proton_trajectories(initial_momentum, initial_angle_mrad, initial_offset, dt, steps):
    initial_angle_rad = initial_angle_mrad * 1e-3  # Convert mrad to radians
    p_mag = initial_momentum
    gamma = np.sqrt(1 + (p_mag / (m_p * c)) ** 2)
    v_mag = p_mag / (gamma * m_p)

    r1 = np.array([-5e-14, -initial_offset / 2])
    r2 = np.array([5e-14, initial_offset / 2])

    v1 = v_mag * np.array([np.cos(initial_angle_rad / 2), np.sin(initial_angle_rad / 2)])
    v2 = -v_mag * np.array([np.cos(initial_angle_rad / 2), np.sin(initial_angle_rad / 2)])

    r1_history = [r1.copy()]
    r2_history = [r2.copy()]

    for _ in range(steps):
        dr = r2 - r1
        distance = np.linalg.norm(dr)
        if distance < 1e-16:
            break

        force_mag = (e ** 2) / (4 * np.pi * epsilon_0 * distance ** 2)
        force_dir = dr / distance
        force = force_mag * force_dir

        a1 = -force / (gamma * m_p)
        a2 = +force / (gamma * m_p)

        v1 += a1 * dt
        v2 += a2 * dt
        r1 += v1 * dt
        r2 += v2 * dt

        r1_history.append(r1.copy())
        r2_history.append(r2.copy())

    return np.array(r1_history), np.array(r2_history)


def main():
    # Simulation parameters
    initial_momentum = 1e-19  # kg·m/s
    initial_angle_mrad = 1  # mrad
    dt = 1e-22 * 0.1  # seconds
    steps = 200
    offsets = [+0.1e-14, -0.1e-14]

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(6, 5), sharex='all', sharey='all')
    fig.suptitle("Coulomb Repulsion of Two Protons", fontsize=14)
    fig.subplots_adjust(hspace=0)

    histories = []
    plot_artists = []

    for ax, offset in zip(axs, offsets):
        r1_hist, r2_hist = simulate_proton_trajectories(
            initial_momentum, initial_angle_mrad, offset, dt, steps
        )
        r1_hist, r2_hist = r1_hist * 1e15, r2_hist * 1e15  # Convert to femtometers
        histories.append((r1_hist, r2_hist))

        ax.axhline(0, color='black', lw=0.5, alpha=0.5, zorder=0)
        ax.set_xlim(-60, 60)
        ax.set_ylim(-29, 29)

        line1, = ax.plot([], [], 'r-', label='Proton 1')
        line2, = ax.plot([], [], 'b-', label='Proton 2')
        dot1, = ax.plot([], [], 'ro')
        dot2, = ax.plot([], [], 'bo')

        # Annotation
        annotation_text = f"Offset: {offset * 1e15:.1f} fm\nAngle: {initial_angle_mrad} mrad"
        ax.text(0.5, 0.95, annotation_text, transform=ax.transAxes, ha='center',
                va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

        plot_artists.append((r1_hist, r2_hist, line1, line2, dot1, dot2))

    axs[-1].set_xlabel("x (fm)")
    for ax in axs:
        ax.set_ylabel("y (fm)")

    def init():
        artists = []
        for _, _, line1, line2, dot1, dot2 in plot_artists:
            for artist in (line1, line2, dot1, dot2):
                artist.set_data([], [])
                artists.append(artist)
        return artists

    def update(frame):
        artists = []
        for r1_hist, r2_hist, line1, line2, dot1, dot2 in plot_artists:
            if frame < len(r1_hist):
                line1.set_data(r1_hist[:frame, 0], r1_hist[:frame, 1])
                line2.set_data(r2_hist[:frame, 0], r2_hist[:frame, 1])
                dot1.set_data(r1_hist[frame, 0], r1_hist[frame, 1])
                dot2.set_data(r2_hist[frame, 0], r2_hist[frame, 1])
            artists += [line1, line2, dot1, dot2]
        return artists

    ani = FuncAnimation(fig, update, frames=steps, init_func=init, blit=True, interval=30)

    print("Saving animation as GIF...")
    ani.save("proton_repulsion.gif", writer=PillowWriter(fps=30))
    print("Saved as proton_repulsion.gif")

    plt.show()
    print('donzo')


if __name__ == '__main__':
    main()

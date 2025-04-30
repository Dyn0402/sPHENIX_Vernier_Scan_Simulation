#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on April 30 18:15 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/beam_beam_kick

@author: Dylan Neff, dn277127
"""

import numpy as np
import matplotlib.pyplot as plt


def beam_beam_kick_nonlinear(x, y, n_protons, gamma, sigma, r0):
    r_squared = x ** 2 + y ** 2
    factor = (2 * r0 * n_protons) / (gamma * r_squared)
    kick = factor * x * (1 - np.exp(-r_squared / (4 * sigma ** 2)))
    return kick


def main():
    # Particle properties
    particles = {
        'proton': {'r0': 1.535e-18, 'm': 0.938272e9},      # classical proton radius [m], mass [eV]
        'electron': {'r0': 2.81794e-15, 'm': 0.511e6},     # classical electron radius [m], mass [eV]
    }
    # x_plot = 'sigmas'
    x_plot = 'microns'

    # Beam configurations to compare
    beam_setups = [
        # {
        #     'name': 'LHC (4 TeV, 9e10 protons/bunch, 95 μm bunch width)',
        #     'particle': 'proton',
        #     'energy': 4e12,
        #     'n_particles_per_bunch': 9e10,
        #     'bunch_width': 95e-6
        # },
        # {
        #     'name': 'SLAC (50 GeV, 1e10 electrons/bunch, 5 μm bunch width)',
        #     'particle': 'electron',
        #     'energy': 50e9,
        #     'n_particles_per_bunch': 1e10,
        #     'bunch_width': 5e-6
        # },
        {
            'name': 'RHIC (200 GeV, 1.4e11 protons/bunch, 150 μm bunch width)',
            'particle': 'proton',
            'energy': 200e9,
            'n_particles_per_bunch': 1.4e11,
            'bunch_width': 150e-6
        },
    ]

    plt.figure(figsize=(8, 5))

    for setup in beam_setups:
        particle = setup['particle']
        r0 = particles[particle]['r0']
        m = particles[particle]['m']
        energy = setup['energy']
        n_particles = setup['n_particles_per_bunch']
        sigma = setup['bunch_width']
        gamma = energy / m

        x_vals = np.linspace(-8 * sigma, 8 * sigma, 1000)
        y_vals = np.zeros_like(x_vals)
        kicks = beam_beam_kick_nonlinear(x_vals, y_vals, n_particles, gamma, sigma, r0)
        kicks_urad = kicks * 1e6

        if x_plot == 'microns':
            plt.plot(x_vals * 1e6, kicks_urad, label=setup['name'], linewidth=2.5)
        else:
            plt.plot(x_vals / sigma, kicks_urad, label=setup['name'], linewidth=2.5)

    # Plot formatting
    plt.title("Beam–Beam Kick Angle vs Transverse Position (x)")
    if x_plot == 'microns':
        plt.xlabel("x position [μm]")
    else:
        plt.xlabel("x position [beam widths]")
    plt.ylabel("Kick angle Δx′ [μrad]")
    plt.ylim(top=plt.ylim()[1] * 1.2)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, zorder=0)
    plt.tight_layout()
    plt.show()

    print('donzo')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 16 15:24 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/solenoid_deflection_test

@author: Dylan Neff, dn277127
"""

import numpy as np
import matplotlib.pyplot as plt


# Physical constants
e_charge = 1.602176634e-19  # Coulombs
c = 299792458  # m/s


def main():
    # Parameters
    B = 1.4  # Tesla
    l = 2.0  # meters
    # m = 0.511e6 * e_charge / c ** 2  # mass of electron in kg
    m = 900e6 * e_charge / c ** 2  # mass of electron in kg
    q = +1  # electron charge
    gamma = 10.0  # relativistic gamma factor
    theta = 1e-3  # angle w.r.t z axis in radians

    x, y, z = simulate_solenoid_trajectory(B, l, m, q, gamma, theta)

    # Plot trajectory
    fig = plt.figure(figsize=(10, 5))

    ax1 = fig.add_subplot(121)
    ax1.plot(z, x, label='x')
    ax1.plot(z, y, label='y')
    ax1.set_xlabel('z [m]')
    ax1.set_ylabel('x, y [m]')
    ax1.set_title('x(z) and y(z)')
    ax1.legend()

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(x, y, z)
    ax2.set_xlabel('x [m]')
    ax2.set_ylabel('y [m]')
    ax2.set_zlabel('z [m]')
    ax2.set_title('3D trajectory')

    plt.tight_layout()
    plt.show()
    print('donzo')


def simulate_solenoid_trajectory(B, l, m, q, gamma, theta, dz=1e-3):
    """
    Simulate a relativistic charged particle moving through a solenoid field.

    Parameters:
        B : float
            Magnetic field strength in Tesla.
        l : float
            Length of solenoid in meters.
        m : float
            Particle mass in kg.
        q : float
            Charge in multiples of elementary charge (e).
        gamma : float
            Lorentz factor.
        theta : float
            Initial angle with respect to the z axis in radians.
        dz : float
            Step size in z in meters.
    """
    # Convert initial angle to radians
    # theta = np.deg2rad(theta_deg)

    # Total speed from gamma
    beta = np.sqrt(1 - 1 / gamma ** 2)
    v = beta * c

    # Initial velocity components
    vz = v * np.cos(theta)
    v_perp = v * np.sin(theta)

    # Cyclotron frequency
    omega = q * e_charge * B / (gamma * m)  # rad/s

    # Radius of helical motion
    r = v_perp / omega

    print(f"Helix radius: {r:.4e} m, Cyclotron freq: {omega:.4e} rad/s, vz: {vz:.4e} m/s")

    # Number of steps
    n_steps = int(l / dz)

    # Allocate arrays
    z = np.linspace(0, l, n_steps)
    x = np.zeros(n_steps)
    y = np.zeros(n_steps)

    # Initial phase
    phi = 0.0

    for i in range(n_steps):
        t = z[i] / vz
        phi = omega * t
        x[i] = r * np.cos(phi)
        y[i] = r * np.sin(phi)

    return x, y, z


if __name__ == '__main__':
    main()

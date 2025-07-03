#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 03 18:01 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/rate_correction_test

@author: Dylan Neff, dn277127
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    # Example usage
    simulation_result = simulate_proton_collisions(
        mu=0.2,
        prob_n=0.1,
        prob_s=0.1,
        prob_ns=0.02,
        n_crossings=100000
    )

    # Print summary statistics
    print(f"Average collisions per crossing: {np.mean(simulation_result['n_collisions']):.4f}")
    print(f"Average N triggers per crossing: {np.mean(simulation_result['n_fired']):.4f}")
    print(f"Average S triggers per crossing: {np.mean(simulation_result['s_fired']):.4f}")
    print(f"Average NS triggers per crossing: {np.mean(simulation_result['ns_fired']):.4f}")
    print('donzo')


def simulate_proton_collisions(
        mu=0.2,
        prob_n=0.1,
        prob_s=0.1,
        prob_ns=0.02,
        n_crossings=100000
):
    """
    Simulate proton bunch crossings with Poisson-distributed number of collisions per crossing.

    Parameters:
        mu: average number of collisions per bunch crossing
        prob_n: probability that detector N fires given a collision
        prob_s: probability that detector S fires given a collision
        prob_ns: probability that both detectors N and S fire together given a collision
        n_crossings: total number of bunch crossings to simulate

    Returns:
        dict with arrays:
            - n_collisions: number of collisions per crossing
            - n_fired: number of times N fired per crossing
            - s_fired: number of times S fired per crossing
            - ns_fired: number of times both N and S fired together per crossing
    """
    # Draw number of collisions per crossing
    n_collisions = np.random.poisson(mu, size=n_crossings)

    # Initialize counters
    n_fired = np.zeros(n_crossings, dtype=int)
    s_fired = np.zeros(n_crossings, dtype=int)
    ns_fired = np.zeros(n_crossings, dtype=int)

    # Simulate each crossing
    for i in range(n_crossings):
        n_c = n_collisions[i]
        if n_c > 0:
            # For each collision, determine if N, S, or NS fire
            fired_n = np.random.rand(n_c) < prob_n
            fired_s = np.random.rand(n_c) < prob_s
            fired_ns = np.random.rand(n_c) < prob_ns

            n_fired[i] = np.sum(fired_n)
            s_fired[i] = np.sum(fired_s)
            ns_fired[i] = np.sum(fired_ns)
        # If n_c == 0, leave counts at zero

    return {
        "n_collisions": n_collisions,
        "n_fired": n_fired,
        "s_fired": s_fired,
        "ns_fired": ns_fired
    }


if __name__ == '__main__':
    main()

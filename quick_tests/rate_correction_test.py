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
import matplotlib.ticker as mticker
from scipy.optimize import root_scalar
from scipy.optimize import curve_fit as cf

from rate_corrections import solve_sasha_equation


def main():
    # run_single_simulation()
    # run_sim_vs_k()
    run_sim_bkg_vs_k()

    print('donzo')


def run_sim_vs_k():
    n_crossings = 100000  # Total number of bunch crossings to simulate
    # prob_nses = np.linspace(0.05, 0.8, 10)  # Probability that both detectors N and S fire together given a collision
    # prob_nses = [0.01, 0.02, 0.025, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8]  # Probability that both detectors N and S fire together given a collision
    ks = np.sort(np.append(np.linspace(7, 0.2, 20), [0.3, 0.4, 0.5, 0.6]))  # K values to test
    prob_nses = 0.083 / ks  # Probability that both detectors N and S fire together given a collision
    prob_n = 0.1  # Probability that detector N fires given a collision
    prob_s = 0.1  # Probability that detector S fires given a collision
    # mu = 0.03
    mu = 0.3

    phenix_zdc_k = 3.5
    phenix_bbc_k = 0.27
    sphenix_zdc_k = 6.8

    avg_ks, p_off_origs, p_off_angelikas, p_off_sashas = [], [], [], []
    for sim_i, prob_ns in enumerate(prob_nses):
        print(f'Running simulation {sim_i + 1} of {len(prob_nses)} with prob_ns={prob_ns:.2f}')
        simulation_result = simulate_proton_collisions(
            mu=mu,
            prob_n=prob_n,
            prob_s=prob_s,
            prob_ns=prob_ns,
            n_crossings=n_crossings,
        )

        ns_fired = np.sum(simulation_result['ns_fired_true'])
        n_eff = np.sum(simulation_result['n_fired_eff'])
        s_eff = np.sum(simulation_result['s_fired_eff'])
        ns_eff = np.sum(simulation_result['ns_fired_eff'])

        ns_acc_cor = ns_eff - (n_eff * s_eff) / n_crossings
        ns_acc_mc_cor = n_crossings * (-np.log(1 - ns_acc_cor / (n_crossings + ns_eff - n_eff - s_eff)))

        ns_sasha_cor = solve_sasha_equation(n_eff, s_eff, ns_eff, n_crossings)

        n_0s = n_eff - ns_eff
        s_0n = s_eff - ns_eff
        ks = s_0n / ns_eff
        kn = n_0s / ns_eff
        avg_k = (ks + kn) / 2

        percent_off_orig = 100 * (ns_eff - ns_fired) / ns_fired
        percent_off_angelika = 100 * (ns_acc_mc_cor - ns_fired) / ns_fired
        percent_off_sasha = 100 * (ns_sasha_cor - ns_fired) / ns_fired

        print(f'k: {avg_k:.4f}, %orig: {percent_off_orig:.2f}%, '
              f'%angelika: {percent_off_angelika:.2f}%, %sasha: {percent_off_sasha:.2f}%')

        avg_ks.append(avg_k)
        p_off_origs.append(percent_off_orig)
        p_off_angelikas.append(percent_off_angelika)
        p_off_sashas.append(percent_off_sasha)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(avg_ks, p_off_origs, marker='o', label='Original Percent Off')
    ax.plot(avg_ks, p_off_angelikas, marker='o', label='Angelika Percent Off')
    ax.plot(avg_ks, p_off_sashas, marker='o', label='Sasha Percent Off')
    ax.axhline(0, color='k', linestyle='-', zorder=-1)
    ax.axvline(phenix_zdc_k, color='salmon', linestyle='--', label='PHENIX ZDC K Value')
    ax.axvline(phenix_bbc_k, color='purple', linestyle='--', label='PHENIX BBC K Value')
    ax.axvline(sphenix_zdc_k, color='salmon', linestyle='-', label='sPHENIX ZDC K Value')
    ax.set_xlabel('Average K Value')
    ax.set_ylabel('Percent Off from True NS Rate')
    ax.set_title('Percent Off vs Average K Value')
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend()
    fig.tight_layout()

    fig_p_k, ax_p_k = plt.subplots(figsize=(10, 6))
    ax_p_k.plot(avg_ks, prob_nses, marker='o', label='Average K Value')

    def inverse_x_fit(x, a):
        return a / x

    popt, _ = cf(inverse_x_fit, avg_ks, prob_nses)
    x_fit = np.linspace(min(avg_ks), max(avg_ks), 100)
    y_fit = inverse_x_fit(x_fit, *popt)
    ax_p_k.plot(x_fit, y_fit, color='red', linestyle='--', label='Exponential Fit')
    ax_p_k.set_xlabel('Average K Value')
    ax_p_k.set_ylabel('Probability of NS Firing Together')
    ax_p_k.set_title('Probability of NS Firing Together vs Average K Value')
    ax_p_k.legend()

    print(f'Fit Parameters: popt = {popt}')

    plt.show()


def run_sim_bkg_vs_k():
    n_crossings = 100000000  # Total number of bunch crossings to simulate
    # prob_nses = np.linspace(0.05, 0.8, 10)  # Probability that both detectors N and S fire together given a collision
    # prob_nses = [0.01, 0.02, 0.025, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8]  # Probability that both detectors N and S fire together given a collision
    # ks = np.sort(np.append(np.linspace(7, 0.2, 20), [0.3, 0.4, 0.5, 0.6]))  # K values to test
    ks = np.array([0.2, 0.3, 0.4, 0.5, 0.6])  # K values to test
    prob_nses = 0.083 / ks  # Probability that both detectors N and S fire together given a collision
    prob_n = 0.1  # Probability that detector N fires given a collision
    prob_s = 0.1  # Probability that detector S fires given a collision
    # mu = 0.03
    mu = 0.003
    prob_bkg_ns = 0.15 * 0.03 * 0.083 / 0.27  # Probability that both detectors N and S fire together given a background collision
    prob_bkg_n = 0.02 * 0.03 * 0.083 / 0.27  # Probability that detector N fires given a background collision
    prob_bkg_s = 0.02 * 0.03 * 0.083 / 0.27  # Probability that detector S fires given a background collision

    phenix_zdc_k = 3.5
    phenix_bbc_k = 0.27
    sphenix_zdc_k = 6.8

    avg_ks, p_off_origs, p_off_angelikas, p_off_sashas = [], [], [], []
    for sim_i, prob_ns in enumerate(prob_nses):
        print(f'Running simulation {sim_i + 1} of {len(prob_nses)} with prob_ns={prob_ns:.2f}')
        simulation_result = simulate_proton_background_collisions(
            mu=mu,
            prob_n=prob_n,
            prob_s=prob_s,
            prob_ns=prob_ns,
            prob_bkg_n=prob_bkg_n,
            prob_bkg_s=prob_bkg_s,
            prob_bkg_ns=prob_bkg_ns,
            n_crossings=n_crossings,
        )

        ns_fired = np.sum(simulation_result['ns_fired_true']) + np.sum(simulation_result['ns_fired_bkg_true'])
        n_eff = np.sum(simulation_result['n_fired_eff'])
        s_eff = np.sum(simulation_result['s_fired_eff'])
        ns_eff = np.sum(simulation_result['ns_fired_eff'])

        ns_acc_cor = ns_eff - (n_eff * s_eff) / n_crossings
        ns_acc_mc_cor = n_crossings * (-np.log(1 - ns_acc_cor / (n_crossings + ns_eff - n_eff - s_eff)))

        ns_sasha_cor = solve_sasha_equation(n_eff, s_eff, ns_eff, n_crossings)

        n_0s = n_eff - ns_eff
        s_0n = s_eff - ns_eff
        ks = s_0n / ns_eff
        kn = n_0s / ns_eff
        avg_k = (ks + kn) / 2

        percent_off_orig = 100 * (ns_eff - ns_fired) / ns_fired
        percent_off_angelika = 100 * (ns_acc_mc_cor - ns_fired) / ns_fired
        percent_off_sasha = 100 * (ns_sasha_cor - ns_fired) / ns_fired

        print(f'k: {avg_k:.4f}, %orig: {percent_off_orig:.2f}%, '
              f'%angelika: {percent_off_angelika:.2f}%, %sasha: {percent_off_sasha:.2f}%')

        avg_ks.append(avg_k)
        p_off_origs.append(percent_off_orig)
        p_off_angelikas.append(percent_off_angelika)
        p_off_sashas.append(percent_off_sasha)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(avg_ks, p_off_origs, marker='o', label='Original Percent Off')
    ax.plot(avg_ks, p_off_angelikas, marker='o', label='Angelika Percent Off')
    ax.plot(avg_ks, p_off_sashas, marker='o', label='Sasha Percent Off')
    ax.axhline(0, color='k', linestyle='-', zorder=-1)
    ax.axvline(phenix_zdc_k, color='salmon', linestyle='--', label='PHENIX ZDC K Value')
    ax.axvline(phenix_bbc_k, color='purple', linestyle='--', label='PHENIX BBC K Value')
    ax.axvline(sphenix_zdc_k, color='salmon', linestyle='-', label='sPHENIX ZDC K Value')
    ax.set_xlabel('Average K Value')
    ax.set_ylabel('Percent Off from True NS Rate')
    ax.set_title('Percent Off vs Average K Value')
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend()
    fig.tight_layout()

    fig_p_k, ax_p_k = plt.subplots(figsize=(10, 6))
    ax_p_k.plot(avg_ks, prob_nses, marker='o', label='Average K Value')

    def inverse_x_fit(x, a):
        return a / x

    popt, _ = cf(inverse_x_fit, avg_ks, prob_nses)
    x_fit = np.linspace(min(avg_ks), max(avg_ks), 100)
    y_fit = inverse_x_fit(x_fit, *popt)
    ax_p_k.plot(x_fit, y_fit, color='red', linestyle='--', label='Exponential Fit')
    ax_p_k.set_xlabel('Average K Value')
    ax_p_k.set_ylabel('Probability of NS Firing Together')
    ax_p_k.set_title('Probability of NS Firing Together vs Average K Value')
    ax_p_k.legend()

    print(f'Fit Parameters: popt = {popt}')

    plt.show()


def run_single_simulation():
    n_crossings = 1000000  # Total number of bunch crossings to simulate

    # ZDC-like
    prob_ns = 0.0  # Probability that both detectors N and S fire together given a collision
    prob_n = 0.4  # Probability that detector N fires given a collision
    prob_s = 0.4  # Probability that detector S fires given a collision

    # MBD-like
    # prob_ns = 0.5  # Probability that both detectors N and S fire together given a collision
    # prob_n = 0.2  # Probability that detector N fires given a collision
    # prob_s = 0.2  # Probability that detector S fires given a collision

    prob_0 = 1.0 - (prob_ns + prob_n + prob_s)
    mu = 0.05

    prob_over_1 = 1 - np.exp(-mu) * (mu + 1)

    simulation_result = simulate_proton_collisions(
        mu=mu,
        prob_n=prob_n,
        prob_s=prob_s,
        prob_ns=prob_ns,
        n_crossings=n_crossings,
    )

    n_col = np.sum(simulation_result['n_collisions'])
    n_fired = np.sum(simulation_result['n_fired_true'])
    s_fired = np.sum(simulation_result['s_fired_true'])
    ns_fired = np.sum(simulation_result['ns_fired_true'])
    n_eff = np.sum(simulation_result['n_fired_eff'])
    s_eff = np.sum(simulation_result['s_fired_eff'])
    ns_eff = np.sum(simulation_result['ns_fired_eff'])

    ns_acc_cor = ns_eff - (n_eff * s_eff) / n_crossings
    ns_acc_mc_cor = n_crossings * (-np.log(1 - ns_acc_cor / (n_crossings + ns_eff - n_eff - s_eff)))

    print()
    print(f"Total collisions: {n_col}, true ns_fired: {ns_fired}")
    print(f"ns_eff: {ns_eff}, n_eff: {n_eff}, s_eff: {s_eff}")
    print(f"ns_acc_cor: {ns_acc_cor}, ns_acc_mc_cor: {ns_acc_mc_cor}")
    print(f'ns_eff vs ns_fired percent difference: {100 * (ns_eff - ns_fired) / ns_fired:.2f}%')
    print(f'Percent difference: {100 * (ns_acc_mc_cor - ns_fired) / ns_fired:.2f}%')
    print()

    ns_acc_cor = ns_eff - prob_over_1 * (n_eff * s_eff) / n_crossings
    ns_acc_mc_cor = n_crossings * (-np.log(1 - ns_acc_cor / (n_crossings + ns_eff - n_eff - s_eff)))

    print()
    print(f"Total collisions: {n_col}, true ns_fired: {ns_fired}")
    print(f"ns_eff: {ns_eff}, n_eff: {n_eff}, s_eff: {s_eff}")
    print(f"ns_acc_cor: {ns_acc_cor}, ns_acc_mc_cor: {ns_acc_mc_cor}")
    print(f'ns_eff vs ns_fired percent difference: {100 * (ns_eff - ns_fired) / ns_fired:.2f}%')
    print(f'Percent difference: {100 * (ns_acc_mc_cor - ns_fired) / ns_fired:.2f}%')
    print()

    # Print summary statistics
    print(f"Average collisions per crossing: {np.mean(simulation_result['n_collisions']):.4f}")
    print(f"Average N triggers per crossing: {np.mean(simulation_result['n_fired_true']):.4f}")
    print(f"Average S triggers per crossing: {np.mean(simulation_result['s_fired_true']):.4f}")
    print(f"Average NS triggers per crossing: {np.mean(simulation_result['ns_fired_true']):.4f}")

    n_0s = n_eff - ns_eff
    s_0n = s_eff - ns_eff
    ks = s_0n / ns_eff
    kn = n_0s / ns_eff
    avg_k = (ks + kn) / 2
    print(f"Average K Value: {avg_k:.4f}")

    plot = False
    solve_sasha_equation(n_eff, s_eff, ns_eff, n_crossings, plot=plot)
    if plot:
        plt.axvline(ns_fired, color='green', linestyle='--', label='True NS Rate')
        plt.legend()
        plt.show()


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
            - n_fired_true: true number of times N fired per crossing
            - s_fired_true: true number of times S fired per crossing
            - ns_fired_true: true number of times both N and S fired together per crossing
            - n_fired_eff: effective number of times N fired per crossing
            - s_fired_eff: effective number of times S fired per crossing
            - ns_fired_eff: effective number of times both N and S fired together per crossing
    """
    # Draw number of collisions per crossing
    n_collisions = np.random.poisson(mu, size=n_crossings)

    # Initialize counters
    n_fired_true = np.zeros(n_crossings, dtype=int)
    s_fired_true = np.zeros(n_crossings, dtype=int)
    ns_fired_true = np.zeros(n_crossings, dtype=int)
    n_fired_eff = np.zeros(n_crossings, dtype=int)
    s_fired_eff = np.zeros(n_crossings, dtype=int)
    ns_fired_eff = np.zeros(n_crossings, dtype=int)

    # Calculate cumulative probabilities
    total_prob = prob_ns + prob_n + prob_s
    if total_prob > 1.0:
        raise ValueError("The sum of prob_ns, prob_n, and prob_s must be <= 1.0")

    thresholds = np.cumsum([prob_ns, prob_n, prob_s])

    # Simulate each crossing
    for i in range(n_crossings):
        # if i % 100000 == 0:
        #     print(f"Processing crossing {i} of {n_crossings}")
        n_c = n_collisions[i]
        if n_c > 0:
            # For each collision, determine if N, S, or NS fire
            random_vals = np.random.rand(n_c)
            fired_ns = random_vals < thresholds[0]  # NS fires
            fired_n = (random_vals >= thresholds[0]) & (random_vals < thresholds[1])  # N fires
            fired_s = (random_vals >= thresholds[1]) & (random_vals < thresholds[2])  # S fires
            ns_eff = np.any(fired_ns) | (np.any(fired_n) & np.any(fired_s))
            n_eff = np.any(np.logical_or(fired_n, fired_ns))
            s_eff = np.any(np.logical_or(fired_s, fired_ns))

            # if n_c > 1:
            #     print(f"Crossing {i}: n_c={n_c}")
            #     print(f"Fired N: {fired_n}, Fired S: {fired_s}, Fired NS: {fired_ns}")
            #     print(f"Effective N: {n_eff}, Effective S: {s_eff}, Effective NS: {ns_eff}")

            n_fired_true[i] = np.sum(fired_n)
            s_fired_true[i] = np.sum(fired_s)
            ns_fired_true[i] = np.sum(fired_ns)
            n_fired_eff[i] = n_eff
            s_fired_eff[i] = s_eff
            ns_fired_eff[i] = ns_eff
        # If n_c == 0, leave counts at zero

    return {
        "n_collisions": n_collisions,
        "n_fired_true": n_fired_true,
        "s_fired_true": s_fired_true,
        "ns_fired_true": ns_fired_true,
        "n_fired_eff": n_fired_eff,
        "s_fired_eff": s_fired_eff,
        "ns_fired_eff": ns_fired_eff
    }


def simulate_proton_background_collisions(
        mu=0.2,
        prob_n=0.1,
        prob_s=0.1,
        prob_ns=0.02,
        prob_bkg_n=0.0,
        prob_bkg_s=0.0,
        prob_bkg_ns=0.0,
        n_crossings=100000
):
    """
    Simulate proton bunch crossings with Poisson-distributed number of collisions per crossing.

    Parameters:
        mu: average number of collisions per bunch crossing
        prob_n: probability that detector N fires given a collision
        prob_s: probability that detector S fires given a collision
        prob_ns: probability that both detectors N and S fire together given a collision
        prob_bkg_n: probability that detector N fires given a background collision
        prob_bkg_s: probability that detector S fires given a background collision
        prob_bkg_ns: probability that both detectors N and S fire together given a background collision
        n_crossings: total number of bunch crossings to simulate

    Returns:
        dict with arrays:
            - n_collisions: number of collisions per crossing
            - n_fired_true: true number of times N fired per crossing
            - s_fired_true: true number of times S fired per crossing
            - ns_fired_true: true number of times both N and S fired together per crossing
            - n_fired_eff: effective number of times N fired per crossing
            - s_fired_eff: effective number of times S fired per crossing
            - ns_fired_eff: effective number of times both N and S fired together per crossing
    """
    # Draw number of collisions per crossing
    n_collisions = np.random.poisson(mu, size=n_crossings)

    # Initialize counters
    n_fired_true = np.zeros(n_crossings, dtype=int)
    s_fired_true = np.zeros(n_crossings, dtype=int)
    ns_fired_true = np.zeros(n_crossings, dtype=int)
    n_fired_bkg_true = np.zeros(n_crossings, dtype=int)
    s_fired_bkg_true = np.zeros(n_crossings, dtype=int)
    ns_fired_bkg_true = np.zeros(n_crossings, dtype=int)
    n_fired_eff = np.zeros(n_crossings, dtype=int)
    s_fired_eff = np.zeros(n_crossings, dtype=int)
    ns_fired_eff = np.zeros(n_crossings, dtype=int)

    # Calculate cumulative probabilities
    total_prob = prob_ns + prob_n + prob_s
    if total_prob > 1.0:
        raise ValueError("The sum of prob_ns, prob_n, and prob_s must be <= 1.0")

    thresholds = np.cumsum([prob_ns, prob_n, prob_s])
    thresholds_bkg = np.cumsum([prob_bkg_ns, prob_bkg_n, prob_bkg_s])

    # Simulate each crossing
    for i in range(n_crossings):
        # if i % 100000 == 0:
        #     print(f"Processing crossing {i} of {n_crossings}")
        n_c = n_collisions[i]
        n_eff, s_eff, ns_eff = 0, 0, 0
        if n_c > 0:
            # For each collision, determine if N, S, or NS fire
            random_vals = np.random.rand(n_c)
            fired_ns = random_vals < thresholds[0]  # NS fires
            fired_n = (random_vals >= thresholds[0]) & (random_vals < thresholds[1])  # N fires
            fired_s = (random_vals >= thresholds[1]) & (random_vals < thresholds[2])  # S fires
            ns_eff = np.any(fired_ns) | (np.any(fired_n) & np.any(fired_s))
            n_eff = np.any(np.logical_or(fired_n, fired_ns))
            s_eff = np.any(np.logical_or(fired_s, fired_ns))

            # if n_c > 1:
            #     print(f"Crossing {i}: n_c={n_c}")
            #     print(f"Fired N: {fired_n}, Fired S: {fired_s}, Fired NS: {fired_ns}")
            #     print(f"Effective N: {n_eff}, Effective S: {s_eff}, Effective NS: {ns_eff}")

            n_fired_true[i] = np.sum(fired_n)
            s_fired_true[i] = np.sum(fired_s)
            ns_fired_true[i] = np.sum(fired_ns)
        # If n_c == 0, leave counts at zero

        random_bkg_val = np.random.rand()
        fired_bkg_ns = random_bkg_val < thresholds_bkg[0]  # NS fires
        fired_bkg_n = (random_bkg_val >= thresholds_bkg[0]) & (random_bkg_val < thresholds_bkg[1])  # N fires
        fired_bkg_s = (random_bkg_val >= thresholds_bkg[1]) & (random_bkg_val < thresholds_bkg[2])  # S fires
        n_fired_bkg_true[i] = int(fired_bkg_n)
        s_fired_bkg_true[i] = int(fired_bkg_s)
        ns_fired_bkg_true[i] = int(fired_bkg_ns)

        ns_eff = ns_eff or fired_bkg_ns
        n_eff = n_eff or fired_bkg_n
        s_eff = s_eff or fired_bkg_s

        n_fired_eff[i] = n_eff
        s_fired_eff[i] = s_eff
        ns_fired_eff[i] = ns_eff

    return {
        "n_collisions": n_collisions,
        "n_fired_true": n_fired_true,
        "s_fired_true": s_fired_true,
        "ns_fired_true": ns_fired_true,
        "n_fired_bkg_true": n_fired_bkg_true,
        "s_fired_bkg_true": s_fired_bkg_true,
        "ns_fired_bkg_true": ns_fired_bkg_true,
        "n_fired_eff": n_fired_eff,
        "s_fired_eff": s_fired_eff,
        "ns_fired_eff": ns_fired_eff
    }


if __name__ == '__main__':
    main()

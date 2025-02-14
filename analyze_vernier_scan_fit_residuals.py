#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on February 14 13:04 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/analyze_vernier_scan_fit_residuals

@author: Dylan Neff, dn277127
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import scipy.optimize as opt
from scipy.interpolate import LinearNDInterpolator, Rbf, griddata


def main():
    scan_dates = ['Aug12']  # ['July11', 'Aug12']
    orientations = ['Horizontal', 'Vertical']
    out_dir_name = 'run_rcf_jobs/output'
    file_name = 'combined_scan_residuals.csv'
    interpolator = 'linear'
    # interpolator = 'cubic'  # 'linear'

    for scan_date in scan_dates:
        df_orientations = []
        for orientation in orientations:
            file_path = f'{out_dir_name}/{scan_date}/{orientation}/{file_name}'
            df = pd.read_csv(file_path)
            df_orientations.append(df)
            analyze_residuals(df, scan_date, orientation, interpolator)
        # Combine orientation results, summing residuals
        df_combined = sum_orientation_residuals(df_orientations)
        analyze_residuals(df_combined, scan_date, 'Combined', interpolator)


    print('donzo')


def sum_orientation_residuals(df_orientations):
    """
    Sum residuals from different orientations
    """
    df_combined = pd.concat(df_orientations).groupby(['bwx', 'bwy', 'betastar'], as_index=False)['residual_mean'].mean()
    return df_combined


def analyze_residuals(df, scan_date, orientation, interpolator):
    """
    Analyze results
    """
    # Extract data
    bwx = df['bwx']
    bwy = df['bwy']
    betastar = df['betastar']
    residuals = df['residual_mean']

    # Interpolator for residuals
    if interpolator == 'linear':
        interp = LinearNDInterpolator(list(zip(bwx, bwy, betastar)), residuals)
    elif interpolator == 'cubic':
        # Get griddata to work with cubic interpolation
        # interp = Rbf(bwx, bwy, betastar, residuals, function='cubic')
        interp = lambda x, y, z: griddata((bwx, bwy, betastar), residuals, (x, y, z), method='cubic')
    else:
        raise ValueError(f'Invalid interpolator: {interpolator}')

    # Objective function to minimize
    def objective(params):
        return interp(*params)

    # Initial guess (take the average)
    x0 = [np.mean(bwx), np.mean(bwy), np.mean(betastar)]

    # Perform minimization
    result = opt.minimize(objective, x0, method='Nelder-Mead')

    # Extract optimal parameters and minimum residual
    bwx_opt, bwy_opt, betastar_opt = result.x
    residual_min = result.fun

    print(f"Optimal parameters: bwx = {bwx_opt:.2f}, bwy = {bwy_opt:.2f}, betastar = {betastar_opt:.2f}")
    print(f"Minimum residual: {residual_min:.6f}")

    # Approximate uncertainty estimation using the Hessian
    hessian_inv = result.hess_inv if hasattr(result, 'hess_inv') else None
    if hessian_inv is not None:
        uncertainties = np.sqrt(np.diag(hessian_inv))
        print(
            f"Estimated uncertainties: bwx = {uncertainties[0]:.2f}, bwy = {uncertainties[1]:.2f}, betastar = {uncertainties[2]:.2f}")
    else:
        print("Hessian not available, cannot estimate uncertainties.")

    # Visualization
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # 2D contours
    param_labels = ['Beam Width X', 'Beam Width Y', 'Beta Star']
    param_units = ['µm', 'µm', 'cm']
    param_vals = [bwx, bwy, betastar]

    for i, (p1_i, p2_i) in enumerate([(0, 1), (0, 2), (1, 2)]):
        p1, p1_lab, p1_unit = param_vals[p1_i], param_labels[p1_i], param_units[p1_i]
        p2, p2_lab, p2_unit = param_vals[p2_i], param_labels[p2_i], param_units[p2_i]
        p1_range = np.linspace(min(p1), max(p1), 100)
        p2_range = np.linspace(min(p2), max(p2), 100)
        P1, P2 = np.meshgrid(p1_range, p2_range)

        # Evaluate residuals on grid
        if i == 0:
            Z = interp(P1, P2, betastar_opt)  # Fix beta*
            opt_x, opt_y = bwx_opt, bwy_opt
            opt_z, opt_z_lab, opt_z_unit = betastar_opt, 'Beta Star', 'cm'
        elif i == 1:
            Z = interp(P1, bwy_opt, P2)  # Fix bwy
            opt_x, opt_y = bwx_opt, betastar_opt
            opt_z, opt_z_lab, opt_z_unit = bwy_opt, 'Beam Width Y', 'µm'
        else:
            Z = interp(bwx_opt, P1, P2)  # Fix bwx
            opt_x, opt_y = bwy_opt, betastar_opt
            opt_z, opt_z_lab, opt_z_unit = bwx_opt, 'Beam Width X', 'µm'

        ax[i].contourf(P1, P2, Z, levels=20, cmap='viridis')
        ax[i].scatter(opt_x, opt_y, color='red', marker='x')
        ax[i].set_xlabel(f'{p1_lab} [{p1_unit}]')
        ax[i].set_ylabel(f'{p2_lab} [{p2_unit}]')
        ax[i].annotate(f'Optimal {opt_z_lab} = {opt_z:.1f} {opt_z_unit}', xy=(0.95, 0.05), xycoords='axes fraction',
                       ha='right', va='bottom', fontsize=12,
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", lw=1, alpha=0.8))

    plt.tight_layout()

    # 3D Scatter plot with interpolated residual color
    fig_3d = plt.figure(figsize=(8, 6))
    ax_3d = fig_3d.add_subplot(111, projection='3d')

    # Generate grid points for interpolation
    bwx_range = np.linspace(min(bwx), max(bwx), 20)
    bwy_range = np.linspace(min(bwy), max(bwy), 20)
    betastar_range = np.linspace(min(betastar), max(betastar), 20)
    Bwx, Bwy, Betastar = np.meshgrid(bwx_range, bwy_range, betastar_range)
    Residuals = interp(Bwx, Bwy, Betastar)

    # Scatter plot of real data points
    sc = ax_3d.scatter(bwx, bwy, betastar, c=residuals, cmap='coolwarm', marker='o', s=100, label="Data Points")
    ax_3d.scatter(bwx_opt, bwy_opt, betastar_opt, color='black', marker='x', s=100, label="Optimal Point")

    ax_3d.set_xlabel('Beam Width X')
    ax_3d.set_ylabel('Beam Width Y')
    ax_3d.set_zlabel('Beta Star')
    ax_3d.set_title('3D Residual Distribution')
    fig_3d.colorbar(sc, label="Residual")
    ax_3d.legend()

    # 2D plots for each unique beta star and optimal beta star
    unique_betastars = np.unique(betastar)
    # Define figure and gridspec
    fig_2d = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 0.1], height_ratios=[1, 1])

    axes_2d = [fig_2d.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

    # Create meshgrid
    p1_range = np.linspace(min(bwx), max(bwx), 100)
    p2_range = np.linspace(min(bwy), max(bwy), 100)
    P1, P2 = np.meshgrid(p1_range, p2_range)
    res_min, res_max = min(residuals), max(residuals)
    cmap = 'jet_r'

    # Plot each subplot
    for i, beta in enumerate(unique_betastars):
        ax = axes_2d[i]
        Z = interp(P1, P2, beta)
        bwx_i = bwx[betastar == beta]
        bwy_i = bwy[betastar == beta]
        residuals_i = residuals[betastar == beta]

        cont = ax.contourf(P1, P2, Z, levels=20, cmap=cmap, vmin=res_min, vmax=res_max)
        sc = ax.scatter(bwx_i, bwy_i, c=residuals_i, edgecolors='k', cmap=cmap, vmin=res_min, vmax=res_max)

        if i > 2:
            ax.set_xlabel('Beam Width X [µm]')
        if i % 3 == 0:
            ax.set_ylabel('Beam Width Y [µm]')

        beta_star_str = rf'$\beta^*=${beta:.1f} cm'
        if beta == betastar_opt:
            beta_star_str += ' (Optimal)'
            ax.scatter(bwx_opt, bwy_opt, color='red', marker='x', s=100, label="Optimal Point")

        ax.annotate(beta_star_str, xy=(0.03, 0.974), xycoords='axes fraction', ha='left', va='top', fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", lw=1, alpha=0.8))

    # Create colorbar axis
    colorbar_ax = fig_2d.add_axes([0.945, 0.1, 0.02, 0.8])  # [left, bottom, width, height]
    fig_2d.colorbar(sc, cax=colorbar_ax, label="Residual")

    fig_2d.tight_layout(rect=[0, 0, 1, 1])  # Leave space for colorbar
    # fig_2d.subplots_adjust(hspace=0.07)

    # fig_2d, axes_2d = plt.subplots(2, 3, figsize=(16, 8), sharex='all')
    # axes_2d = axes_2d.flatten()
    #
    # p1_range = np.linspace(min(bwx), max(bwx), 100)
    # p2_range = np.linspace(min(bwy), max(bwy), 100)
    # P1, P2 = np.meshgrid(p1_range, p2_range)
    # for i, beta in enumerate(unique_betastars):
    #     ax = axes_2d[i]
    #     Z = interp(P1, P2, beta)
    #     bwx_i = bwx[betastar == beta]
    #     bwy_i = bwy[betastar == beta]
    #     residuals_i = residuals[betastar == beta]
    #
    #     cont = ax.contourf(P1, P2, Z, levels=20, cmap='viridis')
    #     ax.scatter(bwx_i, bwy_i, c=residuals_i, edgecolors='k', cmap='viridis')
    #     if i > 2:
    #         ax.set_xlabel('Beam Width X [µm]')
    #     ax.set_ylabel('Beam Width Y [µm]')
    #     beta_star_str = f'Beta Star = {beta:.1f} cm'
    #     if beta == betastar_opt:
    #         beta_star_str += ' (Optimal)'
    #         ax.scatter(bwx_opt, bwy_opt, color='red', marker='x', s=100, label="Optimal Point")
    #     ax.annotate(beta_star_str, xy=(0.1, 0.9), xycoords='axes fraction', ha='left', va='top', fontsize=12,
    #                 bbox=dict(boxstyle="round,pad=0.3", fc="white", lw=1, alpha=0.8))
    #     fig_2d.colorbar(cont, ax=ax, label="Residual")
    #
    # fig_2d.tight_layout()
    # fig_2d.subplots_adjust(hspace=0.07)

    plt.show()


if __name__ == '__main__':
    main()

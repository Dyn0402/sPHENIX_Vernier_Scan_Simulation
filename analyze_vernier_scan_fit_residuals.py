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
from scipy.optimize import curve_fit as cf
from scipy.interpolate import LinearNDInterpolator, Rbf, griddata

from Measure import Measure


def main():
    scan_dates = ['Aug12']  # ['July11', 'Aug12']
    orientations = ['Horizontal', 'Vertical']
    out_dir_name = 'run_rcf_jobs/output'
    file_name = 'combined_scan_residuals.csv'
    # interpolator = 'linear'
    interpolator = 'cubic'  # 'linear'

    for scan_date in scan_dates:
        df_orientations = []
        for orientation in orientations:
            file_path = f'{out_dir_name}/{scan_date}/{orientation}/{file_name}'
            df = pd.read_csv(file_path)
            df_orientations.append(df)
            # analyze_residuals(df, scan_date, orientation, interpolator)
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

    good_betastar = [80, 85, 90, 95, 100, 105]
    bwx = bwx[betastar.isin(good_betastar)]
    bwy = bwy[betastar.isin(good_betastar)]
    residuals = residuals[betastar.isin(good_betastar)]
    betastar = betastar[betastar.isin(good_betastar)]

    # Interpolator for residuals
    if interpolator == 'linear':
        interp = LinearNDInterpolator(list(zip(bwx, bwy, betastar)), residuals)
    elif interpolator == 'cubic':
        # Get griddata to work with cubic interpolation
        interp = Rbf(bwx, bwy, betastar, residuals, function='cubic')
        # interp = lambda x, y, z: griddata((bwx, bwy, betastar), residuals, (x, y, z), method='cubic')
    else:
        raise ValueError(f'Invalid interpolator: {interpolator}')
    lin_interp = LinearNDInterpolator(list(zip(bwx, bwy, betastar)), residuals)

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

    # For each beta star, find the optimal beam width x and y via 2D interpolation
    unique_betastars = np.unique(betastar)
    min_bws, min_resids = {}, {}
    for beta in unique_betastars:
        bwx_i = bwx[betastar == beta]
        bwy_i = bwy[betastar == beta]
        residuals_i = residuals[betastar == beta]
        # 2D cubic interpolation
        interp_i = Rbf(bwx_i, bwy_i, residuals_i, function='cubic')

        def restricted_interp(x, y):
            if x < min(bwx_i) or x > max(bwx_i) or y < min(bwy_i) or y > max(bwy_i):
                return np.nan  # or any other value you want to return for extrapolation
            return interp_i(x, y)

        interp_i_opt = restricted_interp

        # Minimize the interpolated function
        p0 = np.array([np.mean(bwx_i), np.mean(bwy_i)])
        # p0 = np.array([161, 157])
        # bounds = [(min(bwx_i), max(bwx_i)), (min(bwy_i), max(bwy_i))]
        bounds = None
        print(f'p0 = {p0}')
        result_i = opt.minimize(lambda x: interp_i_opt(x[0], x[1]), p0, method='Nelder-Mead', bounds=bounds)
        min_bws[beta] = result_i.x
        min_resids[beta] = interp_i_opt(*result_i.x)
        print(f'Optimal beam widths for beta star {beta:.1f} cm: bwx = {result_i.x[0]:.2f}, bwy = {result_i.x[1]:.2f}')

        # Plot
        # fig, ax = plt.subplots()
        # bwx_interp_vals = np.linspace(min(bwx_i), max(bwx_i), 100)
        # bwy_interp_vals = np.linspace(min(bwy_i), max(bwy_i), 100)
        # Bwx, Bwy = np.meshgrid(bwx_interp_vals, bwy_interp_vals)
        # Z = interp_i(Bwx, Bwy)
        # ax.contourf(Bwx, Bwy, Z, levels=20, cmap='viridis')
        # ax.scatter(bwx_i, bwy_i, c=residuals_i, cmap='viridis')
        # ax.scatter(result_i.x[0], result_i.x[1], color='red', marker='x')
        # ax.set_xlabel('Beam Width X [µm]')
        # ax.set_ylabel('Beam Width Y [µm]')
        # ax.set_title(f'Residuals for Beta Star {beta:.1f} cm')
        # plt.show()

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
            Z = lin_interp(P1, P2, betastar_opt)  # Fix beta*
            opt_x, opt_y = bwx_opt, bwy_opt
            opt_z, opt_z_lab, opt_z_unit = betastar_opt, 'Beta Star', 'cm'
        elif i == 1:
            Z = lin_interp(P1, bwy_opt, P2)  # Fix bwy
            opt_x, opt_y = bwx_opt, betastar_opt
            opt_z, opt_z_lab, opt_z_unit = bwy_opt, 'Beam Width Y', 'µm'
        else:
            Z = lin_interp(bwx_opt, P1, P2)  # Fix bwx
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
        Z = lin_interp(P1, P2, beta)
        bwx_i = bwx[betastar == beta]
        bwy_i = bwy[betastar == beta]
        residuals_i = residuals[betastar == beta]

        cont = ax.contourf(P1, P2, Z, levels=20, cmap=cmap, vmin=res_min, vmax=res_max)
        sc = ax.scatter(bwx_i, bwy_i, c=residuals_i, edgecolors='k', cmap=cmap, vmin=res_min, vmax=res_max)
        min_bwx, min_bwy = min_bws[beta]
        ax.scatter(min_bwx, min_bwy, color='black', marker='x', s=100, label="Optimal Point")

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

        # Do 2D poly fit
        fig2d_fit, ax2d_fit = plt.subplots()
        bwx_i, bwy_i, residuals_i = np.array(bwx_i), np.array(bwy_i), np.array(residuals_i)
        popt, pcov = cf(poly2_2d, (bwx_i, bwy_i), residuals_i)
        perr = np.sqrt(np.diag(pcov))
        meases = [Measure(popt[i], perr[i]) for i in range(len(popt))]
        for par_i, meas in enumerate(meases):
            print(f'Parameter {par_i}: {meas}')
        Z_fit = poly2_2d((P1, P2), *popt)
        ax2d_fit.contour(P1, P2, Z_fit, levels=10, colors='black', linestyles='dashed')
        ax2d_fit.scatter(bwx_i, bwy_i, c=residuals_i, edgecolors='k', cmap='viridis')
        ax2d_fit.scatter(min_bwx, min_bwy, color='red', marker='x', s=100, label="Optimal Point")
        ax2d_fit.set_xlabel('Beam Width X [µm]')
        ax2d_fit.set_ylabel('Beam Width Y [µm]')
        ax2d_fit.set_title('2D Polynomial Fit')
        ax2d_fit.annotate(beta_star_str, xy=(0.03, 0.974), xycoords='axes fraction', ha='left', va='top', fontsize=12,
                          bbox=dict(boxstyle="round,pad=0.2", fc="white", lw=1, alpha=0.8))
        fig2d_fit.colorbar(sc, label="Residual")
        fig2d_fit.tight_layout()

        # ax.scatter(popt[1], popt[2], color='blue', marker='x', s=100, label="Quadratic Fit Optimal Point")

        # Plot a few 1D residuals vs beam width x and beam width y near the optimal point
        fig_1d_resids, ax_1d_resids = plt.subplots(2, 1, figsize=(8, 9), sharex='all')
        ax_1d_resids[-1].set_xlabel('Beam Width [µm]')
        ax_1d_resids[0].set_ylabel('Residual')
        ax_1d_resids[1].set_ylabel('Residual')
        for bwx_val in pd.unique(bwx_i):
            bwy_i_for_x = np.array(bwy_i[bwx_i == bwx_val])
            resid_i_for_x = np.array(residuals_i[bwx_i == bwx_val])
            # popt, pcov = opt.curve_fit(poly2, bwy_i_for_x, resid_i_for_x)
            x_plot = np.linspace(min(bwy_i_for_x), max(bwy_i_for_x), 100)
            y_pol2d = poly2_2d((bwx_val, x_plot), *popt)
            ax_1d_resids[0].scatter(bwy_i_for_x, resid_i_for_x, label=rf'$\sigma_x=${bwx_val}', color='red')
            # ax_1d_resids[0].plot(x_plot, poly2(x_plot, *popt), color='red')
            ax_1d_resids[0].plot(x_plot, y_pol2d, color='red')
        for bwy_val in pd.unique(bwy_i):
            bwx_i_for_y = np.array(bwx_i[bwy_i == bwy_val])
            resid_i_for_y = np.array(residuals_i[bwy_i == bwy_val])
            popt, pcov = opt.curve_fit(poly2, bwx_i_for_y, resid_i_for_y)
            y_plot = np.linspace(min(bwx_i_for_y), max(bwx_i_for_y), 100)
            ax_1d_resids[1].plot(bwx_i_for_y, resid_i_for_y, label=rf'$\sigma_y=${bwy_val}', color='blue', marker='o')
            ax_1d_resids[1].plot(y_plot, poly2(y_plot, *popt), color='blue')
            # ax_1d_resids.plot(np.array(bwx_i), np.array(residuals_i), label='Beam Width X', color='blue', marker='o')
        ax_1d_resids[1].axvline(min_bwx, color='blue', linestyle='--', label='Optimal Beam Width X')
        ax_1d_resids[0].axvline(min_bwy, color='red', linestyle='--', label='Optimal Beam Width Y')
        ax_1d_resids[0].legend()
        ax_1d_resids[1].legend()
        fig_1d_resids.tight_layout()
        fig_1d_resids.subplots_adjust(hspace=0.0)

    # Create colorbar axis
    colorbar_ax = fig_2d.add_axes([0.945, 0.1, 0.02, 0.8])  # [left, bottom, width, height]
    fig_2d.colorbar(sc, cax=colorbar_ax, label="Residual")

    fig_2d.tight_layout(rect=[0, 0, 1, 1])  # Leave space for colorbar

    # Plot 2 rows: 1st row for bwx and bwy, 2nd row for residuals vs beta star
    bw_err = 0.5  # µm  Hardcode 0.5 micron error corresponding to scan point spacing
    err_plt = np.array([bw_err for _ in unique_betastars])
    fig, ax = plt.subplots(2, 1, figsize=(10, 9), sharex='all')

    min_bws_x = [min_bws[beta][0] for beta in unique_betastars]
    min_bws_y = [min_bws[beta][1] for beta in unique_betastars]
    popt_x, pcov_x = cf(linear, unique_betastars, min_bws_x, sigma=err_plt, absolute_sigma=True)
    popt_y, pcov_y = cf(linear, unique_betastars, min_bws_y, sigma=err_plt, absolute_sigma=True)
    perr_x, perr_y = np.sqrt(np.diag(pcov_x)), np.sqrt(np.diag(pcov_y))

    # Plot bwx and bwy together on the same plot
    ax[0].errorbar(unique_betastars, min_bws_x, yerr=err_plt, marker='o', ls='none',
                   color='blue', label='Optimal Beam Widths X')
    ax[0].plot(unique_betastars, linear(unique_betastars, *popt_x), color='blue')
    ax[0].errorbar(unique_betastars, min_bws_y, yerr=err_plt, marker='o', ls='none',
                   color='green', label='Optimal Beam Widths Y')
    ax[0].plot(unique_betastars, linear(unique_betastars, *popt_y), color='green')
    ax[0].set_xlabel('Beta Star [cm]')
    ax[0].set_ylabel('Beam Width [µm]')
    ax[0].legend()

    # Plot residuals on the bottom plot
    ax[1].plot(unique_betastars, [min_resids[beta] for beta in unique_betastars], label='Minimum Residual')
    ax[1].set_xlabel('Beta Star [cm]')
    ax[1].set_ylabel('Residual')
    ax[1].legend()

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.0)

    # Write linear fits and optimal beam widths to file
    if orientation == 'Combined':
        with open(f'run_rcf_jobs/output/{scan_date}/bw_opt_vs_beta_star_fits.txt', 'w') as file:
            file.write(f'Beam Width X Linear Fit: a = {popt_x[0]} +- {perr_x[0]}, b = {popt_x[1]} +- {perr_x[1]}\n')
            file.write(f'Beam Width Y Linear Fit: a = {popt_y[0]} +- {perr_y[0]}, b = {popt_y[1]} +- {perr_y[1]}\n')
        with open(f'run_rcf_jobs/output/{scan_date}/bw_opt_vs_beta_star.txt', 'w') as file:
            file.write('Beta Star [cm], Beam Width X [µm], Beam Width Y [µm]\n')
            for beta in unique_betastars:
                file.write(f'{beta}, {min_bws[beta][0]}, {min_bws[beta][1]}\n')

    plt.show()


def poly2_2d(x, const, x0, y0, x_curve, y_curve, xy_curve):
    x, y = x
    x_shift, y_shift = x - x0, y - y0
    return const + x_curve * x_shift**2 + y_curve * y_shift**2 + xy_curve * x_shift * y_shift


def poly2_2d_old(x, a, b, c, d, e, f):
    x, y = x
    return a + b*x + c*y + d*x**2 + e*x*y + f*y**2


def poly2(x, a, b, c):
    return a + b*x + c*x**2


def linear(x, a, b):
    return a*x + b


if __name__ == '__main__':
    main()

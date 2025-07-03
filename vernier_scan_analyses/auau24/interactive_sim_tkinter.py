#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 02 04:17 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/interactive_sim_tkinter

@author: Dylan Neff, dn277127
"""

import platform
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import uproot

from BunchCollider import BunchCollider
from z_vertex_fitting_common import fit_amp_shift, fit_shift_only, get_profile_path
from common_logistics import set_base_path


def main():
    root = tk.Tk()
    app = PlotSimulatorApp(root)
    root.mainloop()
    print('donzo')


class PlotSimulatorApp:
    def __init__(self, master):
        self.master = master
        master.title("Simulation Viewer")

        base_path = set_base_path()

        base_path_auau = f'{base_path}Vernier_Scans/auau_oct_16_24/'
        self.longitudinal_profiles_dir_path = f'{base_path_auau}profiles/'
        self.z_vertex_data_path = f'{base_path_auau}vertex_data/54733_vertex_distributions_no_zdc_coinc.root'
        self.z_vertex_zdc_data_path = f'{base_path_auau}vertex_data/54733_vertex_distributions.root'
        combined_cad_step_data_csv_path = f'{base_path_auau}combined_cad_step_data.csv'

        self.cad_df = pd.read_csv(combined_cad_step_data_csv_path)
        self.step_0 = self.cad_df[self.cad_df['step'] == 0].iloc[0]
        self.dcct_blue_nom, self.dcct_yellow_nom = self.step_0['blue_dcct_ions'], self.step_0['yellow_dcct_ions']
        self.em_blue_horiz_nom, self.em_blue_vert_nom = self.step_0['blue_horiz_emittance'], self.step_0['blue_vert_emittance']
        self.em_yel_horiz_nom, self.em_yel_vert_nom = self.step_0['yellow_horiz_emittance'], self.step_0['yellow_vert_emittance']

        # Steps: fill this with your real data
        # self.steps = [0, 2, 3, 4, 5]
        self.steps = [0, 6, 12, 18, 24]
        self.raw_data = {}  # Map from step -> (centers_no_zdc, counts_no_zdc, centers, counts)

        self.last_sim_zs = {step: None for step in self.steps}
        self.last_sim_z_dist = {step: None for step in self.steps}

        # Populate self.raw_data here...
        self.load_raw_data()

        # Simulation object placeholder
        self.collider_sim = BunchCollider()  # You assign this externally

        # Define parameters with default values
        self.n_xy_points = tk.IntVar(value=31)
        self.n_z_points = tk.IntVar(value=101)
        self.n_t_points = tk.IntVar(value=31)
        self.bkg = tk.DoubleVar(value=0.0e-17)
        self.gauss_eff_width = tk.DoubleVar(value=500)
        self.mbd_resolution = tk.DoubleVar(value=1.0)
        self.beta_star_x = tk.DoubleVar(value=76.7)
        self.beta_star_y = tk.DoubleVar(value=76.7)
        self.beam_width_x = tk.DoubleVar(value=130.0)
        self.beam_width_y = tk.DoubleVar(value=130.0)
        self.yellow_angle_dx = tk.DoubleVar(value=0.0)
        self.yellow_angle_dy = tk.DoubleVar(value=0.0)
        self.blue_offset_dx = tk.DoubleVar(value=0.0)
        self.blue_offset_dy = tk.DoubleVar(value=0.0)
        self.fit_low = tk.DoubleVar(value=-200.0)
        self.fit_high = tk.DoubleVar(value=200.0)
        self.blue_beta_star_shift_x = tk.DoubleVar(value=0.0)
        self.blue_beta_star_shift_y = tk.DoubleVar(value=0.0)

        self.fit_dist_options = ['MBD&ZDC', 'MBD']
        self.fit_dist = tk.StringVar(value=self.fit_dist_options[0])

        self.show_metrics = tk.BooleanVar(value=True)  # Show metrics on plot

        # GUI layout
        self.create_widgets()
        self.create_plot()

    def create_widgets(self):
        # Top-level horizontal layout: left = controls, right = plot
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True)

        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        self.run_button = ttk.Button(control_frame, text="Run", command=self.run_simulation)
        self.run_button.pack(pady=5)

        # Parameters frame
        param_frame = ttk.LabelFrame(control_frame, text="Parameters")
        param_frame.pack(fill=tk.Y, pady=5)

        def add_param(label_text, variable):
            row = ttk.Frame(param_frame)
            row.pack(anchor='w', pady=2)
            ttk.Label(row, text=label_text, width=15).pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=variable, width=7).pack(side=tk.LEFT)

        dropdown = ttk.Combobox(control_frame, textvariable=self.fit_dist, values=self.fit_dist_options, state="readonly")
        dropdown.pack(anchor='w', pady=2)
        add_param("N XY Points", self.n_xy_points)
        add_param("N Z Points", self.n_z_points)
        add_param("N T Points", self.n_t_points)
        add_param("Fit Low", self.fit_low)
        add_param("Fit High", self.fit_high)
        add_param("Background", self.bkg)
        add_param("Gauss Eff Width", self.gauss_eff_width)
        add_param("MBD Resolution", self.mbd_resolution)
        add_param("Beta*_x", self.beta_star_x)
        add_param("Beta*_y", self.beta_star_y)
        add_param("Beam Width X", self.beam_width_x)
        add_param("Beam Width Y", self.beam_width_y)
        add_param("Yellow Angle DX", self.yellow_angle_dx)
        add_param("Yellow Angle DY", self.yellow_angle_dy)
        add_param("Blue Offset DX", self.blue_offset_dx)
        add_param("Blue Offset DY", self.blue_offset_dy)
        add_param("Blue B* Shift X (cm)", self.blue_beta_star_shift_x)
        add_param("Blue B* Shift Y (cm)", self.blue_beta_star_shift_y)

        # Add show metrics checkbox
        metrics_checkbox = ttk.Checkbutton(control_frame, text="Show Metrics", variable=self.show_metrics)
        metrics_checkbox.pack(anchor='w', pady=2)

    def create_plot(self):
        self.figure, self.axs = plt.subplots(figsize=(6 + 2.5 * len(self.steps), 6), ncols=len(self.steps))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.master.winfo_children()[0])
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)
        self.draw_initial_data()

    def load_raw_data(self):
        for scan_step in self.steps:
            cad_step_row = self.cad_df[self.cad_df['step'] == scan_step].iloc[0]
            with uproot.open(self.z_vertex_data_path) as f:
                hist = f[f'step_{scan_step}']
                centers_no_zdc = hist.axis().centers()
                counts_no_zdc = hist.counts()
                count_errs_no_zdc = hist.errors()
                count_errs_no_zdc[count_errs_no_zdc == 0] = 1

            with uproot.open(self.z_vertex_zdc_data_path) as f:
                hist = f[f'step_{scan_step}']
                centers = hist.axis().centers()
                counts = hist.counts()
                count_errs = hist.errors()
                count_errs[count_errs == 0] = 1

            # Normalize counts to ZDC rate
            zdc_raw_rate = cad_step_row['zdc_cor_rate']
            zdc_hist_counts = np.sum(counts)
            hist_scaling_factor = zdc_raw_rate / zdc_hist_counts

            counts *= hist_scaling_factor
            count_errs *= hist_scaling_factor
            counts_no_zdc *= hist_scaling_factor
            count_errs_no_zdc *= hist_scaling_factor

            # Scale for dcct
            step_dcct_blue, step_dcct_yellow = cad_step_row['blue_dcct_ions'], cad_step_row['yellow_dcct_ions']
            dcct_scale = (self.dcct_blue_nom * self.dcct_yellow_nom) / (step_dcct_blue * step_dcct_yellow)

            counts *= dcct_scale
            count_errs *= dcct_scale
            counts_no_zdc *= dcct_scale
            count_errs_no_zdc *= dcct_scale

            self.raw_data[scan_step] = (centers_no_zdc, counts_no_zdc, count_errs_no_zdc, centers, counts, count_errs)

    def draw_initial_data(self):
        for i, (step, ax) in enumerate(zip(self.steps, self.axs)):
            ax.clear()
            centers_no_zdc, counts_no_zdc, count_errs_no_zdc, centers, counts, count_errs = self.raw_data[step]

            if self.fit_dist.get() == 'MBD&ZDC':
                alpha_no_zdc, alpha_zdc, lw_no_zdc, lw_zdc = 0.3, 1.0, 1, 2
                max_y = np.max(counts[(centers < 180) & (centers > -180)]) * 1.2
            elif self.fit_dist.get() == 'MBD':
                alpha_no_zdc, alpha_zdc, lw_no_zdc, lw_zdc = 1.0, 0.3, 2, 1
                max_y = np.max(counts_no_zdc[(centers_no_zdc < 180) & (centers_no_zdc > -180)]) * 1.2

            ax.step(centers_no_zdc, counts_no_zdc, where='mid', linewidth=lw_no_zdc, alpha=alpha_no_zdc, color='green',
                         label='Data No ZDC')
            ax.step(centers, counts, where='mid', alpha=alpha_zdc, linewidth=lw_zdc, color='blue', label='Data With ZDC')
            if step == self.steps[0]:
                ax.legend()
            ax.set_xlabel('z (cm)')
            if i == 0:
                ax.set_ylabel('Rate (Hz)')
            ax.set_title(f"Step {step}")
            ax.set_ylim(bottom=0, top=max_y)
            ax.grid()
            self.canvas.draw()
        self.figure.tight_layout()

    def run_simulation(self):
        self.draw_initial_data()  # Redraw base data first

        if self.collider_sim is not None:
            self.collider_sim.set_grid_size(self.n_xy_points.get(), self.n_xy_points.get(),
                                            self.n_z_points.get(), self.n_t_points.get())
            self.collider_sim.set_bkg(self.bkg.get())
            self.collider_sim.set_gaus_z_efficiency_width(self.gauss_eff_width.get())
            self.collider_sim.set_gaus_smearing_sigma(self.mbd_resolution.get())
            self.collider_sim.set_bunch_beta_stars(self.beta_star_x.get(), self.beta_star_y.get())
            self.collider_sim.set_bunch_beta_star_shifts(self.blue_beta_star_shift_x.get() * 1e4, 0, self.blue_beta_star_shift_y.get() * 1e4, 0)
            for scan_step, ax in zip(self.steps, self.axs):
                print(f'Running simulation for step {scan_step}...')
                cad_step_row = self.cad_df[self.cad_df['step'] == scan_step].iloc[0]

                em_blue_horiz, em_blue_vert = cad_step_row['blue_horiz_emittance'], cad_step_row['blue_vert_emittance']
                em_yel_horiz, em_yel_vert = cad_step_row['yellow_horiz_emittance'], cad_step_row[
                    'yellow_vert_emittance']

                # blue_widths = np.array([
                #     self.beam_width_x.get() * np.sqrt(em_blue_horiz / self.em_blue_horiz_nom),
                #     self.beam_width_y.get() * np.sqrt(em_blue_vert / self.em_blue_vert_nom)
                # ])
                # yellow_widths = np.array([
                #     self.beam_width_x.get() * np.sqrt(em_yel_horiz / self.em_yel_horiz_nom),
                #     self.beam_width_y.get() * np.sqrt(em_yel_vert / self.em_yel_vert_nom)
                # ])

                blue_widths = np.array([
                    self.beam_width_x.get() * (em_blue_horiz / self.em_blue_horiz_nom),
                    self.beam_width_y.get() * (em_blue_vert / self.em_blue_vert_nom)
                ])
                yellow_widths = np.array([
                    self.beam_width_x.get() * (em_yel_horiz / self.em_yel_horiz_nom),
                    self.beam_width_y.get() * (em_yel_vert / self.em_yel_vert_nom)
                ])

                # blue_widths = np.array([
                #     self.beam_width_x.get() * 1,
                #     self.beam_width_y.get() * 1
                # ])
                # yellow_widths = np.array([
                #     self.beam_width_x.get() * 1,
                #     self.beam_width_y.get() * 1
                # ])

                self.collider_sim.set_bunch_sigmas(blue_widths, yellow_widths)

                blue_angle_x = -cad_step_row['blue angle h'] * 1e-3
                blue_angle_y = -cad_step_row['blue angle v'] * 1e-3
                yellow_angle_x = -cad_step_row['yellow angle h'] * 1e-3
                yellow_angle_y = -cad_step_row['yellow angle v'] * 1e-3

                blue_offset_x, blue_offset_y = cad_step_row['set offset h'] * 1e3, cad_step_row['set offset v'] * 1e3
                yellow_offset_x, yellow_offset_y = 0, 0

                # if scan_step != 0:
                if True:
                    blue_offset_x += self.blue_offset_dx.get()
                    blue_offset_y += self.blue_offset_dy.get()
                    yellow_angle_x += self.yellow_angle_dx.get() * 1e-3
                    yellow_angle_y += self.yellow_angle_dy.get() * 1e-3

                self.collider_sim.set_bunch_crossing(blue_angle_x, blue_angle_y, yellow_angle_x, yellow_angle_y)

                self.collider_sim.set_bunch_offsets(
                    [blue_offset_x, blue_offset_y],
                    [yellow_offset_x, yellow_offset_y]
                )

                profile_path = get_profile_path(
                    self.longitudinal_profiles_dir_path, cad_step_row['start'], cad_step_row['end'], False
                )
                self.collider_sim.set_longitudinal_profiles_from_file(
                    profile_path.replace('COLOR_', 'blue_'),
                    profile_path.replace('COLOR_', 'yellow_')
                )

                self.collider_sim.run_sim_parallel()

                # centers_no_zdc, counts_no_zdc, count_errs_no_zdc, centers, counts, count_errs = self.raw_data[scan_step]
                if self.fit_dist.get() == 'MBD&ZDC':
                    centers, counts, count_errs = self.raw_data[scan_step][3:]
                elif self.fit_dist.get() == 'MBD':
                    centers, counts, count_errs = self.raw_data[scan_step][:3]
                fit_mask = (centers > self.fit_low.get()) & (centers < self.fit_high.get())

                if scan_step == 0:  # Fix the amplitude in the first head-on step
                    fit_amp_shift(self.collider_sim, counts[fit_mask], centers[fit_mask], count_errs[fit_mask])
                else:
                    fit_shift_only(self.collider_sim, counts[fit_mask], centers[fit_mask], count_errs[fit_mask])

                zs, z_dist = self.collider_sim.get_z_density_dist()
                if self.last_sim_zs[scan_step] is not None:
                    ax.plot(self.last_sim_zs[scan_step], self.last_sim_z_dist[scan_step], linewidth=1, ls='--', color='gray',)
                ax.plot(zs, z_dist, linewidth=2, ls='-', color='red')
                self.last_sim_zs[scan_step], self.last_sim_z_dist[scan_step] = zs, z_dist
                interp_sim = np.interp(centers[fit_mask], zs, z_dist)
                if self.show_metrics.get():
                    chi2 = np.sum(((counts[fit_mask] - interp_sim) / count_errs[fit_mask]) ** 2) / len(counts[fit_mask])
                    log_likelihood = np.sum(counts[fit_mask] * np.log(interp_sim) - interp_sim)
                    scaled_residuals = np.sqrt(np.sum(((counts[fit_mask] - interp_sim) / np.mean(counts[fit_mask]))**2))
                    ax.annotate(f'$\chi^2$ / NDF: {chi2:.2f}\nLog Likelihood: {log_likelihood:.2f}\n'
                                f'Scaled Residuals: {scaled_residuals:2f}', xy=(0.05, 0.95), xycoords='axes fraction',
                                fontsize=12, va='top', ha='left', color='black')
                self.canvas.draw()


if __name__ == '__main__':
    main()

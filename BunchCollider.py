#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on August 30 2:32 PM 2024
Created in PyCharm
Created as sphenix_polarimetry/BunchCollider.py

@author: Dylan Neff, Dylan
"""

import numpy as np
import os
# import h5py
from concurrent.futures import ProcessPoolExecutor as Pool
from scipy.ndimage import gaussian_filter1d

from BunchDensity import BunchDensity


class BunchCollider:
    def __init__(self):
        self.bunch1 = BunchDensity()
        self.bunch2 = BunchDensity()

        self.bunch1_beta_original = np.array([0., 0., +1.])
        self.bunch2_beta_original = np.array([0., 0., -1.])
        self.bunch1.set_beta(*self.bunch1_beta_original)
        self.bunch2.set_beta(*self.bunch2_beta_original)

        self.bunch1.set_sigma(150., 150., 1.1e6)
        self.bunch2.set_sigma(150., 150., 1.1e6)  # microns

        self.bunch1.beta_star = 85  # cm
        self.bunch2.beta_star = 85  # cm

        self.bunch1_r_original = np.array([0., 0., -6.e6])
        self.bunch2_r_original = np.array([0., 0., +6.e6])
        self.set_bunch_rs(self.bunch1_r_original, self.bunch2_r_original)

        self.z_shift = 0.  # microns Distance to shift the center of the collisions along beam axis
        self.amplitude = 1.  # arb Scale amplitude of z-distribution by this amount

        self.bkg = 1e-20 * 0.  # Background density to add to the density product, interpret as air density

        self.z_bounds = (-265. * 1e4, 265. * 1e4)  # um Bounds of z-axis to calculate density product over

        self.gaus_smearing_sigma = None
        self.gaus_z_efficiency_width = None

        self.x_lim_sigma = 10
        self.y_lim_sigma = 10
        self.z_lim_sigma = 5

        self.n_points_x = 61
        self.n_points_y = 61
        self.n_points_z = 151
        self.n_points_t = 61

        self.bunch1_longitudinal_fit_parameter_path = None
        self.bunch2_longitudinal_fit_parameter_path = None
        self.bunch1_longitudinal_fit_scaling = 1.
        self.bunch2_longitudinal_fit_scaling = 1.
        self.bunch1_longitudinal_profile_file_path = None
        self.bunch2_longitudinal_profile_file_path = None

        self.x, self.y, self.z = None, None, None
        self.average_density_product_xyz = None
        self.z_dist = None

        self.parallel_threads = os.cpu_count()

    def set_bunch_sigmas(self, sigma1, sigma2):
        self.bunch1.set_sigma(*sigma1)
        self.bunch2.set_sigma(*sigma2)

    def set_bunch_betas(self, beta1, beta2):
        self.bunch1_beta_original = beta1
        self.bunch2_beta_original = beta2
        self.bunch1.set_beta(*self.bunch1_beta_original)
        self.bunch2.set_beta(*self.bunch2_beta_original)

    def set_bunch_rs(self, r1, r2):
        self.bunch1_r_original = r1
        self.bunch2_r_original = r2
        self.bunch1.set_initial_z(self.bunch1_r_original[2])
        self.bunch2.set_initial_z(self.bunch2_r_original[2])
        self.bunch1.set_offsets(*self.bunch1_r_original[:2])
        self.bunch2.set_offsets(*self.bunch2_r_original[:2])

    def set_bunch_offsets(self, offset1, offset2):
        self.bunch1_r_original[:2] = offset1
        self.bunch2_r_original[:2] = offset2
        self.bunch1.set_offsets(*self.bunch1_r_original[:2])
        self.bunch2.set_offsets(*self.bunch2_r_original[:2])

    def set_bunch_beta_stars(self, beta_star1_x, beta_star2_x, beta_star1_y=None, beta_star2_y=None):
        self.bunch1.set_beta_star(beta_star1_x, beta_star1_y)
        self.bunch2.set_beta_star(beta_star2_x, beta_star2_y)

    def set_bunch_beta_star_shifts(self, beta_star1_shift_x, beta_star2_shift_x, beta_star1_shift_y=None, beta_star2_shift_y=None):
        self.bunch1.set_beta_star_shift(beta_star1_shift_x, beta_star1_shift_y)
        self.bunch2.set_beta_star_shift(beta_star2_shift_x, beta_star2_shift_y)

    def set_bunch_crossing(self, crossing_angle1_x, crossing_angle1_y, crossing_angle2_x, crossing_angle2_y):
        self.bunch1.set_angles(crossing_angle1_x, crossing_angle1_y)
        self.bunch2.set_angles(crossing_angle2_x, crossing_angle2_y)

    def set_z_shift(self, z_shift):
        self.z_shift = z_shift

    def set_amplitude(self, amp):
        self.amplitude = amp

    def set_z_bounds(self, z_bounds):
        self.z_bounds = z_bounds

    def set_gaus_smearing_sigma(self, sigma):
        self.gaus_smearing_sigma = sigma

    def set_gaus_z_efficiency_width(self, width):
        self.gaus_z_efficiency_width = width

    def set_bunch_delays(self, delay1, delay2):
        self.bunch1.set_delay(delay1)
        self.bunch2.set_delay(delay2)

    def set_bkg(self, bkg):
        self.bkg = bkg

    def set_longitudinal_fit_parameters_from_file(self, bunch1_path, bunch2_path):
        self.bunch1_longitudinal_fit_parameter_path = bunch1_path
        self.bunch2_longitudinal_fit_parameter_path = bunch2_path
        self.bunch1.read_longitudinal_beam_profile_fit_parameters_from_file(bunch1_path)
        self.bunch2.read_longitudinal_beam_profile_fit_parameters_from_file(bunch2_path)

    def set_longitudinal_fit_scaling(self, scale1, scale2):
        self.bunch1_longitudinal_fit_scaling = scale1
        self.bunch2_longitudinal_fit_scaling = scale2
        self.bunch1.set_longitudinal_beam_profile_scaling(scale1)
        self.bunch2.set_longitudinal_beam_profile_scaling(scale2)

    def set_longitudinal_profiles_from_file(self, bunch1_path, bunch2_path):
        self.bunch1_longitudinal_profile_file_path = bunch1_path
        self.bunch2_longitudinal_profile_file_path = bunch2_path
        self.bunch1.read_longitudinal_beam_profile_from_file(bunch1_path)
        self.bunch2.read_longitudinal_beam_profile_from_file(bunch2_path)

    def set_bunch_lengths(self, length1, length2):
        """
        Set the gaussian longitudinal widths of the bunches in microns.
        :param length1: Length of bunch 1 in microns
        :param length2: Length of bunch 2 in microns
        """
        self.bunch1.set_bunch_length(length1)
        self.bunch2.set_bunch_length(length2)

    def set_grid_size(self, n_points_x=None, n_points_y=None, n_points_z=None, n_points_t=None):
        if n_points_x is not None:
            self.n_points_x = n_points_x
        if n_points_y is not None:
            self.n_points_y = n_points_y
        if n_points_z is not None:
            self.n_points_z = n_points_z
        if n_points_t is not None:
            self.n_points_t = n_points_t

    def check_profile_normalizations(self):
        """
        Check if the longitudinal profiles of the bunches are normalized correctly.
        :return: True if both profiles are normalized, False otherwise.
        """
        total_density1 = self.bunch1.check_profile_normalization()
        total_density2 = self.bunch2.check_profile_normalization()
        print(f'Bunch 1 total density: {total_density1}, Bunch 2 total density: {total_density2}')
        if not np.isclose(total_density1, 1.0, atol=1e-6):
            print(f'Bunch 1 profile is not normalized: {total_density1}')
        if not np.isclose(total_density2, 1.0, atol=1e-6):
            print(f'Bunch 2 profile is not normalized: {total_density2}')

    def generate_grid(self):
        # Set timestep for propagation in nano seconds
        dt = (self.bunch2.initial_z - self.bunch1.initial_z) / self.bunch1.c / self.n_points_t
        self.bunch1.dt = self.bunch2.dt = dt  # ns Timestep to propagate both bunches

        # Create a grid of points for the x-z and y-z planes in microns
        self.x = np.linspace(-self.x_lim_sigma * self.bunch1.transverse_sigma[0],
                             self.x_lim_sigma * self.bunch1.transverse_sigma[0],
                             self.n_points_x)
        self.y = np.linspace(-self.y_lim_sigma * self.bunch1.transverse_sigma[1],
                             self.y_lim_sigma * self.bunch1.transverse_sigma[1],
                             self.n_points_y)
        if self.z_bounds is not None:
            self.z = np.linspace(self.z_bounds[0], self.z_bounds[1], self.n_points_z)
        else:
            min_z = min(self.bunch1_r_original[2], self.bunch2_r_original[2])
            max_z = max(self.bunch1_r_original[2], self.bunch2_r_original[2])
            self.z = np.linspace(min_z - self.z_lim_sigma * self.bunch1.transverse_sigma[2],
                                 max_z + self.z_lim_sigma * self.bunch1.transverse_sigma[2], self.n_points_z)


    def run_sim(self, print_params=False):
        # Reset
        self.set_bunch_rs(self.bunch1_r_original, self.bunch2_r_original)
        self.bunch1.set_beta(*self.bunch1_beta_original)
        self.bunch2.set_beta(*self.bunch2_beta_original)
        self.bunch1.set_angles(self.bunch1.angle_x, self.bunch1.angle_y)
        self.average_density_product_xyz, self.z_dist = None, None

        self.generate_grid()

        x_3d, y_3d, z_3d = np.meshgrid(self.x, self.y, self.z, indexing='ij')  # For 3D space
        self.bunch1.calculate_r_and_beta()
        self.bunch2.calculate_r_and_beta()
        if print_params:
            print(self)

        for i in range(self.n_points_t):
            density1_xyz = self.bunch1.density(x_3d, y_3d, z_3d)
            density2_xyz = self.bunch2.density(x_3d, y_3d, z_3d)

            # Calculate the density product
            density_product_xyz = density1_xyz * density2_xyz

            # Add background
            density_product_xyz += density1_xyz * self.bkg + density2_xyz * self.bkg

            if self.average_density_product_xyz is None:
                self.average_density_product_xyz = density_product_xyz
            else:
                self.average_density_product_xyz += density_product_xyz

            self.bunch1.propagate()
            self.bunch2.propagate()
        self.average_density_product_xyz /= self.n_points_t
        self.z_dist = np.sum(self.average_density_product_xyz, axis=(0, 1))

    def compute_time_step(self, time_step_index):
        """
        Compute the density for a specific time step.
        :param time_step_index: Index of the time step to compute
        :return: Density product for the given time step
        """
        bunch1_copy = self.bunch1.copy()
        bunch2_copy = self.bunch2.copy()
        bunch1_copy.propagate_n_steps(time_step_index)
        bunch2_copy.propagate_n_steps(time_step_index)

        # Create a grid of points for the x-y-z planes
        x_3d, y_3d, z_3d = np.meshgrid(self.x, self.y, self.z, indexing='ij')

        density1_xyz =  bunch1_copy.density(x_3d, y_3d, z_3d)
        density2_xyz = bunch2_copy.density(x_3d, y_3d, z_3d)

        # Calculate the density product
        density_product_xyz = density1_xyz * density2_xyz

        # Add background
        density_product_xyz += density1_xyz * self.bkg + density2_xyz * self.bkg

        return density_product_xyz

    def run_sim_parallel(self, print_params=False):
        # Reset
        self.set_bunch_rs(self.bunch1_r_original, self.bunch2_r_original)
        self.bunch1.set_beta(*self.bunch1_beta_original)
        self.bunch2.set_beta(*self.bunch2_beta_original)
        self.bunch1.set_angles(self.bunch1.angle_x, self.bunch1.angle_y)
        self.average_density_product_xyz, self.z_dist = None, None

        self.generate_grid()

        self.bunch1.calculate_r_and_beta()
        self.bunch2.calculate_r_and_beta()
        if print_params:
            print(self)

        # self.average_density_product_xyz = 0
        # with Pool(max_workers=self.parallel_threads) as pool:
        #     density_products = pool.map(self.compute_time_step, range(self.n_points_t))
        # self.average_density_product_xyz = np.mean([p for p in density_products], axis=0)
        # self.z_dist = np.sum(self.average_density_product_xyz, axis=(0, 1))

        self.average_density_product_xyz = 0
        with Pool(max_workers=self.parallel_threads) as pool:
            density_products = pool.map(self.compute_time_step, range(self.n_points_t))
        density_products = np.array([p for p in density_products])  # shape: (n_points_t, n_points_x, n_points_y, n_points_z)
        integrated_over_t = np.trapezoid(density_products, dx=self.bunch1.dt, axis=0)
        integrated_over_tx = np.trapezoid(integrated_over_t, x=self.x, axis=0)
        integrated_over_txy = np.trapezoid(integrated_over_tx, x=self.y, axis=0)
        self.average_density_product_xyz = integrated_over_t
        self.z_dist = integrated_over_txy

    def get_grid_info(self):
        """
        Calculate grid info for the simulation necessary to compute the integrated density product.
        :return:
        """
        grid_info = {
            'dx': self.x[1] - self.x[0],
            'dy': self.y[1] - self.y[0],
            'dz': self.z[1] - self.z[0],
            'dt': self.bunch1.dt,
            'n_points_t': self.n_points_t,
            'x_range': (self.x[0], self.x[-1]),
            'y_range': (self.y[0], self.y[-1]),
            'z_range': (self.z[0], self.z[-1]),
            'n_points_x': self.n_points_x,
            'n_points_y': self.n_points_y,
            'n_points_z': self.n_points_z
        }
        return grid_info

    def get_beam_sigmas(self):
        return self.bunch1.transverse_sigma, self.bunch2.transverse_sigma

    def get_bunch_crossing_angles(self):
        return self.bunch1.angle_x, self.bunch1.angle_y, self.bunch2.angle_x, self.bunch2.angle_y

    def get_z_density_dist(self):
        z_vals = (self.z - self.z_shift) / 1e4  # um to cm
        z_dist = self.amplitude * self.z_dist
        if self.gaus_z_efficiency_width is not None:
            z_dist = z_dist * gaus(z_vals, 1, 0, self.gaus_z_efficiency_width)
            # z_dist = z_dist * efficiency(z_vals, z_quad=self.gaus_z_efficiency_width, z_switch=200, steepness=-0.2)
        if self.gaus_smearing_sigma is not None:
            z_spacing = z_vals[1] - z_vals[0]
            z_dist = gaussian_filter1d(z_dist, self.gaus_smearing_sigma / z_spacing)
        return z_vals, z_dist

    def get_x_density_dist(self):
        x_vals = self.x
        x_dist = np.sum(self.average_density_product_xyz, axis=(1, 2))
        return x_vals, x_dist

    def get_y_density_dist(self):
        y_vals = self.y
        y_dist = np.sum(self.average_density_product_xyz, axis=(0, 2))
        return y_vals, y_dist

    def get_relativistic_moller_factor(self):
        """
        Calculate the relativistic Moller factor for the given bunches based on their relative velocities
        :return:
        """
        # Calculate initial 3-velocities
        v1 = self.bunch1.beta * self.bunch1.c
        v2 = self.bunch2.beta * self.bunch2.c

        # Calculate Moller factor
        moller_factor = np.sqrt(np.linalg.norm(v1 - v2)**2 - np.linalg.norm(np.cross(v1, v2))**2 / self.bunch1.c**2)
        return moller_factor

    def get_naked_luminosity_old(self):
        """
        Calculate the luminosity corresponding to a single particle in each bunch colliding at 1Hz.
        Assuming head on so Moller factor is 2c.
        :return:
        """
        # integral = np.sum(self.average_density_product_xyz)
        # integral = np.sum(self.z_dist)
        grid_info = self.get_grid_info()
        # luminosity = integral * grid_info['dx'] * grid_info['dy'] * grid_info['dz'] * grid_info['n_points_t'] * grid_info['dt']
        # luminosity = luminosity * 2 * self.bunch1.c  # 2c for head on
        luminosity = np.trapezoid(self.z_dist, self.z) * grid_info['dx'] * grid_info['dy'] * grid_info['n_points_t'] * grid_info['dt']
        # print(f'Old lumi: {luminosity}, new lumi: {new_lumi}')
        luminosity = luminosity * self.get_relativistic_moller_factor()
        return luminosity

    def get_naked_luminosity(self, observed=False):
        """
        Calculate the luminosity corresponding to a single particle in each bunch colliding at 1Hz.
        Assuming head on so Moller factor is 2c.
        :param observed: If True, return the luminosity as observed in the z distribution.
        :return:
        """
        if observed:
            zs, z_dist = self.get_z_density_dist()
            zs = zs * 1e4  # Convert from cm back to um for the luminosity calculation
        else:
            zs, z_dist = self.z, self.z_dist
        luminosity = np.trapezoid(z_dist, zs) * self.get_relativistic_moller_factor()
        return luminosity

    # def get_params(self):
    #     """
    #     Generate a unique hash based on the current simulation parameters.
    #     """
    #     # Collect relevant parameters
    #     params = {
    #         'bunch1_sigma': self.bunch1.transverse_sigma.tolist(),
    #         'bunch2_sigma': self.bunch2.transverse_sigma.tolist(),
    #         'bunch1_longitudinal_scaling': self.bunch1.longitudinal_width_scaling,
    #         'bunch2_longitudinal_scaling': self.bunch2.longitudinal_width_scaling,
    #         'bunch1_initial_z': self.bunch1.initial_z,
    #         'bunch2_initial_z': self.bunch2.initial_z,
    #         'bunch1_offset_x': self.bunch1.offset_x,
    #         'bunch1_offset_y': self.bunch1.offset_y,
    #         'bunch2_offset_x': self.bunch2.offset_x,
    #         'bunch2_offset_y': self.bunch2.offset_y,
    #         'bunch1_angle_x': self.bunch1.angle_x,
    #         'bunch1_angle_y': self.bunch1.angle_y,
    #         'bunch2_angle_x': self.bunch2.angle_x,
    #         'bunch2_angle_y': self.bunch2.angle_y,
    #         'bunch1_beta_star': self.bunch1.beta_star,
    #         'bunch2_beta_star': self.bunch2.beta_star,
    #         'bkg': self.bkg,
    #         'z_bounds': list(self.z_bounds),
    #         'x_lim_sigma': self.x_lim_sigma,
    #         'y_lim_sigma': self.y_lim_sigma,
    #         'z_lim_sigma': self.z_lim_sigma,
    #         'n_points_x': self.n_points_x,
    #         'n_points_y': self.n_points_y,
    #         'n_points_z': self.n_points_z,
    #         'n_points_t': self.n_points_t,
    #     }
    #     return params
    #
    # def save_simulation(self):
    #     """
    #     Saves the simulation parameters and luminosity density data to an HDF5 file.
    #     """
    #     with h5py.File(self.file_path, 'a') as f:
    #         params_group = f.create_group('params')  # Create a group for parameters and save each as an attribute
    #         for key, value in self.get_params().items():
    #             if isinstance(value, list):  # Handle lists (like sigma values) by saving them as numpy arrays
    #                 params_group.attrs[key] = np.array(value)
    #             else:
    #                 params_group.attrs[key] = value
    #
    #         # Save the luminosity density as a compressed dataset
    #         f.create_dataset('luminosity_density', data=self.z_dist, compression='gzip')
    #
    # def load_simulation(self):
    #     """
    #     Loads and returns the simulation parameters and luminosity density data from the HDF5 file.
    #     """
    #     with h5py.File(self.file_path, 'r') as f:
    #         # Retrieve parameters and convert numpy arrays back to lists where needed
    #         params = {key: (
    #             list(f['params'].attrs[key]) if isinstance(f['params'].attrs[key], np.ndarray) else f['params'].attrs[
    #                 key])
    #                   for key in f['params'].attrs}
    #
    #         # Load the luminosity density data
    #         luminosity_density = f['luminosity_density'][:]
    #
    #     return params, luminosity_density
    #
    # def get_params(self):
    #     """
    #     Retrieves only the simulation parameters from the HDF5 file without loading the full luminosity density.
    #     """
    #     with h5py.File(self.file_path, 'r') as f:
    #         params = {key: (
    #             list(f['params'].attrs[key]) if isinstance(f['params'].attrs[key], np.ndarray) else f['params'].attrs[
    #                 key])
    #                   for key in f['params'].attrs}
    #     return params
    #
    # def get_luminosity_density(self):
    #     """
    #     Retrieves only the luminosity density data from the HDF5 file without loading parameters.
    #     """
    #     with h5py.File(self.file_path, 'r') as f:
    #         luminosity_density = f['luminosity_density'][:]
    #     return luminosity_density


    def get_param_string(self):
        param_string = (f'Beta*s: ({self.bunch1.beta_star_x:.1f}, {self.bunch1.beta_star_y:.1f}), ({self.bunch2.beta_star_x:.1f}, {self.bunch2.beta_star_y:.1f}) cm\n'
                        f'Beam Widths: {self.bunch1.transverse_sigma[0]:.1f} (x), {self.bunch1.transverse_sigma[1]:.1f} (y) um\n'
                        f'Beam Lengths: {self.bunch1.get_beam_length() / 1e4:.1f}, {self.bunch2.get_beam_length() / 1e4:.1f} cm\n'
                        f'Crossing Angles y: {self.bunch1.angle_y * 1e3:.2f}, {self.bunch2.angle_y * 1e3:.2f} mrad\n'
                        f'Crossing Angles x: {self.bunch1.angle_x * 1e3:.2f}, {self.bunch2.angle_x * 1e3:.2f} mrad\n'
                        f'Beam Offsets: {self.bunch1_r_original[0]:.0f}, '
                        f'{self.bunch1_r_original[1]:.0f} um')
                        # f'Beam Offsets: {np.sqrt(np.sum(self.bunch1_r_original[:2] ** 2)):.0f}, '
                        # f'{np.sqrt(np.sum(self.bunch2_r_original[:2] ** 2)):.0f} um')
        return param_string

    def __str__(self):
        return (f'BunchCollider:\n'
                f'z_shift: {self.z_shift}, amplitude: {self.amplitude}, x_lim_sigma: {self.x_lim_sigma}, '
                f'y_lim_sigma: {self.y_lim_sigma}, z_lim_sigma: {self.z_lim_sigma}, n_points_x: {self.n_points_x}, '
                f'n_points_y: {self.n_points_y}, n_points_z: {self.n_points_z}, n_points_t: {self.n_points_t}, '
                f'bunch1_r_original: {self.bunch1_r_original}, bunch2_r_original: {self.bunch2_r_original}, '
                f'bunch1_beta_original: {self.bunch1_beta_original}, bunch2_beta_original: {self.bunch2_beta_original}'
                f'\nGaussian Smearing Sigma: {self.gaus_smearing_sigma}\n'
                f'Gaussian Z Efficiency Width: {self.gaus_z_efficiency_width}\n'
                f'Background: {self.bkg}\n'
                f'\n\nbunch1:\n{self.bunch1}\n'
                f'\n\nbunch2:\n{self.bunch2}\n'
                )


def gaus(x, a, b, c):
    return a * np.exp(-(x - b) ** 2 / (2 * c ** 2))


def efficiency(z, z_quad=700, z_switch=200, steepness=-0.1):
    # Quadratic part
    quadratic_part = np.where(np.abs(z) <= z_switch, 1 - (z / z_quad) ** 2, 0)

    # Calculate the value and derivative of the quadratic part at z_switch
    q_sw = 1 - (z_switch / z_quad) ** 2
    dq_sw = -2 * z_switch / z_quad ** 2

    if not 0 < q_sw < 1:
        raise ValueError('Quadratic part must be between 0 and 1 at z_switch.')

    # Sigmoid parameters
    sigmoid_z0 = z_switch - np.log(steepness * q_sw / dq_sw - 1) / steepness
    sigmoid_a = q_sw * (1 + np.exp(-steepness * (z_switch - sigmoid_z0)))

    # Sigmoid part
    sigmoid_part = np.where(np.abs(z) > z_switch,
                            sigmoid_a / (1 + np.exp(-steepness * (np.abs(z) - sigmoid_z0))),
                            0)

    # Combine both parts
    efficiency_value = quadratic_part + sigmoid_part

    # Clip the values to ensure they stay within [0, 1]
    return np.clip(efficiency_value, 0, 1)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 22 09:31 2024
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/generate_sim_training_data

@author: Dylan Neff, dn277127
"""

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from BunchCollider import BunchCollider


def main():
    base_path = 'C:/Users/Dylan/Desktop/vernier_scan/'
    training_set_name = 'training_set_1'
    training_set_dir = os.path.join(base_path + 'training_data/', training_set_name)
    if not os.path.exists(training_set_dir):
        os.mkdir(training_set_dir)
    info_file_path = os.path.join(training_set_dir, 'info.txt')
    training_csv_path = os.path.join(training_set_dir, 'training_data.csv')

    n_sims_to_save = 100

    vernier_scan_date = 'Aug12'

    longitudinal_fit_path = f'{base_path}CAD_Measurements/VernierScan_{vernier_scan_date}_COLOR_longitudinal_fit.dat'
    blue_fit_path = longitudinal_fit_path.replace('_COLOR_', '_blue_')
    yellow_fit_path = longitudinal_fit_path.replace('_COLOR_', '_yellow_')

    amplitude = 1.2048100105307071e+27
    shift = -14507.674093844653

    parameter_ranges = {
        'beta_star': [60, 100],
        'beam_width_x': [130, 190],
        'beam_width_y': [130, 190],
        'beam_length_scale_1': [0.8, 1.2],
        'beam_length_scale_2': [0.8, 1.2],
        'crossing_angle_1x': [-3e-3, 3e-3],
        'crossing_angle_1y': [-1e-3, 1e-3],
        'crossing_angle_2x': [-3e-3, 3e-3],
        'crossing_angle_2y': [-1e-3, 1e-3],
        'offset_1x': [-1000, 1000],
        'offset_1y': [-1000, 1000],
        'bkg': [0, 1e-16],
        'mbd_resolution': [0.01, 10],
        'mbd_z_eff_width': [50, 5000]
    }

    collider_sim = BunchCollider()
    collider_sim.set_longitudinal_fit_parameters_from_file(blue_fit_path, yellow_fit_path)
    collider_sim.set_amplitude(amplitude)
    collider_sim.set_z_shift(shift)

    write_info_file(collider_sim, info_file_path)

    batch_i = 1
    while True:
        df = []

        print(f'Starting batch {batch_i} at {datetime.now()}:')
        sys.stdout.flush()

        for i in tqdm(range(n_sims_to_save), desc='Simulations'):
            random_parameters = generate_random_parameters(parameter_ranges)
            df_i = run_sim_get_df(collider_sim, random_parameters)
            df.append(df_i)

        df = pd.DataFrame(df)

        # Append df to csv. If csv doesn't exist, create it.
        if os.path.exists(training_csv_path):
            df.to_csv(training_csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(training_csv_path, mode='w', header=True, index=False)

        sys.stdout.flush()
        print(f'Batch {batch_i} saved to file')
        batch_i += 1
        sys.stdout.flush()

    print('donzo')


def run_sim_get_df(collider_sim, parameters):
    """
    Run simulation with parameters. Return a dictionary of the parameters and the luminosity density vs z.
    :param collider_sim:
    :param parameters:
    :return:
    """
    collider_sim.set_bunch_beta_stars(parameters['beta_star'], parameters['beta_star'])
    collider_sim.set_bunch_offsets([parameters['offset_1x'], parameters['offset_1y']], [0, 0])
    collider_sim.set_bunch_sigmas([parameters['beam_width_x'], parameters['beam_width_y']],
                                  [parameters['beam_width_x'], parameters['beam_width_y']])
    collider_sim.set_bunch_crossing(parameters['crossing_angle_1x'], parameters['crossing_angle_1y'],
                                    parameters['crossing_angle_2x'], parameters['crossing_angle_2y'])
    collider_sim.set_longitudinal_fit_scaling(parameters['beam_length_scale_1'], parameters['beam_length_scale_2'])
    collider_sim.set_bkg(parameters['bkg'])
    collider_sim.set_gaus_smearing_sigma(parameters['mbd_resolution'])
    collider_sim.set_gaus_z_efficiency_width(parameters['mbd_z_eff_width'])

    collider_sim.run_sim_parallel()
    zs, z_dist = collider_sim.get_z_density_dist()
    z_dict = dict(zip(zs, z_dist))

    # Combine parameters and z_dict into one dictionary
    df = {**parameters, **z_dict}

    return df

def generate_random_parameters(parameter_ranges):
    """
    With the input parameter_ranges dictionary, generate a random set of parameters within the ranges.
    :param parameter_ranges:
    :return:
    """
    random_parameters = {}
    for param, range_vals in parameter_ranges.items():
        if range_vals is not None:
            random_parameters[param] = np.random.uniform(low=range_vals[0], high=range_vals[1])
        else:
            random_parameters[param] = None
    return random_parameters


def write_info_file(collider_sim, file_path):
    """
    Write info file for training data set. Include all parameters which will not be varied in the training data set
    and included in the training data.
    :param collider_sim:
    :param file_path:
    :return:
    """
    collider_sim.generate_grid()
    grid_info = collider_sim.get_grid_info()
    out_str = (
    f"Collider Amplitude: {collider_sim.amplitude}\n"
    f"Collider Shift: {collider_sim.z_shift}\n"
    f"Bunch1 r original: {collider_sim.bunch1_r_original}\n"
    f"Bunch2 r original: {collider_sim.bunch2_r_original}\n"
    f"Bunch1 delay: {collider_sim.bunch1.delay}\n"
    f"Bunch2 delay: {collider_sim.bunch2.delay}\n"
    )
    for param, val in grid_info.items():
        out_str += f'{param}: {val}\n'

    with open(file_path, 'w') as file:
        file.write(out_str)


if __name__ == '__main__':
    main()

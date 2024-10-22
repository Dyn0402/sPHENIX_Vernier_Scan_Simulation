#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 22 09:31 2024
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/generate_sim_training_data

@author: Dylan Neff, dn277127
"""

from BunchCollider import BunchCollider

def main():
    beta_star_range = [60, 100]
    beam_width_range = [130, 190]
    beam_length_scale_range = [0.8, 1.2]
    crossing_angle_range = [-3e-3, 3e-3]
    offset_range = [-1000, 1000]
    bkg_range = None
    mbd_resolution_range = None
    mbd_z_eff_width_range = None

    parameter_ranges = {
        'bet'
    }


    print('donzo')


def print_info_file(collider_sim, file_path):
    # Need to print number of points, initial rs, integration ranges
    pass


if __name__ == '__main__':
    main()

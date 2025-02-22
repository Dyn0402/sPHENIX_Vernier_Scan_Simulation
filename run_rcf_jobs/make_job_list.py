#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on February 10 5:47 PM 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/make_job_list.py

@author: Dylan Neff, Dylan
"""

import numpy as np


def main():
    scan_dates = ['Aug12']
    scan_orientations = ['Horizontal', 'Vertical']
    beam_widths_x = np.arange(150, 170 + 1, 1)
    # beam_widths_x = np.arange(150, 170 + 1, 1)
    # beam_widths_y = np.arange(146, 168 + 1, 1)
    beam_widths_y = np.arange(169, 175 + 1, 1)
    beta_stars  = np.arange(80, 115 + 5, 5)
    memory = '4096MB'
    job_file_name = 'vernier_parameter_scan_jobs.list'

    # Write a job list with all combinations of parameters
    with open(job_file_name, 'w') as file:
        for scan_date in scan_dates:
            for scan_orientation in scan_orientations:
                for beam_width_x in beam_widths_x:
                    for beam_width_y in beam_widths_y:
                        for beta_star in beta_stars:
                            file.write(f'{memory}, {scan_date}, {scan_orientation}, {beam_width_x}, '
                                       f'{beam_width_y}, {beta_star}\n')
    print('Job list created.')


if __name__ == '__main__':
    main()

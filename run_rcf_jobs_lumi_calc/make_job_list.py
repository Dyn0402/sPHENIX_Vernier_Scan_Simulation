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
    num_jobs = 1000
    memory = '4096MB'
    job_file_name = 'lumi_calc_jobs.list'

    # Write a job list with all jobs
    with open(job_file_name, 'w') as file:
        for i in range(num_jobs):
            file.write(f'{memory}, {i}\n')
    print('Job list created.')


if __name__ == '__main__':
    main()

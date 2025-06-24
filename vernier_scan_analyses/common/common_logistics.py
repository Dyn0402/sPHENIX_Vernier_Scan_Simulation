#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 24 3:20 PM 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/common_logistics.py

@author: Dylan Neff, Dylan
"""

import platform


def set_base_path():
    """
    Set the base path depending on the hostname of the machine and platform.
    """
    if platform.system() == 'Windows' and platform.node() == 'DESKTOP-BCED9EL':
        return 'F:/Saclay/'
    elif platform.system() == 'Linux' and platform.node() == 'dylan':
        return '/local/home/dn277127/Bureau/'
    elif platform.system() == 'Linux' and platform.node() == 'dn277127':
        return '/local/home/dn277127/Bureau/'
    else:
        raise ValueError(f"Unknown platform or hostname: {platform.system()} on {platform.node()}")
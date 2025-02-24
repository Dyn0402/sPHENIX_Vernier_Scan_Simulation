#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on February 18 09:45 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/read_cw_rates_from_root

@author: Dylan Neff, dn277127
"""

import platform
import numpy as np
import matplotlib.pyplot as plt
import uproot


def main():
    if platform.system() == 'Linux':
        base_path = '/local/home/dn277127/Bureau/vernier_scan/vertex_data/'
    elif platform.system() == 'Windows':
        base_path = 'C:/Users/Dylan/Desktop/vernier_scan/vertex_data/'
    else:
        base_path = ''
        print('Unknown system, no base path set.')
    path = f'{base_path}MBDNSCount.root'
    bunch_revolution_frequency = 78.4  # kHz
    n_bunches = 111
    write = True

    sets = {'cw': {'Vertical': {'x': [], 'y': [], 'name': 'gr_RTrue_V;1'},
                      'Horizontal': {'x': [], 'y': [], 'name': 'gr_RTrue_H;1'}},
            'cw_method2': {'Vertical': {'x': [], 'y': [], 'name': 'gr_MBDNS_Meas_MMC_V;1'},
                   'Horizontal': {'x': [], 'y': [], 'name': 'gr_MBDNS_Meas_MMC_H;1'}},
            'cw_errors': {'Vertical': {'x': [], 'y': [], 'name': 'gr_MBDNS_Meas_V;1'},
                           'Horizontal': {'x': [], 'y': [], 'name': 'gr_MBDNS_Meas_H;1'}},
            }

    with uproot.open(path) as file:
        for set_type in sets:
            for orientation in sets[set_type]:
                rates_graph = file[sets[set_type][orientation]['name']]
                x_vals = np.array(rates_graph.tojson()['fX'])
                y_vals = np.array(rates_graph.tojson()['fY']) * bunch_revolution_frequency * 1e3 * n_bunches
                sets[set_type][orientation]['x'] = x_vals
                sets[set_type][orientation]['y'] = y_vals
                if 'fEY' in rates_graph.tojson():
                    y_errs = np.array(rates_graph.tojson()['fEY']) * bunch_revolution_frequency * 1e3 * n_bunches
                    print(y_errs)
                    sets[set_type][orientation]['y_err'] = y_errs

    # Plot the data
    for set_type in sets:
        for orientation in sets[set_type]:
            fig, ax = plt.subplots()
            ax.plot(sets[set_type][orientation]['x'], sets[set_type][orientation]['y'], marker='o',
                    label=f'{set_type} {orientation}')
            ax.set_xlabel('Offset (um)')
            ax.set_ylabel('Rate (Hz)')
            ax.set_ylim(bottom=0)
            ax.legend()
            fig.tight_layout()
    plt.show()

    if write:
        # Write results to files
        for set_type in sets:
            with open(f'{set_type}_rates.txt', 'w') as file:
                for orientation in sets[set_type]:
                    column_line = f'{orientation} pos MBDNS Rate (Hz)'
                    column_line += ' Error (Hz)' if 'y_err' in sets[set_type][orientation] else ''
                    file.write(f'{column_line}\n')
                    for i in range(len(sets[set_type][orientation]['x'])):
                        val_line = f'{sets[set_type][orientation]["x"][i]} {sets[set_type][orientation]["y"][i]}'
                        if 'y_err' in sets[set_type][orientation]:
                            val_line += f' {sets[set_type][orientation]["y_err"][i]}'
                        file.write(f'{val_line}\n')

    print('donzo')


if __name__ == '__main__':
    main()

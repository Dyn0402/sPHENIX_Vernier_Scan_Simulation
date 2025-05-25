#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 21 04:22 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/bpm_analysis

@author: Dylan Neff, dn277127
"""

import platform
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def main():
    if platform.system() == 'Windows':
        base_path = 'C:/Users/Dylan/Desktop/'
    else:
        base_path = '/local/home/dn277127/Bureau/'
    scan_path = f'{base_path}Vernier_Scans/auau_oct_16_24/'
    # scan_path = f'{base_path}Vernier_Scans/pp_aug_12_24/'
    # scan_path = f'{base_path}Vernier_Scans/pp_july_11_24/'
    bpm_file_path = f'{scan_path}bpms.dat'

    start_time, end_time = get_start_end_times(scan_path)

    bpm_analysis(bpm_file_path, start_time, end_time, plot=True)
    print('donzo')


def bpm_analysis(bpm_file_path, start_time, end_time, plot=True):
    """
    Read BMP file and extract Vernier scan steps and crossing angles.
    """
    bpm_separation_distance = 16250  # mm
    step_derivative_threshold = 30

    bpm_data = pd.read_csv(bpm_file_path, sep='\t', header=2)
    bpm_data.drop(columns=['Unnamed: 9'], inplace=True)
    bpm_data = bpm_data.rename(columns={'# Time ': 'Time'})
    bpm_data = bpm_data.rename(columns={col: col.replace(' ', '') for col in bpm_data.columns})
    for col in bpm_data.columns:
        if col == 'Time':
            bpm_data[col] = pd.to_datetime(bpm_data[col], errors='coerce')
        else:
            bpm_data[col] = pd.to_numeric(bpm_data[col], errors='coerce')

    bpm_data = bpm_data[(bpm_data['Time'] >= start_time) & (bpm_data['Time'] <= end_time)]

    if plot:
        for c in ['b', 'y']:
            fig, ax = plt.subplots()
            for o in ['h', 'v']:
                ax.plot(np.array(bpm_data['Time']), np.array(bpm_data[f'{c}7bx_{o}']), label=f'{c}7bx_{o}')
                ax.plot(np.array(bpm_data['Time']), np.array(bpm_data[f'{c}8bx_{o}']), label=f'{c}8bx_{o}')
            ax.legend()
            fig.tight_layout()

    # Take derivative of bpms and plot
    if plot:
        fig, ax = plt.subplots(figsize=(8, 5))
    times = np.array(bpm_data['Time'])[:-1]
    masks = []
    for c in ['b', 'y']:
        for n in ['7', '8']:
            for o in ['h', 'v']:
                dbpms = np.abs(np.diff(bpm_data[f'{c}{n}bx_{o}']))
                masks.append(dbpms > step_derivative_threshold)
                if plot:
                    ax.plot(times, dbpms, label=f'{c}{n}bx_{o}')

    times_above_threshold = np.array(bpm_data['Time'])[:-1][np.any(masks, axis=0)]
    times_above_threshold = get_step_bounds(times_above_threshold, max_gap_seconds=30)

    if plot:
        ax.axhline(step_derivative_threshold, color='black', ls='--')
        for start, end in times_above_threshold:
            ax.axvspan(start, end, color='red', alpha=0.5)

    step_bounds = [(start, end) for start, end in times_above_threshold if start >= start_time and end <= end_time]

    steps = [{'step': 0, 'start': start_time}]
    for i, (start, end) in enumerate(step_bounds):
        steps[-1]['end'] = start
        steps.append({'step': i + 1, 'start': end})
    steps[-1]['end'] = end_time

    for step in steps:
        duration = step['end'] - step['start']
        mid_time = (step['start'] + duration / 2.0).round('s')
        step['duration'] = duration.total_seconds()
        step['mid_time'] = mid_time

    blue_xing_h = np.arctan((bpm_data['b7bx_h'] - bpm_data['b8bx_h']) / bpm_separation_distance)
    yellow_xing_h = np.arctan((bpm_data['y7bx_h'] - bpm_data['y8bx_h']) / bpm_separation_distance)
    blue_xing_v = np.arctan((bpm_data['b7bx_v'] - bpm_data['b8bx_v']) / bpm_separation_distance)
    yellow_xing_v = np.arctan((bpm_data['y7bx_v'] - bpm_data['y8bx_v']) / bpm_separation_distance)

    rel_xing_h = blue_xing_h - yellow_xing_h
    rel_xing_v = blue_xing_v - yellow_xing_v

    if plot:
        fig, ax = plt.subplots(figsize=(8, 5))
        for step in steps:
            ax.axvline(step['start'], color='black', ls='-', alpha=0.2)
            ax.axvline(step['end'], color='black', ls='-', alpha=0.2)

        ax.plot(np.array(bpm_data['Time']), np.array(blue_xing_h), color='blue', label='Blue Horizontal')
        ax.plot(np.array(bpm_data['Time']), np.array(yellow_xing_h), color='orange', label='Yellow Horizontal')
        ax.plot(np.array(bpm_data['Time']), np.array(blue_xing_v), color='blue', ls=':', label='Blue Vertical')
        ax.plot(np.array(bpm_data['Time']), np.array(yellow_xing_v), color='orange', ls=':', label='Yellow Vertical')

        ax.plot(np.array(bpm_data['Time']), np.array(rel_xing_h), color='green', label='Relative Horizontal')
        ax.plot(np.array(bpm_data['Time']), np.array(rel_xing_v), color='green', ls=':',
                label='Relative Vertical')
        ax.legend()

    for step in steps:  # Get average crossing angles within each step
        step_start = step['start']
        step_end = step['end']
        step_mask = (bpm_data['Time'] >= step_start) & (bpm_data['Time'] <= step_end)
        step_blue_xing_h = blue_xing_h[step_mask].mean()
        step_yellow_xing_h = yellow_xing_h[step_mask].mean()
        step_blue_xing_v = blue_xing_v[step_mask].mean()
        step_yellow_xing_v = yellow_xing_v[step_mask].mean()
        step['blue angle h'] = step_blue_xing_h
        step['yellow angle h'] = step_yellow_xing_h
        step['blue angle v'] = step_blue_xing_v
        step['yellow angle v'] = step_yellow_xing_v


    # Get horizontal and vertical beam positions at z=0 for each step
    bpm_first_step = bpm_data[(bpm_data['Time'] >= steps[0]['start']) & (bpm_data['Time'] <= steps[0]['end'])]
    blue_nom_pos = {o: (bpm_first_step[f'b7bx_{o}'] + bpm_first_step[f'b8bx_{o}']) / 2 for o in ['h', 'v']}
    yellow_nom_pos = {o: (bpm_first_step[f'y7bx_{o}'] + bpm_first_step[f'y8bx_{o}']) / 2 for o in ['h', 'v']}
    nom_offset = {o: blue_nom_pos[o] - yellow_nom_pos[o] for o in ['h', 'v']}

    for step in steps:
        for o in ['h', 'v']:
            step_start = step['start']
            step_end = step['end']
            step_mask = (bpm_data['Time'] >= step_start) & (bpm_data['Time'] <= step_end)
            step_blue_beam_pos = (bpm_data[f'b7bx_{o}'][step_mask] + bpm_data[f'b8bx_{o}'][step_mask]) / 2
            step_yellow_beam_pos = (bpm_data[f'y7bx_{o}'][step_mask] + bpm_data[f'y8bx_{o}'][step_mask]) / 2
            step[f'blue_{o}_pos'] = step_blue_beam_pos.mean()
            step[f'yellow_{o}_pos'] = step_yellow_beam_pos.mean()
            step[f'{o}_offset'] = step_blue_beam_pos.mean() - step_yellow_beam_pos.mean()
            step[f'{o}_offset_shifted'] = step[f'{o}_offset'] - nom_offset[o].mean()
            if plot:
                print(step)

    if plot:
        fig_offset, ax_offset = plt.subplots()
        fig_offset_shifted, ax_offset_shifted = plt.subplots()
        for o in ['h', 'v']:
            fig, ax = plt.subplots()
            z0_blue_beam_pos = (bpm_data[f'b7bx_{o}'] + bpm_data[f'b8bx_{o}']) / 2
            z0_yellow_beam_pos = (bpm_data[f'y7bx_{o}'] + bpm_data[f'y8bx_{o}']) / 2
            ax.plot(np.array(bpm_data['Time']), np.array(z0_blue_beam_pos), color='blue', label='Blue Beam Position')
            ax.plot(np.array(bpm_data['Time']), np.array(z0_yellow_beam_pos), color='orange', label='Yellow Beam Position')
            ax.legend()
            ax.set_ylabel(f'{o}_beam_pos at z=0 (um)')
            fig.tight_layout()

            z0_beam_offset = z0_blue_beam_pos - z0_yellow_beam_pos
            ax_offset.plot(np.array(bpm_data['Time']), np.array(z0_beam_offset), label=f'{o} Offset')
            ax_offset_shifted.plot(np.array(bpm_data['Time']), np.array(z0_beam_offset) - nom_offset[o].mean(),
                                   label=f'{o} Offset Shifted')
        ax_offset.legend()
        ax_offset.set_ylabel(f'{o}_beam_offset at z=0 (um)')
        fig_offset.tight_layout()
        ax_offset_shifted.legend()
        ax_offset_shifted.set_ylabel(f'{o}_beam_offset at z=0 (um) shifted')
        fig_offset_shifted.tight_layout()

        plt.show()

    return pd.DataFrame(steps)


def get_step_bounds(time_array, max_gap_seconds=2):
    """
    Reduce consecutive timestamps to (start, end) pairs.

    Args:
        time_array (array-like): Sequence of timestamps (strings or datetime).
        max_gap_seconds (int): Max allowed gap (in seconds) between steps in a sequence.

    Returns:
        list of tuples: Each tuple contains (start_time, end_time) of a detected sequence.
    """
    times = pd.to_datetime(time_array)
    times = pd.Series(times).sort_values().reset_index(drop=True)

    time_diffs = times.diff().fillna(pd.Timedelta(seconds=0)).dt.total_seconds()

    boundaries = np.where(time_diffs > max_gap_seconds)[0]

    starts = np.insert(boundaries, 0, 0)
    ends = np.append(boundaries - 1, len(times) - 1)

    return [(times[start], times[end]) for start, end in zip(starts, ends)]


def get_start_end_times(scan_path):
    """
    Get the start and end times of the scan from the BPM file.
    :param scan_path: Path to the scan directory.
    :return: Start and end times as pandas Timestamps.
    """
    start_end_times_path = f'{scan_path}scan_start_end_times.txt'
    with open(start_end_times_path, 'r') as f:
        lines = f.readlines()
    start_time_str, end_time_str = lines[0].split('\t')[-1].strip(), lines[1].split('\t')[-1].strip()
    start_time, end_time = pd.Timestamp(start_time_str), pd.Timestamp(end_time_str)

    return start_time, end_time


if __name__ == '__main__':
    main()

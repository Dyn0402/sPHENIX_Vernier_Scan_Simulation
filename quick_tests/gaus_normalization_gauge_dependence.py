#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on July 24 8:17 PM 2025
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/gaus_normalization_gauge_dependence.py

@author: Dylan Neff, dylan
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    # Simulated analog signal: gaussian
    mu = 52.03  # mean of the Gaussian
    sigma = 1  # standard deviation of the Gaussian
    ampls = [0.8, 0.4]  # amplitude of the Gaussian
    colors = ['blue', 'green']  # colors for the different amplitudes

    # ADC settings
    sampling_rate = 1000  # samples per second
    adc_bits = 7  # ADC resolution in bits
    v_min, v_max = 0, 1  # input voltage range

    # Time points for sampling
    t = np.linspace(0, 100, 10000)  # High-res time for "true" analog signal
    t_sampled = np.linspace(0, 100, sampling_rate, endpoint=False)

    fig_quantized, ax_quantized = plt.subplots(figsize=(10, 5))
    fig_normalized, ax_normalized = plt.subplots(figsize=(10, 5))
    for ampl, color in zip(ampls, colors):
        analog_signal = gaussian(t, ampl, mu, sigma)  # Gaussian function for the analog signal
        sampled_signal = gaussian(t_sampled, ampl, mu, sigma)

        # Quantize the sampled signal
        quantized_signal = quantize_signal(sampled_signal, adc_bits, v_min, v_max)

        # Normalize the quantized signal to a PDF
        normalized_signal = normalize_signal(t_sampled, quantized_signal)

        # Plotting
        ax_quantized.plot(t, analog_signal, label='Analog Signal', color=color, alpha=0.6)
        ax_quantized.plot(t_sampled, sampled_signal, 'o', label='Sampled Points', color=color)
        ax_quantized.step(t_sampled, quantized_signal, label='Quantized (ADC Output)', color=color, ls='--', where='mid')

        ax_normalized.plot(t_sampled, normalized_signal, label=f'Amplitude: {ampl:.1f}', color=color)

    ax_quantized.set_xlabel('Time [s]')
    ax_quantized.set_ylabel('Voltage [V]')
    ax_quantized.set_title(f'{adc_bits}-bit ADC Sampling at {sampling_rate} Hz')
    ax_quantized.grid(True)
    ax_quantized.legend()
    fig_quantized.tight_layout()

    ax_normalized.set_xlabel('Time [s]')
    ax_normalized.set_ylabel('Normalized PDF')
    ax_normalized.set_title('Normalized Quantized Signal to PDF')
    ax_normalized.grid(True)
    ax_normalized.legend()
    fig_normalized.tight_layout()

    plt.show()

    print('donzo')


#Define the Gaussian function
def gaussian(x, ampl, mu, sigma):
    return ampl * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def quantize_signal(signal, adc_bits, v_min=0, v_max=1):
    """
    Quantize the signal based on the ADC resolution.
    """
    adc_levels = 2 ** adc_bits
    step_size = (v_max - v_min) / (adc_levels - 1)
    quantized_signal = np.round((signal - v_min) / step_size) * step_size
    return np.clip(quantized_signal, v_min, v_max)


def normalize_signal(xs, signal):
    """
    Normalize the signal to a pdf
    """
    integral = np.trapezoid(signal, xs)
    return signal / integral



if __name__ == '__main__':
    main()

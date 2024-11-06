#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 22 21:10 2024
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/neural_network

@author: Dylan Neff, dn277127
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Reshape, GaussianNoise
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
from tensorflow.signal import fft
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    train_neural_network()
    print('donzo')


def train_neural_network():
    base_path = '/local/home/dn277127/Bureau/vernier_scan/training_data/'
    # training_set = 'training_set_1'
    # training_set = 'simple_par_training_set_1'
    training_set = 'essential_par_training_set_1'
    training_csv_name = 'training_data.csv'
    retrain = True

    # Load data
    df = pd.read_csv(f'{base_path}{training_set}/{training_csv_name}')
    # n_pars = 14
    # n_pars = 2
    n_pars = 11

    # Define input features and target values
    X = df.iloc[:, :n_pars].values  # First 14 columns are the inputs (up to mbd_z_eff_width)
    y = df.iloc[:, n_pars:].values  # The rest are the z-distributions (targets)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the input features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if retrain:
        # Build a neural network model
        # model = Sequential()
        # model.add(Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)))
        # model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
        # model.add(Dense(64, activation='relu'))
        # model.add(Dense(y_train.shape[1], activation='linear'))  # Initial dense output
        # model.add(Reshape((y_train.shape[1], 1)))  # Reshape to 1D for convolution
        # model.add(Conv1D(32, kernel_size=5, activation='relu', padding='same'))
        # model.add(Flatten())  # Flatten back to original output shape
        # model.add(Dense(y_train.shape[1]))  # Final linear output layer

        # Build the model
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)))
        model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
        model.add(Dense(y_train.shape[1], activation='linear'))
        model.add(Reshape((y_train.shape[1], 1)))  # Reshape to (sequence_length, 1) for Conv1D
        model.add(Conv1D(16, kernel_size=15, activation='relu', padding='same'))
        model.add(Flatten())
        model.add(Dense(y_train.shape[1]))  # Final linear output layer
        model.add(GaussianNoise(0.1))  # Adds Gaussian noise to smooth out oscillations

        # Compile the model
        # model.compile(optimizer='adam', loss='mean_squared_error')
        # model.compile(optimizer='adam', loss=smoothness_loss)
        # model.compile(optimizer='adam', loss=smoothness_freq_loss)
        model.compile(optimizer='adam', loss=smooth_mse)

        # Train the model and store the training history
        history = model.fit(X_train_scaled, y_train, epochs=3000, batch_size=32, validation_split=0.2)

        # Evaluate the model
        loss = model.evaluate(X_test_scaled, y_test)
        print(f"Test Loss: {loss}")

        # Plot training & validation loss values
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss (Mean Squared Error)')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()
    else:
        # Load a saved model
        model = tf.keras.models.load_model(f'{base_path}{training_set}/neural_network_model.keras',
                                           custom_objects={'smoothness_freq_loss': smoothness_freq_loss})

    y_pred = model.predict(X_test_scaled)

    # Plot predicted vs actual for the first test example
    for i in range(15):
        plt.figure(figsize=(10, 6))
        plt.plot(y_test[i], label='Actual Distribution')
        plt.plot(y_pred[i], label='Predicted Distribution')
        plt.title('Predicted vs Actual Z-Distribution for First Test Example')
        plt.xlabel('Z-Distribution Point')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)

        # Plot second derivative of predicted vs actual for the first test example
        plt.figure(figsize=(10, 6))
        diff_pred = y_pred[i, 1:] - y_pred[i, :-1]
        second_diff_pred = diff_pred[1:] - diff_pred[:-1]
        diff_actual = y_test[i, 1:] - y_test[i, :-1]
        second_diff_actual = diff_actual[1:] - diff_actual[:-1]
        plt.plot(second_diff_actual, label='Actual Second Derivative')
        plt.plot(second_diff_pred, label='Predicted Second Derivative')
        plt.title('Predicted vs Actual Second Derivative of Z-Distribution for First Test Example')
        plt.xlabel('Z-Distribution Point')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)


        # Plot the Fourier transform of the predicted distribution
        # fft_pred = fft(tf.cast(y_pred[i], tf.complex64))
        # fft_magnitudes = K.abs(fft_pred)
        # plt.figure(figsize=(10, 6))
        # plt.plot(fft_magnitudes)
        # plt.plot(fft_magnitudes[:-1] - fft_magnitudes[1:])
        # plt.title('Fourier Transform of Predicted Z-Distribution')
        # plt.xlabel('Frequency')
        # plt.ylabel('Magnitude')
        # plt.grid(True)
    plt.show()

    # Save the model
    model.save(f'{base_path}{training_set}/neural_network_model.keras')


def smoothness_loss(y_true, y_pred):
    diff = K.abs(y_pred[:, 1:] - y_pred[:, :-1])
    return K.mean(diff) + K.mean(K.square(y_true - y_pred))  # Penalize roughness + MSE


def smoothness_freq_loss(y_true, y_pred):
    # Difference between consecutive y_pred values
    diff = K.abs(y_pred[:, 1:] - y_pred[:, :-1])

    # Mean squared error between true and predicted values
    mse = K.mean(K.square(y_true - y_pred))

    # Fourier transform to penalize high-frequency oscillations
    # First cast y_pred to complex type for FFT
    y_pred_complex = tf.cast(y_pred, tf.complex64)

    # Perform FFT along the last axis
    fft_pred = tf.signal.fft(y_pred_complex)

    # Take absolute values to get magnitudes of the frequencies
    fft_magnitudes = K.abs(fft_pred)
    fft_derivatives = fft_magnitudes[:-1] - fft_magnitudes[1:]
    fft_diff = K.abs(fft_derivatives)

    # Penalize frequencies higher than the second harmonic (e.g., from index 3 onwards)
    # high_freq_penalty = K.sum(fft_magnitudes[:, 3:])

    # Combine loss terms
    # return K.mean(diff) + mse + high_freq_penalty * 0.01  # Adjust penalty weight as needed
    return K.mean(diff) + mse + K.mean(fft_diff)  # Adjust penalty weight as needed


def smooth_mse(y_true, y_pred):
    # Difference between consecutive y_pred values
    diff = K.abs(y_pred[:, 1:] - y_pred[:, :-1])

    # Mean squared error between true and predicted values
    mse = K.mean(K.square(y_true - y_pred))

    # Approximate the second derivative
    diff1 = y_pred[:, 1:] - y_pred[:, :-1]
    diff2 = diff1[:, 1:] - diff1[:, :-1]
    smooth_penalty = tf.reduce_mean(tf.square(diff2))
    return mse + 0.1 * smooth_penalty


if __name__ == '__main__':
    main()

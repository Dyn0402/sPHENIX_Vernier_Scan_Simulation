#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 22 21:17 2024
Created in PyCharm
Created as sPHENIX_Vernier_Scan_Simulation/gaussian_process

@author: Dylan Neff, dn277127
"""

import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    base_path = '/local/home/dn277127/Bureau/vernier_scan/training_data/training_set_1/'

    # Load data
    df = pd.read_csv(f'{base_path}training_data.csv')

    # Define input features and target values
    X = df.iloc[:200, :14].values  # First 14 columns are the inputs
    y = df.iloc[:200, 14:].values  # The rest are the z-distributions (targets)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the input features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Set up Gaussian Process model with RBF kernel
    kernel = C(1.0, (1e-4, 1e1)) * RBF(length_scale=1.0, length_scale_bounds=(1e-4, 1e1))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

    # Train the Gaussian Process model (fit to the training data)
    gpr.fit(X_train_scaled, y_train)

    # Predict on test data
    y_pred, y_std = gpr.predict(X_test_scaled, return_std=True)

    # Evaluate performance
    mse = np.mean((y_pred - y_test) ** 2)
    print(f"Test Mean Squared Error: {mse}")

    # Save the model
    import joblib
    joblib.dump(gpr, f'{base_path}gaussian_process_model.pkl')

    print('donzo')


if __name__ == '__main__':
    main()

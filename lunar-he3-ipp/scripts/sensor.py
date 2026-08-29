"""
Sensor model: point-sampling with additive Gaussian noise.
z = truth(x, y) + N(0, sigma^2)
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')

import numpy as np


class Sensor:
    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def measure(self, truth_value):
        noise = np.random.normal(0.0, self.sigma)
        return float(truth_value + noise)

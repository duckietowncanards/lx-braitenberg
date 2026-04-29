from typing import Tuple

import numpy as np


X = 30000
Y = 10000

def get_motor_left_matrix(shape: Tuple[int, int]) -> np.ndarray:

    h, w = shape

    mu_y, mu_x = (2*h - 1) / 2, (w - 1) / 2

    y = np.arange(h, dtype=np.float32)[:, None]
    x = np.arange(w, dtype=np.float32)[None, :]

    exponent = -0.5 * ((y - mu_y)**2 / Y + (x - mu_x)**2 / X)
    gaussian = np.exp(exponent) / (2 * np.pi * np.sqrt(Y * X))
    # Favor the right side
    #gaussian[:, :w//2] *= 2
    gaussian[:, w//2:] *= -1

    return gaussian


def get_motor_right_matrix(shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape

    mu_y, mu_x = (2*h - 1) / 2, (w - 1) / 2

    y = np.arange(h, dtype=np.float32)[:, None]
    x = np.arange(w, dtype=np.float32)[None, :]

    exponent = -0.5 * ((y - mu_y)**2 / Y + (x - mu_x)**2 / X)
    gaussian = np.exp(exponent) / (2 * np.pi * np.sqrt(Y * X))

    # Favor the right side
    #gaussian[:, w//2:] *= 2
    gaussian[:, :w//2] *= -1

    return gaussian


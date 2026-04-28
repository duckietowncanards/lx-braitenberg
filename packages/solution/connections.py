from typing import Tuple

import numpy as np


def get_motor_left_matrix(shape: Tuple[int, int]) -> np.ndarray:

    h, w = shape

    mu_y, mu_x = (2*h - 1) / 2, (w - 1) / 2
    var_y, var_x = 40000, 20000

    y = np.arange(h, dtype=np.float32)[:, None]
    x = np.arange(w, dtype=np.float32)[None, :]

    exponent = -0.5 * ((y - mu_y)**2 / var_y + (x - mu_x)**2 / var_x)
    gaussian = np.exp(exponent) / (2 * np.pi * np.sqrt(var_y * var_x))
    # Favor the right side
    gaussian[:, :w//2] *= 2

    return gaussian


def get_motor_right_matrix(shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape

    mu_y, mu_x = (2*h - 1) / 2, (w - 1) / 2
    var_y, var_x = 40000, 20000

    y = np.arange(h, dtype=np.float32)[:, None]
    x = np.arange(w, dtype=np.float32)[None, :]

    exponent = -0.5 * ((y - mu_y)**2 / var_y + (x - mu_x)**2 / var_x)
    gaussian = np.exp(exponent) / (2 * np.pi * np.sqrt(var_y * var_x))

    # Favor the right side
    gaussian[:, w//2:] *= 2

    return gaussian


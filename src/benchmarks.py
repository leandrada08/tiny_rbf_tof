"""
Definitions of benchmark functions.

Author: Francesco Saccani <francesco.saccani@unipr.it>
"""

import numpy as np


def damped_sine_wave(
    x: np.ndarray, amplitude: float = 1, decay: float = 0.3, freq: float = 10
) -> np.ndarray:
    """
    Damped sine wave function

    Args:
        x (np.ndarray): input
        amplitude (float): amplitude; default 1.
        decay (float): decay; default 0.3
        freq (float): frequency; default 10

    Returns:
        np.ndarray: damped sine wave
    """
    return amplitude * np.exp(-decay * x) * np.sin(freq * x)


def hermite_polynomial(x: np.ndarray) -> np.ndarray:
    """
    Hermite polynomial function

    Args:
        x (np.ndarray): input

    Returns:
        np.ndarray: Hermite polynomial
    """
    return (1 - x + 2 * x**2) * np.exp(-(x**2) / 2)


def mackey_glass(
    length: int,
    x0: float | np.ndarray,
    a: float = 0.2,
    b: float = 0.1,
    c: float = 10.0,
    tau: float = 23.0,
    n: int = 1_000,
    sample: float = 0.46,
    discard: int = 250,
):
    """Generate time series using the Mackey-Glass equation.

    Generates time series using the discrete approximation of the Mackey-Glass
    delay differential equation described by Grassberger & Procaccia (1983).

    Args:
        length (int, optional): length of the time series to be generated
        x0 (array): initial condition for the discrete map; in case of an array, it should be of length n
        a (float, optional): constant a in the Mackey-Glass equation; default is 0.2
        b (float, optional): constant b in the Mackey-Glass equation; default is 0.1
        c (float, optional): constant c in the Mackey-Glass equation; default is 10.0
        tau (float, optional): time delay in the Mackey-Glass equation; default is 23.0
        n  (int, optional): the number of discrete steps into which the interval between t and t + tau should be divided; this results in a time step of tau/n and an n + 1 dimensional map; default is 1000
        sample (float, optional): sampling step of the time series; it is useful to pick something between tau/100 and tau/10, with tau/sample being a factor of n; this will make sure that there are only whole number indices; default is 0.46
        discard (int, optional): number of n-steps to discard in order to eliminate transients; a total of n*discard steps will be discarded; default is 250

    Returns:
        np.ndarray: array containing the time series
    """
    sample = int(n * sample / tau)
    grids = n * discard + sample * length
    x = np.empty(grids)
    x[:n] = x0

    A = (2 * n - b * tau) / (2 * n + b * tau)
    B = a * tau / (2 * n + b * tau)

    for i in range(n - 1, grids - 1):
        x[i + 1] = A * x[i] + B * (
            x[i - n] / (1 + x[i - n] ** c) + x[i - n + 1] / (1 + x[i - n + 1] ** c)
        )
    return x[n * discard :: sample]

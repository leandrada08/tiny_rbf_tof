"""
Utility functions.

Author: Francesco Saccani <francesco.saccani@unipr.it>
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from metrics import compute_point_by_point_metrics

import pandas as pd
from tqdm import tqdm


def create_time_series(
    data: np.ndarray, look_back: int = 1, steps: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create dataset with previous `look_back` values with `step` timesteps
    distance between each sample as input and next value as target output

    Args:
        data (np.array): time series data
        look_back (int): look back; default 1
        steps (int): distance in timesteps between each sample; default 1

    Returns:
        x (np.array): input data
        y (np.array): target data
    """
    x, y = [], []
    for i in range(len(data) - look_back * steps):
        x.append(data[i : (i + look_back * steps) : steps])
        y.append(data[i + look_back * steps])
    return np.array(x), np.array(y)


def plot_error_prediction(
    y_true: np.ndarray, y_pred: np.ndarray, scope: int, title: str = "Error Prediction"
):
    """
    Plot the true and predicted values of a time series.

    Args:
        y_true (np.array): true values
        y_pred (np.array): predicted values
        title (str): title of the plot; default "Error Prediction"
    """
    _, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_true*scope, color="tab:green", label="y_true")
    ax.plot(y_pred*scope, color="tab:orange", label="y_test")
    ax.set_title(title)
    ax.set_xlabel("Time Step")
    ax.set_xlim(0, len(y_true))
    ax.set_ylabel("Distance Measurements [Cm]")
    ax.legend()
    ax.grid()
    plt.show()


def plot_absolute_error(
    y_true: np.ndarray, y_pred: np.ndarray, scope: int, error_threshold: float = 0.02
):
    """
    Plot the absolute error of a time series.

    Args:
        y_true (np.array): true values
        y_pred (np.array): predicted values
        error_threshold (float): threshold for the absolute error; default 0.02
    """
    m = compute_point_by_point_metrics(y_true, y_pred,scope)
    _, ax = plt.subplots(figsize=(12, 4))
    ax.plot(m["AE"], color="tab:blue")
    ax.axhline(error_threshold, color="tab:orange", linestyle="--")
    ax.set_title("Absolute Error")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Absolute Error [Cm]")
    ax.grid()
    plt.show()

def plot_absolute_error_histogram(
    y_true: np.ndarray, y_pred: np.ndarray, scope:int, error_threshold: float = 0.02, bins: int = 50
):
    """
    Plot the histogram of the absolute error of a time series.

    Args:
        y_true (np.array): true values
        y_pred (np.array): predicted values
        error_threshold (float): threshold for the absolute error; default 0.02
        bins (int): number of bins for the histogram; default 50
    """
    m = compute_point_by_point_metrics(y_true, y_pred,scope)
    
    # Create a histogram of the absolute error
    _, ax = plt.subplots(figsize=(12, 4))
    ax.hist(m["AE"], bins=bins, color="tab:blue", edgecolor="black", alpha=0.7)
    ax.axvline(error_threshold, color="tab:orange", linestyle="--", label="Error Threshold")
    
    ax.set_title("Absolute Error Histogram")
    ax.set_xlabel("Absolute Error [Cm]")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid()
    plt.show()

def plot_average_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scope: int,
    error_threshold: float = 0.02,
    window_size: int = 1000,
):
    """
    Plot the moving average of the absolute error of a time series.

    Args:
        y_true (np.array): true values
        y_pred (np.array): predicted values
        error_threshold (float): threshold for the average error; default 0.02
        window_size (int): window size for the moving average; default 1000
    """
    m = compute_point_by_point_metrics(y_true, y_pred,scope)
    _, ax = plt.subplots(figsize=(12, 4))
    y = np.convolve(m["AE"], np.ones(window_size) / window_size, mode="valid")
    ax.plot(range(len(y)), y, color="tab:blue", label="AE")
    ax.axhline(error_threshold, color="tab:orange", linestyle="--")
    ax.set_title(f"Moving Average (W={window_size}) of Absolute Error")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Absolute Error [Cm]")
    ax.grid()
    plt.show()


def plot_distance_and_error(
    stats: np.ndarray, ylims: Tuple[float, float] = (5e-3, 5e2)
):
    """
    Plot the evolution of the distance and the error.

    Args:
        stats (np.array): statistics of the learning process
    """
    _, ax = plt.subplots(figsize=(12, 3))
    ax.plot(stats[:, 0], ".", label="d", color="tab:blue", alpha=0.4)
    ax.plot(stats[:, 2], label="d_th", color="darkblue")
    ax.plot(stats[:, 1], ".", label="e", color="tab:orange", alpha=0.4)
    ax.plot(stats[:, 3], label="e_th", color="tab:red")
    ax.set_xlabel("Time Step")
    ax.set_ylim(ylims)
    ax.set_yscale("log")
    ax.set_title("Evolution of the Distance and the Error")
    ax.legend(loc="upper right")
    ax.grid()
    plt.show()


def plot_neurons(stats: np.ndarray):
    """
    Plot the evolution of the number of neurons.

    Args:
        stats (np.array): statistics of the learning process
    """
    _, ax = plt.subplots(figsize=(12, 2))
    ax.plot(stats[:, 4])
    ax.axhline(stats[:, 4].max(), color="tab:orange", linestyle="--")
    ax.text(
        stats.shape[0],
        stats[:, 4].max(),
        f"Max: {stats[:, 4].max():.1f}",
        color="tab:orange",
        horizontalalignment="right",
        verticalalignment="top",
    )
    ax.axhline(stats[:, 4].mean(), color="tab:green", linestyle="--")
    ax.text(
        stats.shape[0],
        stats[:, 4].mean(),
        f"Mean: {stats[:, 4].mean():.1f}",
        color="tab:green",
        horizontalalignment="right",
        verticalalignment="top",
    )
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Number of Neurons")
    ax.set_title("Evolution of the Number of Neurons")
    ax.grid()
    plt.show()






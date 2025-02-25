"""
Metrics for evaluating the performance of a model.

Author: Francesco Saccani <francesco.saccani@unipr.it>
"""

import numpy as np
import pandas as pd
from typing import Dict, Union, Tuple

# MAX_METERS=100

def _process_input(
    actual: Union[np.ndarray, pd.DataFrame, pd.Series],
    predicted: Union[np.ndarray, pd.DataFrame, pd.Series],
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform input to numpy arrays

    Args:
        actual (np.array|pd.DataFrame|pd.Series): actual values
        predicted (np.array|pd.DataFrame|pd.Series): predicted values

    Returns:
        Tuple[np.ndarray, np.ndarray]: actual and predicted values as numpy arrays
    """
    if isinstance(actual, (pd.DataFrame, pd.Series)):
        actual = actual.values
    if isinstance(predicted, (pd.DataFrame, pd.Series)):
        predicted = predicted.values
    return actual.flatten(), predicted.flatten()


def mae(
    actual: Union[np.ndarray, pd.DataFrame, pd.Series],
    predicted: Union[np.ndarray, pd.DataFrame, pd.Series],
    scope: int,
) -> float:
    """Calculate Mean Absolute Error (MAE)

    Args:
        actual (np.array|pd.DataFrame|pd.Series): actual values
        predicted (np.array|pd.DataFrame|pd.Series): predicted values

    Returns:
        float: mean absolute error
    """
    actual, predicted = _process_input(actual, predicted)
    return np.mean(np.abs(predicted - actual)) * scope


def rmse(
    actual: Union[np.ndarray, pd.DataFrame, pd.Series],
    predicted: Union[np.ndarray, pd.DataFrame, pd.Series],
    scope : int ,
) -> float:
    """Calculate Root Mean Squared Error (RMSE)

    Args:
        actual (np.array|pd.DataFrame|pd.Series): actual values
        predicted (np.array|pd.DataFrame|pd.Series): predicted values

    Returns:
        float: root mean squared error
    """
    actual, predicted = _process_input(actual, predicted)
    return np.sqrt(np.mean((predicted - actual) ** 2)) * scope


def mape(
    actual: Union[np.ndarray, pd.DataFrame, pd.Series],
    predicted: Union[np.ndarray, pd.DataFrame, pd.Series],
    eps: float = 1e-6,
) -> float:
    """Calculate Mean Absolute Percentage Error (MAPE)

    Args:
        actual (np.array|pd.DataFrame|pd.Series): actual values
        predicted (np.array|pd.DataFrame|pd.Series): predicted values
        eps (float, optional): small number to prevent division by zero. Defaults to 1e-6.

    Returns:
        float: mean absolute percentage error
    """
    actual, predicted = _process_input(actual, predicted)
    return np.mean(np.abs((predicted - actual) / (actual + eps))) * 100


def smape(
    actual: Union[np.ndarray, pd.DataFrame, pd.Series],
    predicted: Union[np.ndarray, pd.DataFrame, pd.Series],
    eps: float = 1e-6,
) -> float:
    """Calculate Symmetric Mean Absolute Percentage Error (SMAPE)

    Args:
        actual (np.array|pd.DataFrame|pd.Series): actual values
        predicted (np.array|pd.DataFrame|pd.Series): predicted values
        eps (float, optional): small number to prevent division by zero. Defaults to 1e-6.

    Returns:
        float: symmetric mean absolute percentage error
    """
    actual, predicted = _process_input(actual, predicted)
    return (
        np.mean(
            2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted) + eps)
        )
        * 100
    )


def compute_metrics(
    actual: Union[np.ndarray, pd.DataFrame, pd.Series],
    predicted: Union[np.ndarray, pd.DataFrame, pd.Series],
) -> Dict[str, float]:
    """Compute metrics: MAE (Mean Absolute Error), RMSE (Root Mean Squared Error),
    MAPE (Mean Absolute Percentage Error), SMAPE (Symmetric Mean Absolute Percentage Error).

    Args:
        actual (np.array|pd.DataFrame|pd.Series): actual values
        predicted (np.array|pd.DataFrame|pd.Series): predicted values

    Returns:
        Dict[str, float]: MAE, RMSE, MAPE, SMAPE metrics
    """
    return {
        "MAE": mae(actual, predicted),
        "RMSE": rmse(actual, predicted),
        "MAPE": mape(actual, predicted),
        "SMAPE": smape(actual, predicted),
    }


def compute_point_by_point_metrics(
    actual: Union[np.ndarray, pd.DataFrame, pd.Series],
    predicted: Union[np.ndarray, pd.DataFrame, pd.Series],
    scope: int
) -> Dict[str, np.ndarray]:
    """Compute point-by-point metrics: AE (Absolute Error), APE (Absolute
    Percentage Error), SAPE (Symmetric Absolute Percentage Error).

    Args:
        actual (np.array|pd.DataFrame|pd.Series): actual values
        predicted (np.array|pd.DataFrame|pd.Series): predicted values

    Returns:
        Dict[str, np.ndarray]: AE, APE, SAPE metrics
    """
    actual, predicted = _process_input(actual, predicted)
    return {
        "AE": np.abs(predicted - actual)*scope,
        "APE": np.abs((predicted - actual) / actual),
        "SAPE": 2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted)),
    }


def print_metrics(
    actual: Union[np.ndarray, pd.DataFrame, pd.Series],
    predicted: Union[np.ndarray, pd.DataFrame, pd.Series],
    scope: int,
) -> None:
    """Print metrics

    Args:
        actual (np.array|pd.DataFrame|pd.Series): actual values
        predicted (np.array|pd.DataFrame|pd.Series): predicted values
    """
    print(f"MAE: {mae(actual, predicted,scope):.4f} Cm")
    print(f"RMSE: {rmse(actual, predicted,scope):.4f} Cm")
    # print(f"MAPE: {mape(actual, predicted):.4f} %")
    print(f"SMAPE: {smape(actual, predicted,scope):.4f} %")

"""
Implementation of Gaussian Radial Basis Function (RBF) network for Online Learning (OL).

Author: Francesco Saccani <francesco.saccani@unipr.it>
"""

import math
import numpy as np
from sklearn.cluster import KMeans
from typing import Any, List, Optional


class GaussianRBFNode:
    """
    A hidden node in the network that uses a Guassian Radial Basis Function
    as the non-linear activation function
    """

    center: np.ndarray
    """Center of the RBF"""
    radius: float
    """Variance of the RBF"""

    low_activation_count: int = 0
    """Number of times the RBF prouced a normalized activation value below a
    given threshold consecutively"""

    def __init__(self, center: np.ndarray, radius: float) -> None:
        """Initialize the hidden node

        Args:
            center (np.array): center of the RBF
            radius (float): radius of the RBF
        """
        self.center = center
        self.radius = radius

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """Call the hidden node

        Args:
            *args (Any): arguments
            **kwds (Any): keyword arguments
        """
        return self.forward(*args, **kwds)

    def forward(self, x: np.ndarray) -> float:
        return np.exp(-np.linalg.norm(x - self.center) ** 2 / (2 * self.radius**2))


class Network:
    """
    Implementation of a custom algorithm for Online Learning (OL) using a
    Radial Basis Function (RBF) network.
    """

    # Network dimensions
    _input_size: int
    """Number of input nodes"""
    _output_size: int
    """Number of output nodes"""

    # Network weights
    _hidden_nodes: List[GaussianRBFNode]
    """List of neurons in the hidden layer"""
    _bias_weights: np.ndarray
    """Weights for the bias node; shape: (output_size,)"""
    _weights: np.ndarray
    """Weight matrix between hidden and output layers;
    shape: (output_size, len(hidden_nodes))"""

    _e: float = None
    """Prediction error; used only for debugging"""
    _d: float = None
    """Distance to the nearest unit; used only for debugging"""

    _e_th: float
    """Error threshold"""
    _lr: float
    """Learning rate"""
    _overlap: float
    """Overlap factor"""

    _avg_w: Optional[int]
    """Window size for the apprroximated mean and standard deviation"""
    _d_mean: float = 0.0
    """Approximated mean of the distance to the nearest unit"""
    _d_std: float = 0.0
    """Approximated standard deviation of the distance to the nearest unit"""
    _d_th: float
    """Distance to the nearest unit threshold"""
    _k: float
    """Parameter for the distance threshold update"""

    # Pruining parameters
    _act_th: float
    """Activation threshold for the RBF neurons"""
    _pr_w: int
    """Number of consecutive iterations that lead to the elimination of a node
    if its activation is below the threshold"""

    def __init__(
        self,
        input_size: int,
        output_size: int = 1,
        error_threshold: float = 0.01,
        learning_rate: float = 0.2,
        overlap_factor: float = 0.8,
        distance_threshold: float = 1.0,
        average_window: Optional[int] = 100,
        k: float = 2.0,
        activation_threshold: float = 0.01,
        pruning_window: int = 10,
    ) -> None:
        """Initialize an empty Radial Basis Function network

        Args:
            input_size (int): number of input nodes
            output_size (int, optional): number of output nodes; defaults to 1
            error_threshold (float, optional): prediction error tolerance; defaults to 0.01
            learning_rate (float, optional): learning rate; defaults to 0.2
            overlap_factor (float, optional): overlap factor; defaults to 0.8
            distance_threshold (float, optional): initial distance threshold; defaults to 1.0
            average_window (int|None, optional): window size for the approximated mean and standard deviation, if None the distance threshold is fixed; defaults to 100
            k (float, optional): parameter for the distance threshold update; defaults to 2.0
            activation_threshold (float, optional): activation threshold for the RBF neurons; defaults to 0.01
            pruning_window (int, optional): number of consecutive iterations that lead to the elimination of a node if its activation is below the threshold; defaults to 10
        """
        # initialize an empty network
        self._input_size = input_size
        self._output_size = output_size
        self._hidden_nodes = []
        self._bias_weights = np.zeros(output_size)
        self._weights = np.zeros((output_size, 0))
        # set the learning parameters
        self._e_th = error_threshold
        self._lr = learning_rate
        self._overlap = overlap_factor
        self._d_th = distance_threshold
        self._avg_w = average_window
        self._k = k
        self._act_th = activation_threshold
        self._pr_w = pruning_window

    def init_with_kmeans(self, x: np.ndarray, y: np.ndarray, n_nodes: int) -> None:
        """
        Initialize network nodes using k-means clustering for centers, the distance
        from the nearest center for radiuses and least squares for weights.

        Args:
            x (np.ndarray): input data
            y (np.ndarray): output data
            n_nodes (int): number of hidden nodes
        """
        # perform clustering using k-means on input data
        kmeans = KMeans(n_clusters=n_nodes, random_state=42, n_init="auto").fit(x)
        centers = kmeans.cluster_centers_

        # calculate radiuses as the distance from the nearest center
        radiuses = []
        for i in range(len(centers)):
            radius = math.inf
            for j in range(len(centers)):
                if i != j:
                    radius = min(radius, np.linalg.norm(centers[i] - centers[j]))
            radiuses.append(radius)

        # initialize hidden nodes of the network
        self._hidden_nodes = [
            GaussianRBFNode(centers[i], radiuses[i]) for i in range(n_nodes)
        ]

        # calculate activations for each input
        activations = [
            [node(x[i]) for node in self._hidden_nodes] for i in range(len(x))
        ]

        # calculate mean of output data
        y_mean = np.mean(y, axis=0)
        # the weights for the bias node are the mean of the output data
        self._bias_weights = y_mean
        # calculate weights using Least Mean Squares on normalized data
        lstsq = np.linalg.lstsq(activations, y - y_mean, rcond=None)
        self._weights = lstsq[0].reshape(1, -1)

        # initialize parameters for online learning
        self._d_th = min(radiuses)

    def _update_mean_std(self, x: np.ndarray) -> None:
        """Update the approximated mean and standard deviation of the distance
        to the nearest unit

        Args:
            x (np.array): input
        """
        if self._avg_w is not None and len(self._hidden_nodes) > 0:
            self._d = np.min([np.linalg.norm(x - n.center) for n in self._hidden_nodes])
            self._d_mean -= self._d_mean / self._avg_w
            self._d_mean += self._d / self._avg_w
            self._d_std -= self._d_std / self._avg_w
            self._d_std += (self._d - self._d_mean) ** 2 / self._avg_w

    def inference(self, x: np.ndarray) -> np.ndarray:
        """Inference the network

        Args:
            x (np.array): input

        Returns:
            np.array: predicted output
        """
        assert len(x) == self._input_size, "Input size does not match"
        self._update_mean_std(x)

        # compute the activations of hidden nodes
        activations = [node(x) for node in self._hidden_nodes]
        # add bias node weights to the weighted sum of hidden nodes activations
        return self._bias_weights + np.dot(self._weights, activations)

    def learning(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Perform learning on the network

        Args:
            x (np.array): input
            y (np.array): target output

        Returns:
            np.array: predicted output
        """
        assert len(x) == self._input_size, "Input size does not match"
        assert len(y) == self._output_size, "Output size does not match"
        self._update_mean_std(x)

        # Make copy of the inputs
        x = x.copy()
        y = y.copy()

        # Perform inference
        activations = [node(x) for node in self._hidden_nodes]
        y_pred = self._bias_weights + np.dot(self._weights, activations)

        # Compute prediction error
        self._e = y - y_pred

        if len(self._hidden_nodes) == 0:
            # If the network has no hidden units, we add a new RBF node
            self._hidden_nodes.append(GaussianRBFNode(center=x, radius=self._d_th))
            # Then, we add the corresponding weights for the output layer
            self._weights = np.append(
                self._weights,
                self._e.reshape((self._output_size, 1)),
                axis=1,
            )
            return y_pred

        # Compute the distance to the nearest RBF neuron
        distances = np.array([np.linalg.norm(x - n.center) for n in self._hidden_nodes])
        nr_idx = np.argmin(distances)
        self._d = distances[nr_idx]

        if self._avg_w is not None:
            # Update distance threshold using approximated mean and standard deviation
            self._d_th = self._d_mean + self._k * math.sqrt(self._d_std)

        if np.linalg.norm(self._e) > self._e_th and self._d > self._d_th:
            # If the prediction error is above the threshold and the distance
            # to the nearest neuron is above the threshold, we allocate a new neuron
            self._hidden_nodes.append(
                GaussianRBFNode(center=x, radius=self._overlap * self._d)
            )
            # Then, we add the corresponding weights for the output layer
            self._weights = np.append(
                self._weights,
                self._e.reshape((self._output_size, 1)),
                axis=1,
            )
            return y_pred

        # Otherwise, we update network parameters using gradient descent
        self._bias_weights += self._lr * self._e
        for i, n in enumerate(self._hidden_nodes):
            old_weights = self._weights[:, i].copy()
            old_center = n.center.copy()

            self._hidden_nodes[i].center += (
                self._lr
                * (2 / n.radius**2)
                * activations[i]
                * old_weights.dot(self._e)
                * (x - old_center)
            )
            self._weights[:, i] += self._lr * self._e * activations[i]

        # Pruning strategy
        act_norm = activations / np.max(activations)
        for i, n in enumerate(self._hidden_nodes):
            # Check if the normalized activation is below the threshold
            if act_norm[i] < self._act_th:
                n.low_activation_count += 1
            else:
                n.low_activation_count = 0

            # If the activation is below the threshold for a given number of
            # iterations, we remove the neuron from the network
            if n.low_activation_count > self._pr_w:
                self._hidden_nodes.pop(i)
                self._weights = np.delete(self._weights, i, axis=1)

        return y_pred

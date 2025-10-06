from __future__ import annotations
import numpy as np
from sklearn.metrics import pairwise_distances
import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch_geometric.data import Data
import mne
from typing import List, Optional, Union


class EEGGraphBuilder:
    """
    Class to build EEG graphs from electrode positions.

    Parameters
    ----------
    positions : np.ndarray
        Array of shape (n_channels, 3) with electrode coordinates.
    ch_names : list of str
        List of electrode names matching the positions.
        The electrode 'Cz' should be deleted beforehand if not desired.
    """

    def __init__(self, positions: np.ndarray, ch_names: List[str]) -> None:
        self.positions: np.ndarray = positions
        self.ch_names: List[str] = ch_names
        self.n_channels: int = len(ch_names)
        self.distance_matrix: Optional[np.ndarray] = None
        self.adjacency_matrix: Optional[np.ndarray] = None

    def compute_distances(self) -> np.ndarray:
        """Compute pairwise Euclidean distances between electrodes."""
        self.distance_matrix = pairwise_distances(self.positions, metric='euclidean')
        return self.distance_matrix

    def build_adjacency(
        self,
        method: str = 'gaussian',
        k: int = 4,
        sigma: Optional[float] = None,
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Build adjacency matrix using either Gaussian weighting or kNN.

        Parameters
        ----------
        method : str
            'gaussian' (default) or 'knn'.
        k : int
            Number of neighbors for kNN.
        sigma : float, optional
            Gaussian kernel width. Defaults to mean of distances.
        threshold : float, optional
            Minimum edge weight.

        Returns
        -------
        np.ndarray
            Adjacency matrix of shape (n_channels, n_channels).
        """
        if self.distance_matrix is None:
            self.compute_distances()

        if method == 'gaussian':
            if sigma is None:
                sigma = float(np.mean(self.distance_matrix))
            A: np.ndarray = np.exp(-self.distance_matrix**2 / (2 * sigma**2))
            np.fill_diagonal(A, 0)
            if threshold is not None:
                A[A < threshold] = 0

        elif method == 'knn':
            A = np.zeros_like(self.distance_matrix)
            for i in range(self.n_channels):
                neighbors = np.argsort(self.distance_matrix[i])[1:k+1]
                A[i, neighbors] = 1
            A = np.maximum(A, A.T)  # make symmetric

        else:
            raise ValueError("method must be 'gaussian' or 'knn'")

        self.adjacency_matrix = A
        return A

    def visualize(self, show_weights: bool = False) -> None:
        """Visualize the graph layout using NetworkX."""
        if self.adjacency_matrix is None:
            raise RuntimeError("Build adjacency first with .build_adjacency()")

        G = nx.from_numpy_array(self.adjacency_matrix)
        pos_2d = {i: self.positions[i][:2] for i in range(self.n_channels)}
        plt.figure(figsize=(6, 6))
        nx.draw(G, pos=pos_2d, node_size=100, with_labels=False)
        if show_weights:
            labels = {(i, j): f"{w:.2f}" for i, j, w in G.edges.data('weight')}
            nx.draw_networkx_edge_labels(G, pos_2d, edge_labels=labels, font_size=8)
        plt.title("EEG Graph")
        plt.show()

    def to_pyg(self, node_features: Optional[np.ndarray] = None) -> Data:
        """
        Convert the adjacency into a PyTorch Geometric Data object.

        Parameters
        ----------
        node_features : np.ndarray, optional
            Node features of shape (n_channels, n_features).
            Defaults to a column vector of zeros if None.

        Returns
        -------
        Data
            PyTorch Geometric Data object containing x, edge_index, and edge_attr.
        """
        if self.adjacency_matrix is None:
            raise RuntimeError("Build adjacency first with .build_adjacency()")

        edge_index_np = np.array(np.nonzero(self.adjacency_matrix))
        edge_attr_np = self.adjacency_matrix[self.adjacency_matrix > 0]

        edge_index: Tensor = torch.tensor(edge_index_np, dtype=torch.long)
        edge_attr: Tensor = torch.tensor(edge_attr_np, dtype=torch.float32)
        if node_features is None:
            node_features = np.zeros((self.n_channels, 1))
        x: Tensor = torch.tensor(node_features, dtype=torch.float32)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

if __name__ == "__main__":
    montage = mne.channels.make_standard_montage('GSN-HydroCel-128')
    ch_names = montage.ch_names
    positions = np.array([montage.get_positions()['ch_pos'][ch] for ch in ch_names])

    builder = EEGGraphBuilder(positions, ch_names)
    A = builder.build_adjacency(method='knn', k=7)
    builder.visualize()

    graph = builder.to_pyg()
    print(graph)
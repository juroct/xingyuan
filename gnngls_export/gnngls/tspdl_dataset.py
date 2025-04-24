"""
TSPDL Dataset module.

This module provides a PyTorch Dataset for TSPDL instances.
"""

import os
import pickle
import torch
import dgl
import numpy as np
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Optional, Union, Any

from .tspdl import TSPDLInstance
from .tspdl_gnn_gls import create_graph_from_instance, update_graph_with_tour


class TSPDLDataset(Dataset):
    """
    Dataset for TSPDL instances.

    Attributes:
        instances_file: Path to the file containing instance filenames
        data_dir: Directory containing the instances
        instances: List of instance filenames
    """

    def __init__(self, instances_file: str):
        """
        Initialize the dataset.

        Args:
            instances_file: Path to the file containing instance filenames
        """
        self.instances_file = instances_file
        self.data_dir = os.path.dirname(instances_file)

        # Load instance filenames
        with open(instances_file, 'r') as f:
            self.instances = [line.strip() for line in f]

    def __len__(self) -> int:
        """
        Get the number of instances in the dataset.

        Returns:
            length: Number of instances
        """
        return len(self.instances)

    def __getitem__(self, idx: int) -> Tuple[dgl.DGLGraph, torch.Tensor]:
        """
        Get an instance and its labels.

        Args:
            idx: Index of the instance

        Returns:
            graph: DGL graph representing the instance
            labels: Edge labels (1 if in optimal tour, 0 otherwise)
        """
        # Load instance
        instance_filename = self.instances[idx]
        instance_path = os.path.join(self.data_dir, instance_filename)
        print(f"Loading instance from {instance_filename}")
        with open(instance_path, 'rb') as f:
            instance = pickle.load(f)

        # Generate a tour using nearest neighbor
        from .tspdl_algorithms import nearest_neighbor_tspdl
        tour = nearest_neighbor_tspdl(instance)

        # Create graph
        graph = create_graph_from_instance(instance)

        # Update graph with tour
        graph = update_graph_with_tour(graph, instance, tour)

        # Create edge labels
        edge_labels = []

        for i, j in zip(graph.edges()[0], graph.edges()[1]):
            i, j = i.item(), j.item()

            # Skip self-loops
            if i == j:
                edge_labels.append(0)
                continue

            # Check if edge is in tour
            in_tour = 0
            for k in range(len(tour) - 1):
                if (tour[k] == i and tour[k+1] == j) or (tour[k] == j and tour[k+1] == i):
                    in_tour = 1
                    break

            edge_labels.append(in_tour)

        return graph, torch.tensor(edge_labels, dtype=torch.float32)

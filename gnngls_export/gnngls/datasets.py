import json
import os
from pathlib import Path

import dgl
import networkx as nx
import numpy as np
import torch
from torch.utils.data import Dataset


class TSPDataset(Dataset):
    def __init__(self, file_path):
        """
        Dataset for TSP instances.

        Args:
            file_path: Path to the file containing TSP instances
        """
        self.file_path = Path(file_path)
        self.instances = []

        with open(file_path, 'r') as f:
            for line in f:
                self.instances.append(json.loads(line))

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        instance = self.instances[idx]

        # Create a complete graph
        G = nx.complete_graph(len(instance['coordinates']))

        # Add node features (coordinates)
        for i, coords in enumerate(instance['coordinates']):
            G.nodes[i]['coords'] = coords

        # Add edge features (distances)
        for i, j in G.edges:
            x1, y1 = G.nodes[i]['coords']
            x2, y2 = G.nodes[j]['coords']

            # Euclidean distance
            dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            G.edges[i, j]['weight'] = dist

            # Add regret if available
            if 'regret' in instance:
                G.edges[i, j]['regret'] = instance['regret'][i][j]

            # Add in_solution if available
            if 'solution' in instance:
                edges = list(zip(instance['solution'][:-1], instance['solution'][1:]))
                G.edges[i, j]['in_solution'] = 1.0 if (i, j) in edges or (j, i) in edges else 0.0

        # Convert to DGL graph - for undirected graphs, we need to convert to directed first
        G_directed = G.to_directed()
        dgl_G = dgl.from_networkx(G_directed, edge_attrs=['weight', 'regret', 'in_solution'])

        # Add node features
        coords = torch.tensor([G.nodes[i]['coords'] for i in range(len(G.nodes))], dtype=torch.float32)
        dgl_G.ndata['coords'] = coords

        # Add edge features
        edge_feats = []
        edges = dgl_G.edges()
        for i in range(edges[0].shape[0]):
            u, v = edges[0][i].item(), edges[1][i].item()
            x1, y1 = G.nodes[u]['coords']
            x2, y2 = G.nodes[v]['coords']

            # Edge features: distance, coordinates of both nodes
            if (u, v) in G.edges:
                weight = G.edges[u, v]['weight']
            else:  # For directed graph, the reverse edge
                weight = G.edges[v, u]['weight']
            edge_feat = [weight, x1, y1, x2, y2]
            edge_feats.append(edge_feat)

        dgl_G.edata['features'] = torch.tensor(edge_feats, dtype=torch.float32)

        # Add target values if available
        if 'regret' in instance:
            regret = []
            edges = dgl_G.edges()
            for i in range(edges[0].shape[0]):
                u, v = edges[0][i].item(), edges[1][i].item()
                if (u, v) in G.edges:
                    r = G.edges[u, v]['regret']
                else:  # For directed graph, the reverse edge
                    r = G.edges[v, u]['regret']
                regret.append(r)
            dgl_G.edata['regret'] = torch.tensor(regret, dtype=torch.float32)

        if 'solution' in instance:
            in_solution = []
            edges = dgl_G.edges()
            for i in range(edges[0].shape[0]):
                u, v = edges[0][i].item(), edges[1][i].item()
                if (u, v) in G.edges:
                    s = G.edges[u, v]['in_solution']
                else:  # For directed graph, the reverse edge
                    s = G.edges[v, u]['in_solution']
                in_solution.append(s)
            dgl_G.edata['in_solution'] = torch.tensor(in_solution, dtype=torch.float32)

        return dgl_G


def generate_tsp_instance(n_nodes, seed=None):
    """
    Generate a random TSP instance.

    Args:
        n_nodes: Number of nodes
        seed: Random seed

    Returns:
        instance: Dictionary containing the TSP instance
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate random coordinates in [0, 1]
    coordinates = np.random.rand(n_nodes, 2).tolist()

    return {
        'coordinates': coordinates
    }


def save_tsp_instances(instances, file_path):
    """
    Save TSP instances to a file.

    Args:
        instances: List of TSP instances
        file_path: Path to save the instances
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'w') as f:
        for instance in instances:
            f.write(json.dumps(instance) + '\n')

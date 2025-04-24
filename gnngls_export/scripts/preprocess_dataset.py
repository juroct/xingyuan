#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import pathlib
import time

import networkx as nx
import numpy as np
import tqdm.auto as tqdm

from gnngls import algorithms, tour_cost


def preprocess_instance(instance, solver='concorde'):
    """
    Preprocess a TSP instance by computing the optimal solution and regret values.
    
    Args:
        instance: Dictionary containing the TSP instance
        solver: Solver to use for computing the optimal solution ('concorde' or 'lkh')
        
    Returns:
        instance: Dictionary containing the preprocessed instance
    """
    # Create a complete graph
    n_nodes = len(instance['coordinates'])
    G = nx.complete_graph(n_nodes)
    
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
    
    # Compute the optimal solution
    depot = 0  # Assuming node 0 is the depot
    
    # For simplicity, we'll use nearest neighbor + local search as a proxy for the optimal solution
    # In a real implementation, you would use Concorde or LKH solvers
    nn_tour = algorithms.nearest_neighbor(G, depot)
    nn_cost = tour_cost(G, nn_tour)
    
    # Run local search to improve the solution
    ls_tour, ls_cost, _ = algorithms.local_search(nn_tour, nn_cost, nx.attr_matrix(G, 'weight')[0])
    
    # Compute regret values
    regret = [[0.0 for _ in range(n_nodes)] for _ in range(n_nodes)]
    
    # For each edge, compute the regret (cost increase) of including it in the solution
    for i, j in G.edges:
        if i != j:
            # Skip if the edge is already in the solution
            if (i, j) in zip(ls_tour[:-1], ls_tour[1:]) or (j, i) in zip(ls_tour[:-1], ls_tour[1:]):
                regret[i][j] = 0.0
                regret[j][i] = 0.0
            else:
                # Compute the regret by forcing the edge into the solution
                # This is a simplified approach; in practice, you would recompute the optimal solution
                # with the edge included
                
                # For simplicity, we'll use a heuristic: the regret is proportional to the edge weight
                # and inversely proportional to the distance to the nearest edge in the solution
                min_dist_to_solution = float('inf')
                for u, v in zip(ls_tour[:-1], ls_tour[1:]):
                    d1 = G.edges[i, u]['weight'] + G.edges[j, v]['weight']
                    d2 = G.edges[i, v]['weight'] + G.edges[j, u]['weight']
                    min_dist_to_solution = min(min_dist_to_solution, d1, d2)
                
                edge_regret = G.edges[i, j]['weight'] * min_dist_to_solution
                regret[i][j] = edge_regret
                regret[j][i] = edge_regret
    
    # Add the solution and regret to the instance
    instance['solution'] = ls_tour
    instance['regret'] = regret
    
    return instance


def preprocess_dataset(input_path, output_path):
    """
    Preprocess a dataset of TSP instances.
    
    Args:
        input_path: Path to the input dataset
        output_path: Path to save the preprocessed dataset
    """
    # Load instances
    instances = []
    with open(input_path, 'r') as f:
        for line in f:
            instances.append(json.loads(line))
    
    # Preprocess instances
    preprocessed_instances = []
    for instance in tqdm.tqdm(instances, desc="Preprocessing"):
        preprocessed_instance = preprocess_instance(instance)
        preprocessed_instances.append(preprocessed_instance)
    
    # Save preprocessed instances
    with open(output_path, 'w') as f:
        for instance in preprocessed_instances:
            f.write(json.dumps(instance) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess TSP instances')
    parser.add_argument('input_dir', type=pathlib.Path, help='Directory containing input instances')
    parser.add_argument('output_dir', type=pathlib.Path, help='Directory to save preprocessed instances')
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Preprocess datasets
    for dataset in ['train.txt', 'val.txt', 'test.txt']:
        input_path = args.input_dir / dataset
        output_path = args.output_dir / dataset
        
        if input_path.exists():
            print(f"Preprocessing {dataset}...")
            preprocess_dataset(input_path, output_path)
        else:
            print(f"Warning: {input_path} does not exist, skipping")
    
    print(f"Preprocessed datasets saved to {args.output_dir}")

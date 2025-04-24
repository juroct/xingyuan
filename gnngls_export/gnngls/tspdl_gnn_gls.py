"""
GNN-guided local search for TSPDL.

This module provides functions for using a trained GNN model to guide
local search for TSPDL problems.
"""

import time
import numpy as np
import torch
import networkx as nx
import dgl
from typing import List, Tuple, Dict, Optional, Union, Any

from .tspdl import TSPDLInstance, TSPDLSolution
from .tspdl_models import TSPDLEdgeModel
from .tspdl_algorithms import nearest_neighbor_tspdl, local_search_tspdl


def create_graph_from_instance(instance: TSPDLInstance) -> dgl.DGLGraph:
    """
    Create a DGL graph from a TSPDL instance.

    Args:
        instance: TSPDL instance

    Returns:
        g: DGL graph
    """
    # Create a complete graph
    g = dgl.graph(([], []), num_nodes=instance.problem_size)

    # Add self-loops
    g = dgl.add_self_loop(g)

    # Add edges between all nodes
    src, dst = [], []
    for i in range(instance.problem_size):
        for j in range(instance.problem_size):
            if i != j:
                src.append(i)
                dst.append(j)

    g.add_edges(src, dst)

    # Move graph to the same device as instance
    g = g.to(instance.device)

    # Add node features
    g.ndata['coords'] = instance.node_xy
    g.ndata['demand'] = instance.node_demand.unsqueeze(1)
    g.ndata['draft_limit'] = instance.node_draft_limit.unsqueeze(1)

    # Precompute reachability mask based on draft limits
    # For each edge (i,j), check if going from i to j with current load would exceed j's draft limit
    total_demand = instance.node_demand.sum().item()

    # Add edge features
    edge_feats = []
    reachability_mask = []

    for i, j in zip(g.edges()[0], g.edges()[1]):
        i, j = i.item(), j.item()

        # Skip self-loops
        if i == j:
            edge_feats.append([0, 0, 0])
            reachability_mask.append(1.0)  # Self-loops are always reachable
            continue

        # Calculate distance
        x1, y1 = instance.node_xy[i]
        x2, y2 = instance.node_xy[j]
        dist = torch.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2).item()

        # Calculate reachability based on draft limits
        # We consider different scenarios:

        # 1. From depot (i=0) to any node: check if initial load exceeds draft limit
        if i == 0:
            # When starting from depot, we have full load
            reachable = total_demand <= instance.node_draft_limit[j].item()
        # 2. From any node to depot: always reachable (depot has no draft limit)
        elif j == 0:
            reachable = True
        # 3. Between non-depot nodes: conservative estimate
        else:
            # Estimate remaining load after visiting i
            # This is a conservative estimate assuming we visit i directly from depot
            remaining_load = total_demand - instance.node_demand[i].item()
            reachable = remaining_load <= instance.node_draft_limit[j].item()

        # Edge features: [distance, in_tour, draft_limit_respected]
        # Initially, no edge is in tour and all respect draft limits
        edge_feats.append([dist, 0, 1])

        # Add reachability to mask (1.0 if reachable, 0.0 if not)
        reachability_mask.append(1.0 if reachable else 0.0)

    g.edata['features'] = torch.tensor(edge_feats, dtype=torch.float32, device=instance.device)

    # Add reachability mask as a separate edge feature
    g.edata['reachability_mask'] = torch.tensor(reachability_mask, dtype=torch.float32, device=instance.device).unsqueeze(1)

    return g


def update_graph_with_tour(
    g: dgl.DGLGraph,
    instance: TSPDLInstance,
    tour: List[int]
) -> dgl.DGLGraph:
    """
    Update a DGL graph with a tour.

    Args:
        g: DGL graph
        instance: TSPDL instance
        tour: Tour

    Returns:
        g: Updated DGL graph
    """
    # Get edge features
    edge_feats = g.edata['features']

    # Reset in_tour feature
    edge_feats[:, 1] = 0

    # Create edge to index mapping
    edge_to_idx = {}
    for idx, (i, j) in enumerate(zip(g.edges()[0], g.edges()[1])):
        i, j = i.item(), j.item()
        edge_to_idx[(i, j)] = idx

    # Update in_tour feature
    for i in range(len(tour) - 1):
        u, v = tour[i], tour[i+1]
        if (u, v) in edge_to_idx:
            edge_feats[edge_to_idx[(u, v)], 1] = 1

    # Update draft_limit_respected feature
    solution = TSPDLSolution(instance, tour)

    for i in range(1, len(tour)):
        prev_node = tour[i-1]
        node = tour[i]

        # Check if draft limit is exceeded
        if solution.out_of_draft_limit[i-1] > 0:
            if (prev_node, node) in edge_to_idx:
                edge_feats[edge_to_idx[(prev_node, node)], 2] = 0

    # Update edge features
    g.edata['features'] = edge_feats

    return g


def predict_edge_scores(
    model: TSPDLEdgeModel,
    g: dgl.DGLGraph,
    instance: TSPDLInstance
) -> torch.Tensor:
    """
    Predict edge scores using a trained GNN model.

    Args:
        model: Trained GNN model
        g: DGL graph
        instance: TSPDL instance

    Returns:
        edge_scores: Edge scores
    """
    # Move graph to device
    g = g.to(instance.device)

    # Get edge features
    edge_feats = g.edata['features']

    # Get node features
    node_feats = torch.cat([
        g.ndata['coords'],
        g.ndata['demand'],
        g.ndata['draft_limit']
    ], dim=1)

    # Predict edge scores
    model.eval()
    with torch.no_grad():
        edge_scores = torch.sigmoid(model(g, edge_feats, node_feats))

    # Apply reachability mask to edge scores
    # Unreachable edges (mask=0) will have score=0
    reachability_mask = g.edata['reachability_mask'].squeeze()
    edge_scores = edge_scores.squeeze() * reachability_mask

    return edge_scores


def gnn_guided_local_search_tspdl(
    instance: TSPDLInstance,
    model: TSPDLEdgeModel,
    initial_tour: Optional[List[int]] = None,
    time_limit: float = 10.0,
    lambda_factor: float = 0.1,
    perturbation_moves: int = 5,
    first_improvement: bool = False,
    max_iterations: int = 100,
    track_progress: bool = False
) -> Tuple[List[int], float, Dict]:
    """
    GNN-guided local search for TSPDL.

    Args:
        instance: TSPDL instance
        model: Trained GNN model
        initial_tour: Initial tour (if None, nearest neighbor is used)
        time_limit: Time limit in seconds
        lambda_factor: Lambda factor for penalizing edges
        perturbation_moves: Number of random moves for perturbation
        first_improvement: Whether to use first improvement strategy
        max_iterations: Maximum number of iterations
        track_progress: Whether to track progress

    Returns:
        tour: Improved tour
        cost: Cost of the improved tour
        info: Additional information
    """
    # Create initial tour if not provided
    if initial_tour is None:
        initial_tour = nearest_neighbor_tspdl(instance)

    # Convert instance to NetworkX graph for compatibility with operators
    G = instance.to_networkx()

    # Get distance matrix
    edge_weight, _ = nx.attr_matrix(G, 'weight')

    # Initialize penalties
    penalties = np.zeros_like(edge_weight)

    # Initialize tour and cost
    tour = initial_tour.copy()
    solution = TSPDLSolution(instance, tour)
    cost = solution.cost

    # Initialize best tour and cost
    best_tour = tour.copy()
    best_cost = cost
    best_solution = solution

    # Initialize tracking variables
    if track_progress:
        progress = {
            'tours': [tour.copy()],
            'costs': [cost],
            'penalties': [penalties.copy()],
            'out_of_draft_limit': [solution.total_out_of_draft_limit()],
            'infeasible_nodes': [solution.count_out_of_draft_limit_nodes()]
        }

    # Create DGL graph
    g = create_graph_from_instance(instance)

    # Initialize start time and iteration counter
    start_time = time.time()
    iteration = 0

    # GNN-GLS loop
    while time.time() - start_time < time_limit and iteration < max_iterations:
        iteration += 1
        print(f"GNN-GLS iteration {iteration}")

        # Apply local search
        tour, cost, _ = local_search_tspdl(
            instance, tour, max_iterations=1000, first_improvement=first_improvement
        )

        # Update solution
        solution = TSPDLSolution(instance, tour)
        cost = solution.cost

        # Update best solution if better
        if cost < best_cost and solution.feasible:
            best_tour = tour.copy()
            best_cost = cost
            best_solution = solution

        # Track progress
        if track_progress:
            progress['tours'].append(tour.copy())
            progress['costs'].append(cost)
            progress['penalties'].append(penalties.copy())
            progress['out_of_draft_limit'].append(solution.total_out_of_draft_limit())
            progress['infeasible_nodes'].append(solution.count_out_of_draft_limit_nodes())

        # Update graph with current tour
        g = update_graph_with_tour(g, instance, tour)

        # Predict edge scores
        edge_scores = predict_edge_scores(model, g, instance)

        # Create edge to index mapping
        edge_to_idx = {}
        for idx, (i, j) in enumerate(zip(g.edges()[0], g.edges()[1])):
            i, j = i.item(), j.item()
            edge_to_idx[(i, j)] = idx

        # Update penalties based on GNN predictions
        for i in range(len(tour) - 1):
            u, v = tour[i], tour[i+1]

            # Get edge score
            if (u, v) in edge_to_idx:
                edge_idx = edge_to_idx[(u, v)]
                edge_score = edge_scores[edge_idx].item()

                # Calculate utility
                utility = G.edges[u, v]['weight'] / (1 + penalties[u, v])

                # Add penalty for edges in the tour with low scores
                penalty = lambda_factor * utility * (1 - edge_score)
                penalties[u, v] += penalty
                penalties[v, u] += penalty

        # Add penalties for exceeding draft limits
        for i in range(1, len(tour)):
            node = tour[i]
            prev_node = tour[i-1]

            # Check if draft limit is exceeded
            if solution.out_of_draft_limit[i-1] > 0:
                # Add penalty for the edge leading to this node
                utility = G.edges[prev_node, node]['weight'] / (1 + penalties[prev_node, node])
                penalties[prev_node, node] += lambda_factor * utility * 2  # Double penalty for draft limit violation
                penalties[node, prev_node] += lambda_factor * utility * 2

        # Perturb the solution
        for _ in range(perturbation_moves):
            # Randomly choose between 2-opt and relocate
            if np.random.random() < 0.5:
                # Random 2-opt move
                i = np.random.randint(1, len(tour) - 1)
                j = np.random.randint(1, len(tour) - 1)
                if i != j:
                    new_tour = tour.copy()
                    if i > j:
                        i, j = j, i
                    new_tour[i:j+1] = new_tour[i:j+1][::-1]

                    # Check if the new tour is feasible
                    new_solution = TSPDLSolution(instance, new_tour)
                    if new_solution.feasible:
                        tour = new_tour
            else:
                # Random relocate move
                i = np.random.randint(1, len(tour) - 1)
                j = np.random.randint(1, len(tour))
                if i != j and j != i - 1:
                    new_tour = tour.copy()
                    node = new_tour.pop(i)
                    new_tour.insert(j if j < i else j - 1, node)

                    # Check if the new tour is feasible
                    new_solution = TSPDLSolution(instance, new_tour)
                    if new_solution.feasible:
                        tour = new_tour

    # Return best tour and cost
    elapsed_time = time.time() - start_time
    info = {
        'time': elapsed_time,
        'iterations': iteration
    }
    if track_progress:
        info['progress'] = progress

    print(f"GNN-GLS completed: {iteration} iterations in {elapsed_time:.2f} seconds")
    print(f"Best cost: {best_cost:.4f}, Feasible: {best_solution.feasible}")

    return best_tour, best_cost, info

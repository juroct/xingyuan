"""
Algorithms for solving the Traveling Salesperson Problem with Draft Limits (TSPDL).

This module extends the GNNGLS algorithms to handle TSPDL constraints.
"""

import time
import numpy as np
import torch
import networkx as nx
from typing import List, Tuple, Dict, Optional, Union, Any

from .tspdl import TSPDLInstance, TSPDLSolution
from .operators import two_opt_a2a, relocate_a2a, two_opt_o2a, relocate_o2a


def nearest_neighbor_tspdl(
    instance: TSPDLInstance,
    start_node: int = 0
) -> List[int]:
    """
    Nearest neighbor algorithm for TSPDL.

    Args:
        instance: TSPDL instance
        start_node: Starting node (depot)

    Returns:
        tour: List of node indices representing the tour
    """
    problem_size = instance.problem_size
    node_xy = instance.node_xy
    node_demand = instance.node_demand
    node_draft_limit = instance.node_draft_limit

    # Initialize tour with start node
    tour = [start_node]

    # Initialize current load
    current_load = 0.0

    # Create a set of unvisited nodes
    unvisited = set(range(problem_size))
    unvisited.remove(start_node)

    # Current node is the start node
    current_node = start_node

    # While there are unvisited nodes
    while unvisited:
        # Find the nearest unvisited node that respects draft limit
        min_dist = float('inf')
        nearest_node = None

        for node in unvisited:
            # Check if draft limit is respected
            if current_load > node_draft_limit[node]:
                continue

            # Calculate distance
            x1, y1 = node_xy[current_node]
            x2, y2 = node_xy[node]
            dist = torch.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

            # Update nearest node if closer
            if dist < min_dist:
                min_dist = dist
                nearest_node = node

        # If no feasible node is found, break
        if nearest_node is None:
            break

        # Add nearest node to tour
        tour.append(nearest_node)

        # Update current node and load
        current_node = nearest_node
        current_load += node_demand[nearest_node]

        # Remove from unvisited
        unvisited.remove(nearest_node)

    # Return to depot
    if tour[-1] != start_node:
        tour.append(start_node)

    return tour


def insertion_tspdl(
    instance: TSPDLInstance,
    start_node: int = 0
) -> List[int]:
    """
    Insertion algorithm for TSPDL.

    Args:
        instance: TSPDL instance
        start_node: Starting node (depot)

    Returns:
        tour: List of node indices representing the tour
    """
    problem_size = instance.problem_size
    node_xy = instance.node_xy
    node_demand = instance.node_demand
    node_draft_limit = instance.node_draft_limit

    # Initialize tour with start node (depot loop)
    tour = [start_node, start_node]

    # Initialize loads at each position
    loads = [0.0, 0.0]

    # Create a set of unvisited nodes
    unvisited = set(range(problem_size))
    unvisited.remove(start_node)

    # While there are unvisited nodes
    while unvisited:
        # Find the best insertion
        min_cost_increase = float('inf')
        best_insertion = None
        best_node = None

        for node in unvisited:
            # Try all possible insertion positions
            for i in range(1, len(tour)):
                # Check if draft limit is respected
                if loads[i-1] > node_draft_limit[node]:
                    continue

                # Calculate cost increase
                prev_node = tour[i-1]
                next_node = tour[i]

                # Distance before insertion
                x1, y1 = node_xy[prev_node]
                x2, y2 = node_xy[next_node]
                dist_before = torch.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

                # Distance after insertion
                x3, y3 = node_xy[node]
                dist_after = torch.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2) + \
                             torch.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)

                # Cost increase
                cost_increase = dist_after - dist_before

                # Update best insertion if better
                if cost_increase < min_cost_increase:
                    min_cost_increase = cost_increase
                    best_insertion = i
                    best_node = node

        # If no feasible insertion is found, break
        if best_insertion is None:
            break

        # Insert best node
        tour.insert(best_insertion, best_node)

        # Update loads
        new_loads = loads.copy()
        new_loads.insert(best_insertion, loads[best_insertion-1] + node_demand[best_node])

        # Update loads after insertion
        for i in range(best_insertion + 1, len(new_loads)):
            new_loads[i] += node_demand[best_node]

        loads = new_loads

        # Remove from unvisited
        unvisited.remove(best_node)

    return tour


def local_search_tspdl(
    instance: TSPDLInstance,
    initial_tour: List[int],
    max_iterations: int = 1000,
    first_improvement: bool = False
) -> Tuple[List[int], float, int]:
    """
    Local search algorithm for TSPDL.

    Args:
        instance: TSPDL instance
        initial_tour: Initial tour
        max_iterations: Maximum number of iterations
        first_improvement: Whether to use first improvement strategy

    Returns:
        tour: Improved tour
        cost: Cost of the improved tour
        iterations: Number of iterations performed
    """
    # Convert instance to NetworkX graph for compatibility with operators
    G = instance.to_networkx()

    # Get distance matrix
    edge_weight, _ = nx.attr_matrix(G, 'weight')

    # Initialize tour and cost
    tour = initial_tour.copy()
    cost = TSPDLSolution(instance, tour).cost

    # Initialize iteration counter
    iterations = 0

    # Local search loop
    while iterations < max_iterations:
        iterations += 1

        # Try 2-opt moves
        delta, new_tour = two_opt_a2a(tour, edge_weight, first_improvement)

        # Check if the new tour is feasible
        if delta < 0:
            solution = TSPDLSolution(instance, new_tour)
            if solution.feasible:
                tour = new_tour
                cost += delta
                continue

        # Try relocate moves
        delta, new_tour = relocate_a2a(tour, edge_weight, first_improvement)

        # Check if the new tour is feasible
        if delta < 0:
            solution = TSPDLSolution(instance, new_tour)
            if solution.feasible:
                tour = new_tour
                cost += delta
                continue

        # If no improvement, break
        break

    return tour, cost, iterations


def guided_local_search_tspdl(
    instance: TSPDLInstance,
    initial_tour: List[int],
    time_limit: float,
    lambda_factor: float = 0.1,
    perturbation_moves: int = 5,
    first_improvement: bool = False,
    guides: List[str] = ['weight'],
    track_progress: bool = False,
    max_iterations: int = 100  # Add maximum iterations as a safety measure
) -> Tuple[List[int], float, Dict]:
    """
    Guided local search algorithm for TSPDL.

    Args:
        instance: TSPDL instance
        initial_tour: Initial tour
        time_limit: Time limit in seconds
        lambda_factor: Lambda factor for penalizing edges
        perturbation_moves: Number of random moves for perturbation
        first_improvement: Whether to use first improvement strategy
        guides: List of edge attributes to use as guides
        track_progress: Whether to track progress

    Returns:
        tour: Improved tour
        cost: Cost of the improved tour
        info: Additional information
    """
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

    # Initialize start time and iteration counter
    start_time = time.time()
    iteration = 0

    # GLS loop
    while time.time() - start_time < time_limit and iteration < max_iterations:
        iteration += 1
        print(f"GLS iteration {iteration}")
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

        # Update penalties
        for i in range(len(tour) - 1):
            u, v = tour[i], tour[i+1]

            # Calculate utility
            utility = G.edges[u, v]['weight'] / (1 + penalties[u, v])

            # Add penalty for edges in the tour
            penalties[u, v] += lambda_factor * utility
            penalties[v, u] += lambda_factor * utility

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

    print(f"GLS completed: {iteration} iterations in {elapsed_time:.2f} seconds")
    print(f"Best cost: {best_cost:.4f}, Feasible: {best_solution.feasible}")

    return best_tour, best_cost, info


def reinforcement_learning_tspdl(
    instance: TSPDLInstance,
    model,
    time_limit: float,
    lambda_factor: float = 0.1,
    perturbation_moves: int = 5,
    first_improvement: bool = False,
    track_progress: bool = False
) -> Tuple[List[int], float, Dict]:
    """
    Reinforcement learning guided local search for TSPDL.

    Args:
        instance: TSPDL instance
        model: Trained GNN model
        time_limit: Time limit in seconds
        lambda_factor: Lambda factor for penalizing edges
        perturbation_moves: Number of random moves for perturbation
        first_improvement: Whether to use first improvement strategy
        track_progress: Whether to track progress

    Returns:
        tour: Improved tour
        cost: Cost of the improved tour
        info: Additional information
    """
    # Convert instance to NetworkX graph
    G = instance.to_networkx()

    # Get distance matrix
    edge_weight, _ = nx.attr_matrix(G, 'weight')

    # Generate initial tour using nearest neighbor
    initial_tour = nearest_neighbor_tspdl(instance)

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
            'out_of_draft_limit': [solution.total_out_of_draft_limit()],
            'infeasible_nodes': [solution.count_out_of_draft_limit_nodes()]
        }

    # Initialize start time
    start_time = time.time()

    # RL-GLS loop
    while time.time() - start_time < time_limit:
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
            progress['out_of_draft_limit'].append(solution.total_out_of_draft_limit())
            progress['infeasible_nodes'].append(solution.count_out_of_draft_limit_nodes())

        # Use model to predict edge scores
        # TODO: Implement model prediction

        # Perturb the solution based on model predictions
        # TODO: Implement perturbation based on model predictions

    # Return best tour and cost
    info = {'time': time.time() - start_time}
    if track_progress:
        info['progress'] = progress

    return best_tour, best_cost, info

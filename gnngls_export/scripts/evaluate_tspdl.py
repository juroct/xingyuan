#!/usr/bin/env python
# coding: utf-8

"""
Evaluate TSPDL algorithms.

This script evaluates different algorithms for solving TSPDL problems.
"""

import argparse
import json
import os
import pathlib
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm.auto as tqdm

from gnngls import (
    TSPDLInstance,
    TSPDLSolution,
    nearest_neighbor_tspdl,
    insertion_tspdl,
    local_search_tspdl,
    guided_local_search_tspdl,
    TSPDLEdgeModel,
    gnn_guided_local_search_tspdl
)


def evaluate_algorithms(
    instances_dir,
    output_dir,
    model_path=None,
    time_limit=10.0,
    max_iterations=50,
    use_gpu=False
):
    """
    Evaluate different algorithms for solving TSPDL problems.

    Args:
        instances_dir: Directory containing instances
        output_dir: Output directory
        model_path: Path to trained model (optional)
        time_limit: Time limit for search algorithms
        max_iterations: Maximum iterations for search algorithms
        use_gpu: Whether to use GPU
    """
    # Create output directory
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model if provided
    model = None
    if model_path is not None:
        model_path = pathlib.Path(model_path)

        # Create model
        model = TSPDLEdgeModel(
            in_dim=3,  # [distance, in_tour, draft_limit_respected]
            embed_dim=64,
            out_dim=1,
            n_layers=2,
            n_heads=4
        ).to(device)

        # Load model weights
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()

        print(f"Model loaded from {model_path}")

    # Load instances
    instances_dir = pathlib.Path(instances_dir)
    instance_files = list(instances_dir.glob("*.pkl"))

    if len(instance_files) == 0:
        print(f"No instances found in {instances_dir}")
        return

    print(f"Found {len(instance_files)} instances")

    # Initialize results
    results = {
        'instance': [],
        'problem_size': [],
        'nearest_neighbor': {
            'cost': [],
            'feasible': [],
            'time': []
        },
        'insertion': {
            'cost': [],
            'feasible': [],
            'time': []
        },
        'local_search': {
            'cost': [],
            'feasible': [],
            'time': []
        },
        'guided_local_search': {
            'cost': [],
            'feasible': [],
            'time': []
        }
    }

    if model is not None:
        results['gnn_guided_local_search'] = {
            'cost': [],
            'feasible': [],
            'time': []
        }

    # Evaluate algorithms
    for instance_file in tqdm.tqdm(instance_files):
        # Load instance
        with open(instance_file, 'rb') as f:
            instance = pickle.load(f)

        # Move instance to device if needed
        if hasattr(instance, 'to'):
            instance.to(device)

        # Store instance metadata
        results['instance'].append(instance_file.name)
        results['problem_size'].append(instance.problem_size)

        # Nearest neighbor
        start_time = time.time()
        nn_tour = nearest_neighbor_tspdl(instance)
        nn_time = time.time() - start_time
        nn_solution = TSPDLSolution(instance, nn_tour)

        results['nearest_neighbor']['cost'].append(nn_solution.cost)
        results['nearest_neighbor']['feasible'].append(nn_solution.feasible)
        results['nearest_neighbor']['time'].append(nn_time)

        # Insertion
        start_time = time.time()
        ins_tour = insertion_tspdl(instance)
        ins_time = time.time() - start_time
        ins_solution = TSPDLSolution(instance, ins_tour)

        results['insertion']['cost'].append(ins_solution.cost)
        results['insertion']['feasible'].append(ins_solution.feasible)
        results['insertion']['time'].append(ins_time)

        # Local search
        start_time = time.time()
        ls_tour, ls_cost, _ = local_search_tspdl(instance, nn_tour)
        ls_time = time.time() - start_time
        ls_solution = TSPDLSolution(instance, ls_tour)

        results['local_search']['cost'].append(ls_solution.cost)
        results['local_search']['feasible'].append(ls_solution.feasible)
        results['local_search']['time'].append(ls_time)

        # Guided local search
        t_limit = time.time() + time_limit

        start_time = time.time()
        gls_tour, gls_cost, _ = guided_local_search_tspdl(
            instance, nn_tour, t_limit, max_iterations=max_iterations
        )
        gls_time = time.time() - start_time
        gls_solution = TSPDLSolution(instance, gls_tour)

        results['guided_local_search']['cost'].append(gls_solution.cost)
        results['guided_local_search']['feasible'].append(gls_solution.feasible)
        results['guided_local_search']['time'].append(gls_time)

        # GNN-guided local search
        if model is not None:
            t_limit = time.time() + time_limit

            start_time = time.time()
            gnn_gls_tour, gnn_gls_cost, _ = gnn_guided_local_search_tspdl(
                instance, model, nn_tour, t_limit, max_iterations=max_iterations
            )
            gnn_gls_time = time.time() - start_time
            gnn_gls_solution = TSPDLSolution(instance, gnn_gls_tour)

            results['gnn_guided_local_search']['cost'].append(gnn_gls_solution.cost)
            results['gnn_guided_local_search']['feasible'].append(gnn_gls_solution.feasible)
            results['gnn_guided_local_search']['time'].append(gnn_gls_time)

    # Save results
    results_path = output_dir / "evaluation_results.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {results_path}")

    # Analyze results
    analyze_results(results, output_dir, model is not None)


def analyze_results(results, output_dir, gnn_available):
    """
    Analyze evaluation results.

    Args:
        results: Evaluation results
        output_dir: Output directory
        gnn_available: Whether GNN model is available
    """
    print("\nAnalyzing results...")

    # Convert to numpy arrays for easier analysis
    problem_sizes = np.array(results['problem_size'])

    methods = ['nearest_neighbor', 'insertion', 'local_search', 'guided_local_search']
    if gnn_available:
        methods.append('gnn_guided_local_search')

    # Initialize summary
    summary = {
        'problem_size': [],
    }

    for method in methods:
        summary[method] = {
            'avg_cost': [],
            'feasible_rate': [],
            'avg_time': []
        }

    # Compute summary statistics for each problem size
    for problem_size in np.unique(problem_sizes):
        # Filter results for this problem size
        mask = problem_sizes == problem_size

        # Add problem size to summary
        summary['problem_size'].append(problem_size)

        # Compute statistics for each method
        for method in methods:
            costs = np.array(results[method]['cost'])[mask]
            feasible = np.array(results[method]['feasible'])[mask]
            times = np.array(results[method]['time'])[mask]

            # Only consider feasible solutions for cost
            if np.any(feasible):
                avg_cost = np.mean(costs[feasible])
            else:
                avg_cost = float('inf')

            feasible_rate = np.mean(feasible)
            avg_time = np.mean(times)

            # Add to summary
            summary[method]['avg_cost'].append(avg_cost)
            summary[method]['feasible_rate'].append(feasible_rate)
            summary[method]['avg_time'].append(avg_time)

    # Save summary
    summary_path = output_dir / "evaluation_summary.json"

    # Convert numpy types to Python native types
    json_summary = {}
    for key, value in summary.items():
        if isinstance(value, dict):
            json_summary[key] = {}
            for k, v in value.items():
                if isinstance(v, list):
                    json_summary[key][k] = [float(x) if isinstance(x, (np.float32, np.float64, np.int32, np.int64)) else x for x in v]
                else:
                    json_summary[key][k] = float(v) if isinstance(v, (np.float32, np.float64, np.int32, np.int64)) else v
        elif isinstance(value, list):
            json_summary[key] = [int(x) if isinstance(x, (np.int32, np.int64)) else x for x in value]
        else:
            json_summary[key] = value

    with open(summary_path, 'w') as f:
        json.dump(json_summary, f, indent=4)
    print(f"Summary saved to {summary_path}")

    # Print summary
    print("\nSummary:")
    for i in range(len(summary['problem_size'])):
        problem_size = summary['problem_size'][i]

        print(f"\nProblem Size: {problem_size}")

        for method in methods:
            avg_cost = summary[method]['avg_cost'][i]
            feasible_rate = summary[method]['feasible_rate'][i]
            avg_time = summary[method]['avg_time'][i]

            print(f"  {method}:")
            print(f"    Avg Cost: {avg_cost:.4f}")
            print(f"    Feasible Rate: {feasible_rate:.2f}")
            print(f"    Avg Time: {avg_time:.4f}s")

    # Plot results
    plot_results(summary, methods, output_dir)


def plot_results(summary, methods, output_dir):
    """
    Plot evaluation results.

    Args:
        summary: Summary statistics
        methods: Methods to plot
        output_dir: Output directory
    """
    print("\nPlotting results...")

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    # Convert to numpy arrays
    problem_sizes = np.array(summary['problem_size'])

    # Define colors and markers
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    markers = ['o', 's', '^', 'D', 'v']

    # Plot average cost
    ax = axes[0]

    for i, method in enumerate(methods):
        costs = np.array(summary[method]['avg_cost'])

        # Skip if all costs are inf
        if np.all(np.isinf(costs)):
            continue

        ax.plot(problem_sizes, costs, color=colors[i], marker=markers[i], linestyle='-', label=method)

    ax.set_title("Average Cost (Feasible Solutions Only)")
    ax.set_xlabel("Problem Size")
    ax.set_ylabel("Average Cost")
    ax.grid(True)
    ax.legend()

    # Plot feasible rate
    ax = axes[1]

    for i, method in enumerate(methods):
        rates = np.array(summary[method]['feasible_rate'])
        ax.plot(problem_sizes, rates, color=colors[i], marker=markers[i], linestyle='-', label=method)

    ax.set_title("Feasible Solution Rate")
    ax.set_xlabel("Problem Size")
    ax.set_ylabel("Feasible Rate")
    ax.grid(True)
    ax.legend()

    # Plot average time
    ax = axes[2]

    for i, method in enumerate(methods):
        times = np.array(summary[method]['avg_time'])
        ax.plot(problem_sizes, times, color=colors[i], marker=markers[i], linestyle='-', label=method)

    ax.set_title("Average Computation Time")
    ax.set_xlabel("Problem Size")
    ax.set_ylabel("Average Time (s)")
    ax.grid(True)
    ax.legend()

    # Save figure
    plt.tight_layout()
    fig_path = output_dir / "evaluation_results.png"
    plt.savefig(fig_path, dpi=300)
    print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TSPDL algorithms")
    parser.add_argument("instances_dir", type=str, help="Directory containing instances")
    parser.add_argument("output_dir", type=str, help="Output directory")
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained model")
    parser.add_argument("--time_limit", type=float, default=10.0, help="Time limit for search algorithms")
    parser.add_argument("--max_iterations", type=int, default=50, help="Maximum iterations for search algorithms")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available")

    args = parser.parse_args()

    evaluate_algorithms(
        args.instances_dir,
        args.output_dir,
        args.model_path,
        args.time_limit,
        args.max_iterations,
        args.use_gpu
    )

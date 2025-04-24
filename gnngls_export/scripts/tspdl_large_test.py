#!/usr/bin/env python
# coding: utf-8

"""
Test TSPDL model on large instances.
"""

import argparse
import os
import pathlib
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from gnngls import (
    TSPDLInstance,
    TSPDLSolution,
    TSPDLEdgeModel,
    nearest_neighbor_tspdl,
    local_search_tspdl,
    guided_local_search_tspdl,
    gnn_guided_local_search_tspdl,
    plot_tspdl_solution
)


def run_large_test(
    output_dir,
    problem_size=50,
    hardness='medium',
    model_path=None,
    time_limit=30.0,
    max_iterations=200,
    n_instances=5,
    use_gpu=False
):
    """
    Run TSPDL test on large instances.

    Args:
        output_dir: Output directory
        problem_size: Problem size
        hardness: Hardness level ('easy', 'medium', 'hard')
        model_path: Path to the trained model
        time_limit: Time limit for search algorithms
        max_iterations: Maximum number of iterations
        n_instances: Number of instances to test
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

        # Create model with the same architecture as during training
        model = TSPDLEdgeModel(
            in_dim=3,  # [distance, in_tour, draft_limit_respected]
            embed_dim=128,  # Same as training
            out_dim=1,
            n_layers=3,  # Same as training
            n_heads=8  # Same as training
        ).to(device)

        # Load model weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        print(f"Model loaded from {model_path}")

    # Results storage
    results = {
        'instance_id': [],
        'algorithm': [],
        'cost': [],
        'feasible': [],
        'time': [],
        'iterations': []
    }

    # Test on multiple instances
    for i in range(n_instances):
        print(f"\n=== Testing on instance {i+1}/{n_instances} ===")

        # Create a TSPDL instance
        print("Creating a TSPDL instance...")
        instance = TSPDLInstance.generate_random(
            problem_size=problem_size,
            hardness=hardness,
            normalized=True,
            device=device
        )

        # Generate initial tour using nearest neighbor
        print("\nGenerating initial tour using nearest neighbor...")
        start_time = time.time()
        nn_tour = nearest_neighbor_tspdl(instance)
        nn_time = time.time() - start_time
        nn_solution = TSPDLSolution(instance, nn_tour)
        nn_cost = nn_solution.cost
        print(f"Nearest neighbor tour cost: {nn_cost:.4f}")
        print(f"Feasible: {nn_solution.feasible}")
        print(f"Time: {nn_time:.2f}s")

        # Store results
        results['instance_id'].append(i)
        results['algorithm'].append('NN')
        results['cost'].append(nn_cost)
        results['feasible'].append(nn_solution.feasible)
        results['time'].append(nn_time)
        results['iterations'].append(0)

        # Apply local search
        print("\nApplying local search...")
        start_time = time.time()
        ls_tour, ls_cost, ls_iterations = local_search_tspdl(
            instance, nn_tour, max_iterations=max_iterations
        )
        ls_time = time.time() - start_time
        ls_solution = TSPDLSolution(instance, ls_tour)
        print(f"Local search tour cost: {ls_cost:.4f}")
        print(f"Feasible: {ls_solution.feasible}")
        print(f"Time: {ls_time:.2f}s")
        print(f"Iterations: {ls_iterations}")

        # Store results
        results['instance_id'].append(i)
        results['algorithm'].append('LS')
        results['cost'].append(ls_cost)
        results['feasible'].append(ls_solution.feasible)
        results['time'].append(ls_time)
        results['iterations'].append(ls_iterations)

        # Apply guided local search
        print("\nApplying guided local search...")
        start_time = time.time()
        t_limit = time.time() + time_limit
        gls_tour, gls_cost, gls_iterations = guided_local_search_tspdl(
            instance, nn_tour, t_limit, max_iterations=max_iterations
        )
        gls_time = time.time() - start_time
        gls_solution = TSPDLSolution(instance, gls_tour)
        print(f"Guided local search tour cost: {gls_cost:.4f}")
        print(f"Feasible: {gls_solution.feasible}")
        print(f"Time: {gls_time:.2f}s")
        print(f"Iterations: {gls_iterations}")

        # Store results
        results['instance_id'].append(i)
        results['algorithm'].append('GLS')
        results['cost'].append(gls_cost)
        results['feasible'].append(gls_solution.feasible)
        results['time'].append(gls_time)
        # Extract iterations count from dict if needed
        if isinstance(gls_iterations, dict) and 'iterations' in gls_iterations:
            results['iterations'].append(gls_iterations['iterations'])
        else:
            results['iterations'].append(gls_iterations)

        # Apply GNN-guided local search if model is provided
        if model is not None:
            print("\nApplying GNN-guided local search...")
            start_time = time.time()
            t_limit = time.time() + time_limit
            gnn_gls_tour, gnn_gls_cost, gnn_gls_iterations = gnn_guided_local_search_tspdl(
                instance, model, nn_tour, t_limit, max_iterations=max_iterations
            )
            gnn_gls_time = time.time() - start_time
            gnn_gls_solution = TSPDLSolution(instance, gnn_gls_tour)
            print(f"GNN-guided local search tour cost: {gnn_gls_cost:.4f}")
            print(f"Feasible: {gnn_gls_solution.feasible}")
            print(f"Time: {gnn_gls_time:.2f}s")
            print(f"Iterations: {gnn_gls_iterations}")

            # Store results
            results['instance_id'].append(i)
            results['algorithm'].append('GNN-GLS')
            results['cost'].append(gnn_gls_cost)
            results['feasible'].append(gnn_gls_solution.feasible)
            results['time'].append(gnn_gls_time)
            # Extract iterations count from dict if needed
            if isinstance(gnn_gls_iterations, dict) and 'iterations' in gnn_gls_iterations:
                results['iterations'].append(gnn_gls_iterations['iterations'])
            else:
                results['iterations'].append(gnn_gls_iterations)

        # Plot solutions for the first instance
        if i == 0:
            print("\nPlotting solutions...")

            # Plot nearest neighbor solution
            plt.figure(figsize=(10, 10))
            plot_tspdl_solution(nn_solution)
            plt.title(f"Nearest Neighbor Solution (Cost: {nn_cost:.4f})")
            plt.savefig(output_dir / "nn_solution.png")

            # Plot local search solution
            plt.figure(figsize=(10, 10))
            plot_tspdl_solution(ls_solution)
            plt.title(f"Local Search Solution (Cost: {ls_cost:.4f})")
            plt.savefig(output_dir / "ls_solution.png")

            # Plot guided local search solution
            plt.figure(figsize=(10, 10))
            plot_tspdl_solution(gls_solution)
            plt.title(f"Guided Local Search Solution (Cost: {gls_cost:.4f})")
            plt.savefig(output_dir / "gls_solution.png")

            # Plot GNN-guided local search solution if model is provided
            if model is not None:
                plt.figure(figsize=(10, 10))
                plot_tspdl_solution(gnn_gls_solution)
                plt.title(f"GNN-Guided Local Search Solution (Cost: {gnn_gls_cost:.4f})")
                plt.savefig(output_dir / "gnn_gls_solution.png")

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    results_df.to_csv(output_dir / "results.csv", index=False)

    # Analyze results
    print("\n=== Results Analysis ===")

    # Group by algorithm
    grouped = results_df.groupby('algorithm')

    # Calculate statistics
    stats = grouped.agg({
        'cost': ['mean', 'std', 'min', 'max'],
        'feasible': 'mean',
        'time': ['mean', 'std', 'min', 'max'],
        'iterations': ['mean', 'std', 'min', 'max']
    })

    print("\nCost statistics:")
    print(stats['cost'])

    print("\nFeasibility rate:")
    print(stats['feasible'])

    print("\nTime statistics (seconds):")
    print(stats['time'])

    print("\nIterations statistics:")
    print(stats['iterations'])

    # Calculate improvement over NN
    nn_mean_cost = grouped.get_group('NN')['cost'].mean()

    improvements = {}
    for alg in grouped.groups.keys():
        if alg != 'NN':
            alg_mean_cost = grouped.get_group(alg)['cost'].mean()
            improvement = (nn_mean_cost - alg_mean_cost) / nn_mean_cost * 100
            improvements[alg] = improvement

    print("\nImprovement over Nearest Neighbor (%):")
    for alg, imp in improvements.items():
        print(f"{alg}: {imp:.2f}%")

    # Plot cost comparison
    plt.figure(figsize=(10, 6))

    # Box plot
    plt.subplot(1, 2, 1)
    results_df.boxplot(column='cost', by='algorithm', grid=False, ax=plt.gca())
    plt.title("Solution Cost by Algorithm")
    plt.suptitle("")
    plt.ylabel("Cost")
    plt.xlabel("Algorithm")

    # Bar plot of mean costs
    plt.subplot(1, 2, 2)
    mean_costs = grouped['cost'].mean()
    mean_costs.plot(kind='bar', ax=plt.gca())
    plt.title("Mean Solution Cost by Algorithm")
    plt.ylabel("Mean Cost")
    plt.xlabel("Algorithm")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / "cost_comparison.png")

    # Plot time comparison
    plt.figure(figsize=(10, 6))

    # Box plot
    plt.subplot(1, 2, 1)
    results_df.boxplot(column='time', by='algorithm', grid=False, ax=plt.gca())
    plt.title("Computation Time by Algorithm")
    plt.suptitle("")
    plt.ylabel("Time (seconds)")
    plt.xlabel("Algorithm")

    # Bar plot of mean times
    plt.subplot(1, 2, 2)
    mean_times = grouped['time'].mean()
    mean_times.plot(kind='bar', ax=plt.gca())
    plt.title("Mean Computation Time by Algorithm")
    plt.ylabel("Mean Time (seconds)")
    plt.xlabel("Algorithm")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / "time_comparison.png")

    # Save statistics
    with open(output_dir / "statistics.txt", "w") as f:
        f.write(f"Problem size: {problem_size}\n")
        f.write(f"Hardness: {hardness}\n")
        f.write(f"Time limit: {time_limit} seconds\n")
        f.write(f"Max iterations: {max_iterations}\n")
        f.write(f"Number of instances: {n_instances}\n")
        f.write("\n=== Cost Statistics ===\n")
        f.write(str(stats['cost']))
        f.write("\n\n=== Feasibility Rate ===\n")
        f.write(str(stats['feasible']))
        f.write("\n\n=== Time Statistics (seconds) ===\n")
        f.write(str(stats['time']))
        f.write("\n\n=== Iterations Statistics ===\n")
        f.write(str(stats['iterations']))
        f.write("\n\n=== Improvement over Nearest Neighbor (%) ===\n")
        for alg, imp in improvements.items():
            f.write(f"{alg}: {imp:.2f}%\n")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test TSPDL model on large instances")
    parser.add_argument("output_dir", type=str, help="Output directory")
    parser.add_argument("--problem_size", type=int, default=50, help="Problem size")
    parser.add_argument("--hardness", type=str, default="medium", choices=["easy", "medium", "hard"], help="Hardness level")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the trained model")
    parser.add_argument("--time_limit", type=float, default=30.0, help="Time limit for search algorithms")
    parser.add_argument("--max_iterations", type=int, default=200, help="Maximum number of iterations")
    parser.add_argument("--n_instances", type=int, default=5, help="Number of instances to test")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available")

    args = parser.parse_args()

    run_large_test(
        args.output_dir,
        args.problem_size,
        args.hardness,
        args.model_path,
        args.time_limit,
        args.max_iterations,
        args.n_instances,
        args.use_gpu
    )

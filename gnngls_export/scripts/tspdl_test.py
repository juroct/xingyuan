#!/usr/bin/env python
# coding: utf-8

"""
Test TSPDL model.
"""

import argparse
import os
import pathlib
import time
import torch
import matplotlib.pyplot as plt

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


def run_test(
    output_dir,
    problem_size=20,
    hardness='medium',
    model_path=None,
    time_limit=10.0,
    max_iterations=100,
    use_gpu=False
):
    """
    Run TSPDL test.

    Args:
        output_dir: Output directory
        problem_size: Problem size
        hardness: Hardness level ('easy', 'medium', 'hard')
        model_path: Path to the trained model
        time_limit: Time limit for search algorithms
        max_iterations: Maximum number of iterations
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
            embed_dim=64,  # Same as training
            out_dim=1,
            n_layers=2,  # Same as training
            n_heads=4  # Same as training
        ).to(device)

        # Load model weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        print(f"Model loaded from {model_path}")

    # Create a TSPDL instance
    print("\nCreating a TSPDL instance...")
    instance = TSPDLInstance.generate_random(
        problem_size=problem_size,
        hardness=hardness,
        normalized=True,
        device=device
    )

    # Generate initial tour using nearest neighbor
    print("\nGenerating initial tour using nearest neighbor...")
    nn_tour = nearest_neighbor_tspdl(instance)
    nn_solution = TSPDLSolution(instance, nn_tour)
    nn_cost = nn_solution.cost
    print(f"Nearest neighbor tour cost: {nn_cost:.4f}")
    print(f"Feasible: {nn_solution.feasible}")

    # Apply local search
    print("\nApplying local search...")
    ls_tour, ls_cost, _ = local_search_tspdl(instance, nn_tour, max_iterations=1000)
    ls_solution = TSPDLSolution(instance, ls_tour)
    print(f"Local search tour cost: {ls_cost:.4f}")
    print(f"Feasible: {ls_solution.feasible}")

    # Apply guided local search
    print("\nApplying guided local search...")
    t_limit = time.time() + time_limit
    gls_tour, gls_cost, _ = guided_local_search_tspdl(
        instance, nn_tour, t_limit, max_iterations=max_iterations
    )
    gls_solution = TSPDLSolution(instance, gls_tour)
    print(f"Guided local search tour cost: {gls_cost:.4f}")
    print(f"Feasible: {gls_solution.feasible}")

    # Apply GNN-guided local search if model is provided
    if model is not None:
        print("\nApplying GNN-guided local search...")
        t_limit = time.time() + time_limit
        gnn_gls_tour, gnn_gls_cost, _ = gnn_guided_local_search_tspdl(
            instance, model, nn_tour, t_limit, max_iterations=max_iterations
        )
        gnn_gls_solution = TSPDLSolution(instance, gnn_gls_tour)
        print(f"GNN-guided local search tour cost: {gnn_gls_cost:.4f}")
        print(f"Feasible: {gnn_gls_solution.feasible}")

    # Plot solutions
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

    # Compare results
    print("\nResults summary:")
    print(f"Nearest neighbor tour cost: {nn_cost:.4f} (Feasible: {nn_solution.feasible})")
    print(f"Local search tour cost: {ls_cost:.4f} (Feasible: {ls_solution.feasible})")
    print(f"Guided local search tour cost: {gls_cost:.4f} (Feasible: {gls_solution.feasible})")
    if model is not None:
        print(f"GNN-guided local search tour cost: {gnn_gls_cost:.4f} (Feasible: {gnn_gls_solution.feasible})")

    # Save results
    with open(output_dir / "results.txt", "w") as f:
        f.write(f"Problem size: {problem_size}\n")
        f.write(f"Hardness: {hardness}\n")
        f.write(f"Time limit: {time_limit} seconds\n")
        f.write(f"Max iterations: {max_iterations}\n")
        f.write("\n")
        f.write(f"Nearest neighbor tour cost: {nn_cost:.4f} (Feasible: {nn_solution.feasible})\n")
        f.write(f"Local search tour cost: {ls_cost:.4f} (Feasible: {ls_solution.feasible})\n")
        f.write(f"Guided local search tour cost: {gls_cost:.4f} (Feasible: {gls_solution.feasible})\n")
        if model is not None:
            f.write(f"GNN-guided local search tour cost: {gnn_gls_cost:.4f} (Feasible: {gnn_gls_solution.feasible})\n")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test TSPDL model")
    parser.add_argument("output_dir", type=str, help="Output directory")
    parser.add_argument("--problem_size", type=int, default=20, help="Problem size")
    parser.add_argument("--hardness", type=str, default="medium", choices=["easy", "medium", "hard"], help="Hardness level")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the trained model")
    parser.add_argument("--time_limit", type=float, default=10.0, help="Time limit for search algorithms")
    parser.add_argument("--max_iterations", type=int, default=100, help="Maximum number of iterations")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available")

    args = parser.parse_args()

    run_test(
        args.output_dir,
        args.problem_size,
        args.hardness,
        args.model_path,
        args.time_limit,
        args.max_iterations,
        args.use_gpu
    )

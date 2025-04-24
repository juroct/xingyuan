#!/usr/bin/env python
# coding: utf-8

"""
Demonstrate TSPDL algorithms.

This script demonstrates different algorithms for solving TSPDL problems.
"""

import argparse
import os
import pathlib
import time

import matplotlib.pyplot as plt
import torch

from gnngls import (
    TSPDLInstance,
    TSPDLSolution,
    nearest_neighbor_tspdl,
    insertion_tspdl,
    local_search_tspdl,
    guided_local_search_tspdl,
    TSPDLEdgeModel,
    gnn_guided_local_search_tspdl,
    plot_tspdl_instance,
    plot_tspdl_solution,
    plot_tspdl_algorithm_comparison,
    plot_tspdl_gls_progress
)


def run_demo(
    output_dir,
    problem_size=20,
    hardness='medium',
    model_path=None,
    time_limit=5.0,
    max_iterations=20,
    use_gpu=False
):
    """
    Run TSPDL demo.
    
    Args:
        output_dir: Output directory
        problem_size: Problem size
        hardness: Hardness level ('easy', 'medium', 'hard')
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
            embed_dim=128,
            out_dim=1,
            n_layers=3,
            n_heads=8
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
    
    # Visualize the instance
    print("Visualizing the instance...")
    plot_tspdl_instance(
        instance, 
        title=f"TSPDL Instance ({problem_size} nodes, {hardness} hardness)",
        save_path=output_dir / "instance.png",
        show=False
    )
    
    # Apply different algorithms
    print("\nApplying different algorithms...")
    
    # Nearest neighbor
    print("\nApplying nearest neighbor...")
    start_time = time.time()
    nn_tour = nearest_neighbor_tspdl(instance)
    nn_time = time.time() - start_time
    nn_solution = TSPDLSolution(instance, nn_tour)
    
    print(f"Nearest Neighbor:")
    print(f"  Cost: {nn_solution.cost:.4f}")
    print(f"  Feasible: {nn_solution.feasible}")
    print(f"  Time: {nn_time:.4f}s")
    
    plot_tspdl_solution(
        nn_solution, 
        title="Nearest Neighbor Solution",
        save_path=output_dir / "nearest_neighbor_solution.png",
        show=False
    )
    
    # Insertion
    print("\nApplying insertion...")
    start_time = time.time()
    ins_tour = insertion_tspdl(instance)
    ins_time = time.time() - start_time
    ins_solution = TSPDLSolution(instance, ins_tour)
    
    print(f"Insertion:")
    print(f"  Cost: {ins_solution.cost:.4f}")
    print(f"  Feasible: {ins_solution.feasible}")
    print(f"  Time: {ins_time:.4f}s")
    
    plot_tspdl_solution(
        ins_solution, 
        title="Insertion Solution",
        save_path=output_dir / "insertion_solution.png",
        show=False
    )
    
    # Local search
    print("\nApplying local search...")
    start_time = time.time()
    ls_tour, ls_cost, _ = local_search_tspdl(instance, nn_tour)
    ls_time = time.time() - start_time
    ls_solution = TSPDLSolution(instance, ls_tour)
    
    print(f"Local Search:")
    print(f"  Cost: {ls_solution.cost:.4f}")
    print(f"  Feasible: {ls_solution.feasible}")
    print(f"  Time: {ls_time:.4f}s")
    
    plot_tspdl_solution(
        ls_solution, 
        title="Local Search Solution",
        save_path=output_dir / "local_search_solution.png",
        show=False
    )
    
    # Guided local search
    print("\nApplying guided local search...")
    t_limit = time.time() + time_limit
    
    start_time = time.time()
    gls_tour, gls_cost, gls_info = guided_local_search_tspdl(
        instance, nn_tour, t_limit, 
        track_progress=True,
        max_iterations=max_iterations
    )
    gls_time = time.time() - start_time
    gls_solution = TSPDLSolution(instance, gls_tour)
    
    print(f"Guided Local Search:")
    print(f"  Cost: {gls_solution.cost:.4f}")
    print(f"  Feasible: {gls_solution.feasible}")
    print(f"  Time: {gls_time:.4f}s")
    
    plot_tspdl_solution(
        gls_solution, 
        title="Guided Local Search Solution",
        save_path=output_dir / "guided_local_search_solution.png",
        show=False
    )
    
    # GNN-guided local search
    if model is not None:
        print("\nApplying GNN-guided local search...")
        t_limit = time.time() + time_limit
        
        start_time = time.time()
        gnn_gls_tour, gnn_gls_cost, gnn_gls_info = gnn_guided_local_search_tspdl(
            instance, model, nn_tour, t_limit, 
            track_progress=True,
            max_iterations=max_iterations
        )
        gnn_gls_time = time.time() - start_time
        gnn_gls_solution = TSPDLSolution(instance, gnn_gls_tour)
        
        print(f"GNN-Guided Local Search:")
        print(f"  Cost: {gnn_gls_solution.cost:.4f}")
        print(f"  Feasible: {gnn_gls_solution.feasible}")
        print(f"  Time: {gnn_gls_time:.4f}s")
        
        plot_tspdl_solution(
            gnn_gls_solution, 
            title="GNN-Guided Local Search Solution",
            save_path=output_dir / "gnn_guided_local_search_solution.png",
            show=False
        )
    
    # Compare all algorithms
    print("\nComparing all algorithms...")
    algorithm_tours = {
        "Nearest Neighbor": nn_tour,
        "Insertion": ins_tour,
        "Local Search": ls_tour,
        "Guided Local Search": gls_tour
    }
    
    if model is not None:
        algorithm_tours["GNN-Guided Local Search"] = gnn_gls_tour
    
    plot_tspdl_algorithm_comparison(
        instance, 
        algorithm_tours,
        save_path=output_dir / "algorithm_comparison.png",
        show=False
    )
    
    # Visualize GLS progress
    if 'progress' in gls_info:
        print("\nVisualizing GLS progress...")
        progress = gls_info['progress']
        
        plot_tspdl_gls_progress(
            instance,
            progress['tours'],
            progress['costs'],
            progress['out_of_draft_limit'],
            progress['infeasible_nodes'],
            save_path=output_dir / "gls_progress.png",
            show=False
        )
    
    # Visualize GNN-GLS progress
    if model is not None and 'progress' in gnn_gls_info:
        print("\nVisualizing GNN-GLS progress...")
        progress = gnn_gls_info['progress']
        
        plot_tspdl_gls_progress(
            instance,
            progress['tours'],
            progress['costs'],
            progress['out_of_draft_limit'],
            progress['infeasible_nodes'],
            save_path=output_dir / "gnn_gls_progress.png",
            show=False
        )
    
    print("\nDemo completed!")
    print(f"All results have been saved to the '{output_dir}' directory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demonstrate TSPDL algorithms")
    parser.add_argument("output_dir", type=str, help="Output directory")
    parser.add_argument("--problem_size", type=int, default=20, help="Problem size")
    parser.add_argument("--hardness", type=str, default="medium", choices=["easy", "medium", "hard"], help="Hardness level")
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained model")
    parser.add_argument("--time_limit", type=float, default=5.0, help="Time limit for search algorithms")
    parser.add_argument("--max_iterations", type=int, default=20, help="Maximum iterations for search algorithms")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available")
    
    args = parser.parse_args()
    
    run_demo(
        args.output_dir,
        args.problem_size,
        args.hardness,
        args.model_path,
        args.time_limit,
        args.max_iterations,
        args.use_gpu
    )

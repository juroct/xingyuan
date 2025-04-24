#!/usr/bin/env python
# coding: utf-8

import argparse
import pathlib

import numpy as np
import tqdm.auto as tqdm

from gnngls import datasets


def generate_dataset(n_instances, n_nodes, seed=None):
    """Generate a dataset of TSP instances."""
    if seed is not None:
        np.random.seed(seed)
    
    instances = []
    for i in tqdm.trange(n_instances, desc="Generating instances"):
        instance = datasets.generate_tsp_instance(n_nodes, seed=seed + i if seed is not None else None)
        instances.append(instance)
    
    return instances


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate TSP instances')
    parser.add_argument('output_dir', type=pathlib.Path, help='Directory to save instances')
    parser.add_argument('--n_train', type=int, default=1000, help='Number of training instances')
    parser.add_argument('--n_val', type=int, default=100, help='Number of validation instances')
    parser.add_argument('--n_test', type=int, default=100, help='Number of test instances')
    parser.add_argument('--n_nodes', type=int, default=20, help='Number of nodes in each instance')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate datasets
    train_instances = generate_dataset(args.n_train, args.n_nodes, seed=args.seed)
    val_instances = generate_dataset(args.n_val, args.n_nodes, seed=args.seed + args.n_train)
    test_instances = generate_dataset(args.n_test, args.n_nodes, seed=args.seed + args.n_train + args.n_val)
    
    # Save datasets
    datasets.save_tsp_instances(train_instances, args.output_dir / 'train.txt')
    datasets.save_tsp_instances(val_instances, args.output_dir / 'val.txt')
    datasets.save_tsp_instances(test_instances, args.output_dir / 'test.txt')
    
    print(f"Generated {args.n_train} training, {args.n_val} validation, and {args.n_test} test instances")
    print(f"Saved to {args.output_dir}")

#!/usr/bin/env python
# coding: utf-8

"""
Generate TSPDL instances.

This script generates TSPDL instances for training and testing.
"""

import argparse
import os
import pathlib
import pickle
import uuid

import numpy as np
import tqdm.auto as tqdm

from gnngls import TSPDLInstance


def generate_instances(
    output_dir,
    problem_size,
    n_instances,
    hardness='medium',
    seed=None
):
    """
    Generate TSPDL instances.

    Args:
        output_dir: Output directory
        problem_size: Problem size
        n_instances: Number of instances
        hardness: Hardness level ('easy', 'medium', 'hard')
        seed: Random seed
    """
    if seed is not None:
        np.random.seed(seed)

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    instances = []

    for i in tqdm.trange(n_instances, desc=f"Generating {problem_size}-node instances"):
        # Set random seed for reproducibility
        instance_seed = seed + i if seed is not None else None

        # Generate instance
        if instance_seed is not None:
            np.random.seed(instance_seed)

        instance = TSPDLInstance.generate_random(
            problem_size=problem_size,
            hardness=hardness,
            normalized=True
        )

        # Save instance
        instance_id = uuid.uuid4().hex
        instance_path = output_dir / f"{instance_id}.pkl"

        with open(instance_path, 'wb') as f:
            pickle.dump(instance, f)

        instances.append(instance_path.name)

    # Save instance list
    with open(output_dir / "instances.txt", 'w') as f:
        for instance in instances:
            f.write(f"{instance}\n")

    print(f"Generated {n_instances} instances with {problem_size} nodes")
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate TSPDL instances")
    parser.add_argument("output_dir", type=str, help="Output directory")
    parser.add_argument("--problem_size", type=int, default=20, help="Problem size")
    parser.add_argument("--n_instances", type=int, default=100, help="Number of instances")
    parser.add_argument("--hardness", type=str, default="medium", choices=["easy", "medium", "hard"], help="Hardness level")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    args = parser.parse_args()

    generate_instances(
        args.output_dir,
        args.problem_size,
        args.n_instances,
        args.hardness,
        args.seed
    )

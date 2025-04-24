"""
External solvers for TSP.
"""

import os
import tempfile
import shutil
import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Any, Optional, Union

# Check if manually installed solvers are available
CONCORDE_AVAILABLE = shutil.which("concorde") is not None
LKH_AVAILABLE = shutil.which("LKH") is not None

# Import wrapper modules if available
if CONCORDE_AVAILABLE:
    try:
        from .concorde_wrapper import solve_tsp_concorde as _solve_tsp_concorde
    except ImportError:
        try:
            # Try to import from the same directory
            from concorde_wrapper import solve_tsp_concorde as _solve_tsp_concorde
        except ImportError:
            CONCORDE_AVAILABLE = False
            print("Warning: Concorde wrapper not found. Make sure concorde_wrapper.py is in the PYTHONPATH")
else:
    print("Warning: Concorde solver not found in PATH. Make sure it's installed and in the PATH")

if LKH_AVAILABLE:
    try:
        from .lkh_wrapper import solve_tsp_lkh as _solve_tsp_lkh
    except ImportError:
        try:
            # Try to import from the same directory
            from lkh_wrapper import solve_tsp_lkh as _solve_tsp_lkh
        except ImportError:
            LKH_AVAILABLE = False
            print("Warning: LKH wrapper not found. Make sure lkh_wrapper.py is in the PYTHONPATH")
else:
    print("Warning: LKH solver not found in PATH. Make sure it's installed and in the PATH")

try:
    import tsplib95
    TSPLIB_AVAILABLE = True
except ImportError:
    TSPLIB_AVAILABLE = False
    print("Warning: TSPLIB not available. Install with 'pip install tsplib95'")

# Define a flag to check if any solver is available
ANY_SOLVER_AVAILABLE = CONCORDE_AVAILABLE or LKH_AVAILABLE


def calculate_tour_cost(
    coordinates: List[Tuple[float, float]],
    tour: List[int]
) -> float:
    """
    Calculate the cost of a tour.
    
    Args:
        coordinates: List of (x, y) coordinates
        tour: Tour as a list of node indices
        
    Returns:
        cost: Tour cost
    """
    cost = 0.0
    for i in range(len(tour) - 1):
        x1, y1 = coordinates[tour[i]]
        x2, y2 = coordinates[tour[i + 1]]
        cost += ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    return cost


def solve_tsp_concorde(
    coordinates: List[Tuple[float, float]],
    time_limit: Optional[int] = None,
    verbose: bool = False
) -> Tuple[List[int], float]:
    """
    Solve TSP using Concorde solver.
    
    Args:
        coordinates: List of (x, y) coordinates
        time_limit: Time limit in seconds (None for no limit)
        verbose: Whether to print solver output
        
    Returns:
        tour: Optimal tour
        cost: Optimal tour cost
    """
    if not CONCORDE_AVAILABLE:
        raise ImportError("Concorde solver not available. Make sure it's installed and in the PATH")
    
    # Use the wrapper function
    return _solve_tsp_concorde(coordinates, time_limit, verbose)


def solve_tsp_lkh(
    coordinates: List[Tuple[float, float]],
    time_limit: Optional[int] = None,
    max_trials: int = 1000,
    verbose: bool = False
) -> Tuple[List[int], float]:
    """
    Solve TSP using LKH solver.
    
    Args:
        coordinates: List of (x, y) coordinates
        time_limit: Time limit in seconds (None for no limit)
        max_trials: Maximum number of trials
        verbose: Whether to print solver output
        
    Returns:
        tour: Optimal tour
        cost: Optimal tour cost
    """
    if not LKH_AVAILABLE:
        raise ImportError("LKH solver not available. Make sure it's installed and in the PATH")
    
    # Use the wrapper function
    return _solve_tsp_lkh(coordinates, max_trials, time_limit, verbose)


def solve_tsp_with_fixed_edges(
    coordinates: List[Tuple[float, float]],
    fixed_edges: List[Tuple[int, int]],
    solver: str = 'concorde',
    time_limit: Optional[int] = None,
    verbose: bool = False
) -> Tuple[List[int], float]:
    """
    This function is not fully implemented for manual solvers yet.
    It will be implemented in a future version.
    """
    raise NotImplementedError("solve_tsp_with_fixed_edges is not implemented for manual solvers yet.")


def get_optimal_tour(
    coordinates: List[Tuple[float, float]],
    solver: str = 'concorde',
    time_limit: Optional[int] = None,
    verbose: bool = False
) -> Tuple[List[int], float]:
    """
    Get the optimal tour for a TSP instance.
    
    Args:
        coordinates: List of (x, y) coordinates
        solver: Solver to use ('concorde' or 'lkh')
        time_limit: Time limit in seconds (None for no limit)
        verbose: Whether to print solver output
        
    Returns:
        tour: Optimal tour
        cost: Optimal tour cost
    """
    if solver == 'concorde':
        return solve_tsp_concorde(coordinates, time_limit, verbose)
    elif solver == 'lkh':
        return solve_tsp_lkh(coordinates, time_limit, verbose=verbose)
    else:
        raise ValueError(f"Unknown solver: {solver}. Use 'concorde' or 'lkh'")


def get_optimal_cost(
    coordinates: List[Tuple[float, float]],
    solver: str = 'concorde',
    time_limit: Optional[int] = None,
    verbose: bool = False
) -> float:
    """
    Get the optimal cost for a TSP instance.
    
    Args:
        coordinates: List of (x, y) coordinates
        solver: Solver to use ('concorde' or 'lkh')
        time_limit: Time limit in seconds (None for no limit)
        verbose: Whether to print solver output
        
    Returns:
        cost: Optimal tour cost
    """
    _, cost = get_optimal_tour(coordinates, solver, time_limit, verbose)
    return cost

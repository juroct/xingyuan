import networkx as nx
import numpy as np


def tour_to_edge_attribute(G, tour):
    in_tour = {}
    tour_edges = list(zip(tour[:-1], tour[1:]))
    for e in G.edges:
        in_tour[e] = e in tour_edges or tuple(reversed(e)) in tour_edges
    return in_tour


def tour_cost(G, tour, weight='weight'):
    c = 0
    for e in zip(tour[:-1], tour[1:]):
        c += G.edges[e][weight]
    return c


def is_equivalent_tour(tour_a, tour_b):
    if tour_a == tour_b[::-1]:
        return True
    if tour_a == tour_b:
        return True
    return False


def is_valid_tour(G, tour):
    if tour[0] != 0:
        return False
    if tour[-1] != 0:
        return False
    for n in G.nodes:
        c = tour.count(n)
        if n == 0:
            if c != 2:
                return False
        elif c != 1:
            return False
    return True


# Import core modules
from .models import EdgePropertyPredictionModel
from .algorithms import (
    nearest_neighbor,
    probabilistic_nearest_neighbour,
    best_probabilistic_nearest_neighbour,
    insertion,
    local_search,
    guided_local_search,
)
from .operators import two_opt_a2a, relocate_a2a, two_opt_o2a, relocate_o2a
from .datasets import TSPDataset

# Import solvers if available
try:
    from .solvers import (
        get_optimal_tour,
        get_optimal_cost,
        solve_tsp_concorde,
        solve_tsp_lkh,
        solve_tsp_with_fixed_edges,
        ANY_SOLVER_AVAILABLE,
    )
    SOLVERS_AVAILABLE = ANY_SOLVER_AVAILABLE
except ImportError:
    SOLVERS_AVAILABLE = False

# Import visualization functions
try:
    from .visualization import (
        plot_tsp_instance,
        plot_tour,
        plot_algorithm_comparison,
        plot_gnn_predictions,
        plot_guided_local_search_progress,
        create_animation,
    )
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# Import TSPDL functions
try:
    from .tspdl import (
        TSPDLInstance,
        TSPDLSolution,
        generate_tspdl_dataset,
        load_tspdl_dataset,
    )
    from .tspdl_algorithms import (
        nearest_neighbor_tspdl,
        insertion_tspdl,
        local_search_tspdl,
        guided_local_search_tspdl,
        reinforcement_learning_tspdl,
    )
    from .tspdl_models import (
        TSPDLEdgeModel,
        TSPDLNodeModel,
        TSPDLRLModel,
    )
    from .tspdl_simple_model import (
        SimpleGNNModel,
    )
    from .tspdl_env import (
        TSPDLEnv,
        TSPDLBatchEnv,
    )
    from .tspdl_rl import (
        TSPDLRLAgent,
        train_tspdl_rl_agent,
    )
    from .tspdl_gnn_gls import (
        create_graph_from_instance,
        update_graph_with_tour,
        predict_edge_scores,
        gnn_guided_local_search_tspdl,
    )
    from .tspdl_visualization import (
        plot_tspdl_instance,
        plot_tspdl_solution,
        plot_tspdl_algorithm_comparison,
        plot_tspdl_gls_progress,
    )
    from .tspdl_dataset import (
        TSPDLDataset,
    )
    TSPDL_AVAILABLE = True
except ImportError:
    TSPDL_AVAILABLE = False

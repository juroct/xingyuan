"""
TSPDL (Traveling Salesperson Problem with Draft Limits) module.

This module extends the GNNGLS framework to handle TSP with draft limits,
where each node has a demand and a draft limit, and the vehicle's load
must not exceed the draft limit of the next node to be visited.
"""

import os
import pickle
import numpy as np
import torch
import networkx as nx
from typing import List, Tuple, Dict, Optional, Union, Any

from .solvers import ANY_SOLVER_AVAILABLE
if ANY_SOLVER_AVAILABLE:
    from .solvers import get_optimal_tour


class TSPDLInstance:
    """
    Class representing a TSPDL instance.

    Attributes:
        node_xy: Node coordinates
        node_demand: Demand of each node
        node_draft_limit: Draft limit of each node
        problem_size: Number of nodes
        device: Device to use for tensor operations
    """

    def __init__(
        self,
        node_xy: torch.Tensor,
        node_demand: torch.Tensor,
        node_draft_limit: torch.Tensor,
        device: torch.device = None
    ):
        """
        Initialize a TSPDL instance.

        Args:
            node_xy: Node coordinates, shape (problem_size, 2)
            node_demand: Demand of each node, shape (problem_size,)
            node_draft_limit: Draft limit of each node, shape (problem_size,)
            device: Device to use for tensor operations
        """
        self.node_xy = node_xy
        self.node_demand = node_demand
        self.node_draft_limit = node_draft_limit
        self.problem_size = node_xy.size(0)
        self.device = device if device is not None else torch.device('cpu')

        # Move tensors to device
        self.node_xy = self.node_xy.to(self.device)
        self.node_demand = self.node_demand.to(self.device)
        self.node_draft_limit = self.node_draft_limit.to(self.device)

    def to_networkx(self) -> nx.Graph:
        """
        Convert the TSPDL instance to a NetworkX graph.

        Returns:
            G: NetworkX graph representing the TSPDL instance
        """
        G = nx.complete_graph(self.problem_size)

        # Add node attributes
        for i in range(self.problem_size):
            G.nodes[i]['coords'] = tuple(self.node_xy[i].cpu().numpy())
            G.nodes[i]['demand'] = float(self.node_demand[i].cpu().numpy())
            G.nodes[i]['draft_limit'] = float(self.node_draft_limit[i].cpu().numpy())

        # Add edge weights (distances)
        for i, j in G.edges:
            x1, y1 = G.nodes[i]['coords']
            x2, y2 = G.nodes[j]['coords']
            dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            G.edges[i, j]['weight'] = dist

        return G

    @classmethod
    def from_networkx(cls, G: nx.Graph, device: torch.device = None) -> 'TSPDLInstance':
        """
        Create a TSPDL instance from a NetworkX graph.

        Args:
            G: NetworkX graph representing the TSPDL instance
            device: Device to use for tensor operations

        Returns:
            instance: TSPDL instance
        """
        problem_size = G.number_of_nodes()
        node_xy = torch.zeros((problem_size, 2))
        node_demand = torch.zeros(problem_size)
        node_draft_limit = torch.zeros(problem_size)

        for i in range(problem_size):
            node_xy[i] = torch.tensor(G.nodes[i]['coords'])
            node_demand[i] = G.nodes[i]['demand']
            node_draft_limit[i] = G.nodes[i]['draft_limit']

        return cls(node_xy, node_demand, node_draft_limit, device)

    @classmethod
    def generate_random(
        cls,
        problem_size: int,
        hardness: str = 'medium',
        normalized: bool = True,
        device: torch.device = None,
        max_attempts: int = 10
    ) -> 'TSPDLInstance':
        """
        Generate a random TSPDL instance with at least one feasible solution.

        Args:
            problem_size: Number of nodes
            hardness: Difficulty level ('easy', 'medium', 'hard')
            normalized: Whether to normalize demands and draft limits
            device: Device to use for tensor operations
            max_attempts: Maximum number of attempts to generate a feasible instance

        Returns:
            instance: Random TSPDL instance with at least one feasible solution
        """
        dl_percent_dict = {
            "hard": 90,
            "medium": 75,
            "easy": 50,
        }
        dl_percent = dl_percent_dict[hardness]

        # Try multiple times to generate a feasible instance
        for attempt in range(max_attempts):
            # Generate random coordinates
            node_xy = torch.rand(size=(problem_size, 2))

            # Generate demands (depot has 0 demand, others have 1)
            node_demand = torch.cat([
                torch.zeros(1),
                torch.ones(problem_size - 1)
            ])

            # Calculate demand sum
            demand_sum = node_demand.sum().unsqueeze(0)

            # Generate a random feasible tour first
            # We'll use a greedy approach based on node indices
            # Sort nodes by increasing demand (non-depot nodes all have demand=1 here)
            # This ensures we visit nodes in a specific order
            sorted_nodes = list(range(1, problem_size))
            np.random.shuffle(sorted_nodes)  # Randomize order
            feasible_tour = [0] + sorted_nodes + [0]  # Start and end at depot

            # Calculate loads at each step of the tour
            loads = [0.0]  # Start with zero load at depot
            current_load = 0.0
            for node in feasible_tour[1:]:  # Skip depot at beginning
                if node != 0:  # Not returning to depot
                    current_load += node_demand[node].item()
                loads.append(current_load)

            # Now set draft limits based on this feasible tour
            # Each node's draft limit must be at least its load when visited
            node_draft_limit = torch.ones(problem_size) * demand_sum  # Start with maximum

            # For non-depot nodes that should have lower draft limits
            lower_dl_idx = np.random.choice(
                range(1, problem_size),
                size=problem_size * dl_percent // 100,
                replace=False
            )

            # For each node in lower_dl_idx, find its position in the tour
            # and set its draft limit slightly above the load at that position
            for node in lower_dl_idx:
                pos = feasible_tour.index(node)
                load_at_visit = loads[pos-1]  # Load before visiting this node

                # Set draft limit to a random value between load_at_visit and demand_sum
                # with a small margin to ensure feasibility
                margin = 0.05  # 5% margin
                min_limit = load_at_visit * (1.0 + margin)
                max_limit = demand_sum.item()

                if min_limit >= max_limit:
                    # If margin pushes beyond max, just use the load plus a tiny amount
                    draft_limit = load_at_visit + 0.01
                else:
                    # Random value between min_limit and max_limit
                    draft_limit = min_limit + torch.rand(1).item() * (max_limit - min_limit)

                node_draft_limit[node] = draft_limit

            # Create the instance
            instance = cls(node_xy, node_demand, node_draft_limit, device)

            # Verify that our feasible_tour is actually feasible
            # We'll check feasibility manually to avoid circular imports
            feasible = True
            current_load = 0.0
            for i in range(1, len(feasible_tour)):
                node = feasible_tour[i]
                if node != 0:  # Not returning to depot
                    # Check if load exceeds draft limit
                    if current_load > node_draft_limit[node].item():
                        feasible = False
                        break
                    # Update load after visiting
                    current_load += node_demand[node].item()

            if feasible:
                # Normalize if requested
                if normalized:
                    instance.node_demand = instance.node_demand / demand_sum
                    instance.node_draft_limit = instance.node_draft_limit / demand_sum

                print(f"Generated feasible TSPDL instance on attempt {attempt+1}")
                return instance

            print(f"Attempt {attempt+1} failed to generate a feasible instance, retrying...")

        # If we reach here, we couldn't generate a feasible instance
        # Fall back to the original method
        print("Failed to generate a feasible instance after maximum attempts, falling back to original method")

        # Generate random coordinates
        node_xy = torch.rand(size=(problem_size, 2))

        # Generate demands (depot has 0 demand, others have 1)
        node_demand = torch.cat([
            torch.zeros(1),
            torch.ones(problem_size - 1)
        ])

        # Calculate demand sum
        demand_sum = node_demand.sum().unsqueeze(0)

        # Initialize draft limits to demand sum
        node_draft_limit = torch.ones(problem_size) * demand_sum

        # Randomly lower draft limits for some nodes
        lower_dl_idx = np.random.choice(
            range(1, problem_size),
            size=problem_size * dl_percent // 100,
            replace=False
        )

        # Ensure feasible draft limits
        feasible_dl = False
        while not feasible_dl:
            lower_dl = torch.randint(
                1, demand_sum.int().item(),
                size=(problem_size * dl_percent // 100,)
            )
            cnt = torch.bincount(lower_dl)
            cnt_cumsum = torch.cumsum(cnt, dim=0)
            feasible_dl = (cnt_cumsum <= torch.arange(0, cnt.size(0))).all()

        node_draft_limit[lower_dl_idx] = lower_dl.float()

        # Normalize if requested
        if normalized:
            node_demand = node_demand / demand_sum
            node_draft_limit = node_draft_limit / demand_sum

        return cls(node_xy, node_demand, node_draft_limit, device)

    @classmethod
    def load_dataset(
        cls,
        path: str,
        offset: int = 0,
        num_samples: int = 1000,
        device: torch.device = None
    ) -> List['TSPDLInstance']:
        """
        Load TSPDL instances from a dataset file.

        Args:
            path: Path to the dataset file
            offset: Offset in the dataset
            num_samples: Number of samples to load
            device: Device to use for tensor operations

        Returns:
            instances: List of TSPDL instances
        """
        assert os.path.splitext(path)[1] == ".pkl", "Unsupported file type (.pkl needed)."

        with open(path, 'rb') as f:
            data = pickle.load(f)[offset: offset + num_samples]

        instances = []
        for item in data:
            node_xy = torch.tensor(item[0], dtype=torch.float32)
            node_demand = torch.tensor(item[1], dtype=torch.float32)
            node_draft_limit = torch.tensor(item[2], dtype=torch.float32)

            # Scale to [0,1]
            demand_sum = node_demand.sum().view(-1)
            node_demand = node_demand / demand_sum
            node_draft_limit = node_draft_limit / demand_sum

            instances.append(cls(node_xy, node_demand, node_draft_limit, device))

        return instances

    @staticmethod
    def save_dataset(instances: List[Tuple], path: str):
        """
        Save TSPDL instances to a dataset file.

        Args:
            instances: List of (node_xy, node_demand, node_draft_limit) tuples
            path: Path to save the dataset
        """
        filedir = os.path.split(path)[0]
        if not os.path.isdir(filedir):
            os.makedirs(filedir)

        dataset = []
        for node_xy, node_demand, node_draft_limit in instances:
            if isinstance(node_xy, torch.Tensor):
                node_xy = node_xy.cpu().tolist()
            if isinstance(node_demand, torch.Tensor):
                node_demand = node_demand.cpu().tolist()
            if isinstance(node_draft_limit, torch.Tensor):
                node_draft_limit = node_draft_limit.cpu().tolist()

            dataset.append((node_xy, node_demand, node_draft_limit))

        with open(path, 'wb') as f:
            pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

        print(f"Saved TSPDL dataset to {path}")

    @staticmethod
    def generate_dataset(
        num_samples: int,
        problem_size: int,
        path: str,
        hardness: str = 'medium',
        normalized: bool = True
    ):
        """
        Generate and save a TSPDL dataset.

        Args:
            num_samples: Number of instances to generate
            problem_size: Number of nodes in each instance
            path: Path to save the dataset
            hardness: Difficulty level ('easy', 'medium', 'hard')
            normalized: Whether to normalize demands and draft limits
        """
        instances = []
        for _ in range(num_samples):
            instance = TSPDLInstance.generate_random(
                problem_size, hardness, normalized
            )
            instances.append((
                instance.node_xy,
                instance.node_demand,
                instance.node_draft_limit
            ))

        TSPDLInstance.save_dataset(instances, path)


class TSPDLSolution:
    """
    Class representing a solution to a TSPDL instance.

    Attributes:
        instance: TSPDL instance
        tour: List of node indices representing the tour
        load: Current load at each step of the tour
        feasible: Whether the solution is feasible
        cost: Total cost of the tour
    """

    def __init__(
        self,
        instance: TSPDLInstance,
        tour: List[int]
    ):
        """
        Initialize a TSPDL solution.

        Args:
            instance: TSPDL instance
            tour: List of node indices representing the tour
        """
        self.instance = instance
        self.tour = tour
        self.load = self._calculate_load()
        self.feasible = self._check_feasibility()
        self.cost = self._calculate_cost()
        self.out_of_draft_limit = self._calculate_out_of_draft_limit()

    def _calculate_load(self) -> List[float]:
        """
        Calculate the load at each step of the tour.

        Returns:
            load: List of loads at each step
        """
        load = [0.0]  # Start with zero load
        current_load = 0.0

        for node in self.tour[1:]:  # Skip depot at the beginning
            current_load += self.instance.node_demand[node].item()
            load.append(current_load)

        return load

    def _check_feasibility(self) -> bool:
        """
        Check if the solution is feasible.

        Returns:
            feasible: Whether the solution is feasible
        """
        for i in range(1, len(self.tour)):
            prev_node = self.tour[i-1]
            node = self.tour[i]

            # Check if load exceeds draft limit
            if self.load[i-1] > self.instance.node_draft_limit[node].item():
                return False

        return True

    def _calculate_cost(self) -> float:
        """
        Calculate the total cost of the tour.

        Returns:
            cost: Total cost of the tour
        """
        cost = 0.0
        for i in range(len(self.tour) - 1):
            node1 = self.tour[i]
            node2 = self.tour[i+1]

            # Calculate Euclidean distance
            x1, y1 = self.instance.node_xy[node1]
            x2, y2 = self.instance.node_xy[node2]

            dist = torch.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            cost += dist.item()

        return cost

    def _calculate_out_of_draft_limit(self) -> List[float]:
        """
        Calculate how much the load exceeds the draft limit at each step.

        Returns:
            out_of_draft_limit: List of excesses at each step
        """
        out_of_draft_limit = []

        for i in range(1, len(self.tour)):
            node = self.tour[i]
            draft_limit = self.instance.node_draft_limit[node].item()
            excess = max(0, self.load[i-1] - draft_limit)
            out_of_draft_limit.append(excess)

        return out_of_draft_limit

    def total_out_of_draft_limit(self) -> float:
        """
        Calculate the total excess over draft limits.

        Returns:
            total_excess: Total excess over all draft limits
        """
        return sum(self.out_of_draft_limit)

    def count_out_of_draft_limit_nodes(self) -> int:
        """
        Count the number of nodes where draft limit is exceeded.

        Returns:
            count: Number of nodes with exceeded draft limit
        """
        return sum(1 for excess in self.out_of_draft_limit if excess > 0)

    def to_networkx(self) -> nx.Graph:
        """
        Convert the solution to a NetworkX graph.

        Returns:
            G: NetworkX graph representing the solution
        """
        G = self.instance.to_networkx()

        # Add solution attributes
        for i, node in enumerate(self.tour):
            G.nodes[node]['visit_order'] = i
            if i < len(self.load):
                G.nodes[node]['load'] = self.load[i]

        # Add tour edges
        for i in range(len(self.tour) - 1):
            u, v = self.tour[i], self.tour[i+1]
            G.edges[u, v]['in_tour'] = True

            # Add excess information if applicable
            if i+1 < len(self.tour) and i < len(self.out_of_draft_limit):
                G.edges[u, v]['excess'] = self.out_of_draft_limit[i]

        return G


def generate_tspdl_dataset(
    num_samples: int,
    problem_size: int,
    path: str,
    hardness: str = 'medium',
    normalized: bool = True
):
    """
    Generate and save a TSPDL dataset.

    Args:
        num_samples: Number of instances to generate
        problem_size: Number of nodes in each instance
        path: Path to save the dataset
        hardness: Difficulty level ('easy', 'medium', 'hard')
        normalized: Whether to normalize demands and draft limits
    """
    TSPDLInstance.generate_dataset(
        num_samples, problem_size, path, hardness, normalized
    )


def load_tspdl_dataset(
    path: str,
    offset: int = 0,
    num_samples: int = 1000,
    device: torch.device = None
) -> List[TSPDLInstance]:
    """
    Load TSPDL instances from a dataset file.

    Args:
        path: Path to the dataset file
        offset: Offset in the dataset
        num_samples: Number of samples to load
        device: Device to use for tensor operations

    Returns:
        instances: List of TSPDL instances
    """
    return TSPDLInstance.load_dataset(path, offset, num_samples, device)

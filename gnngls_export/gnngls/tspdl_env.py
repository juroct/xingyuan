"""
Environment for the Traveling Salesperson Problem with Draft Limits (TSPDL).

This module provides an environment for reinforcement learning on TSPDL.
"""

import numpy as np
import torch
import networkx as nx
import dgl
from typing import List, Tuple, Dict, Optional, Union, Any

from .tspdl import TSPDLInstance, TSPDLSolution


class TSPDLEnv:
    """
    Environment for TSPDL.
    
    This environment allows an agent to solve TSPDL instances using reinforcement learning.
    """
    
    def __init__(
        self,
        instance: TSPDLInstance,
        device: torch.device = None
    ):
        """
        Initialize the environment.
        
        Args:
            instance: TSPDL instance
            device: Device to use for tensor operations
        """
        self.instance = instance
        self.device = device if device is not None else torch.device('cpu')
        
        # Problem size
        self.problem_size = instance.problem_size
        
        # Node features
        self.node_xy = instance.node_xy
        self.node_demand = instance.node_demand
        self.node_draft_limit = instance.node_draft_limit
        
        # Reset environment
        self.reset()
    
    def reset(self, start_node: int = 0) -> Dict:
        """
        Reset the environment.
        
        Args:
            start_node: Starting node (depot)
            
        Returns:
            state: Initial state
        """
        # Initialize tour with start node
        self.tour = [start_node]
        
        # Initialize current load
        self.current_load = 0.0
        
        # Initialize visited nodes
        self.visited = set([start_node])
        
        # Initialize current node
        self.current_node = start_node
        
        # Initialize step counter
        self.step_count = 0
        
        # Initialize done flag
        self.done = False
        
        # Initialize reward
        self.total_reward = 0.0
        
        # Initialize state
        self.state = self._get_state()
        
        return self.state
    
    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Node to visit next
            
        Returns:
            state: New state
            reward: Reward
            done: Whether the episode is done
            info: Additional information
        """
        # Check if action is valid
        if action in self.visited:
            reward = -1.0  # Penalty for visiting already visited node
            return self.state, reward, self.done, {'valid': False}
        
        # Check if draft limit is respected
        if self.current_load > self.instance.node_draft_limit[action].item():
            reward = -1.0  # Penalty for exceeding draft limit
            return self.state, reward, self.done, {'valid': False}
        
        # Add node to tour
        self.tour.append(action)
        
        # Update current load
        self.current_load += self.instance.node_demand[action].item()
        
        # Update visited nodes
        self.visited.add(action)
        
        # Update current node
        prev_node = self.current_node
        self.current_node = action
        
        # Calculate reward (negative distance)
        x1, y1 = self.instance.node_xy[prev_node]
        x2, y2 = self.instance.node_xy[action]
        distance = torch.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2).item()
        reward = -distance
        
        # Update total reward
        self.total_reward += reward
        
        # Increment step counter
        self.step_count += 1
        
        # Check if all nodes are visited
        if len(self.visited) == self.problem_size:
            # Return to depot
            self.tour.append(0)  # Depot
            
            # Calculate final reward (negative distance to depot)
            x1, y1 = self.instance.node_xy[self.current_node]
            x2, y2 = self.instance.node_xy[0]  # Depot
            distance = torch.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2).item()
            reward -= distance
            
            # Update total reward
            self.total_reward += reward
            
            # Set done flag
            self.done = True
        
        # Update state
        self.state = self._get_state()
        
        return self.state, reward, self.done, {'valid': True}
    
    def _get_state(self) -> Dict:
        """
        Get the current state.
        
        Returns:
            state: Current state
        """
        # Create mask for valid actions
        mask = torch.zeros(self.problem_size, device=self.device)
        
        for node in range(self.problem_size):
            if node not in self.visited and self.current_load <= self.instance.node_draft_limit[node].item():
                mask[node] = 1.0
        
        # If all nodes are visited, allow returning to depot
        if len(self.visited) == self.problem_size - 1:
            mask[0] = 1.0  # Depot
        
        # Create state dictionary
        state = {
            'current_node': self.current_node,
            'current_load': self.current_load,
            'visited': list(self.visited),
            'mask': mask,
            'node_xy': self.instance.node_xy,
            'node_demand': self.instance.node_demand,
            'node_draft_limit': self.instance.node_draft_limit,
            'step': self.step_count
        }
        
        return state
    
    def get_solution(self) -> TSPDLSolution:
        """
        Get the current solution.
        
        Returns:
            solution: Current solution
        """
        return TSPDLSolution(self.instance, self.tour)
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Args:
            mode: Rendering mode
            
        Returns:
            img: Rendered image (if mode is 'rgb_array')
        """
        if mode == 'human':
            # Print current state
            print(f"Step: {self.step_count}")
            print(f"Tour: {self.tour}")
            print(f"Current node: {self.current_node}")
            print(f"Current load: {self.current_load}")
            print(f"Visited nodes: {self.visited}")
            print(f"Total reward: {self.total_reward}")
            print(f"Done: {self.done}")
            
            return None
        
        elif mode == 'rgb_array':
            # TODO: Implement rendering as RGB array
            return np.zeros((10, 10, 3))
        
        else:
            raise ValueError(f"Unknown rendering mode: {mode}")
    
    def to_dgl(self) -> dgl.DGLGraph:
        """
        Convert the environment to a DGL graph.
        
        Returns:
            g: DGL graph
        """
        # Create a complete graph
        g = dgl.graph(([], []), num_nodes=self.problem_size)
        
        # Add self-loops
        g = dgl.add_self_loop(g)
        
        # Add edges between all nodes
        src, dst = [], []
        for i in range(self.problem_size):
            for j in range(self.problem_size):
                if i != j:
                    src.append(i)
                    dst.append(j)
        
        g.add_edges(src, dst)
        
        # Add node features
        g.ndata['coords'] = self.instance.node_xy
        g.ndata['demand'] = self.instance.node_demand.unsqueeze(1)
        g.ndata['draft_limit'] = self.instance.node_draft_limit.unsqueeze(1)
        
        # Add node features indicating whether the node is visited
        visited = torch.zeros(self.problem_size, 1, device=self.device)
        for node in self.visited:
            visited[node] = 1.0
        g.ndata['visited'] = visited
        
        # Add node feature indicating whether the node is the current node
        current = torch.zeros(self.problem_size, 1, device=self.device)
        current[self.current_node] = 1.0
        g.ndata['current'] = current
        
        # Add edge features
        edge_feats = []
        for i, j in zip(g.edges()[0], g.edges()[1]):
            i, j = i.item(), j.item()
            
            # Calculate distance
            x1, y1 = self.instance.node_xy[i]
            x2, y2 = self.instance.node_xy[j]
            dist = torch.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2).item()
            
            # Check if edge is in tour
            in_tour = 0.0
            for k in range(len(self.tour) - 1):
                if self.tour[k] == i and self.tour[k+1] == j:
                    in_tour = 1.0
                    break
            
            # Check if draft limit is respected
            load_after_i = 0.0
            for node in self.tour:
                if node == i:
                    break
                load_after_i += self.instance.node_demand[node].item()
            
            draft_limit_respected = 1.0 if load_after_i <= self.instance.node_draft_limit[j].item() else 0.0
            
            # Edge features
            edge_feat = [dist, in_tour, draft_limit_respected]
            edge_feats.append(edge_feat)
        
        g.edata['features'] = torch.tensor(edge_feats, device=self.device)
        
        return g


class TSPDLBatchEnv:
    """
    Batch environment for TSPDL.
    
    This environment allows an agent to solve multiple TSPDL instances in parallel.
    """
    
    def __init__(
        self,
        instances: List[TSPDLInstance],
        device: torch.device = None
    ):
        """
        Initialize the batch environment.
        
        Args:
            instances: List of TSPDL instances
            device: Device to use for tensor operations
        """
        self.instances = instances
        self.device = device if device is not None else torch.device('cpu')
        
        # Create environments for each instance
        self.envs = [TSPDLEnv(instance, device) for instance in instances]
        
        # Problem size (assuming all instances have the same size)
        self.problem_size = instances[0].problem_size
        
        # Batch size
        self.batch_size = len(instances)
        
        # Reset environments
        self.reset()
    
    def reset(self, start_node: int = 0) -> List[Dict]:
        """
        Reset all environments.
        
        Args:
            start_node: Starting node (depot)
            
        Returns:
            states: Initial states
        """
        self.states = [env.reset(start_node) for env in self.envs]
        self.dones = [False] * self.batch_size
        
        return self.states
    
    def step(self, actions: List[int]) -> Tuple[List[Dict], List[float], List[bool], List[Dict]]:
        """
        Take a step in all environments.
        
        Args:
            actions: Actions for each environment
            
        Returns:
            states: New states
            rewards: Rewards
            dones: Whether the episodes are done
            infos: Additional information
        """
        assert len(actions) == self.batch_size, "Number of actions must match batch size"
        
        states, rewards, dones, infos = [], [], [], []
        
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            if self.dones[i]:
                # If environment is already done, return the same state
                states.append(self.states[i])
                rewards.append(0.0)
                dones.append(True)
                infos.append({'valid': False})
            else:
                # Take a step in the environment
                state, reward, done, info = env.step(action)
                
                states.append(state)
                rewards.append(reward)
                dones.append(done)
                infos.append(info)
                
                # Update done flag
                self.dones[i] = done
        
        # Update states
        self.states = states
        
        return states, rewards, dones, infos
    
    def get_solutions(self) -> List[TSPDLSolution]:
        """
        Get the current solutions.
        
        Returns:
            solutions: Current solutions
        """
        return [env.get_solution() for env in self.envs]
    
    def to_dgl(self) -> List[dgl.DGLGraph]:
        """
        Convert the environments to DGL graphs.
        
        Returns:
            graphs: DGL graphs
        """
        return [env.to_dgl() for env in self.envs]

"""
Reinforcement learning for the Traveling Salesperson Problem with Draft Limits (TSPDL).

This module provides reinforcement learning algorithms for solving TSPDL.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import dgl
from typing import List, Tuple, Dict, Optional, Union, Any

from .tspdl import TSPDLInstance, TSPDLSolution
from .tspdl_env import TSPDLEnv, TSPDLBatchEnv
from .tspdl_models import TSPDLRLModel
from .tspdl_algorithms import nearest_neighbor_tspdl, insertion_tspdl, local_search_tspdl


class TSPDLRLAgent:
    """
    Reinforcement learning agent for TSPDL.
    
    This agent uses a GNN model to solve TSPDL instances.
    """
    
    def __init__(
        self,
        model: TSPDLRLModel,
        device: torch.device = None,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        entropy_weight: float = 0.01
    ):
        """
        Initialize the agent.
        
        Args:
            model: GNN model
            device: Device to use for tensor operations
            learning_rate: Learning rate
            gamma: Discount factor
            entropy_weight: Weight for entropy regularization
        """
        self.model = model
        self.device = device if device is not None else torch.device('cpu')
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Hyperparameters
        self.gamma = gamma
        self.entropy_weight = entropy_weight
        
        # Training history
        self.history = {
            'loss': [],
            'reward': [],
            'entropy': [],
            'policy_loss': [],
            'value_loss': []
        }
    
    def select_action(
        self,
        state: Dict,
        env: TSPDLEnv,
        deterministic: bool = False
    ) -> int:
        """
        Select an action based on the current state.
        
        Args:
            state: Current state
            env: Environment
            deterministic: Whether to select the action deterministically
            
        Returns:
            action: Selected action
        """
        # Convert environment to DGL graph
        g = env.to_dgl()
        
        # Get node features
        node_feats = torch.cat([
            g.ndata['coords'],
            g.ndata['demand'],
            g.ndata['draft_limit'],
            g.ndata['visited']
        ], dim=1)
        
        # Get edge features
        edge_feats = g.edata['features']
        
        # Get action mask
        mask = state['mask']
        
        # Predict edge scores
        edge_scores = self.model.predict_edge_scores(g, edge_feats, node_feats)
        
        # Predict node scores
        node_scores = self.model.predict_node_scores(g, edge_feats, node_feats).squeeze(1)
        
        # Apply mask
        node_scores = node_scores * mask
        
        # Select action
        if deterministic:
            action = torch.argmax(node_scores).item()
        else:
            # Add small epsilon to avoid zero probabilities
            node_probs = torch.softmax(node_scores + 1e-10, dim=0)
            
            # Sample action
            action = torch.multinomial(node_probs, 1).item()
        
        return action
    
    def train_step(
        self,
        states: List[Dict],
        actions: List[int],
        rewards: List[float],
        next_states: List[Dict],
        dones: List[bool],
        envs: List[TSPDLEnv]
    ) -> Dict:
        """
        Perform a training step.
        
        Args:
            states: Current states
            actions: Actions taken
            rewards: Rewards received
            next_states: Next states
            dones: Whether the episodes are done
            envs: Environments
            
        Returns:
            info: Training information
        """
        # Convert to tensors
        rewards_tensor = torch.tensor(rewards, device=self.device)
        actions_tensor = torch.tensor(actions, device=self.device)
        dones_tensor = torch.tensor(dones, device=self.device, dtype=torch.float32)
        
        # Convert environments to DGL graphs
        graphs = [env.to_dgl() for env in envs]
        
        # Get node features
        node_feats_list = []
        for g in graphs:
            node_feats = torch.cat([
                g.ndata['coords'],
                g.ndata['demand'],
                g.ndata['draft_limit'],
                g.ndata['visited']
            ], dim=1)
            node_feats_list.append(node_feats)
        
        # Get edge features
        edge_feats_list = [g.edata['features'] for g in graphs]
        
        # Get action masks
        masks = [state['mask'] for state in states]
        masks_tensor = torch.stack(masks)
        
        # Predict edge scores and node scores
        edge_scores_list = []
        node_scores_list = []
        
        for g, edge_feats, node_feats in zip(graphs, edge_feats_list, node_feats_list):
            edge_scores = self.model.predict_edge_scores(g, edge_feats, node_feats)
            node_scores = self.model.predict_node_scores(g, edge_feats, node_feats).squeeze(1)
            
            edge_scores_list.append(edge_scores)
            node_scores_list.append(node_scores)
        
        node_scores_tensor = torch.stack(node_scores_list)
        
        # Apply masks
        node_scores_tensor = node_scores_tensor * masks_tensor
        
        # Calculate action probabilities
        node_probs = torch.softmax(node_scores_tensor + 1e-10, dim=1)
        
        # Calculate log probabilities
        log_probs = torch.log(node_probs + 1e-10)
        
        # Get log probabilities of selected actions
        action_log_probs = log_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        
        # Calculate entropy
        entropy = -torch.sum(node_probs * log_probs, dim=1).mean()
        
        # Calculate policy loss
        policy_loss = -action_log_probs.mean()
        
        # Calculate total loss
        loss = policy_loss - self.entropy_weight * entropy
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update history
        self.history['loss'].append(loss.item())
        self.history['reward'].append(rewards_tensor.mean().item())
        self.history['entropy'].append(entropy.item())
        self.history['policy_loss'].append(policy_loss.item())
        
        # Return training information
        info = {
            'loss': loss.item(),
            'reward': rewards_tensor.mean().item(),
            'entropy': entropy.item(),
            'policy_loss': policy_loss.item()
        }
        
        return info
    
    def train(
        self,
        env: TSPDLBatchEnv,
        num_episodes: int,
        max_steps: int = 1000,
        eval_interval: int = 10,
        save_path: Optional[str] = None
    ) -> Dict:
        """
        Train the agent.
        
        Args:
            env: Batch environment
            num_episodes: Number of episodes
            max_steps: Maximum number of steps per episode
            eval_interval: Evaluation interval
            save_path: Path to save the model
            
        Returns:
            history: Training history
        """
        # Training loop
        for episode in range(num_episodes):
            # Reset environment
            states = env.reset()
            
            # Initialize episode variables
            episode_rewards = []
            episode_actions = []
            episode_states = []
            episode_dones = []
            
            # Episode loop
            for step in range(max_steps):
                # Select actions
                actions = [
                    self.select_action(state, env.envs[i])
                    for i, state in enumerate(states)
                ]
                
                # Take a step in the environment
                next_states, rewards, dones, infos = env.step(actions)
                
                # Store episode data
                episode_rewards.append(rewards)
                episode_actions.append(actions)
                episode_states.append(states)
                episode_dones.append(dones)
                
                # Update states
                states = next_states
                
                # Check if all episodes are done
                if all(dones):
                    break
            
            # Calculate returns
            returns = []
            for t in range(len(episode_rewards)):
                ret = 0
                for k in range(t, len(episode_rewards)):
                    ret += self.gamma ** (k - t) * episode_rewards[k]
                returns.append(ret)
            
            # Flatten episode data
            flat_states = [state for states_t in episode_states for state in states_t]
            flat_actions = [action for actions_t in episode_actions for action in actions_t]
            flat_returns = [ret for returns_t in returns for ret in returns_t]
            flat_next_states = [state for states_t in episode_states[1:] + [next_states] for state in states_t]
            flat_dones = [done for dones_t in episode_dones for done in dones_t]
            flat_envs = [env.envs[i] for _ in range(len(episode_states)) for i in range(env.batch_size)]
            
            # Train on episode data
            train_info = self.train_step(
                flat_states, flat_actions, flat_returns, flat_next_states, flat_dones, flat_envs
            )
            
            # Print episode information
            if (episode + 1) % eval_interval == 0:
                print(f"Episode {episode + 1}/{num_episodes}")
                print(f"  Loss: {train_info['loss']:.4f}")
                print(f"  Reward: {train_info['reward']:.4f}")
                print(f"  Entropy: {train_info['entropy']:.4f}")
                print(f"  Policy Loss: {train_info['policy_loss']:.4f}")
                
                # Evaluate agent
                eval_rewards = self.evaluate(env, 5)
                print(f"  Eval Reward: {eval_rewards:.4f}")
                
                # Save model
                if save_path is not None:
                    self.save(save_path)
        
        return self.history
    
    def evaluate(
        self,
        env: TSPDLBatchEnv,
        num_episodes: int
    ) -> float:
        """
        Evaluate the agent.
        
        Args:
            env: Batch environment
            num_episodes: Number of episodes
            
        Returns:
            avg_reward: Average reward
        """
        total_rewards = []
        
        for _ in range(num_episodes):
            # Reset environment
            states = env.reset()
            
            # Initialize episode variables
            episode_rewards = []
            
            # Episode loop
            done = False
            while not done:
                # Select actions
                actions = [
                    self.select_action(state, env.envs[i], deterministic=True)
                    for i, state in enumerate(states)
                ]
                
                # Take a step in the environment
                next_states, rewards, dones, infos = env.step(actions)
                
                # Store rewards
                episode_rewards.append(rewards)
                
                # Update states
                states = next_states
                
                # Check if all episodes are done
                done = all(dones)
            
            # Calculate total reward
            total_reward = sum(sum(rewards) for rewards in episode_rewards)
            total_rewards.append(total_reward)
        
        # Calculate average reward
        avg_reward = sum(total_rewards) / len(total_rewards)
        
        return avg_reward
    
    def solve(
        self,
        instance: TSPDLInstance,
        max_steps: int = 1000,
        deterministic: bool = True
    ) -> TSPDLSolution:
        """
        Solve a TSPDL instance.
        
        Args:
            instance: TSPDL instance
            max_steps: Maximum number of steps
            deterministic: Whether to select actions deterministically
            
        Returns:
            solution: Solution
        """
        # Create environment
        env = TSPDLEnv(instance, self.device)
        
        # Reset environment
        state = env.reset()
        
        # Episode loop
        for _ in range(max_steps):
            # Select action
            action = self.select_action(state, env, deterministic)
            
            # Take a step in the environment
            next_state, reward, done, info = env.step(action)
            
            # Update state
            state = next_state
            
            # Check if episode is done
            if done:
                break
        
        # Get solution
        solution = env.get_solution()
        
        return solution
    
    def save(self, path: str):
        """
        Save the agent.
        
        Args:
            path: Path to save the agent
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, path)
    
    def load(self, path: str):
        """
        Load the agent.
        
        Args:
            path: Path to load the agent
        """
        # Load model
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load history
        self.history = checkpoint['history']


def train_tspdl_rl_agent(
    train_instances: List[TSPDLInstance],
    val_instances: List[TSPDLInstance],
    embed_dim: int = 128,
    n_layers: int = 3,
    n_heads: int = 8,
    learning_rate: float = 1e-4,
    gamma: float = 0.99,
    entropy_weight: float = 0.01,
    num_episodes: int = 1000,
    max_steps: int = 1000,
    eval_interval: int = 10,
    save_path: Optional[str] = None,
    device: torch.device = None
) -> TSPDLRLAgent:
    """
    Train a TSPDL RL agent.
    
    Args:
        train_instances: Training instances
        val_instances: Validation instances
        embed_dim: Embedding dimension
        n_layers: Number of GNN layers
        n_heads: Number of attention heads
        learning_rate: Learning rate
        gamma: Discount factor
        entropy_weight: Weight for entropy regularization
        num_episodes: Number of episodes
        max_steps: Maximum number of steps per episode
        eval_interval: Evaluation interval
        save_path: Path to save the model
        device: Device to use for tensor operations
        
    Returns:
        agent: Trained agent
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = TSPDLRLModel(
        in_dim=3,  # [distance, in_tour, draft_limit_respected]
        embed_dim=embed_dim,
        n_layers=n_layers,
        n_heads=n_heads
    )
    
    # Create agent
    agent = TSPDLRLAgent(
        model=model,
        device=device,
        learning_rate=learning_rate,
        gamma=gamma,
        entropy_weight=entropy_weight
    )
    
    # Create training environment
    train_env = TSPDLBatchEnv(train_instances, device)
    
    # Create validation environment
    val_env = TSPDLBatchEnv(val_instances, device)
    
    # Train agent
    agent.train(
        env=train_env,
        num_episodes=num_episodes,
        max_steps=max_steps,
        eval_interval=eval_interval,
        save_path=save_path
    )
    
    return agent
